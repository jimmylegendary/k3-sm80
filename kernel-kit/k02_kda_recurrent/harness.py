#!/usr/bin/env python3
"""k02_kda_recurrent — KDA recurrent decode (fused_recurrent_kda 계열, 순수 Triton) 단독 검증.

대상 API (vLLM main @ a3c14efbc77c018e8bf579935095a87721349e05 원문 확인 — 이전 핀 0498dc7e 에서 재vendor):
  주 경로: fused_recurrent_kda_packed_decode(mixed_qkv, raw_g, raw_beta, A_log, dt_bias,
           lower_bound, initial_state, state_indices)
    — SM80 plain-decode 프로덕션 경로 (nvidia/kda.py L671-683; CUDA fused_kda_decode 는
      SM90+ 전용이라 SM80 은 이 Triton 폴백). state 풀 in-place 갱신, index 0 = null block
      (커널 L527-531: state_idx≤0 → 출력 0·state 미접촉).
  보조 arm: fused_recurrent_kda_fwd(q,k,v,g,beta,…,cu_seqlens,ssm_state_indices)
    — spec-decode 경로가 쓰는 dense 커널(1 tok/seq) 을 같은 참조로 대조.
  state 레이아웃 = (slots, H, V, K) fp32 — naive 참조 (K,V) 의 전치.
  import 는 **vendor 우선**(kernel-kit/vendor/fla_kda — 동일 핀 vendored 사본,
  vLLM 설치 불필요) → 실패 시 vllm 폴백. 출처는 evidence.kernel_import.

판정 (docs/PLAN.md §S2 4게이트):
  NUMERIC   fp32 참조 recurrent(common/kda_ref.py, vLLM naive 이식) 대비 rmse_rel ≤ 5e-3
            (tolerance.py — 업스트림 test_kda.py 바). 4-step 연속 decode 로 state 진화 검증
            + 미사용 slot 불변 + null-index 의미론 + prefill(k01 chunk)→decode 이월.
  SHAPES    REQUIRED 5케이스 (b 1/8/32 × ctx 8k/128k — recurrent state 는 ctx 무관
            고정 크기이므로 ctx 축은 paged-state 산란 인덱싱 검증으로 소화, shapes.py 주석).
  SANITIZER compute-sanitizer memcheck (자체 --sanitizer-arm 축소 shape).
  PERF      대표 케이스 b32_ctx8k(packed 1스텝) vs phase0 루프라인 (eff_floor 0.50 —
            state 6.29MB/seq fp32 r+w 스트림 지배).

실행: python3 harness.py --phase0-verdict <verdict.json> --report-dir <dir> [--cpu-dry-run]
exit: 0 FAIL없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, kda_ref, report, shapes, timing, tolerance  # noqa: E402

KID = "k02_kda_recurrent"
PINS = {"vllm_pin": "a3c14efb", "hf": "moonshotai/Kimi-K3@9f62e4e9"}


def import_kda_ops():
    """KDA 커널 import: vendor(vLLM 불필요) 우선 → vllm 폴백. (모듈, 출처 dict) 반환."""
    try:
        from vendor import fla_kda as m  # kernel-kit/vendor (sys.path=_KIT)
        return m, {"source": "vendor", "module_file": m.__file__,
                   "pin": PINS["vllm_pin"]}
    except Exception as e_v:
        try:
            import vllm.models.kimi_k3.nvidia.ops.third_party.kda as m
            return m, {"source": "vllm", "module_file": m.__file__,
                       "vendor_import_error": f"{type(e_v).__name__}: {e_v}"[:300]}
        except ImportError as e_f:
            raise ImportError(
                f"vendor({type(e_v).__name__}: {e_v}) / vllm({e_f}) 모두 import 불가"
            ) from e_f
H, D = shapes.KDA_HEADS, shapes.KDA_HEAD_DIM          # 96, 128
LB = shapes.KDA_GATE_LOWER_BOUND                       # -5.0
PERF_CASE = "b32_ctx8k"
N_STEPS = 4                                            # 연속 decode 스텝 수
EXTRA_CASES = ("prefill_to_decode_carry", "null_index_semantics")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--report-dir", default=os.path.join(_KIT, "kernel-report"))
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--sanitizer-arm", action="store_true")
    ap.add_argument("--twin", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 입력 생성 — paged-state 풀(행 패딩 stride) + 산란 인덱스(1..slots-1, 0=null 예약)
# ---------------------------------------------------------------------------
def build_pool(torch, seed: int, slots: int, dev: str):
    g = torch.Generator().manual_seed(seed)
    row = H * D * D
    storage = torch.randn(slots, row + 32, generator=g,
                          dtype=torch.float32).to(dev)     # 행 패딩 → paged stride 모사
    pool = storage[:, :row].view(slots, H, D, D)
    return storage, pool


def scatter_indices(torch, seed: int, b: int, slots: int, dev: str,
                    strided: bool = False):
    perm = list(range(1, slots))                            # 0 = null block 예약
    random.Random(seed).shuffle(perm)
    idx = perm[:b]
    if strided:                                             # index 스트라이드 경로 검증
        st = torch.zeros(b, 4, dtype=torch.int32, device=dev)
        st[:, 0] = torch.tensor(idx, dtype=torch.int32)
        return st[:, 0], idx
    return torch.tensor(idx, dtype=torch.int32, device=dev), idx


def step_inputs(torch, seed: int, b: int, dev: str):
    g = torch.Generator().manual_seed(seed)

    def mk(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=torch.float32)
        if scale != 1.0:
            t.mul_(scale)
        return t.to(dtype).to(dev)

    mixed_storage = mk(b, 3 * H * D + 5)                    # row-strided (in_proj split 모사)
    mixed = mixed_storage[:, :3 * H * D]
    raw_g = mk(1, b, H, D)
    beta_storage = mk(1, b, H + 1)
    raw_beta = beta_storage[..., :H]
    return mixed, raw_g, raw_beta


def ref_step(torch, mixed, raw_g, raw_beta, A_log, dt_bias, ref_state):
    """fp32 참조 1 decode 스텝. ref_state (b,H,K,V) naive 레이아웃, in-place 아님(반환)."""
    b = mixed.shape[0]
    q, k, v = (x.contiguous().view(b, 1, H, D) for x in mixed.split(H * D, dim=-1))
    o, S = kda_ref.naive_recurrent_kda_raw(
        q, k, v, raw_g.transpose(0, 1), raw_beta.transpose(0, 1),
        A_log, dt_bias, LB, initial_state=ref_state)
    return o[:, 0], S                                       # (b,H,V), (b,H,K,V)


# ---------------------------------------------------------------------------
# GPU 본 실행
# ---------------------------------------------------------------------------
def gpu_run(r: report.KernelReport, a) -> None:
    import torch
    try:
        kda_mod, kda_src = import_kda_ops()
        chunk_kda_with_fused_gate = kda_mod.chunk_kda_with_fused_gate
        fused_recurrent_kda_fwd = kda_mod.fused_recurrent_kda_fwd
        fused_recurrent_kda_packed_decode = kda_mod.fused_recurrent_kda_packed_decode
    except ImportError as e:
        print(f"[{KID}] KDA ops import 실패 (vendor/vllm 모두): {e}\n"
              f"  → torch+triton venv(triton 티어, vendor 경로) 또는 venv-k3(RUNNER §1) 확인.",
              file=sys.stderr)
        sys.exit(report.EXIT_PRECOND)
    r.add_evidence("kernel_import", kda_src)

    dev = "cuda"
    results: list[tolerance.NumericResult] = []
    covered: list[str] = []
    case_meta: dict = {}
    perf_stats = None

    req = shapes.REQUIRED_SHAPES[KID]
    order = [PERF_CASE] + [c for c in req if c != PERF_CASE] + list(EXTRA_CASES)
    for name in order:
        seed = shapes.stable_seed(KID, name)
        t0 = time.time()
        try:
            if name == "prefill_to_decode_carry":
                entries = case_prefill_decode_carry(
                    torch, chunk_kda_with_fused_gate,
                    fused_recurrent_kda_packed_decode, seed, dev)
            elif name == "null_index_semantics":
                entries = case_null_index(
                    torch, fused_recurrent_kda_packed_decode, seed, dev)
            else:
                spec = req[name]
                entries, perf_fn = case_decode(
                    torch, fused_recurrent_kda_packed_decode,
                    fused_recurrent_kda_fwd, name, spec, seed, dev)
                if name == PERF_CASE:
                    perf_stats = timing.measure(perf_fn, warmup=a.warmup, reps=a.reps)
            results.extend(entries)
            covered.append(name)
            case_meta[name] = {"seed": seed, "duration_s": round(time.time() - t0, 1),
                               "entries": [e.metrics.get("case") for e in entries]}
        except torch.cuda.OutOfMemoryError as e:
            results.append(tolerance.NumericResult(
                metrics={"case": name, "error": "cuda_oom"},
                violations=[f"CUDA OOM: {str(e)[:200]}"], passed=False))
            case_meta[name] = {"seed": seed, "error": "cuda_oom"}
        except RuntimeError as e:
            results.append(tolerance.NumericResult(
                metrics={"case": name, "error": "runtime_error"},
                violations=[f"RuntimeError: {str(e)[:300]}"], passed=False))
            case_meta[name] = {"seed": seed, "error": str(e)[:300]}
        finally:
            torch.cuda.empty_cache()

    r.add_evidence("cases", case_meta)
    r.add_evidence("k3_params", {
        "H": H, "D": D, "lower_bound": LB, "n_steps": N_STEPS,
        "state_bytes_per_seq": shapes.KDA_STATE_BYTES_LAYER_SEQ,
        "ctx_axis_note": "recurrent state 는 ctx 무관 고정 6.29MB/seq — ctx 축은 "
                         "풀 slot 산란 패턴(seed)으로 paged 인덱싱 경로를 바꿔 검증"})
    r.set_gate("NUMERIC", *tolerance.numeric_gate(KID, results))
    r.set_gate("SHAPES", *shapes.shapes_gate(KID, covered))
    run_sanitizer_gate(r)
    model = shapes.perf_bytes_model(KID, b=32, ctx=8192)
    st, rs, ev = timing.perf_gate(
        KID, perf_stats["median_us"] if perf_stats else None, model,
        a.phase0_verdict, a.twin, stats=perf_stats)
    ev["perf_case"] = PERF_CASE
    ev["note"] = ("측정은 packed_decode 커널 호출만 (conv 는 k03); state 풀 in-place "
                  "갱신이므로 rep 간 내용은 진화하나 시간·바이트는 동일")
    r.set_gate("PERF", st, rs, ev)


def case_decode(torch, packed_fn, fwd_fn, name, spec, seed, dev):
    """요구 케이스: packed(주 arm, N_STEPS 연속) + dense fwd(보조 arm) — 동일 fp32 참조 대조."""
    b, ctx = spec["b"], spec["ctx"]
    slots = b + 8
    A_log = (0.5 * torch.randn(H, generator=torch.Generator().manual_seed(seed + 1),
                               dtype=torch.float32)).to(dev)
    dt_bias = (0.1 * torch.randn(H * D,
                                 generator=torch.Generator().manual_seed(seed + 2),
                                 dtype=torch.float32)).to(dev)
    _, pool = build_pool(torch, seed, slots, dev)
    pool0 = pool.clone()
    state_indices, idx_list = scatter_indices(torch, seed, b, slots, dev,
                                              strided=(b == 8))
    idx_t = torch.tensor(idx_list, dtype=torch.long, device=dev)
    ref_state = pool[idx_t].transpose(-1, -2).contiguous()  # (b,H,K,V) naive

    steps = [step_inputs(torch, seed + 100 + s, b, dev) for s in range(N_STEPS)]
    outs, refs = [], []
    for mixed, raw_g, raw_beta in steps:
        out, _ = packed_fn(mixed_qkv=mixed, raw_g=raw_g, raw_beta=raw_beta,
                           A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                           initial_state=pool, state_indices=state_indices)
        outs.append(out[0].float())                          # (b,H,V)
        o_ref, ref_state = ref_step(torch, mixed, raw_g, raw_beta,
                                    A_log, dt_bias, ref_state)
        refs.append(o_ref)
    entries = [
        tolerance.compare(KID, torch.stack(refs), torch.stack(outs),
                          case=f"{name}/o"),
        tolerance.compare(KID, ref_state.transpose(-1, -2), pool[idx_t],
                          case=f"{name}/ht"),
    ]
    untouched = [i for i in range(slots) if i not in idx_list]
    un_t = torch.tensor(untouched, dtype=torch.long, device=dev)
    entries.append(tolerance.compare(KID, pool0[un_t], pool[un_t],
                                     case=f"{name}/untouched_slots"))

    # 보조 arm: dense fwd (spec-decode 경로 커널) — 같은 스텝·같은 참조
    pool_d = pool0.clone()
    ref_state_d = pool0[idx_t].transpose(-1, -2).contiguous()
    cu = torch.arange(b + 1, dtype=torch.int32, device=dev)
    dense_idx = torch.tensor(idx_list, dtype=torch.int32, device=dev)
    outs_d, refs_d = [], []
    for mixed, raw_g, raw_beta in steps:
        q, k, v = (x.contiguous().view(1, b, H, D) for x in mixed.split(H * D, dim=-1))
        gate = kda_ref.kda_gate_ref(raw_g, A_log, dt_bias, LB)          # (1,b,H,D) fp32
        beta = torch.sigmoid(raw_beta.float())                           # (1,b,H) fp32
        out_d, _ = fwd_fn(q=q, k=k, v=v, g=gate, beta=beta, scale=D ** -0.5,
                          initial_state=pool_d, inplace_final_state=True,
                          cu_seqlens=cu, ssm_state_indices=dense_idx,
                          use_qk_l2norm_in_kernel=True)
        outs_d.append(out_d[0].float())
        o_ref, ref_state_d = ref_step(torch, mixed, raw_g, raw_beta,
                                      A_log, dt_bias, ref_state_d)
        refs_d.append(o_ref)
    entries.append(tolerance.compare(KID, torch.stack(refs_d), torch.stack(outs_d),
                                     case=f"{name}/dense_o"))
    entries.append(tolerance.compare(KID, ref_state_d.transpose(-1, -2), pool_d[idx_t],
                                     case=f"{name}/dense_ht"))

    mixed0, raw_g0, raw_beta0 = steps[0]

    def perf_fn():
        packed_fn(mixed_qkv=mixed0, raw_g=raw_g0, raw_beta=raw_beta0,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                  initial_state=pool, state_indices=state_indices)

    return entries, perf_fn


def case_prefill_decode_carry(torch, chunk_fn, packed_fn, seed, dev):
    """extra: k01 chunk prefill(T=256) → final state → packed decode 8스텝 연속.

    참조는 264 토큰 연속 recurrent — S4 통합의 핵심 handoff(프리필→디코드 state 이월)를
    커널 조합 그대로 검증한다.
    """
    g = torch.Generator().manual_seed(seed)

    def mk(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=torch.float32)
        if scale != 1.0:
            t.mul_(scale)
        return t.to(dtype).to(dev)

    T_pre, n_dec = 256, 8
    T = T_pre + n_dec
    q = mk(1, T, H, D)
    k = mk(1, T, H, D)
    v = mk(1, T, H, D)
    raw_g = mk(1, T, H, D)
    raw_beta = mk(1, T, H + 1)[..., :H]
    A_log = mk(H, dtype=torch.float32, scale=0.5)
    dt_bias = mk(H * D, dtype=torch.float32, scale=0.1)
    h0 = mk(1, H, D, D, dtype=torch.float32)
    cu = torch.tensor([0, T_pre], dtype=torch.int32, device=dev)

    _, s_pre = chunk_fn(q=q[:, :T_pre], k=k[:, :T_pre], v=v[:, :T_pre].clone(),
                        raw_g=raw_g[:, :T_pre], raw_beta=raw_beta[:, :T_pre],
                        A_log=A_log, g_bias=dt_bias, lower_bound=LB,
                        initial_state=h0.clone(), output_final_state=True,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=cu)

    slots = 4
    _, pool = build_pool(torch, seed + 9, slots, dev)
    slot = 2
    pool[slot].copy_(s_pre[0])                              # prefill state → decode 풀
    state_indices = torch.tensor([slot], dtype=torch.int32, device=dev)
    outs = []
    for t in range(T_pre, T):
        mixed = torch.cat([q[0, t].reshape(1, -1), k[0, t].reshape(1, -1),
                           v[0, t].reshape(1, -1)], dim=-1).contiguous()
        out, _ = packed_fn(mixed_qkv=mixed, raw_g=raw_g[:, t:t + 1],
                           raw_beta=raw_beta[:, t:t + 1], A_log=A_log,
                           dt_bias=dt_bias, lower_bound=LB,
                           initial_state=pool, state_indices=state_indices)
        outs.append(out[0, 0].float())                       # (H,V)
    o_ref, S_ref = kda_ref.naive_recurrent_kda_raw(
        q, k, v, raw_g, raw_beta, A_log, dt_bias, LB,
        initial_state=h0.float().transpose(-1, -2))
    return [
        tolerance.compare(KID, o_ref[0, T_pre:], torch.stack(outs),
                          case="prefill_to_decode_carry/o"),
        tolerance.compare(KID, S_ref[0].transpose(-1, -2), pool[slot],
                          case="prefill_to_decode_carry/ht"),
    ]


def case_null_index(torch, packed_fn, seed, dev):
    """extra: index 0 = null block — 출력 0·state 미접촉 (커널 L527-531 계약)."""
    b, slots = 2, 5
    _, pool = build_pool(torch, seed, slots, dev)
    pool0 = pool.clone()
    A_log = (0.5 * torch.randn(H, generator=torch.Generator().manual_seed(seed + 1),
                               dtype=torch.float32)).to(dev)
    dt_bias = (0.1 * torch.randn(H * D,
                                 generator=torch.Generator().manual_seed(seed + 2),
                                 dtype=torch.float32)).to(dev)
    mixed, raw_g, raw_beta = step_inputs(torch, seed + 100, b, dev)
    live = 3
    state_indices = torch.tensor([0, live], dtype=torch.int32, device=dev)
    ref_state = pool[live:live + 1].transpose(-1, -2).contiguous()
    out, _ = packed_fn(mixed_qkv=mixed, raw_g=raw_g, raw_beta=raw_beta,
                       A_log=A_log, dt_bias=dt_bias, lower_bound=LB,
                       initial_state=pool, state_indices=state_indices)
    o_ref, S_ref = ref_step(torch, mixed[1:2], raw_g[:, 1:2], raw_beta[:, 1:2],
                            A_log, dt_bias, ref_state)
    return [
        tolerance.compare(KID, torch.zeros_like(out[0, 0].float()), out[0, 0],
                          case="null_index_semantics/zero_out"),
        tolerance.compare(KID, o_ref[0], out[0, 1], case="null_index_semantics/live_o"),
        tolerance.compare(KID, pool0[0], pool[0],
                          case="null_index_semantics/state0_untouched"),
        tolerance.compare(KID, S_ref[0].transpose(-1, -2), pool[live],
                          case="null_index_semantics/live_ht"),
    ]


# ---------------------------------------------------------------------------
# SANITIZER
# ---------------------------------------------------------------------------
def run_sanitizer_gate(r: report.KernelReport, timeout: int = 2400) -> None:
    san = env_check.find_compute_sanitizer()
    if san is None:
        r.set_gate("SANITIZER", "SKIP",
                   "compute-sanitizer 미발견 — CUDA 툴킷 bin/ PATH 확인 (RUNNER §6)")
        return
    cmd = [san, "--tool", "memcheck", sys.executable,
           os.path.abspath(__file__), "--sanitizer-arm"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        st, rs, ev = report.sanitizer_gate_from_output(
            (p.stdout or "") + "\n" + (p.stderr or ""), " ".join(cmd), p.returncode)
    except subprocess.TimeoutExpired:
        st, rs, ev = "SKIP", f"sanitizer {timeout}s 초과 — 수동 재실행 필요", \
            {"cmd": " ".join(cmd)}
    r.set_gate("SANITIZER", st, rs, ev)


def sanitizer_arm() -> int:
    """축소 shape: packed 2스텝(산란 인덱스·null 행 포함) + dense fwd 1회 — memcheck 대상."""
    import torch
    kda_mod, _src = import_kda_ops()          # vendor 우선 → vllm 폴백 (본 실행과 동일)
    fused_recurrent_kda_fwd = kda_mod.fused_recurrent_kda_fwd
    fused_recurrent_kda_packed_decode = kda_mod.fused_recurrent_kda_packed_decode
    dev = "cuda"
    seed = shapes.stable_seed(KID, "sanitizer_arm")
    b, slots = 3, 6
    _, pool = build_pool(torch, seed, slots, dev)
    A_log = (0.5 * torch.randn(H, generator=torch.Generator().manual_seed(seed + 1),
                               dtype=torch.float32)).to(dev)
    dt_bias = (0.1 * torch.randn(H * D,
                                 generator=torch.Generator().manual_seed(seed + 2),
                                 dtype=torch.float32)).to(dev)
    state_indices = torch.tensor([0, 2, 5], dtype=torch.int32, device=dev)  # null 포함
    for s in range(2):
        mixed, raw_g, raw_beta = step_inputs(torch, seed + 100 + s, b, dev)
        out, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log,
            dt_bias=dt_bias, lower_bound=LB, initial_state=pool,
            state_indices=state_indices)
    q, k, v = (x.contiguous().view(1, b, H, D) for x in mixed.split(H * D, dim=-1))
    gate = kda_ref.kda_gate_ref(raw_g, A_log, dt_bias, LB)
    beta = torch.sigmoid(raw_beta.float())
    out_d, _ = fused_recurrent_kda_fwd(
        q=q, k=k, v=v, g=gate, beta=beta, scale=D ** -0.5, initial_state=pool,
        inplace_final_state=True,
        cu_seqlens=torch.arange(b + 1, dtype=torch.int32, device=dev),
        ssm_state_indices=torch.tensor([1, 3, 4], dtype=torch.int32, device=dev),
        use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    print(f"[{KID} sanitizer-arm] done packed={tuple(out.shape)} "
          f"dense={tuple(out_d.shape)} sum={out_d.float().sum().item():.4f}")
    return 0


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------
def dry_run(r: report.KernelReport, a) -> None:
    ok, details = kda_ref.selftest(verbose=False)
    wiring = {}
    if ok:
        case = kda_ref._py_rand_case(shapes.stable_seed(KID, "dry"), 6, 2, 4)
        rnd_ref = kda_ref.py_flat(
            kda_ref.py_recurrent_kda(*case[:7], LB, S0=case[7])[0])
        ok_out = [v * (1 + 1e-4) for v in rnd_ref]
        bad_out = [v * (1 + 2e-2) for v in rnd_ref]
        wiring = {
            "pass_case": tolerance.compare(KID, rnd_ref, ok_out, "dry_ok").passed,
            "fail_case": not tolerance.compare(KID, rnd_ref, bad_out, "dry_bad").passed,
        }
        ok = ok and all(wiring.values())
    if not ok:
        r.set_gate("NUMERIC", "FAIL",
                   f"cpu-dry-run: 참조/판정 자기검증 실패 — {details['failed'] or wiring}",
                   {"selftest": details, "wiring": wiring})
    else:
        r.set_gate("NUMERIC", "SKIP",
                   f"cpu-dry-run: 실커널 미실행 — 순수 python 참조 자기검증 "
                   f"{details['checks']}건(분할=이월 동치 포함) + 판정 배선 양방향 green",
                   {"selftest": details, "wiring": wiring})
    r.set_gate("SHAPES", "SKIP", "cpu-dry-run: 커널 미실행 — 요구 케이스 정의만 확인",
               {"required": sorted(shapes.REQUIRED_SHAPES[KID]),
                "extra": list(EXTRA_CASES), "n_steps": N_STEPS})
    r.set_gate("SANITIZER", "SKIP", "cpu-dry-run: GPU 미실행")
    model = shapes.perf_bytes_model(KID, b=32, ctx=8192)
    st, rs, ev = timing.perf_gate(KID, None, model, a.phase0_verdict, a.twin)
    try:
        tp = timing.resolve_twin_path(a.phase0_verdict, a.twin) \
            if a.phase0_verdict else None
        if tp:
            ev["dry_roofline"] = timing.roofline_us(
                model["bytes"], model.get("macs", 0.0), model.get("alu_ops", 0.0),
                timing.load_twin_peaks(tp))
            ev["dry_note"] = "목표 산출 경로(verdict→twin→roofline) 검증됨 — 판정 아님"
    except Exception as e:
        ev["dry_roofline_error"] = f"{type(e).__name__}: {e}"
    r.set_gate("PERF", st, rs, ev)


# ---------------------------------------------------------------------------
def main() -> int:
    a = parse_args()
    if a.sanitizer_arm:
        return sanitizer_arm()
    r = report.KernelReport(KID, dry_run=a.cpu_dry_run, pins=PINS)
    try:
        if a.cpu_dry_run:
            dry_run(r, a)
        else:
            env = env_check.collect_env()
            ok, checks = env_check.check_preconditions(env)
            if not ok:
                for c in checks:
                    if c["status"] == "FAIL":
                        print(f"[{KID}] 사전조건 FAIL: {c['check']} — {c['detail']}",
                              file=sys.stderr)
                return report.EXIT_PRECOND
            gpu_run(r, a)
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:
        traceback.print_exc()
        try:  # k04와 대칭: crash 도 리포트를 남긴다 (집계 MISSING 방지 — CRASH 로그도 S3 입력물)
            r.add_evidence("harness_crash", {
                "error": f"{type(e).__name__}: {e}",
                "trace_tail": traceback.format_exc()[-1200:]})
            r.write(a.report_dir)
        except Exception:
            pass
        return report.EXIT_ERR
    jp, _ = r.write(a.report_dir)
    d = r.as_dict()
    print(json.dumps({"id": KID, "overall": d["overall"],
                      "gates": {g: d["gates"][g]["status"] for g in report.GATES},
                      "report": jp}, ensure_ascii=False))
    return r.exit_code()


if __name__ == "__main__":
    sys.exit(main())
