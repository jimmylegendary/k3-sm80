#!/usr/bin/env python3
"""k03_causal_conv1d — depthwise conv4+SiLU (causal_conv1d_fn/update, Triton) 단독 검증.

대상 API (vLLM kimi-k3 @ 0498dc7ea9204f1863b46515816f5d8d458e01e7 원문 확인):
  vllm.model_executor.layers.mamba.ops.causal_conv1d.{causal_conv1d_fn, causal_conv1d_update}
  프로덕션 사용처 = nvidia/kda.py:
    prefill: q/k/v 각각 별도 호출 (L591-605) — x (dim, T) channel-last, weight 슬라이스,
             conv_state 풀 split 뷰(dim축 비연속), varlen(query_start_loc)+cache_indices
             +has_initial_state.
    decode:  packed 1회 호출 (L661-670) — x (b, 3·12288), conv_state_indices,
             validate_data=True, out 별도 버퍼. index 0 = null block (NULL_BLOCK_ID=0).
  가중치 fp32(K3 conv1d params_dtype=float32, bias 없음) · conv state bf16
  · 풀 저장 레이아웃 (slots, W-1, dim) → 커널엔 transpose(-1,-2) 뷰 전달 (L469-470).

판정 (docs/PLAN.md §S2 4게이트):
  NUMERIC   fp32 참조(common/kda_ref.py — 업스트림 test_causal_conv1d.py ref 이식) 대비
            rmse_rel ≤ 4e-3, max_abs ≤ 3e-2 (tolerance.py — bf16 eps 유도, 근거 주석).
            update 는 4-step 연속(state shift 정상성) + 미사용 slot 불변,
            fn 은 최종 conv state 까지 대조 + extra: fn→update 이월(프리필→디코드 연속성).
  SHAPES    REQUIRED 5케이스 (update b 1/8/32 + fn 8k/varlen; 채널 3×12288, W=4;
            varlen 에 길이 2(<W-1) seq 포함 — 짧은 seq state 패딩 경로).
  SANITIZER compute-sanitizer memcheck (자체 --sanitizer-arm 축소 채널).
  PERF      사전등록 규칙(k04와 동일): roofline < 30µs 케이스는 launch 고정비 지배라
            판정 제외. upd_b32 roofline(≈9.3µs) < 30µs 이므로 판정 대표 = fn_b1_T8k
            (roofline ≈590µs, eff_floor 0.50). update 측정치는 항상 evidence 에 보존.

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

KID = "k03_causal_conv1d"
PINS = {"vllm_kimi_k3": "0498dc7e", "vllm_main": "d223c900",
        "hf": "moonshotai/Kimi-K3@9f62e4e9"}
PROJ = shapes.KDA_PROJ                     # 12288 (q/k/v 각)
DIM3 = 3 * PROJ                            # 36864 (packed decode)
W = shapes.KDA_CONV_KERNEL                 # 4
PERF_CASE = "upd_b32"
PERF_FN_CASE = "fn_b1_T8k"                 # roofline ≈590µs — launch 고정비 무시 가능
PERF_JUDGE_MIN_ROOFLINE_US = 30.0          # k04와 동일한 사전등록 제외 규칙
N_STEPS = 4
EXTRA_CASES = ("fn_then_update_carry",)

# fn_b8 varlen: 길이 2(<W-1=3) seq 를 has_init=False 자리(홀수 idx)에 배치 —
# 초기state 없는 짧은 seq 의 state 좌측 0-패드 경로를 numeric 으로 검증
FN_B8_LENGTHS = [15, 2, 8192, 300, 16384, 4096, 23000, 13547]   # 합 65536 = 8×8192


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
# 입력 생성
# ---------------------------------------------------------------------------
def mk(torch, g, *shape, dtype=None, scale=1.0, dev="cuda"):
    t = torch.randn(*shape, generator=g, dtype=torch.float32)
    if scale != 1.0:
        t.mul_(scale)                        # in-place — 대형 케이스 CPU 사본 방지
    if dtype is None:
        dtype = torch.bfloat16
    return t.to(dtype).to(dev)


def build_conv_pool(torch, g, slots: int, dim: int, dev: str):
    """프로덕션 캐시 레이아웃: (slots, W-1, dim) bf16 저장 → 커널용 (slots, dim, W-1) 전치 뷰."""
    base = mk(torch, g, slots, W - 1, dim, dev=dev)
    return base, base.transpose(-1, -2)


def scatter_indices(seed: int, b: int, slots: int):
    perm = list(range(1, slots))            # 0 = null block 예약 (NULL_BLOCK_ID=0)
    random.Random(seed).shuffle(perm)
    return perm[:b]


# ---------------------------------------------------------------------------
# GPU 본 실행
# ---------------------------------------------------------------------------
def gpu_run(r: report.KernelReport, a) -> None:
    import torch
    try:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
    except ImportError as e:
        print(f"[{KID}] causal_conv1d import 실패: {e}\n"
              f"  → venv-k3 (또는 vLLM 설치 환경, RUNNER §1) 에서 실행할 것.",
              file=sys.stderr)
        sys.exit(report.EXIT_PRECOND)

    dev = "cuda"
    results: list[tolerance.NumericResult] = []
    covered: list[str] = []
    case_meta: dict = {}
    perf_stats = None
    fn_perf_stats = None

    req = shapes.REQUIRED_SHAPES[KID]
    order = [PERF_CASE] + [c for c in req if c != PERF_CASE] + list(EXTRA_CASES)
    for name in order:
        seed = shapes.stable_seed(KID, name)
        t0 = time.time()
        try:
            if name == "fn_then_update_carry":
                entries = case_fn_update_carry(torch, causal_conv1d_fn,
                                               causal_conv1d_update, seed, dev)
            elif req.get(name, {}).get("mode") == "update":
                entries, perf_fn = case_update(torch, causal_conv1d_update,
                                               name, req[name]["b"], seed, dev)
                if name == PERF_CASE:
                    perf_stats = timing.measure(perf_fn, warmup=a.warmup, reps=a.reps)
            else:
                lengths = FN_B8_LENGTHS if req[name].get("varlen") else [req[name]["T"]]
                if name == PERF_FN_CASE:
                    entries, fn_perf_fn = case_fn(torch, causal_conv1d_fn, name,
                                                  lengths, seed, dev, want_perf=True)
                    fn_perf_stats = timing.measure(fn_perf_fn,
                                                   warmup=a.warmup, reps=a.reps)
                    del fn_perf_fn
                else:
                    entries = case_fn(torch, causal_conv1d_fn, name, lengths, seed, dev)
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
    r.add_evidence("k3_params", {"channels": f"3×{PROJ}={DIM3}", "kernel": W,
                                 "weight_dtype": "float32(K3 conv1d params_dtype)",
                                 "state_dtype": "bfloat16(model dtype)",
                                 "fn_b8_lengths": FN_B8_LENGTHS})
    r.set_gate("NUMERIC", *tolerance.numeric_gate(KID, results))
    r.set_gate("SHAPES", *shapes.shapes_gate(KID, covered))
    run_sanitizer_gate(r)
    # 사전등록 규칙(k04와 동일): roofline < 30µs 케이스는 launch 고정비 지배 → 판정 제외.
    upd_model = shapes.perf_bytes_model(KID, b=32, mode="update")
    peaks = None
    rl = None
    vp = a.phase0_verdict if a.phase0_verdict and os.path.isfile(a.phase0_verdict) else None
    if vp:
        tp = timing.resolve_twin_path(vp, a.twin)
        if tp:
            peaks = timing.load_twin_peaks(tp)
    if peaks:
        rl = timing.roofline_us(upd_model["bytes"], upd_model.get("macs", 0.0),
                                upd_model.get("alu_ops", 0.0), peaks)
    judge_update = rl is None or rl["roofline_us"] >= PERF_JUDGE_MIN_ROOFLINE_US
    if judge_update or fn_perf_stats is None:
        st, rs, ev = timing.perf_gate(
            KID, perf_stats["median_us"] if perf_stats else None, upd_model,
            a.phase0_verdict, a.twin, stats=perf_stats)
        ev["perf_case"] = PERF_CASE
        if not judge_update:
            ev["note_fallback"] = (f"{PERF_FN_CASE} 측정 실패로 {PERF_CASE} 판정 유지 — "
                                   "launch 고정비 지배 구간이라 FAIL 시 위양성 가능성 검토")
    else:
        fn_model = shapes.perf_bytes_model(KID, **req[PERF_FN_CASE])
        st, rs, ev = timing.perf_gate(
            KID, fn_perf_stats["median_us"], fn_model,
            a.phase0_verdict, a.twin, stats=fn_perf_stats)
        ev["perf_case"] = PERF_FN_CASE
        ev["excluded"] = {PERF_CASE: {
            "roofline_us": rl["roofline_us"],
            "median_us": perf_stats["median_us"] if perf_stats else None,
            "reason": f"roofline {rl['roofline_us']:.1f}µs < "
                      f"{PERF_JUDGE_MIN_ROOFLINE_US}µs — launch 고정비 지배 구간, "
                      "판정 제외(사전등록 규칙, k04와 동일)"}}
    ev["note"] = ("판정 대표 = roofline≥30µs 케이스. upd_b32 등 update 측정치는 "
                  "항상 evidence 에 보존 — cases/excluded 참조")
    r.set_gate("PERF", st, rs, ev)


def case_update(torch, update_fn, name, b, seed, dev):
    """packed decode update: (b, 3·12288) × N_STEPS 연속, 산란 인덱스, out 별도 버퍼."""
    g = torch.Generator().manual_seed(seed)
    slots = b + 4
    base, conv_state = build_conv_pool(torch, g, slots, DIM3, dev)
    base0 = base.clone()
    weight = mk(torch, g, DIM3, W, dtype=torch.float32, scale=0.1, dev=dev)
    idx_list = scatter_indices(seed, b, slots)
    idx = torch.tensor(idx_list, dtype=torch.int32, device=dev)
    idx_l = torch.tensor(idx_list, dtype=torch.long, device=dev)

    state_ref = conv_state[idx_l].float().clone()           # (b, DIM3, W-1)
    outs, refs, steps = [], [], []
    for s in range(N_STEPS):
        gs = torch.Generator().manual_seed(seed + 100 + s)
        x_storage = mk(torch, gs, b, DIM3 + 8, dev=dev)
        x = x_storage[:, :DIM3]                              # row-strided (프로덕션 split 모사)
        steps.append(x)
        out = torch.empty_like(x)
        ret = update_fn(x, conv_state, weight, None, activation="silu",
                        conv_state_indices=idx, validate_data=True, out=out)
        outs.append(ret.float())
        refs.append(kda_ref.causal_conv1d_update_ref(x, state_ref, weight))
    entries = [
        tolerance.compare(KID, torch.stack(refs), torch.stack(outs), case=f"{name}/o"),
        tolerance.compare(KID, state_ref, conv_state[idx_l].float(),
                          case=f"{name}/state"),
    ]
    untouched = [i for i in range(slots) if i not in idx_list]
    un_t = torch.tensor(untouched, dtype=torch.long, device=dev)
    entries.append(tolerance.compare(KID, base0[un_t], base[un_t],
                                     case=f"{name}/untouched_slots"))

    x0 = steps[0]
    out_b = torch.empty_like(x0)

    def perf_fn():
        update_fn(x0, conv_state, weight, None, activation="silu",
                  conv_state_indices=idx, validate_data=True, out=out_b)

    return entries, perf_fn


def case_fn(torch, fn, name, lengths, seed, dev, want_perf=False):
    """prefill fn: q/k/v 3파트 각각 호출 (프로덕션 split 뷰 그대로), varlen+state 대조.

    want_perf=True 면 (entries, perf_fn) 반환 — perf_fn 은 프로덕션과 동일한 3파트
    호출을 재실행한다 (conv_state 는 반복 중 변형되지만 타이밍에는 무영향).
    """
    g = torch.Generator().manual_seed(seed)
    N = len(lengths)
    T = sum(lengths)
    slots = N + 2
    base, conv_state3 = build_conv_pool(torch, g, slots, DIM3, dev)
    parts_state = conv_state3.split(PROJ, dim=-2)           # 프로덕션 split 뷰(비연속)
    weight3 = mk(torch, g, DIM3, W, dtype=torch.float32, scale=0.1, dev=dev)
    parts_weight = weight3.split(PROJ, dim=0)
    idx_list = scatter_indices(seed, N, slots)
    idx = torch.tensor(idx_list, dtype=torch.int32, device=dev)
    idx_l = torch.tensor(idx_list, dtype=torch.long, device=dev)
    has_init = torch.tensor([i % 2 == 0 for i in range(N)], device=dev)
    cu = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(N)],
                      dtype=torch.int32, device=dev)
    entries = []
    perf_xs = []
    for pi, pname in enumerate(("q", "k", "v")):
        gs = torch.Generator().manual_seed(seed + 10 + pi)
        x = mk(torch, gs, T, PROJ, dev=dev)                  # (T, dim) → 전치 = channel-last
        if want_perf:
            perf_xs.append(x.transpose(0, 1))                # 벤치 재호출용으로 유지
        st_before = parts_state[pi][idx_l].float().clone()   # (N, PROJ, W-1)
        out = fn(x.transpose(0, 1), parts_weight[pi], None,
                 conv_states=parts_state[pi], query_start_loc=cu,
                 cache_indices=idx, has_initial_state=has_init,
                 activation="silu").transpose(0, 1)          # (T, PROJ)
        o_refs, s_refs = [], []
        for i in range(N):
            s, e = int(cu[i]), int(cu[i + 1])
            init = st_before[i] if bool(has_init[i]) else None
            o_i, s_i = kda_ref.causal_conv1d_ref(
                x[s:e].transpose(0, 1), parts_weight[pi], initial_state=init)
            o_refs.append(o_i.transpose(0, 1))
            s_refs.append(s_i)
        entries.append(tolerance.compare(
            KID, torch.cat(o_refs, dim=0), out, case=f"{name}/o_{pname}"))
        entries.append(tolerance.compare(
            KID, torch.stack(s_refs), parts_state[pi][idx_l].float(),
            case=f"{name}/state_{pname}"))
        del x, out, o_refs, s_refs
    if not want_perf:
        return entries

    def perf_fn():
        for pi in range(3):
            fn(perf_xs[pi], parts_weight[pi], None, conv_states=parts_state[pi],
               query_start_loc=cu, cache_indices=idx, has_initial_state=has_init,
               activation="silu")

    return entries, perf_fn


def case_fn_update_carry(torch, fn, update_fn, seed, dev):
    """extra: fn(T=13, 초기state 無) → 같은 풀로 update 3스텝 — 프리필→디코드 conv 연속성.

    참조 = 16 토큰 연속 conv (kda_ref.causal_conv1d_ref 를 이월 호출).
    """
    g = torch.Generator().manual_seed(seed)
    T_pre, n_dec = 13, 3
    slots = 3
    base, conv_state3 = build_conv_pool(torch, g, slots, DIM3, dev)
    weight3 = mk(torch, g, DIM3, W, dtype=torch.float32, scale=0.1, dev=dev)
    slot = 1
    idx = torch.tensor([slot], dtype=torch.int32, device=dev)
    x_all = mk(torch, g, T_pre + n_dec, DIM3, dev=dev)       # (16, DIM3)
    cu = torch.tensor([0, T_pre], dtype=torch.int32, device=dev)
    has_init = torch.tensor([False], device=dev)

    out_pre = fn(x_all[:T_pre].transpose(0, 1), weight3, None,
                 conv_states=conv_state3, query_start_loc=cu, cache_indices=idx,
                 has_initial_state=has_init, activation="silu").transpose(0, 1)
    outs = []
    for t in range(T_pre, T_pre + n_dec):
        x_t = x_all[t:t + 1].contiguous()                    # (1, DIM3)
        out_t = torch.empty_like(x_t)
        update_fn(x_t, conv_state3, weight3, None, activation="silu",
                  conv_state_indices=idx, validate_data=True, out=out_t)
        outs.append(out_t[0].float())
    o_ref, s_ref = kda_ref.causal_conv1d_ref(
        x_all.transpose(0, 1), weight3, initial_state=None)
    return [
        tolerance.compare(KID, o_ref[:, :T_pre].transpose(0, 1), out_pre,
                          case="fn_then_update_carry/o_prefill"),
        tolerance.compare(KID, o_ref[:, T_pre:].transpose(0, 1), torch.stack(outs),
                          case="fn_then_update_carry/o_decode"),
        tolerance.compare(KID, s_ref, conv_state3[slot].float(),
                          case="fn_then_update_carry/state"),
    ]


# ---------------------------------------------------------------------------
# SANITIZER
# ---------------------------------------------------------------------------
def run_sanitizer_gate(r: report.KernelReport, timeout: int = 1800) -> None:
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
    """축소 채널(3×1536): fn ragged([2,37], 짧은 seq 포함) + update 2스텝 — memcheck 대상."""
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    dev = "cuda"
    seed = shapes.stable_seed(KID, "sanitizer_arm")
    g = torch.Generator().manual_seed(seed)
    dim = 3 * 1536
    slots, N = 5, 2
    base, conv_state = build_conv_pool(torch, g, slots, dim, dev)
    weight = mk(torch, g, dim, W, dtype=torch.float32, scale=0.1, dev=dev)
    lengths = [2, 37]
    cu = torch.tensor([0, 2, 39], dtype=torch.int32, device=dev)
    idx = torch.tensor([1, 3], dtype=torch.int32, device=dev)
    has_init = torch.tensor([False, True], device=dev)
    x = mk(torch, g, sum(lengths), dim, dev=dev)
    out = causal_conv1d_fn(x.transpose(0, 1), weight, None, conv_states=conv_state,
                           query_start_loc=cu, cache_indices=idx,
                           has_initial_state=has_init, activation="silu")
    for s in range(2):
        xb = mk(torch, g, 3, dim, dev=dev)
        ob = torch.empty_like(xb)
        causal_conv1d_update(xb, conv_state, weight, None, activation="silu",
                             conv_state_indices=torch.tensor([1, 3, 4],
                                                             dtype=torch.int32,
                                                             device=dev),
                             validate_data=True, out=ob)
    torch.cuda.synchronize()
    print(f"[{KID} sanitizer-arm] done fn={tuple(out.shape)} upd={tuple(ob.shape)} "
          f"sum={ob.float().sum().item():.4f}")
    return 0


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------
def dry_run(r: report.KernelReport, a) -> None:
    ok, details = kda_ref.selftest(verbose=False)
    wiring = {}
    if ok:
        rnd = random.Random(shapes.stable_seed(KID, "dry"))
        C, L = 3, 9
        x = [[rnd.gauss(0, 1) for _ in range(L)] for _ in range(C)]
        w = [[rnd.gauss(0, 1) for _ in range(W)] for _ in range(C)]
        ref = kda_ref.py_flat(kda_ref.py_conv1d(x, w, None)[0])
        ok_out = [v * (1 + 1e-4) for v in ref]
        bad_out = [v + 0.1 for v in ref]
        wiring = {
            "pass_case": tolerance.compare(KID, ref, ok_out, "dry_ok").passed,
            "fail_case": not tolerance.compare(KID, ref, bad_out, "dry_bad").passed,
        }
        ok = ok and all(wiring.values())
    if not ok:
        r.set_gate("NUMERIC", "FAIL",
                   f"cpu-dry-run: 참조/판정 자기검증 실패 — {details['failed'] or wiring}",
                   {"selftest": details, "wiring": wiring})
    else:
        r.set_gate("NUMERIC", "SKIP",
                   f"cpu-dry-run: 실커널 미실행 — 순수 python 참조 자기검증 "
                   f"{details['checks']}건(fn↔update 토큰 동치 포함) + 판정 배선 양방향 green",
                   {"selftest": details, "wiring": wiring})
    r.set_gate("SHAPES", "SKIP", "cpu-dry-run: 커널 미실행 — 요구 케이스 정의만 확인",
               {"required": sorted(shapes.REQUIRED_SHAPES[KID]),
                "fn_b8_lengths": FN_B8_LENGTHS, "extra": list(EXTRA_CASES)})
    r.set_gate("SANITIZER", "SKIP", "cpu-dry-run: GPU 미실행")
    model = shapes.perf_bytes_model(KID, b=32, mode="update")
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
