#!/usr/bin/env python3
"""k01_kda_chunk — KDA chunk prefill (chunk_kda_with_fused_gate, 순수 Triton) 단독 검증.

대상 API (vLLM kimi-k3 @ 0498dc7ea9204f1863b46515816f5d8d458e01e7 원문 확인):
  vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_kda_with_fused_gate(
      q, k, v, raw_g, raw_beta, A_log, g_bias, lower_bound,
      initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)
  — SM80 prefill 프로덕션 경로 (nvidia/kda.py L639-652: flashkda 미지원 아치의 폴백).
  주의: 커널은 출력 o를 **v 버퍼에 in-place로 기록**한다(chunk.py `o=v`) → v는 clone 전달.
  initial/final state 레이아웃 = (N, H, V, K) — naive 참조 (K,V)의 전치 (test_kda.py L177-192).

판정 (docs/PLAN.md §S2 4게이트):
  NUMERIC   fp32 참조 recurrent(common/kda_ref.py, vLLM naive 이식) 대비
            rmse_rel ≤ 5e-3 (tolerance.py TOL — 업스트림 test_kda.py 바 그대로).
            chunk 커널 vs 참조 recurrent 의 일치가 곧 chunk/recurrent 동치의 수학 검증.
  SHAPES    K3 실측 H=96, dk=dv=128 고정 + REQUIRED_SHAPES 4케이스(varlen·ragged 포함)
            + extra: state 이월(분할 prefill) 케이스.
  SANITIZER compute-sanitizer memcheck (자체 --sanitizer-arm 축소 shape).
  PERF      대표 케이스 b1_T8k 측정 vs phase0 루프라인 (timing.py, eff_floor 0.35).

실행: python3 harness.py --phase0-verdict <verdict.json> --report-dir <dir> [--cpu-dry-run]
exit: 0 FAIL없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, kda_ref, report, shapes, timing, tolerance  # noqa: E402

KID = "k01_kda_chunk"
PINS = {"vllm_kimi_k3": "0498dc7e", "vllm_main": "d223c900",
        "hf": "moonshotai/Kimi-K3@9f62e4e9"}
H, D = shapes.KDA_HEADS, shapes.KDA_HEAD_DIM          # 96, 128 (K3 실측)
LB = shapes.KDA_GATE_LOWER_BOUND                       # -5.0 (K3 config)
PERF_CASE = "b1_T8k"                                   # PERF 대표(단일 seq 8k prefill)

# 케이스 → varlen 길이 목록 (합 = b×T; 64 비배수 ragged 로 청크 경계 검증)
CASE_LENGTHS = {
    "b1_T8k": [8192],
    "b1_T128k": [131072],
    "b8_T8k_varlen": [15, 8192, 300, 16384, 4096, 12288, 8000, 16261],  # 합 65536
    # b32 는 T=2048/seq (합 65,536) — 구 [8192]×32 는 A100 80GB OOM 확정
    # (shapes._PREFILL_CASES 사전등록 주석·VERIFICATION.md #2 근거).
    "b32_T2k": [2048] * 32,
}
EXTRA_CASES = ("state_carry_2x256",)


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
# 케이스 입력 생성 (CPU generator 고정 seed → 디바이스 이동; 재실행·호스트 무관 동일)
# ---------------------------------------------------------------------------
def build_inputs(torch, seed: int, lengths: list[int], dev: str,
                 has_init_pattern: str = "all"):
    g = torch.Generator().manual_seed(seed)
    T = sum(lengths)
    N = len(lengths)

    def mk(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=torch.float32)
        if scale != 1.0:
            t.mul_(scale)                    # in-place — 대형 케이스 CPU 사본 방지
        return t.to(dtype).to(dev)

    q = mk(1, T, H, D)
    k = mk(1, T, H, D)
    v = mk(1, T, H, D)
    raw_g = mk(1, T, H, D)
    beta_storage = mk(1, T, H + 1)          # row-strided (프로덕션 in_proj split 모사)
    raw_beta = beta_storage[..., :H]
    A_log = mk(H, dtype=torch.float32, scale=0.5)
    dt_bias = mk(H * D, dtype=torch.float32, scale=0.1)
    h0 = mk(N, H, D, D, dtype=torch.float32)          # 커널 레이아웃 (V,K)로 해석
    has_init = [True] * N if has_init_pattern == "all" else \
        [i % 2 == 0 for i in range(N)]
    for i, hi in enumerate(has_init):
        if not hi:
            h0[i].zero_()                    # 프로덕션 gather_initial_states 의 0 채움 모사
    cu = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(N)],
                      dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
                A_log=A_log, dt_bias=dt_bias, h0=h0, cu=cu,
                has_init=has_init, T=T, N=N)


def run_kernel(chunk_fn, inp, torch):
    v_in = inp["v"].clone()                  # 커널이 o를 v에 in-place 기록
    o, ht = chunk_fn(
        q=inp["q"], k=inp["k"], v=v_in,
        raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
        A_log=inp["A_log"], g_bias=inp["dt_bias"], lower_bound=LB,
        initial_state=inp["h0"].clone(),     # contiguous 요구 + 원본 보존
        output_final_state=True, use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu"])
    return o, ht


# ---------------------------------------------------------------------------
# GPU 본 실행
# ---------------------------------------------------------------------------
def gpu_run(r: report.KernelReport, a) -> None:
    import torch
    try:
        from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
            chunk_kda_with_fused_gate,
        )
    except ImportError as e:
        print(f"[{KID}] vLLM kimi-k3 브랜치 ops import 실패: {e}\n"
              f"  → venv-k3(Stage 2 소스빌드, RUNNER §1) 에서 실행할 것.", file=sys.stderr)
        sys.exit(report.EXIT_PRECOND)

    dev = "cuda"
    results: list[tolerance.NumericResult] = []
    covered: list[str] = []
    case_meta: dict = {}
    perf_stats = None

    order = [PERF_CASE] + [c for c in CASE_LENGTHS if c != PERF_CASE] + list(EXTRA_CASES)
    for name in order:
        seed = shapes.stable_seed(KID, name)
        t0 = time.time()
        try:
            if name == "state_carry_2x256":
                entries = case_state_carry(torch, chunk_kda_with_fused_gate, seed, dev)
            else:
                lengths = CASE_LENGTHS[name]
                pattern = "alt" if "varlen" in name else "all"
                inp = build_inputs(torch, seed, lengths, dev, pattern)
                o, ht = run_kernel(chunk_kda_with_fused_gate, inp, torch)
                o_ref, ht_ref = kda_ref.kda_reference_varlen(
                    inp["q"], inp["k"], inp["v"], inp["raw_g"], inp["raw_beta"],
                    inp["A_log"], inp["dt_bias"], LB, inp["h0"], inp["cu"],
                    progress=16384 if inp["T"] >= 32768 else None)
                entries = [tolerance.compare(KID, o_ref, o, case=f"{name}/o"),
                           tolerance.compare(KID, ht_ref, ht, case=f"{name}/ht")]
                if name == PERF_CASE:
                    v_bench = inp["v"].clone()
                    h_bench = inp["h0"].clone()

                    def bench():
                        chunk_kda_with_fused_gate(
                            q=inp["q"], k=inp["k"], v=v_bench,
                            raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
                            A_log=inp["A_log"], g_bias=inp["dt_bias"],
                            lower_bound=LB, initial_state=h_bench,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu"])

                    perf_stats = timing.measure(bench, warmup=a.warmup, reps=a.reps)
                del inp, o, ht, o_ref, ht_ref
            results.extend(entries)
            covered.append(name)   # 실행 완료 = 커버 (오차 판정은 NUMERIC 게이트 몫)
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
    r.add_evidence("k3_params", {"H": H, "D": D, "lower_bound": LB, "chunk_size": 64,
                                 "int32_overflow_watch":
                                 "단일호출 최대 elem = b1_T128k 131,072×12288 = 1.61e9 < 2^31. "
                                 "elem ≥ 2^31 단일호출(≥174,763 tok)은 중간 텐서 상주로 "
                                 "A100 80GB 초과라 S2에서 실행 불가·프로덕션 chunked-prefill "
                                 "에도 없음 — recon/kda-fla.md §5.3 int32 리스크는 상한 케이스 "
                                 "b1_T128k 로 감시(사전등록; 구 b32_T8k 262k tok 는 OOM 확정 "
                                 "→ b32_T2k 교체, VERIFICATION.md #2)"})
    r.set_gate("NUMERIC", *tolerance.numeric_gate(KID, results))
    r.set_gate("SHAPES", *shapes.shapes_gate(KID, covered))
    run_sanitizer_gate(r)
    model = shapes.perf_bytes_model(KID, b=1, T=8192)
    st, rs, ev = timing.perf_gate(
        KID, perf_stats["median_us"] if perf_stats else None, model,
        a.phase0_verdict, a.twin, stats=perf_stats)
    ev["perf_case"] = PERF_CASE
    ev["note"] = "측정은 chunk 커널 호출만(v 버퍼 재사용 — 내용 무의미, 시간만 유효); conv 는 k03"
    r.set_gate("PERF", st, rs, ev)


def case_state_carry(torch, chunk_fn, seed: int, dev: str):
    """extra: 분할 prefill state 이월 — 512 단일호출 vs 256+256 이월, 참조 대비 + 상호대비."""
    inp = build_inputs(torch, seed, [512], dev)
    o_full, _ht_full = run_kernel(chunk_fn, inp, torch)
    cu256 = torch.tensor([0, 256], dtype=torch.int32, device=dev)
    v1 = inp["v"][:, :256].clone()
    o1, s1 = chunk_fn(q=inp["q"][:, :256], k=inp["k"][:, :256], v=v1,
                      raw_g=inp["raw_g"][:, :256], raw_beta=inp["raw_beta"][:, :256],
                      A_log=inp["A_log"], g_bias=inp["dt_bias"], lower_bound=LB,
                      initial_state=inp["h0"].clone(), output_final_state=True,
                      use_qk_l2norm_in_kernel=True, cu_seqlens=cu256)
    v2 = inp["v"][:, 256:].clone().contiguous()
    o2, s2 = chunk_fn(q=inp["q"][:, 256:].contiguous(), k=inp["k"][:, 256:].contiguous(),
                      v=v2, raw_g=inp["raw_g"][:, 256:].contiguous(),
                      raw_beta=inp["raw_beta"][:, 256:], A_log=inp["A_log"],
                      g_bias=inp["dt_bias"], lower_bound=LB,
                      initial_state=s1, output_final_state=True,
                      use_qk_l2norm_in_kernel=True, cu_seqlens=cu256)
    o_carry = torch.cat([o1, o2], dim=1)
    o_ref, ht_ref = kda_ref.kda_reference_varlen(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], inp["raw_beta"],
        inp["A_log"], inp["dt_bias"], LB, inp["h0"], inp["cu"])
    return [
        tolerance.compare(KID, o_ref, o_carry, case="state_carry_2x256/o"),
        tolerance.compare(KID, ht_ref, s2, case="state_carry_2x256/ht"),
        # 커널 상호대비: 단일호출 vs 이월호출 (둘 다 커널 — fp32 state 이월이면 근사 일치)
        tolerance.compare(KID, o_full, o_carry, case="state_carry_2x256/carry_vs_single"),
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
    """축소 shape(H=96 유지, T 축소·ragged) 로 chunk 커널 1회 — memcheck 대상."""
    import torch
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
    inp = build_inputs(torch, shapes.stable_seed(KID, "sanitizer_arm"),
                       [15, 305], "cuda", "alt")
    o, ht = run_kernel(chunk_kda_with_fused_gate, inp, torch)
    torch.cuda.synchronize()
    print(f"[{KID} sanitizer-arm] done o={tuple(o.shape)} ht={tuple(ht.shape)} "
          f"sum={o.float().sum().item():.4f}")
    return 0


# ---------------------------------------------------------------------------
# dry-run (호스트: 순수 python 참조 자기검증 + 판정 배선 + PERF 경로 검증)
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
                   f"{details['checks']}건 + 판정 배선 양방향 green",
                   {"selftest": details, "wiring": wiring})
    r.set_gate("SHAPES", "SKIP", "cpu-dry-run: 커널 미실행 — 요구 케이스 정의만 확인",
               {"required": sorted(shapes.REQUIRED_SHAPES[KID]),
                "planned_lengths": {k: (v if len(v) <= 8 else f"[{v[0]}]×{len(v)}")
                                    for k, v in CASE_LENGTHS.items()},
                "extra": list(EXTRA_CASES)})
    r.set_gate("SANITIZER", "SKIP", "cpu-dry-run: GPU 미실행")
    model = shapes.perf_bytes_model(KID, b=1, T=8192)
    st, rs, ev = timing.perf_gate(KID, None, model, a.phase0_verdict, a.twin)
    try:
        tp = timing.resolve_twin_path(a.phase0_verdict, a.twin) \
            if a.phase0_verdict else None
        if tp:
            ev["dry_roofline"] = timing.roofline_us(
                model["bytes"], model.get("macs", 0.0), model.get("alu_ops", 0.0),
                timing.load_twin_peaks(tp))
            ev["dry_note"] = "목표 산출 경로(verdict→twin→roofline) 검증됨 — 판정 아님"
    except Exception as e:  # 경로 검증 실패는 evidence 로만
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
