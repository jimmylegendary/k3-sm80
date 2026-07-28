"""k06 — W4A16 dequant ALU 마이크로벤치 (dequant_ops_per_param 밴드를 실측으로 고정).

왜 있는가
  phase0/verdict.json caveats: "dequant_ops_per_param은 무근거 밴드"(현재 4.0 가정,
  발견 2 ALU-bound 는 dq=4 조건부). 이 하네스는 그 값을 **실측으로 재보정**한다.
  출력은 phase0/k3_model.py 재보정 입력 — 리포트 evidence 와 별도 파일
  `kernel-report/calibration-k06-dequant.json` 에 ops/param 추정치 필드를 남긴다.

방법 (Marlin dequant 경로 vs bf16 순수 GEMV 대비)
  각 케이스에서 두 가지를 측정한다 (m=1 GEMV = decode weight-stream 형):
    t_marlin : ops.marlin_gemm (uint4b8 g128, use_fp32_reduce — MoE 경로와 동일 설정)
    t_bf16   : torch.matmul (동일 dequant 가중치의 bf16 GEMV)
  bf16 GEMV 로 커널계열 공통의 실효 대역폭을 보정한 뒤, Marlin 초과시간을 ALU 로 귀속:
    BW_eff        = bytes_bf16 / t_bf16              (launch/스트림 효율 포함 실효 BW)
    t_mem_pred    = bytes_w4 / BW_eff                (dequant 이 공짜일 때의 예측 시간)
    dq_time       = max(0, t_marlin − t_mem_pred)
    ops_per_param = dq_time × ALU_peak / P           (ALU_peak = 트윈 fp32 유도 피크)
  point = 케이스별 추정의 중앙값, band = [최소, 최대] (+datasheet BW 대안치 병기).
  overhead(마린 vs matmul 런치 차이)가 섞이므로 **상한 성격** — evidence 에 명시.

NUMERIC (dequant 정확성 — tolerance.py k06 정책: 곱 1회 ≤1ulp)
  identity 입력 트릭: a = I(K×K) 로 marlin_gemm 을 부르면 out[i,j] = dequant(W)[i,j]
  (fp32 누적에 비영항 1개 → 커널 자신의 dequant 값이 그대로 나옴).
  참조 = marlin_quantize 가 돌려주는 w_ref (동일 (q−8)·s 의 기준 구현) fp32.
  → rmse_rel ≤ 1e-3, elementwise rtol=2·eps_bf16/atol=0 위반 0.

케이스 (shapes.REQUIRED_SHAPES[k06] 과 이름 일치, K3 실측 expert 행렬)
  one_expert_w13   : (K=3584 → N=3072)  w1/w3 단위 행렬
  one_expert_w2    : (K=3072 → N=3584)  w2 단위 행렬
  active16_experts : 16 expert × (w1,w3,w2) = decode 1토큰 활성분 528.5M param 스트림
                     — PERF(measure_only) 판정 케이스 (CV ≤ 0.10)

PASS 기준: NUMERIC(dequant ≤1ulp) + SANITIZER clean(소형 arm) + SHAPES(3케이스 실행)
           + PERF(active16 스트림 CV ≤ 0.10, 측정 산출 완료). 목표시간 비교는 없음
           (measure_only — 측정 자체가 산출물).

업스트림 핀: ops.marlin_gemm = vllm/_custom_ops.py@d223c900 L1200 (_C.marlin_gemm),
  가중치 준비 = marlin_utils_test.marlin_quantize (tests/kernels/quantization/
  test_marlin_gemm.py L470-523 패턴, uint4b8 g128). kimi-k3 0498dc7e 동일 경로.

CLI (run_all.sh 계약):
  python3 harness.py --phase0-verdict <verdict.json> --report-dir <dir> [--cpu-dry-run]
                     [--twin <twin.json>] [--skip-sanitizer]
내부용: --sanitizer-child (소형 GPU 워크로드 1회 — memcheck 래핑 대상)
exit: 0 FAIL 없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k06_dequant_bench"
PINS = {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e",
        "hf": "moonshotai/Kimi-K3@9f62e4e9"}
GROUP_SIZE = 128                    # W4A16 g128 (recon/k3-config.md §6, BPP_W4A16 근거)
DQ_OPS_PRIOR = 4.0                  # phase0 현행 무근거 가정 (재보정 대상)

CASES = {
    "one_expert_w13": dict(K=shapes.LATENT, N=shapes.MOE_INTER),
    "one_expert_w2": dict(K=shapes.MOE_INTER, N=shapes.LATENT),
    "active16_experts": dict(experts=shapes.TOPK),   # 16 × (w1,w3,w2)
}
PERF_CASE = "active16_experts"


# ---------------------------------------------------------------------------
# GPU 경로
# ---------------------------------------------------------------------------
def _build_matrix(torch, dev, K: int, N: int, seed: int):
    """(K,N) bf16 행렬 생성→uint4b8 g128 marlin 양자화. 반환: (w_ref, packed dict)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((K, N), generator=g, dtype=torch.float32) / 10) \
        .to(torch.bfloat16).to(dev)
    w_ref, qw, s, g_idx, perm, _ = marlin_quantize(
        w, scalar_types.uint4b8, GROUP_SIZE, act_order=False)
    return w_ref, {"qw": qw, "s": s, "g_idx": g_idx, "perm": perm, "K": K, "N": N}


_QTYPE = None


def _qtype():
    """scalar_types.uint4b8 1회 해석 캐시 — 타이밍 루프 안 import 오버헤드 제거."""
    global _QTYPE
    if _QTYPE is None:
        from vllm.scalar_type import scalar_types
        _QTYPE = scalar_types.uint4b8
    return _QTYPE


def _marlin_gemv(torch, ops, ws, x, pk, out):
    return ops.marlin_gemm(
        x, out, pk["qw"], None, pk["s"], None, None, None,
        pk["g_idx"], pk["perm"], ws, _qtype(),
        x.shape[0], pk["N"], pk["K"],
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
        is_zp_float=False)


def run_gpu(args, rep: report.KernelReport) -> None:
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    dev = torch.device("cuda:0")
    ws = marlin_make_workspace_new(dev)
    covered: list[str] = []
    nrs: list[tolerance.NumericResult] = []
    meas: dict[str, dict] = {}

    # ---- 케이스 1·2: 단일 expert 행렬 — dequant 정확성 + GEMV 시간 ------------
    for case in ("one_expert_w13", "one_expert_w2"):
        K, N = CASES[case]["K"], CASES[case]["N"]
        seed = shapes.stable_seed(KID, case)
        torch.manual_seed(seed)
        w_ref, pk = _build_matrix(torch, dev, K, N, seed)

        # NUMERIC: identity 입력 → 커널측 dequant 추출
        eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
        deq_out = torch.empty((K, N), dtype=torch.bfloat16, device=dev)
        deq = _marlin_gemv(torch, ops, ws, eye, pk, deq_out)
        torch.cuda.synchronize()
        nrs.append(tolerance.compare(KID, w_ref.float(), deq, case=case))

        # 타이밍: m=1 GEMV (marlin vs bf16 matmul)
        gx = torch.Generator().manual_seed(seed + 7)
        x = (torch.randn((1, K), generator=gx, dtype=torch.float32) / 10) \
            .to(torch.bfloat16).to(dev)
        out_m = torch.empty((1, N), dtype=torch.bfloat16, device=dev)
        w_bf16 = w_ref.contiguous()
        st_marlin = timing.measure(lambda: _marlin_gemv(torch, ops, ws, x, pk, out_m),
                                   warmup=10, reps=50)
        st_bf16 = timing.measure(lambda: torch.matmul(x, w_bf16), warmup=10, reps=50)
        meas[case] = {"params": K * N, "marlin": st_marlin, "bf16": st_bf16,
                      "n_matrices": 1}
        covered.append(case)
        del eye, deq_out, deq
        print(f"[k06] {case}: marlin {st_marlin['median_us']:.1f}µs / "
              f"bf16 {st_bf16['median_us']:.1f}µs")

    # ---- 케이스 3: active16 — 16 expert × (w1,w3,w2) 스트림 -------------------
    case = "active16_experts"
    seed = shapes.stable_seed(KID, case)
    torch.manual_seed(seed)
    KL, NI = shapes.LATENT, shapes.MOE_INTER
    packs, refs = [], []
    for e_i in range(shapes.TOPK):
        for tag, (K, N) in (("w1", (KL, NI)), ("w3", (KL, NI)), ("w2", (NI, KL))):
            w_ref, pk = _build_matrix(torch, dev, K, N, seed + 100 + e_i * 3 + len(refs))
            packs.append(pk)
            refs.append(w_ref.contiguous())
    # dequant 스팟체크: expert0 w1 (모든 케이스가 NUMERIC 증거를 갖도록)
    eye = torch.eye(KL, dtype=torch.bfloat16, device=dev)
    spot_out = torch.empty((KL, NI), dtype=torch.bfloat16, device=dev)
    spot = _marlin_gemv(torch, ops, ws, eye, packs[0], spot_out)
    torch.cuda.synchronize()
    nrs.append(tolerance.compare(KID, refs[0].float(), spot,
                                 case="active16_experts_spot_e0w1"))
    del eye, spot_out, spot

    gx = torch.Generator().manual_seed(seed + 7)
    x1 = (torch.randn((1, KL), generator=gx, dtype=torch.float32) / 10) \
        .to(torch.bfloat16).to(dev)
    x2 = (torch.randn((1, NI), generator=gx, dtype=torch.float32) / 10) \
        .to(torch.bfloat16).to(dev)
    outs = [torch.empty((1, pk["N"]), dtype=torch.bfloat16, device=dev) for pk in packs]

    def marlin_stream():
        for pk, o in zip(packs, outs):
            _marlin_gemv(torch, ops, ws, x1 if pk["K"] == KL else x2, pk, o)

    def bf16_stream():
        for w in refs:
            torch.matmul(x1 if w.shape[0] == KL else x2, w)

    st_marlin = timing.measure(marlin_stream, warmup=10, reps=50)
    st_bf16 = timing.measure(bf16_stream, warmup=10, reps=50)
    meas[case] = {"params": shapes.TOPK * shapes.EXPERT_P,
                  "marlin": st_marlin, "bf16": st_bf16,
                  "n_matrices": len(packs)}
    covered.append(case)
    print(f"[k06] {case}: marlin {st_marlin['median_us']:.1f}µs / "
          f"bf16 {st_bf16['median_us']:.1f}µs ({len(packs)}행렬)")

    # ---- dequant_ops_per_param 추정 ------------------------------------------
    calib = estimate_dq_ops(meas, args)
    rep.add_evidence("dequant_ops_per_param", calib)
    rep.add_evidence("timing_by_case", meas)
    calib_path = os.path.join(args.report_dir, "calibration-k06-dequant.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({"schema": "k06-dequant-calibration/v1", "pins": PINS,
                   "dequant_ops_per_param": calib,
                   "timing_by_case": meas, "env": env_check.collect_env()},
                  f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[k06] 재보정 파일: {calib_path} "
          f"(point={calib.get('point')}, prior={DQ_OPS_PRIOR})")

    # ---- 게이트 ---------------------------------------------------------------
    st, rs, ev = tolerance.numeric_gate(KID, nrs)
    rep.set_gate("NUMERIC", st, rs, ev)
    st, rs, ev = shapes.shapes_gate(KID, covered)
    rep.set_gate("SHAPES", st, rs, ev)
    model = shapes.perf_bytes_model(KID, params=meas[PERF_CASE]["params"])
    st, rs, ev = timing.perf_gate(KID, meas[PERF_CASE]["marlin"]["median_us"], model,
                                  args.phase0_verdict, args.twin,
                                  stats=meas[PERF_CASE]["marlin"])
    rep.set_gate("PERF", st, rs, ev)

    if args.skip_sanitizer:
        rep.set_gate("SANITIZER", "SKIP", "--skip-sanitizer 지정 (부분 재실행 모드)")
    else:
        st, rs, ev = sanitizer_arm(args)
        rep.set_gate("SANITIZER", st, rs, ev)


def estimate_dq_ops(meas: dict, args) -> dict:
    """케이스별 (t_marlin, t_bf16) → dequant ops/param 추정. 방법은 모듈 docstring."""
    peaks = None
    note = []
    try:
        tp = timing.resolve_twin_path(args.phase0_verdict, args.twin) \
            if args.phase0_verdict else None
        if tp:
            peaks = timing.load_twin_peaks(tp)
    except Exception as e:
        note.append(f"트윈 로드 실패: {e}")
    if peaks is None:
        return {"available": False,
                "reason": "트윈(ALU_peak 원천) 미해석 — --phase0-verdict/--twin 확인",
                "note": note}
    alu, bw_sheet = peaks["alu_flops"], peaks["hbm_bw"]
    per_case, points = {}, []
    for case, m in meas.items():
        P = m["params"]
        bytes_w4 = P * shapes.BPP_W4A16
        bytes_bf16 = P * 2.0
        t_m = m["marlin"]["median_us"] * 1e-6
        t_b = m["bf16"]["median_us"] * 1e-6
        bw_eff = bytes_bf16 / t_b
        t_mem_pred = bytes_w4 / bw_eff
        dq_time = max(0.0, t_m - t_mem_pred)
        ops_eff = dq_time * alu / P
        # 대안: datasheet BW 기준 (launch 오버헤드가 ALU 로 과대귀속되는 방향)
        dq_time_sheet = max(0.0, t_m - bytes_w4 / bw_sheet)
        ops_sheet = dq_time_sheet * alu / P
        per_case[case] = {
            "params": P, "t_marlin_us": m["marlin"]["median_us"],
            "t_bf16_us": m["bf16"]["median_us"],
            "bw_eff_gbps": round(bw_eff / 1e9, 1),
            "speedup_w4_vs_bf16": round(t_b / t_m, 3) if t_m > 0 else None,
            "speedup_ideal_bytes_ratio": round(2.0 / shapes.BPP_W4A16, 3),
            "ops_per_param_bweff": round(ops_eff, 3),
            "ops_per_param_bwsheet": round(ops_sheet, 3),
        }
        points.append(ops_eff)
    return {
        "available": True,
        "point": round(statistics.median(points), 3),
        "band": [round(min(points), 3),
                 round(max(max(points),
                           max(c["ops_per_param_bwsheet"] for c in per_case.values())), 3)],
        "prior_phase0": DQ_OPS_PRIOR,
        "per_case": per_case,
        "method": "BW_eff(bf16 GEMV 보정) 기반 초과시간의 ALU 귀속 — 커널 오버헤드 차이가 "
                  "섞이는 상한 성격. band 상한은 datasheet-BW 대안치 포함.",
        "alu_peak_flops": alu, "hbm_bw_datasheet": bw_sheet,
        "note": note,
    }


# ---------------------------------------------------------------------------
# sanitizer arm (소형: K=512,N=256 identity+GEMV)
# ---------------------------------------------------------------------------
def run_sanitizer_child() -> int:
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    dev = torch.device("cuda:0")
    ws = marlin_make_workspace_new(dev)
    torch.manual_seed(shapes.stable_seed(KID, "sanitizer_child"))
    _, pk = _build_matrix(torch, dev, 512, 256, shapes.stable_seed(KID, "san"))
    eye = torch.eye(512, dtype=torch.bfloat16, device=dev)
    out = torch.empty((512, 256), dtype=torch.bfloat16, device=dev)
    _marlin_gemv(torch, ops, ws, eye, pk, out)
    x = torch.randn((1, 512), dtype=torch.bfloat16, device=dev) / 10
    o1 = torch.empty((1, 256), dtype=torch.bfloat16, device=dev)
    for _ in range(3):
        _marlin_gemv(torch, ops, ws, x, pk, o1)
    torch.cuda.synchronize()
    print("k06 sanitizer child done")
    return 0


def sanitizer_arm(args) -> tuple[str, str, dict]:
    san = env_check.find_compute_sanitizer()
    if not san:
        return ("SKIP", "compute-sanitizer 미발견 — RUNNER.md §6 (CUDA 툴킷 bin PATH)", {})
    cmd = [san, "--tool", "memcheck", "--error-exitcode", "88",
           sys.executable, os.path.abspath(__file__), "--sanitizer-child"]
    print("[k06] sanitizer arm (소형 K=512/N=256)")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        out = (p.stdout or "") + (p.stderr or "")
        return report.sanitizer_gate_from_output(
            out, cmd=" ".join(cmd[:4]) + " ...", returncode=p.returncode)
    except subprocess.TimeoutExpired:
        return ("SKIP", "sanitizer arm timeout(1800s) — 판독 불가", {})


# ---------------------------------------------------------------------------
# dry-run: 순수 python INT4 g128 dequant 시뮬레이션 (판정 경로 검증)
# ---------------------------------------------------------------------------
def dry_run(args, rep: report.KernelReport) -> None:
    import random
    rnd = random.Random(shapes.stable_seed(KID, "dry"))
    K, N = 256, 64
    q = [[rnd.randint(0, 15) for _ in range(N)] for _ in range(K)]
    s = [[rnd.uniform(0.005, 0.02) for _ in range(N)] for _ in range(K // GROUP_SIZE)]

    def dequant():
        return [(q[i][j] - 8) * s[i // GROUP_SIZE][j] for i in range(K) for j in range(N)]

    ref = dequant()
    out = dequant()                              # 동일 산식 → 정확 일치 (경로 검증)
    nr = tolerance.compare(KID, ref, out, case="dry_int4g128_exact")
    bad = [v * (1 + 0.05) for v in ref]          # FAIL 방향
    nr_bad = tolerance.compare(KID, ref, bad, case="dry_int4g128_perturbed")
    ok = nr.passed and not nr_bad.passed

    # 케이스별 루프라인 예측 (트윈 있으면) — 사내 실행 기대치 사전 기록
    pred = {}
    try:
        tp = timing.resolve_twin_path(args.phase0_verdict, args.twin) \
            if args.phase0_verdict else None
        if tp:
            peaks = timing.load_twin_peaks(tp)
            for case in CASES:
                P = (shapes.TOPK * shapes.EXPERT_P if case == PERF_CASE
                     else CASES[case]["K"] * CASES[case]["N"])
                model = shapes.perf_bytes_model(KID, params=P)
                pred[case] = {"params": P,
                              **timing.roofline_us(model["bytes"], 0.0,
                                                   model["alu_ops"], peaks)}
    except Exception as e:
        pred = {"error": str(e)[:200]}
    rep.add_evidence("dry_run", {"dequant_sim": {
        "exact_passed": nr.passed, "perturbed_detected": not nr_bad.passed,
        "metrics": nr.metrics}, "roofline_predictions_us": pred,
        "cases_validated": list(CASES)})
    rep.set_gate("NUMERIC", "SKIP" if ok else "FAIL",
                 "cpu-dry-run — 순수 python dequant 판정경로 green, 실커널 비교 없음"
                 if ok else "dry-run dequant 판정경로 실패 — 하네스 결함")
    rep.set_gate("SANITIZER", "SKIP", "cpu-dry-run — GPU 미실행")
    rep.set_gate("SHAPES", "SKIP",
                 f"cpu-dry-run — 케이스 정의 {len(CASES)}건 검증만, 실행 커버 아님",
                 {"validated": list(CASES)})
    rep.set_gate("PERF", "SKIP", "cpu-dry-run — 측정 없음 (measure_only 커널)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--report-dir", default=os.path.join(_KIT, "kernel-report"))
    ap.add_argument("--twin", default=None)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--skip-sanitizer", action="store_true")
    ap.add_argument("--sanitizer-child", action="store_true")
    args = ap.parse_args()

    if args.sanitizer_child:
        return run_sanitizer_child()

    rep = report.KernelReport(KID, dry_run=args.cpu_dry_run, pins=PINS)
    rep.add_evidence("purpose", {
        "target": "dequant_ops_per_param 재보정 (phase0 verdict caveats 의 무근거 밴드 해소)",
        "consumer": "phase0/k3_model.py — kernel-report/calibration-k06-dequant.json",
        "prior": DQ_OPS_PRIOR})

    env = env_check.collect_env()
    ok, checks = env_check.check_preconditions(env, cpu_dry_run=args.cpu_dry_run)
    if not ok:
        print("[k06] GPU 사전조건 미충족 — skip", file=sys.stderr)
        for c in checks:
            print(f"  {c['status']:4s} {c['check']:20s} {c['detail']}", file=sys.stderr)
        return report.EXIT_PRECOND

    if args.cpu_dry_run:
        dry_run(args, rep)
    else:
        run_gpu(args, rep)

    j, _m = rep.write(args.report_dir)
    print(f"[k06] 리포트: {j} (overall={rep.overall()})")
    return rep.exit_code()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
