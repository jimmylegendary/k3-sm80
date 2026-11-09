"""k07_situ_act — SiTU-GLU 활성화: 언퓨즈드 순수 PyTorch 참조 + vLLM 퓨즈드 op 대조 (S2).

SiTU 정의 (1차 출처 원문 확인, 2026-07-28 fetch):
  HF moonshotai/Kimi-K3 @ 9f62e4e9 `modeling_kimi_linear.py` L64-91 `SituAndMul`:
      d    = x.shape[-1] // 2
      gate = x[..., :d].float();  up = x[..., d:].float()          # 내부 fp32
      situ_a = β·tanh(gate/β)·σ(gate)                              # σ = sigmoid
      up     = lβ·tanh(up/lβ)          (linear_beta 설정 시 — soft clip)
      out    = (situ_a · up).to(x.dtype)
  β = 4.0, lβ = 25.0 — config `activation_situ_beta`/`activation_situ_linear_beta`
  (recon/k3-config.md §1 `hidden_act: situ`; shapes.SITU_BETA/SITU_LINEAR_BETA).

vLLM 측 구현 (main `a3c14efb` 원문 확인 — 이전 핀 0498dc7e 와 forward_native 본문 동일):
  vllm/model_executor/layers/activation.py L162-204 `SituAndMul(CustomOp)` (이전 핀 L161-203):
    - forward_native = 위 HF 수식과 동일 (fp32 내부 → x.dtype 캐스트) — **언퓨즈드 경로**
    - forward_cuda  = 퓨즈드 CUDA op `torch.ops._C.situ_and_mul(out, x, β, lβ)`
      (lβ 미설정은 -1.0 센티널; "no fp32 temporaries")
  커널 본체: csrc/libtorch_stable/activation_kernels.cu L467-637 (situ_and_mul_kernel)
    — elementwise, 아치 제네릭 → SM80 소스빌드에 포함. (SM 게이트가 있는 것은
    MegaMoE trtllmgen 융합 경로이며 그쪽은 SM80 무관 — recon/vllm-upstream.md (4).)
  K3 모델 결합: vllm/models/kimi_k3/nvidia/model.py L158-167 (MLP/shared/dense 가
  SituAndMul 사용; MoE routed 도 비-MegaMoE 경로에선 activation="situ" 전달).

S2 대상과 arm:
  PLAN 커널 표: "SiTU activation — 언퓨즈드 참조로 대체(성능 후순위)".
  - arm `native`  (항상): 언퓨즈드 순수 PyTorch(위 수식, fp32 내부→bf16) — SM80 폴백 경로.
    구현은 **vendor 우선**(kernel-kit/vendor/situ.py = 동일 핀 forward_native 발췌,
    vLLM 불필요) → import 실패 시 하네스 내장 동일 수식. 출처는 evidence.native_arm_source.
  - arm `fused`   (존재 시): `torch.ops._C.situ_and_mul` — C-확장이라 vendor 불가,
    venv-k3(브랜치 소스빌드)에서만 존재. torch+triton 만 있는 venv(triton 티어)에선
    미등록 → arm SKIP 사유 기록(게이트 FAIL 아님).
  NUMERIC 은 두 arm 모두 fp32 참조 대비 tolerance.TOL[k07] 로 판정.

참조 (NUMERIC 기준): 동일 bf16 입력을 fp32 로 승격해 위 수식을 fp32 로 계산한 값.

입력 분포: randn + 매 7번째 원소 ×32 — tanh/σ 포화 구간과 lβ=25 soft-clip 구간을
  반드시 커버 (β·tanh(g/β)는 |g|≳12 에서 ±4 포화, lβ·tanh(u/lβ)는 |u|≳60 에서 ±25 포화).

PERF: 판정 arm = fused 존재 시 fused(=S4 실경로), 없으면 native(그 경우 낮은 효율이
  곧 실배포 신호). PERF_POLICY k07 (mode=optional, eff_floor 0.30): "측정 제공 시에만
  바 적용" — 본 하네스는 기본으로 측정을 제공해 k07 이 PASS 에 도달 가능하게 한다
  (--no-perf 로 정책상 SKIP 경로 선택 가능). roofline < 30µs 케이스는 launch 고정비
  지배로 판정 제외(사전등록; m32 3케이스 해당) — 판정은 shared_inter6144_m8192.

CPU dry-run: 순수 python 으로 동일 수식 fp64 참조 vs bf16 반올림 모사 출력을
  실제 tol 정책으로 대조 + 음성 대조(β 왜곡 → FAIL 확인). 실커널 미실행 —
  4게이트 SKIP(사유), overall INCOMPLETE 상한. 자체점검 실패 시 exit 3.

exit: 0 FAIL 없음 / 1 FAIL / 2 사전조건 미충족 / 3 하네스 오류.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import struct
import subprocess
import sys
import traceback

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k07_situ_act"
BETA = shapes.SITU_BETA                # 4.0  (k3-config §1)
LINEAR_BETA = shapes.SITU_LINEAR_BETA  # 25.0 (k3-config §1)
PINS = {"vllm_pin": "a3c14efb", "hf_kimi_k3": "9f62e4e9"}

PERF_JUDGE_MIN_ROOFLINE_US = 30.0      # k04 와 동일 사전등록 규칙
SATURATE_STRIDE, SATURATE_MULT = 7, 32.0

FORMULA = ("out = [β·tanh(gate/β)·sigmoid(gate)] · [lβ·tanh(up/lβ)], "
           "gate=x[...,:d], up=x[...,d:], β=4.0, lβ=25.0 — "
           "HF modeling_kimi_linear.py@9f62e4e9 L64-91 = vLLM activation.py@a3c14efb "
           "SituAndMul.forward_native (동일 수식 원문 대조)")


# ===========================================================================
# 수식 (torch 경로)
# ===========================================================================
def _situ_fp32(torch, x_f32):
    """fp32 참조: HF/vLLM forward_native 수식을 fp32 입력에 그대로."""
    d = x_f32.shape[-1] // 2
    gate, up = x_f32[..., :d], x_f32[..., d:]
    a = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    u = LINEAR_BETA * torch.tanh(up / LINEAR_BETA)
    return a * u


# native arm 구현: vendor(핀 발췌) 우선 → 실패 시 하네스 내장 동일 수식 (모듈 import 는
# 호스트 dry-run 을 깨지 않도록 가드 — vendor/situ.py 는 torch 를 요구한다)
try:
    from vendor.situ import situ_and_mul_native as _vendor_situ  # noqa: E402
    _NATIVE_ARM_SOURCE = "vendor/situ.py@a3c14efb (SituAndMul.forward_native 발췌)"
except Exception as _e_vendor_situ:
    _vendor_situ = None
    _NATIVE_ARM_SOURCE = ("harness 내장 수식 (vendor import 실패: "
                          f"{type(_e_vendor_situ).__name__}: {_e_vendor_situ})"[:300])


def _situ_native(torch, x_bf16):
    """언퓨즈드 배포 경로 = vLLM forward_native: bf16 입력 → fp32 내부 → bf16 출력."""
    if _vendor_situ is not None:
        return _vendor_situ(x_bf16, BETA, LINEAR_BETA)
    return _situ_fp32(torch, x_bf16.float()).to(x_bf16.dtype)


def _probe_fused_op():
    """vLLM 브랜치 소스빌드의 `_C.situ_and_mul` 퓨즈드 CUDA op 탐색."""
    try:
        import vllm  # noqa: F401 — _C custom op 등록 부수효과
    except Exception as e:
        return None, f"vllm import 불가(torch-only venv?) — fused arm 생략: {type(e).__name__}: {e}"
    import torch
    try:
        op = torch.ops._C.situ_and_mul
        _ = op  # 존재 확인 (스키마 미등록이면 AttributeError/RuntimeError)
        return op, None
    except Exception as e:
        return None, ("_C.situ_and_mul 미등록 — kimi-k3 브랜치 소스빌드가 아님(RUNBOOK §3 "
                      f"venv-k3 확인): {type(e).__name__}: {e}")


def _fused_call(torch, op, x_bf16):
    """vLLM forward_cuda 재현 (activation.py L192-200): out 할당 + op(out, x, β, lβ)."""
    d = x_bf16.shape[-1] // 2
    out = torch.empty(x_bf16.shape[:-1] + (d,), dtype=x_bf16.dtype, device=x_bf16.device)
    op(out, x_bf16, BETA, LINEAR_BETA)
    return out


def _build_x(torch, case_name: str, m: int, n: int, dev):
    seed = shapes.stable_seed(KID, case_name)
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn((m, 2 * n), generator=g, dtype=torch.float32)
    x[:, ::SATURATE_STRIDE] *= SATURATE_MULT    # 포화/soft-clip 구간 커버 (양쪽 half 모두)
    return seed, x.to(torch.bfloat16).to(dev)


# ===========================================================================
# GPU 경로
# ===========================================================================
def run_gpu(args, rep: report.KernelReport) -> int:
    import torch
    dev = "cuda"
    op, op_err = _probe_fused_op()
    rep.add_evidence("formula", FORMULA)
    rep.add_evidence("native_arm_source", _NATIVE_ARM_SOURCE)
    rep.add_evidence("fused_op", {"available": op is not None,
                                  "detail": op_err or "torch.ops._C.situ_and_mul (브랜치 소스빌드)"})
    rep.add_evidence("input_dist",
                     f"randn + [::{SATURATE_STRIDE}]×{SATURATE_MULT} — tanh/sigmoid 포화 및 "
                     f"lβ={LINEAR_BETA} soft-clip 구간 포함 (seed=stable_seed)")

    vp = args.phase0_verdict if args.phase0_verdict and os.path.isfile(args.phase0_verdict) else None
    peaks = None
    if vp:
        tp = timing.resolve_twin_path(vp, args.twin)
        if tp:
            peaks = timing.load_twin_peaks(tp)

    wanted = list(shapes.REQUIRED_SHAPES[KID].items())
    if args.cases:
        wanted = [(nm, c) for nm, c in wanted if nm in args.cases]

    numeric_results: list[tolerance.NumericResult] = []
    covered: list[str] = []
    case_ev: dict = {}
    perf_judged: dict = {}
    perf_excluded: dict = {}
    case_errors: dict = {}
    judge_arm = "fused" if op is not None else "native"

    for name, cfg in wanted:
        m, n = cfg["m"], cfg["n"]
        try:
            seed, x = _build_x(torch, name, m, n, dev)
            ref = _situ_fp32(torch, x.float())              # fp32 참조

            out_native = _situ_native(torch, x)
            r_nat = tolerance.compare(KID, ref, out_native, case=f"{name}/native")
            numeric_results.append(r_nat)
            arms = {"native": {"passed": r_nat.passed, "rmse_rel": r_nat.metrics["rmse_rel"],
                               "mismatch_frac": r_nat.metrics.get("mismatch_frac")}}
            if op is not None:
                out_fused = _fused_call(torch, op, x)
                torch.cuda.synchronize()
                r_fu = tolerance.compare(KID, ref, out_fused, case=f"{name}/fused")
                numeric_results.append(r_fu)
                arms["fused"] = {"passed": r_fu.passed, "rmse_rel": r_fu.metrics["rmse_rel"],
                                 "mismatch_frac": r_fu.metrics.get("mismatch_frac"),
                                 "max_abs_native_vs_fused":
                                     (out_fused.float() - out_native.float()).abs().max().item()}
                del out_fused
            del ref, out_native

            # PERF 측정 (두 arm 다 기록, 판정은 judge_arm)
            model = shapes.perf_bytes_model(KID, m=m, n=n)
            fn_native = lambda: _situ_native(torch, x)                     # noqa: E731
            stats_n = timing.measure(fn_native, warmup=args.warmup, reps=args.reps)
            meas = {"native": {k: stats_n[k] for k in ("median_us", "cv", "reps")}}
            stats_j = stats_n
            if op is not None:
                fn_fused = lambda: _fused_call(torch, op, x)               # noqa: E731
                stats_f = timing.measure(fn_fused, warmup=args.warmup, reps=args.reps)
                meas["fused"] = {k: stats_f[k] for k in ("median_us", "cv", "reps")}
                stats_j = stats_f
            ev = {"seed": seed, "m": m, "n": n, "arms": arms, "measured": meas}
            if peaks and not args.no_perf:
                rl = timing.roofline_us(model["bytes"], model.get("macs", 0.0),
                                        model.get("alu_ops", 0.0), peaks)
                ev["roofline_us"] = rl["roofline_us"]
                if rl["roofline_us"] >= PERF_JUDGE_MIN_ROOFLINE_US:
                    perf_judged[name] = timing.perf_gate(
                        KID, stats_j["median_us"], model, vp, args.twin, stats=stats_j)
                else:
                    perf_excluded[name] = {
                        "roofline_us": rl["roofline_us"],
                        "median_us": {a: s["median_us"] for a, s in meas.items()},
                        "reason": f"roofline {rl['roofline_us']:.2f}µs < "
                                  f"{PERF_JUDGE_MIN_ROOFLINE_US}µs — launch 고정비 지배, 판정 제외(사전등록)"}
            covered.append(name)
            case_ev[name] = ev
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            case_errors[name] = {"error": f"{type(e).__name__}: {e}",
                                 "trace_tail": traceback.format_exc()[-1200:]}
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    rep.add_evidence("cases", case_ev)
    if case_errors:
        rep.add_evidence("case_errors", case_errors)

    rep.set_gate("NUMERIC", *tolerance.numeric_gate(KID, numeric_results))
    rep.set_gate("SHAPES", *shapes.shapes_gate(KID, covered))
    rep.set_gate("SANITIZER", *_sanitizer_gate(args))

    if args.no_perf:
        rep.set_gate("PERF", "SKIP",
                     "성능 후순위(PLAN 커널 표) — --no-perf 지정, 측정치는 evidence 참고",
                     {"measured_cases": {nm: e.get("measured") for nm, e in case_ev.items()}})
    elif vp is None:
        rep.set_gate("PERF", "SKIP",
                     f"PERF 목표 원천(verdict.json) 없음: {args.phase0_verdict} — 임의 목표 판정 금지",
                     {"measured_cases": {nm: e.get("measured") for nm, e in case_ev.items()}})
    elif peaks is None:
        rep.set_gate("PERF", "SKIP", "트윈(HW 피크 원천) 미해석 — verdict files/--twin 확인",
                     {"measured_cases": {nm: e.get("measured") for nm, e in case_ev.items()}})
    else:
        pev = {"judge_arm": judge_arm,
               "judge_rule": f"판정 arm={judge_arm}(fused=S4 실경로 우선), "
                             f"roofline < {PERF_JUDGE_MIN_ROOFLINE_US}µs 케이스 제외(사전등록)",
               "judged": {nm: {"status": g[0], "reason": g[1], **g[2]} for nm, g in perf_judged.items()},
               "excluded": perf_excluded}
        fails = {nm: g for nm, g in perf_judged.items() if g[0] == "FAIL"}
        if fails:
            n0, g0 = next(iter(fails.items()))
            rep.set_gate("PERF", "FAIL",
                         f"[arm={judge_arm}] {len(fails)}/{len(perf_judged)} 판정 케이스 미달 — "
                         f"예: [{n0}] {g0[1]}"
                         + (" (언퓨즈드 eager 다중패스 — fused op 빌드/torch.compile 검토 신호)"
                            if judge_arm == "native" else ""), pev)
        elif perf_judged:
            rep.set_gate("PERF", "PASS",
                         f"[arm={judge_arm}] 판정 {len(perf_judged)}케이스 floor 충족 "
                         f"(제외 {len(perf_excluded)}건 evidence)", pev)
        else:
            rep.set_gate("PERF", "SKIP", "판정 가능 케이스 없음(전부 roofline<임계 또는 실행 실패)", pev)

    rep.write(args.report_dir)
    return rep.exit_code()


# ---------------------------------------------------------------------------
# SANITIZER
# ---------------------------------------------------------------------------
def _sanitizer_gate(args) -> tuple:
    san = env_check.find_compute_sanitizer()
    if not san:
        return ("SKIP", "compute-sanitizer 미발견 — RUNBOOK §1.5 (CUDA 툴킷 bin/)", {})
    cmd = [san, "--tool", "memcheck", sys.executable,
           os.path.abspath(__file__), "--sanitizer-inner"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.sanitizer_timeout)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return report.sanitizer_gate_from_output(out, " ".join(cmd), p.returncode)
    except subprocess.TimeoutExpired:
        return ("SKIP", f"sanitizer {args.sanitizer_timeout}s 타임아웃 — 판독 불가(수동 재실행 필요)",
                {"cmd": " ".join(cmd)})


def _sanitizer_inner() -> int:
    """sanitizer 하 1회: native + (존재 시) fused — m=64, n=3072(routed 폭)."""
    import torch
    dev = "cuda"
    _, x = _build_x(torch, "sanitizer_m64_n3072", 64, shapes.MOE_INTER, dev)
    _ = _situ_native(torch, x)
    op, err = _probe_fused_op()
    ran = ["native"]
    if op is not None:
        _ = _fused_call(torch, op, x)
        ran.append("fused")
    torch.cuda.synchronize()
    print(f"[k07 sanitizer-inner] 실행 arm={ran}" + (f" (fused 생략: {err})" if err else ""))
    return 0


# ===========================================================================
# CPU dry-run — 순수 python
# ===========================================================================
def _bf16(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    lo, hi = b & 0xFFFF, b >> 16
    if lo > 0x8000 or (lo == 0x8000 and (hi & 1)):
        hi += 1
    return struct.unpack("<f", struct.pack("<I", (hi << 16) & 0xFFFFFFFF))[0]


def _situ_py(gate: float, up: float, beta: float = BETA, lbeta: float = LINEAR_BETA) -> float:
    a = beta * math.tanh(gate / beta) * (1.0 / (1.0 + math.exp(-gate)))
    u = lbeta * math.tanh(up / lbeta)
    return a * u


def run_dry(args, rep: report.KernelReport) -> int:
    results = []
    scaled_cases = []
    for name, cfg in shapes.REQUIRED_SHAPES[KID].items():
        m, n = min(cfg["m"], 8), 16                 # 축소
        seed = shapes.stable_seed(KID, name)
        rnd = random.Random(seed)
        ref, out = [], []
        fixed = [0.0, 0.1, -0.1, 1.0, -1.0, 5.0, -5.0, 30.0, -30.0, 80.0, -80.0]
        for _ in range(m):
            for j in range(n):
                gate = _bf16(fixed[j % len(fixed)] if j % 3 == 0 else rnd.gauss(0, 1.5))
                up = _bf16(fixed[(j + 5) % len(fixed)] if j % 3 == 1 else rnd.gauss(0, 1.5))
                y = _situ_py(gate, up)              # 동일 bf16 입력, fp64 수식
                ref.append(y)
                out.append(_bf16(y))                # bf16 출력 반올림 모사(배포 경로 semantics)
        r = tolerance.compare(KID, ref, out, case=f"{name}(scaled)")
        results.append(r)
        scaled_cases.append({"case": name, "m_scaled": m, "n_scaled": n, "seed": seed,
                             "passed": r.passed, "rmse_rel": r.metrics["rmse_rel"],
                             "mismatch_frac": r.metrics.get("mismatch_frac")})
    # 음성 대조: β=3.7 로 왜곡 — 포화 구간(±4→±3.7, ~7.5% 오차)에서 FAIL 해야 함
    bad_out = []
    rnd = random.Random(shapes.stable_seed(KID, "negative_control"))
    ref_neg = []
    for _ in range(512):
        gate = _bf16(rnd.choice([15.0, -15.0, 30.0, 2.0, 0.5]) * (1 + 0.1 * rnd.random()))
        up = _bf16(rnd.gauss(0, 2))
        ref_neg.append(_situ_py(gate, up))
        bad_out.append(_bf16(_situ_py(gate, up, beta=3.7)))
    bad = tolerance.compare(KID, ref_neg, bad_out, case="negative_control(beta=3.7)")
    selfcheck_ok = all(r.passed for r in results) and (not bad.passed)

    rep.add_evidence("formula", FORMULA)
    rep.add_evidence("dry_run", {
        "mode": "cpu-dry-run(순수 python, torch/GPU 불필요)",
        "check": "fp64 수식 참조 vs bf16 반올림 모사 출력 — 실제 TOL[k07] 정책으로 판정",
        "scaled_cases": scaled_cases,
        "negative_control_failed_as_expected": not bad.passed,
        "selfcheck": "PASS" if selfcheck_ok else "FAIL"})
    rep.set_gate("NUMERIC", "SKIP",
                 "cpu-dry-run: 실커널 미실행 — 수식·판정 파이프 자체점검만 수행(evidence.dry_run)")
    rep.set_gate("SHAPES", "SKIP",
                 "cpu-dry-run: 축소 shape 실행 — K3 실측 shape 판정은 GPU 모드 전용")
    rep.set_gate("SANITIZER", "SKIP", "cpu-dry-run: GPU 없음")
    rep.set_gate("PERF", "SKIP", "cpu-dry-run: 측정 없음")
    rep.write(args.report_dir)
    print(f"[k07 dry-run] selfcheck={'PASS' if selfcheck_ok else 'FAIL'} "
          f"(scaled {len(results)}케이스, 음성대조 FAIL 확인={not bad.passed})")
    return report.EXIT_OK if selfcheck_ok else report.EXIT_ERR


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=f"{KID} S2 하네스")
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--twin", default=None)
    ap.add_argument("--report-dir", default=os.path.join(_KIT, "kernel-report"))
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--no-perf", action="store_true",
                    help="PERF 게이트를 정책상 SKIP(성능 후순위)으로 남김")
    ap.add_argument("--sanitizer-inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sanitizer-timeout", type=int, default=900)
    args = ap.parse_args()

    if args.sanitizer_inner:
        return _sanitizer_inner()

    rep = report.KernelReport(KID, dry_run=args.cpu_dry_run, pins=PINS)
    if args.cpu_dry_run:
        return run_dry(args, rep)

    env = env_check.collect_env()
    ok, checks = env_check.check_preconditions(env, cpu_dry_run=False)
    if not ok:
        print(f"[{KID}] GPU 사전조건 미충족 — --cpu-dry-run 으로 참조 경로만 검증 가능",
              file=sys.stderr)
        for c in checks:
            if c["status"] == "FAIL":
                print(f"  FAIL {c['check']}: {c['detail']}", file=sys.stderr)
        return report.EXIT_PRECOND
    return run_gpu(args, rep)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
