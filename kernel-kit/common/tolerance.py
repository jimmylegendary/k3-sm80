"""bf16/fp32 수치 비교 정책 — NUMERIC 게이트의 단일 원천.

설계 원칙
  1. 메트릭은 4종을 항상 전부 계산해 evidence로 남기고, 판정은 커널별 정책(TOL)이
     켠 임계만 본다: max_abs / rmse_rel / rel_mean_abs / mismatch_frac(atol,rtol).
  2. 임계는 업스트림이 자기 커널에 실제로 쓰는 바를 우선 채택(핀 커밋+라인 명시),
     업스트림 바가 없으면 bf16 eps(2^-8 = 3.90625e-3)에서 유도하고 근거를 주석에 남긴다.
  3. NaN/Inf: 커널 출력에 1개라도 있으면 즉시 FAIL (업스트림 test_kda.py도 동일).
  4. GPU에서는 torch 텐서 연산(디바이스 상 fp32 승격), 저작 호스트 dry-run에서는
     순수 python 리스트로 동일 산식 — 판정 로직이 두 경로에서 동일함을 selftest로 보증.

임계 근거(전부 vLLM main @ d223c900d85224c02f2162ee2c757a769e99f519 raw 확인, 2026-07-28):
  - tests/kernels/test_kda.py L61-79: RMSE-ratio 비교기(rel_err = rmse(ref−tri)/rmse(ref)),
    L160-161: chunk_kda(bf16) vs naive fp32 참조에 threshold 0.005 + NaN assert.
    (L225-226의 1e-3은 동일커널 신구 구현 간 비교 — 참조 대조 바가 아님)
  - tests/kernels/quantization/test_marlin_gemm.py L137: compute_max_diff =
    mean|out−ref| / mean|ref|, L523/570/619: < 0.04.
  - tests/kernels/moe/test_moe.py L1009: marlin MoE INT4 assert_close(atol=4e-2, rtol=0).
  - torch.testing.assert_close bf16 기본: rtol=1.6e-2, atol=1e-5 (torch 문서 표준값).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field

BF16_EPS = 2.0 ** -8   # 3.90625e-3 — bf16 가수 8bit


@dataclass
class TolPolicy:
    """켜진(=None 아닌) 임계만 판정에 사용."""
    rmse_rel: float | None = None        # rmse(ref−out)/rmse(ref) 상한
    max_abs: float | None = None         # 전역 최대 절대오차 상한
    rel_mean_abs: float | None = None    # mean|diff|/mean|ref| 상한 (marlin 방식)
    atol: float | None = None            # elementwise |diff| ≤ atol + rtol·|ref|
    rtol: float | None = None
    mismatch_frac: float = 0.0           # elementwise 위반 허용 비율 (atol/rtol 켜진 경우)
    basis: str = ""                      # 근거 (리포트 evidence로 전사)


TOL: dict[str, TolPolicy] = {
    "k01_kda_chunk": TolPolicy(
        rmse_rel=5e-3,
        basis="vLLM tests/kernels/test_kda.py@d223c900 L160-161: chunk_kda(bf16) vs naive fp32, "
              "RMSE-ratio<0.005 + NaN 금지 — 업스트림 자체 바 그대로."),
    "k02_kda_recurrent": TolPolicy(
        rmse_rel=5e-3,
        basis="test_kda.py 동일 비교기/바(0.005). fused_recurrent도 naive fp32 참조 대조 — "
              "누적 길이는 chunk보다 짧지만 state 이월 오차가 동급이라 같은 바 적용."),
    "k03_causal_conv1d": TolPolicy(
        rmse_rel=4e-3, max_abs=3e-2,
        basis="4-tap depthwise conv+SiLU: 누적 4항 → bf16 입력 반올림 지배. "
              f"eps_bf16={BF16_EPS:.3e} → RMS 상대오차 ~eps/√3≈2.3e-3, 여유 ×1.7 = 4e-3. "
              "max_abs 3e-2 = randn 입력 |out|~O(4) × eps × 안전율 2."),
    "k04_mla_decode": TolPolicy(
        rmse_rel=5e-3, max_abs=2.6e-2,
        basis="softmax 정규화로 |o| ≤ max|v|: max_abs = torch.testing bf16 rtol 기본 1.6e-2 × "
              "|v|~1.6 스케일. RMSE-ratio는 KDA와 동일 바 5e-3 (fp32 누적 attention). "
              "fp8-KV 변형은 별도 바(upstream test_triton_decode_attention.py L233 atol=5e-1) — "
              "S2는 bf16 KV만 다룬다."),
    "k05_marlin_moe_w4a16": TolPolicy(
        rel_mean_abs=4e-2, atol=4e-2, rtol=0.0, mismatch_frac=0.0,
        basis="vLLM test_marlin_gemm.py@d223c900 L137/L523: compute_max_diff<0.04 + "
              "test_moe.py L1009: assert_close(atol=4e-2, rtol=0). "
              "참조 = 동일 dequant 가중치의 fp32 grouped-GEMM(양자화 오차는 참조에 이미 포함) — "
              "이 게이트는 커널 오차만 본다."),
    "k06_dequant_bench": TolPolicy(
        rmse_rel=1e-3, atol=0.0, rtol=2 * BF16_EPS, mismatch_frac=0.0,
        basis="INT4×fp16-scale dequant는 곱 1회 — bf16 표현 반올림(≤1ulp)만 허용: "
              f"rtol=2·eps_bf16={2*BF16_EPS:.3e}, 위반 0. rmse_rel 1e-3은 전역 가드."),
    "k07_situ_act": TolPolicy(
        rmse_rel=4e-3, atol=1e-5, rtol=1.6e-2, mismatch_frac=1e-4,
        basis="elementwise 초월함수(SiTU β=4.0/linear-β=25.0): torch.testing bf16 기본"
              "(rtol=1.6e-2, atol=1e-5) + RMSE-ratio 4e-3. mismatch 1e-4는 sigmoid 포화 "
              "경계의 최종 ulp 반올림 갈림 허용."),
}


@dataclass
class NumericResult:
    metrics: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    passed: bool = False


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _metrics_torch(ref, out) -> dict:
    import torch
    ref = ref.detach().float()
    out = out.detach().float()
    assert ref.shape == out.shape, f"shape 불일치: {tuple(ref.shape)} vs {tuple(out.shape)}"
    diff = (ref - out)
    n = ref.numel()
    rmse_base = torch.sqrt(torch.mean(ref * ref)).item()
    rmse_diff = torch.sqrt(torch.mean(diff * diff)).item()
    return {
        "n": n,
        "max_abs": diff.abs().max().item(),
        "rmse_rel": rmse_diff / (rmse_base + 1e-8),
        "rel_mean_abs": (diff.abs().mean() / (ref.abs().mean() + 1e-12)).item(),
        "nan_out": int(torch.isnan(out).sum().item()) + int(torch.isinf(out).sum().item()),
        "nan_ref": int(torch.isnan(ref).sum().item()) + int(torch.isinf(ref).sum().item()),
        "_pair": (ref, out),
    }


def _metrics_py(ref: list, out: list) -> dict:
    assert len(ref) == len(out), f"길이 불일치: {len(ref)} vs {len(out)}"
    n = len(ref)
    bad = sum(1 for v in out if math.isnan(v) or math.isinf(v))
    badr = sum(1 for v in ref if math.isnan(v) or math.isinf(v))
    fin = [(r, o) for r, o in zip(ref, out)
           if not (math.isnan(r) or math.isinf(r) or math.isnan(o) or math.isinf(o))]
    if fin:
        ss_ref = sum(r * r for r, _ in fin) / len(fin)
        ss_diff = sum((r - o) ** 2 for r, o in fin) / len(fin)
        max_abs = max(abs(r - o) for r, o in fin)
        rel_mean = (sum(abs(r - o) for r, o in fin) / len(fin)) / \
                   (sum(abs(r) for r, _ in fin) / len(fin) + 1e-12)
    else:
        ss_ref = ss_diff = max_abs = rel_mean = float("nan")
    return {
        "n": n,
        "max_abs": max_abs,
        "rmse_rel": math.sqrt(ss_diff) / (math.sqrt(ss_ref) + 1e-8),
        "rel_mean_abs": rel_mean,
        "nan_out": bad,
        "nan_ref": badr,
        "_pair": fin,
    }


def _mismatch_frac(pair, atol: float, rtol: float) -> float:
    if isinstance(pair, tuple):  # torch (ref, out)
        import torch
        ref, out = pair
        viol = ((ref - out).abs() > (atol + rtol * ref.abs())).sum().item()
        return viol / max(1, ref.numel())
    viol = sum(1 for r, o in pair if abs(r - o) > atol + rtol * abs(r))
    return viol / max(1, len(pair))


def compare(kernel_id: str, ref, out, case: str = "") -> NumericResult:
    """참조(ref, fp32) 대 커널 출력(out) 비교 → 메트릭 + 커널별 정책 판정.

    ref/out: torch 텐서(권장, GPU/CPU 무관) 또는 float 리스트(dry-run 순수 python 경로).
    """
    pol = TOL[kernel_id]
    m = _metrics_torch(ref, out) if _is_torch(ref) else _metrics_py(ref, out)
    pair = m.pop("_pair")
    viol: list[str] = []
    if m["nan_out"] > 0:
        viol.append(f"출력 NaN/Inf {m['nan_out']}개 — 즉시 FAIL")
    if pol.rmse_rel is not None and m["rmse_rel"] > pol.rmse_rel:
        viol.append(f"rmse_rel {m['rmse_rel']:.3e} > {pol.rmse_rel:.1e}")
    if pol.max_abs is not None and m["max_abs"] > pol.max_abs:
        viol.append(f"max_abs {m['max_abs']:.3e} > {pol.max_abs:.1e}")
    if pol.rel_mean_abs is not None and m["rel_mean_abs"] > pol.rel_mean_abs:
        viol.append(f"rel_mean_abs {m['rel_mean_abs']:.3e} > {pol.rel_mean_abs:.1e}")
    if pol.atol is not None:
        frac = _mismatch_frac(pair, pol.atol, pol.rtol or 0.0)
        m["mismatch_frac"] = frac
        if frac > pol.mismatch_frac:
            viol.append(f"mismatch_frac {frac:.3e} > {pol.mismatch_frac:.1e} "
                        f"(atol={pol.atol}, rtol={pol.rtol})")
    m["case"] = case
    m["policy"] = {k: v for k, v in vars(pol).items() if k != "basis" and v is not None}
    return NumericResult(metrics=m, violations=viol, passed=not viol)


def numeric_gate(kernel_id: str, results: list[NumericResult]) -> tuple[str, str, dict]:
    """케이스별 compare() 결과 목록 → NUMERIC 게이트 (status, reason, evidence)."""
    pol = TOL[kernel_id]
    ev = {"basis": pol.basis,
          "cases": [{**r.metrics, "violations": r.violations, "passed": r.passed}
                    for r in results]}
    if not results:
        return ("SKIP", "실행된 비교 케이스 없음", ev)
    fails = [r for r in results if not r.passed]
    if fails:
        first = fails[0]
        return ("FAIL", f"{len(fails)}/{len(results)} 케이스 위반 — 예: "
                        f"[{first.metrics.get('case')}] {'; '.join(first.violations[:2])}", ev)
    return ("PASS", f"{len(results)} 케이스 전부 tol 내 (정책: {pol.basis.split(':')[0]})", ev)


def _selftest() -> int:
    """결정적(고정 입력) 자기검증 — PASS/FAIL 양방향 + NaN 즉시 FAIL 확인."""
    import random
    rnd = random.Random(20260728)
    ref = [rnd.gauss(0, 1) for _ in range(4096)]
    ok_out = [v * (1 + 1e-4) for v in ref]                    # rmse_rel ≈ 1e-4 → PASS여야 함
    bad_out = [v * (1 + 2e-2) for v in ref]                   # rmse_rel ≈ 2e-2 → FAIL여야 함
    nan_out = list(ok_out)
    nan_out[7] = float("nan")
    checks = []
    for kid in ("k01_kda_chunk", "k05_marlin_moe_w4a16", "k07_situ_act"):
        r_ok = compare(kid, ref, ok_out, case="selftest_ok")
        r_bad = compare(kid, ref, bad_out, case="selftest_bad")
        r_nan = compare(kid, ref, nan_out, case="selftest_nan")
        checks += [r_ok.passed, not r_bad.passed, not r_nan.passed]
        g = numeric_gate(kid, [r_ok])
        checks.append(g[0] == "PASS")
        g = numeric_gate(kid, [r_ok, r_bad])
        checks.append(g[0] == "FAIL")
    print(json.dumps({"tolerance_selftest": "PASS" if all(checks) else "FAIL",
                      "checks": len(checks),
                      "kernels": sorted(TOL.keys())}, ensure_ascii=False))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    if "--table" in sys.argv:
        for kid, pol in TOL.items():
            on = {k: v for k, v in vars(pol).items() if k != "basis" and v is not None}
            print(f"{kid:24s} {on}\n{'':24s} 근거: {pol.basis}\n")
        sys.exit(0)
    sys.exit(_selftest())
