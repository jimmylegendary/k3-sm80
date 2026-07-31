"""CUDA event 타이밍(warmup/rep) + Phase 0 verdict 기반 PERF 게이트.

PERF 목표 계약 (docs/PLAN.md §S2 완료판정 4)
  - 목표 원천 = Phase 0 산출물. **경로는 하드코딩하지 않는다** — 하네스/CLI가
    `--phase0-verdict <path>` 로 넘긴 verdict.json 이 유일한 진입점이고, 하드웨어
    피크(HBM/TC)는 verdict["files"] 에 등재된 트윈(twin_a100_x32.json)을 verdict 위치
    기준으로 해석해 읽는다(--twin 으로 오버라이드 가능). 원천이 없으면 PERF는
    임의값으로 판정하지 않고 SKIP(사유) 처리한다.
  - 커널별 목표: target_us = roofline_us(bytes, macs, alu) / eff_floor.
    roofline_us = max(bytes/HBM_BW, macs/TC_PEAK, alu/ALU_PEAK).
    eff_floor(임계 비율)는 아래 PERF_POLICY 에 커널별 근거와 함께 명시.
  - verdict 의 best_real/best_opt 비율(≈0.536)은 Phase 0 이 "real" 시나리오에서
    가정한 종합 효율 마진 — 기본 floor 의 출처로 evidence 에 전사한다.

타이밍 계약: warmup 후 rep 반복을 CUDA event 로 개별 측정, 통계(median 중심)로 판정.
             측정 자체는 비결정이지만 임계 판정 + 고정 rep/warmup 으로 재현 가능 범위 유지.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import shapes  # noqa: E402


# ---------------------------------------------------------------------------
# 1) Phase 0 원천 로드 (경로는 전부 인자)
# ---------------------------------------------------------------------------
def load_verdict(verdict_path: str) -> dict:
    with open(verdict_path, encoding="utf-8") as f:
        v = json.load(f)
    for key in ("gate", "pins"):
        if key not in v:
            raise ValueError(f"verdict.json 형식 불일치: '{key}' 없음 ({verdict_path})")
    return v


def resolve_twin_path(verdict_path: str, twin_path: str | None = None) -> str | None:
    """트윈 경로 해석: 명시 인자 > verdict["files"]의 twin 항목(repo 루트 기준 상대경로)."""
    if twin_path:
        return twin_path if os.path.isfile(twin_path) else None
    v = load_verdict(verdict_path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(verdict_path)))
    for rel in v.get("files", []):
        if "twin" in os.path.basename(rel):
            cand = os.path.join(repo_root, rel)
            if os.path.isfile(cand):
                return cand
    return None


def load_twin_peaks(twin_path: str) -> dict:
    """트윈에서 단일 GPU 피크 추출: HBM bytes/s, TC MACs/s, (유도) fp32 ALU FLOPs/s.

    트윈 계층(phase0/twin_a100_x32.json): ib → nvlink → hbm → l2 → smem → pe.
    - hbm.bandwidth_bps = 2039e9 (A100 80GB SXM, vendor_datasheet)
    - pe.peak_macs_per_s = SM당 bf16 TC MACs/s (1.444e12), pe.instances = 108
      → TC 피크 = 1.444e12 × 108 ≈ 1.56e14 MACs/s (= 312 TFLOPS bf16)
    - fp32 ALU 피크(dequant 등 비-TC 연산)는 트윈 필드로 유도:
      clock_hz × instances × 64 FMA/clk × 2 = 1.41e9 × 108 × 128 ≈ 1.95e13 FLOPs/s
      (A100 SM당 FP32 64 lane — CUDA Ampere Tuning Guide; 유도식임을 evidence에 명시)
    """
    with open(twin_path, encoding="utf-8") as f:
        twin = json.load(f)

    found: dict = {}

    def walk(node: dict):
        lid = node.get("level_id")
        if lid == "hbm":
            found["hbm_bw"] = float(node["bandwidth_bps"])
        if lid == "pe":
            found["tc_macs"] = float(node["peak_macs_per_s"]) * int(node.get("instances", 1))
            found["alu_flops"] = float(node["clock_hz"]) * int(node.get("instances", 1)) * 128.0
            found["sm_count"] = int(node.get("instances", 1))
        for c in node.get("children", []):
            walk(c)

    walk(twin)
    if "hbm_bw" not in found or "tc_macs" not in found:
        raise ValueError(f"트윈에서 hbm/pe 레벨을 찾지 못함: {twin_path}")
    found["twin_path"] = twin_path
    return found


# ---------------------------------------------------------------------------
# 2) 루프라인 + 커널별 PERF 정책
# ---------------------------------------------------------------------------
def roofline_us(bytes_moved: float, macs: float = 0.0, alu_ops: float = 0.0,
                peaks: dict | None = None) -> dict:
    t_mem = bytes_moved / peaks["hbm_bw"] * 1e6
    t_tc = macs / peaks["tc_macs"] * 1e6 if macs else 0.0
    t_alu = alu_ops / peaks["alu_flops"] * 1e6 if alu_ops else 0.0
    t = max(t_mem, t_tc, t_alu)
    bound = {t_mem: "mem", t_tc: "tc", t_alu: "alu"}[t]
    return {"roofline_us": t, "bound": bound,
            "t_mem_us": t_mem, "t_tc_us": t_tc, "t_alu_us": t_alu}


# eff_floor = "측정시간 ≤ roofline/eff_floor 이면 PASS" (즉 달성효율 ≥ floor).
PERF_POLICY = {
    "k01_kda_chunk": dict(
        eff_floor=0.35, goal=0.50,
        basis="FLA Triton은 H100 튜닝 기준 — A100(smem 164KB) 미튜닝 초기 바 0.35. "
              "FlashKDA가 Triton 대비 1.72–2.22×(H20, recon/kda-fla.md §2) → Triton "
              "달성가능선 ~0.5를 튜닝 목표(goal)로 병기. S2 작업 자체가 이 타일 튜닝."),
    "k02_kda_recurrent": dict(
        eff_floor=0.50,
        basis="state 6.29MB/seq fp32 r+w 스트림 지배(memory-bound) — 50% 미만이면 "
              "launch/인덱싱 병목 의심. Phase 0 real 마진(0.536)과 동수준."),
    "k03_causal_conv1d": dict(
        eff_floor=0.50, basis="순수 스트림 커널(3탭 state r/w) — k02와 동일 논거."),
    "k04_mla_decode": dict(
        eff_floor=0.50,
        basis="latent KV 576B/tok 스트림 지배(특히 ctx 128k). vLLM TRITON_MLA는 "
              "SM80 지원 명시(PLAN 커널 표) — 스트림 효율 50%를 바닥으로."),
    "k05_marlin_moe_w4a16": dict(
        eff_floor=0.40,
        basis="max(HBM, dequant-ALU) 루프라인 대비 40% — marlin은 잘 튜닝된 커널이나 "
              "SM80 MoE 경로는 crash(#35922) 수정 후 첫 실측이라 보수적 바. "
              "dequant ops/param은 무근거 밴드(verdict caveats) → k06 재보정 후 목표 갱신."),
    "k06_dequant_bench": dict(
        mode="measure_only", cv_max=0.10,
        basis="목적이 dequant_ops/param '측정' 자체(phase0 모델 재보정용, verdict caveats의 "
              "무근거 밴드 해소). 판정은 반복 안정성 CV ≤ 10% + 산출값 기록."),
    "k07_situ_act": dict(
        mode="optional", eff_floor=0.30,
        basis="PLAN 커널 표: SiTU는 언퓨즈드 참조로 대체(성능 후순위) — 측정 없으면 "
              "SKIP(성능 후순위), 측정 제공 시에만 0.30 바 적용."),
    "k08_mxfp4_marlin_moe": dict(
        eff_floor=0.40,
        basis="원본 MXFP4 weight-only Marlin MoE의 최초 SM80 gate. k05 W4A16과 같은 "
              "보수적 루프라인 효율 0.40을 사용하고 TP8/EP32 양 arm을 별도 기록."),
}


def perf_gate(kernel_id: str, measured_us: float | None, model: dict | None,
              verdict_path: str | None, twin_path: str | None = None,
              stats: dict | None = None) -> tuple[str, str, dict]:
    """PERF 게이트 판정.

    measured_us: 측정 중앙값(µs). None이면 (optional 커널만) SKIP.
    model: shapes.perf_bytes_model() 반환 dict (bytes/macs/alu_ops/note).
    verdict_path: Phase 0 verdict.json 경로 (없으면 SKIP — 임의 목표 금지).
    stats: measure() 반환 통계(있으면 evidence 전사; k06 CV 판정에 사용).
    """
    pol = PERF_POLICY[kernel_id]
    ev: dict = {"basis": pol["basis"], "policy": {k: v for k, v in pol.items() if k != "basis"}}

    if verdict_path is None or not os.path.isfile(verdict_path):
        return ("SKIP", f"PERF 목표 원천(verdict.json) 없음: {verdict_path} — 임의 목표 판정 금지", ev)
    v = load_verdict(verdict_path)
    ev["verdict"] = {"path": os.path.abspath(verdict_path),
                     "gate": v["gate"]["verdict"], "pins": v["pins"],
                     "real_opt_margin": (v["best_opt"]["step_ms"] / v["best_real"]["step_ms"])
                     if "best_real" in v and "best_opt" in v else None}

    if pol.get("mode") == "measure_only":
        if measured_us is None or stats is None:
            return ("SKIP", "측정 미수행 — measure_only 커널은 stats 필수", ev)
        cv = stats.get("cv", float("inf"))
        ev["stats"] = stats
        if cv <= pol["cv_max"]:
            return ("PASS", f"측정 산출 완료 (median {measured_us:.1f}µs, CV {cv:.3f} ≤ "
                            f"{pol['cv_max']}) — 재보정 데이터로 반출", ev)
        return ("FAIL", f"반복 불안정: CV {cv:.3f} > {pol['cv_max']} — 측정 신뢰 불가", ev)

    if measured_us is None:
        if pol.get("mode") == "optional":
            return ("SKIP", "성능 후순위(PLAN 커널 표) — 측정 미수행", ev)
        return ("SKIP", "측정값 없음 (GPU 미실행)", ev)

    tp = resolve_twin_path(verdict_path, twin_path)
    if tp is None:
        return ("SKIP", "트윈(HW 피크 원천) 미해석 — verdict files/--twin 확인", ev)
    peaks = load_twin_peaks(tp)
    rl = roofline_us(model["bytes"], model.get("macs", 0.0), model.get("alu_ops", 0.0), peaks)
    floor = pol["eff_floor"]
    target_us = rl["roofline_us"] / floor
    eff = rl["roofline_us"] / measured_us if measured_us > 0 else 0.0
    ev.update({
        "model": model, "roofline": rl,
        "peaks": {k: peaks[k] for k in ("hbm_bw", "tc_macs", "alu_flops", "sm_count")},
        "twin_path": os.path.abspath(tp),
        "measured_us": measured_us, "target_us": target_us,
        "achieved_eff": eff, "eff_floor": floor,
    })
    if stats:
        ev["stats"] = stats
    if measured_us <= target_us:
        msg = f"측정 {measured_us:.1f}µs ≤ 목표 {target_us:.1f}µs (달성효율 {eff:.2f} ≥ {floor})"
        if pol.get("goal") and eff < pol["goal"]:
            msg += f" — 단 튜닝 목표({pol['goal']}) 미달, S2 튜닝 항목"
        return ("PASS", msg, ev)
    return ("FAIL", f"측정 {measured_us:.1f}µs > 목표 {target_us:.1f}µs "
                    f"(달성효율 {eff:.2f} < {floor}, bound={rl['bound']})", ev)


# ---------------------------------------------------------------------------
# 3) 측정기
# ---------------------------------------------------------------------------
def measure(fn, warmup: int = 10, reps: int = 50) -> dict:
    """CUDA event 기반 측정. GPU 전용 — 호스트에서는 measure_cpu 를 쓸 것."""
    import torch
    assert torch.cuda.is_available(), "measure()는 CUDA 필수 — dry-run은 measure_cpu()"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)   # ms → µs
    return _stats(times, warmup, reps, "cuda_event")


def measure_cpu(fn, warmup: int = 2, reps: int = 5) -> dict:
    """dry-run용 wall-clock 측정 — 판정용 아님(리포트 경로 검증·참조구현 소요 파악용)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return _stats(times, warmup, reps, "cpu_wall")


def _stats(times_us: list, warmup: int, reps: int, method: str) -> dict:
    med = statistics.median(times_us)
    mean = statistics.fmean(times_us)
    sd = statistics.pstdev(times_us) if len(times_us) > 1 else 0.0
    qs = sorted(times_us)
    return {"method": method, "warmup": warmup, "reps": reps,
            "median_us": med, "mean_us": mean, "min_us": qs[0], "max_us": qs[-1],
            "p90_us": qs[min(len(qs) - 1, math.ceil(0.9 * len(qs)) - 1)],
            "cv": (sd / mean) if mean > 0 else 0.0}


# ---------------------------------------------------------------------------
# 4) CLI / selftest (CPU에서 목표 산출 경로 전체 검증)
# ---------------------------------------------------------------------------
def _selftest(verdict_path: str, twin_path: str | None) -> int:
    rows = []
    ok = True
    for kid in shapes.KERNEL_IDS:
        case_name, case = next(iter(shapes.REQUIRED_SHAPES[kid].items()))
        model = shapes.perf_bytes_model(kid, **{k: v for k, v in case.items() if k != "note"})
        pol = PERF_POLICY[kid]
        if pol.get("mode") == "measure_only":
            st_fake = {"cv": 0.05, "median_us": 100.0}
            g = perf_gate(kid, 100.0, model, verdict_path, twin_path, stats=st_fake)
            ok &= g[0] == "PASS"
            g2 = perf_gate(kid, 100.0, model, verdict_path, twin_path, stats={"cv": 0.5})
            ok &= g2[0] == "FAIL"
            rows.append((kid, case_name, "-", g[0]))
            continue
        tp = resolve_twin_path(verdict_path, twin_path)
        if tp is None:
            print(f"[timing selftest] 트윈 미해석 — {kid} SKIP 경로 확인")
            g = perf_gate(kid, 1.0, model, verdict_path, twin_path)
            ok &= g[0] == "SKIP"
            continue
        peaks = load_twin_peaks(tp)
        rl = roofline_us(model["bytes"], model.get("macs", 0.0), model.get("alu_ops", 0.0), peaks)
        floor = pol.get("eff_floor", 0.5)
        # 합성 측정값으로 PASS/FAIL 양방향 검증
        g_pass = perf_gate(kid, rl["roofline_us"] / floor * 0.9, model, verdict_path, twin_path)
        g_fail = perf_gate(kid, rl["roofline_us"] / floor * 1.5, model, verdict_path, twin_path)
        g_none = perf_gate(kid, None, model, verdict_path, twin_path)
        ok &= g_pass[0] == "PASS" and g_fail[0] == "FAIL" and g_none[0] == "SKIP"
        rows.append((kid, case_name, f"{rl['roofline_us']:.1f}µs/{rl['bound']}", g_pass[0]))
    print(f"{'커널':24s} {'대표케이스':18s} {'루프라인':>16s} {'합성판정':>8s}")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:18s} {r[2]:>16s} {r[3]:>8s}")
    print(f"timing selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    vp = None
    tp = None
    if "--phase0-verdict" in args:
        vp = args[args.index("--phase0-verdict") + 1]
    if "--twin" in args:
        tp = args[args.index("--twin") + 1]
    if vp is None:
        print("사용법: python3 timing.py --selftest --phase0-verdict <verdict.json> [--twin <twin.json>]",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(_selftest(vp, tp))
