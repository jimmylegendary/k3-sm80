#!/usr/bin/env bash
# PERF 게이트 교정 측정 — 이 A100 의 **실제 달성 가능** 피크를 재고, 커널 효율을 그에 대해 재표현.
#
# 왜: PERF 목표는 트윈의 이론 피크(HBM 2039GB/s, TC 312TFLOPS)를 eff_floor 로 나눈 값이다.
#     서로 다른 커널이 일제히 30~32% 에 몰리면 '커널이 느리다'보다 '이론 피크 대비 floor 가
#     과공격적'이거나 '이 기기의 달성 피크가 이론보다 낮다'가 원인일 수 있다. 추정으로 floor 를
#     낮추면 게이트를 무력화하는 것이므로, 물리 피크를 **측정**해 근거를 만든다
#     (HATIR 원칙: magic derate 금지, 물리 인자 분해).
#
# 사용:   bash run/27-perf-calib.sh [triton|full]
# 결과:   C0~C9 (≤12줄) — 기기 정보 / 측정 달성 HBM·TC 피크 / 커널별 효율(이론 vs 달성 기준).
#         이 출력 전체를 전달하면 floor 재조정 여부를 근거로 판정한다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-triton}"
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit:$TIER"
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "[calib] $IMG 없음 — 먼저: bash run/10-load-image.sh $TIER" >&2; exit 3; }

echo "[calib] 시작: tier=$TIER image=$IMG (C0~C9 진행상황은 즉시 출력)"
docker run --rm -i --gpus all -v "$ROOT":/work \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton \
  "$IMG" python3 -u - <<'EOF'
import json, os, sys
import torch

W = 76
def line(s): print(str(s)[:W])

d = torch.cuda.get_device_properties(0)
line(f"C0 {d.name} cap={d.major}.{d.minor} SM={d.multi_processor_count} "
     f"mem={d.total_memory/2**30:.0f}GiB")
try:
    clk = torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else None
except Exception:
    clk = None
line(f"C1 torch={torch.__version__} cuda={torch.version.cuda} clk={clk}")

def bench(fn, warmup=5, reps=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        ev0.record(); fn(); ev1.record(); torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1) * 1e3)      # µs
    ts.sort()
    return ts[len(ts)//2]

# --- 달성 HBM 대역폭: 대형 copy_ (read+write) ---
n = 1 << 28                                          # 256Mi elem × 2B = 512MiB
a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
b = torch.empty_like(a)
us = bench(lambda: b.copy_(a))
hbm_meas = (2 * n * 2) / (us * 1e-6) / 1e9           # GB/s
line(f"C2 HBM 달성={hbm_meas:.0f}GB/s (이론 2039 대비 {hbm_meas/2039:.2f}) copy512MiB {us:.0f}us")
del a, b; torch.cuda.empty_cache()

# --- 달성 BF16 TC: 대형 GEMM ---
m = 8192
x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
y = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
us = bench(lambda: torch.mm(x, y))
tflops = (2 * m**3) / (us * 1e-6) / 1e12
line(f"C3 BF16 GEMM 달성={tflops:.0f}TFLOPS (이론 312 대비 {tflops/312:.2f}) 8192^3 {us:.0f}us")
del x, y; torch.cuda.empty_cache()

# --- 커널 리포트의 PERF evidence 재표현 ---
rd = "/work/kernel-kit/kernel-report"
i = 0
for fn_ in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
    if not (fn_.startswith("k0") and fn_.endswith(".json")): continue
    try:
        dd = json.load(open(os.path.join(rd, fn_), encoding="utf-8"))
        ev = ((dd.get("gates") or {}).get("PERF") or {}).get("evidence") or {}
        # k04/k07 은 케이스별(judged) 구조, 나머지는 평면 구조
        flat = [(dd.get("id", fn_)[:3], ev)] if "measured_us" in ev else [
            (f"{dd.get('id', fn_)[:3]}/{k[:10]}", v)
            for k, v in (ev.get("judged") or {}).items() if isinstance(v, dict)]
        for tag, e in flat:
            mu, rl = e.get("measured_us"), (e.get("roofline") or {}).get("roofline_us")
            if not (mu and rl): continue
            bound = (e.get("roofline") or {}).get("bound", "?")
            scale = hbm_meas/2039 if bound == "mem" else (tflops/312 if bound == "tc" else 1.0)
            i += 1
            line(f"C{3+i} {tag} m={mu:.0f}us rl={rl:.0f} eff_이론={rl/mu:.2f} "
                 f"eff_달성={rl*scale/mu:.2f} bound={bound}")
    except Exception as ex:
        line(f"C? {fn_}: {type(ex).__name__}")
line("C9 판독: eff_달성 >= 0.50 이면 floor 는 이론피크 오조정 → 목표 재정의 근거")
EOF
