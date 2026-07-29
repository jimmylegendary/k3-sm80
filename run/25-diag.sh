#!/usr/bin/env bash
# 특정 커널의 진단 요약을 화면 전달용 ≤12줄로 출력한다 (RELAY 의 상세판).
#
# 사용:   bash run/25-diag.sh <커널id>          예: bash run/25-diag.sh k01_kda_chunk
# 결과:   D0~D9 줄 — overall/게이트별 사유/crash·케이스 에러/커버된 케이스/로그 에러줄.
#         이 출력 전체를 그대로 전달하면 된다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K="${1:?사용: bash run/25-diag.sh <커널id>}"
RD="$ROOT/kernel-kit/kernel-report"

python3 - "$K" "$RD" <<'EOF'
import json, os, re, sys
k, rd = sys.argv[1], sys.argv[2]
W = 76
def s(x, n): return re.sub(r"\s+", " ", str(x)).strip()[:n]
p = os.path.join(rd, f"{k}.json")
if not os.path.isfile(p):
    print(f"D0 {k}: 리포트 파일 없음 — 실행 전 단계에서 죽었거나 미실행")
    sys.exit(0)
d = json.load(open(p, encoding="utf-8"))
print(s(f"D0 {k} overall={d.get('overall')} dry={d.get('dry_run', False)}", W))
for i, g in enumerate(("NUMERIC", "SANITIZER", "SHAPES", "PERF")):
    gd = (d.get("gates") or {}).get(g) or {}
    print(s(f"D{i+1} {g}={gd.get('status')} | {gd.get('reason','')}", W))
ev = d.get("evidence", {}) or {}
hc = ev.get("harness_crash")
if hc:
    print(s(f"D5 CRASH: {hc.get('error','')}", W))
ce = ev.get("case_errors") or {}
for n, (c, v) in enumerate(list(ce.items())[:2]):
    print(s(f"D6.{n} ERR@{c}: {(v or {}).get('error','')}", W))
cases = ev.get("cases") or {}
done = [c for c in cases if not (isinstance(cases[c], dict) and cases[c].get("error"))]
print(s(f"D7 covered({len(done)}): {','.join(done)}", W))
bad = [c for c in cases if isinstance(cases[c], dict) and cases[c].get("error")]
if bad:
    print(s(f"D8 case-fail: {','.join(bad)}", W))
EOF

L="$RD/$K.log"
if [[ -f $L ]]; then
  echo "D9 log-err(마지막 3줄):"
  grep -nE "Error|error|Traceback|illegal|assert|CUDA" "$L" | tail -3 | cut -c1-76 || true
fi
exit 0
