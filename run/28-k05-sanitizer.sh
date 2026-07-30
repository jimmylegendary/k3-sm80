#!/usr/bin/env bash
# k05 Marlin MoE 의 compute-sanitizer 오류 상세를 화면 전달용으로 뽑는다 (#35922 계열 진단).
#
# 배경: k05 는 m1 에서 이미 memcheck 오류 4건 — crash 없이 진행되지만 메모리 위반이 실재한다.
#       (crash 만 보면 '정상'으로 오판하는 케이스 — sanitizer 게이트를 둔 이유)
#
# 사용:   bash run/28-k05-sanitizer.sh [full]      (sanitizer 는 full 이미지에만 동봉)
# 결과:   S0~S9 (≤14줄) — 오류 종류/발생 커널명/스택 상위 프레임/접근 크기·주소 종류.
#         이 출력 전체를 전달하면 수정 지점(c_tmp 워크스페이스 OOB 가설 등)을 특정한다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RD="$ROOT/kernel-kit/kernel-report"
L="$RD/k05_marlin_moe_w4a16.log"

echo "S0 log=$([[ -f $L ]] && echo present || echo MISSING)"
if [[ -f $L ]]; then
  # sanitizer 원문에서 진단 가치가 높은 줄만: 오류 종류 / 위반 커널 / 접근 정보
  echo "S1 오류 종류(빈도순):"
  grep -oE "(Invalid|Illegal) [a-z ]*(read|write|access)[a-z ]*" "$L" 2>/dev/null \
    | sort | uniq -c | sort -rn | head -3 | cut -c1-76 || true
  echo "S2 위반 커널(kernel 심볼):"
  grep -oE "in [A-Za-z_][A-Za-z0-9_:<>, ]{4,60}" "$L" 2>/dev/null \
    | sort -u | head -4 | cut -c1-76 || true
  echo "S3 접근 상세(size/주소 종류):"
  grep -oE "of size [0-9]+ bytes|in (global|shared|local) memory|Address 0x[0-9a-f]+ is" "$L" 2>/dev/null \
    | sort | uniq -c | sort -rn | head -4 | cut -c1-76 || true
  echo "S4 요약줄:"
  grep -E "ERROR SUMMARY" "$L" 2>/dev/null | tail -2 | cut -c1-76 || true
  echo "S5 케이스 진행(어느 m 에서):"
  grep -oE "\[k05[^]]*\] *(m[0-9]+|m[0-9]+_prefill)[^|]{0,30}" "$L" 2>/dev/null \
    | tail -4 | cut -c1-76 || true
fi

# 리포트 JSON 쪽 evidence (sanitizer 게이트가 기록한 요약)
python3 - "$RD" <<'EOF' || true
import json, os, re, sys
rd = sys.argv[1]
p = os.path.join(rd, "k05_marlin_moe_w4a16.json")
if os.path.isfile(p):
    d = json.load(open(p, encoding="utf-8"))
    g = (d.get("gates") or {}).get("SANITIZER") or {}
    print(("S6 gate=" + str(g.get("status")) + " | " + re.sub(r"\s+", " ", str(g.get("reason", ""))))[:76])
    ev = g.get("evidence") or {}
    for k in ("cmd", "returncode", "error_count", "tail"):
        if k in ev:
            print(("S7." + k + ": " + re.sub(r"\s+", " ", str(ev[k])))[:76])
    cases = (d.get("evidence") or {}).get("cases") or {}
    print(("S8 covered: " + ",".join(cases.keys()))[:76])
EOF
echo "S9 (sanitizer 원문 전체는 kernel-report/k05_marlin_moe_w4a16.log)"
