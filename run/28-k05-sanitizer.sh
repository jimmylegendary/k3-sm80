#!/usr/bin/env bash
# k05 Marlin MoE 의 compute-sanitizer 오류 상세 추출 (#35922 계열 진단). GPU 불필요 — 리포트만 읽는다.
#
# 배경: k05 는 NUMERIC/SHAPES/PERF 통과, SANITIZER 만 FAIL(m1 에서 memcheck 오류 4건).
#       crash 없이 메모리 위반이 실재하는 케이스 — raw 출력이 수정 지점을 가리킨다.
#       raw tail 저장 위치 = gates.SANITIZER.evidence.runs[*].evidence.tail
#
# 사용:   bash run/28-k05-sanitizer.sh
# 결과:   S0~S9 (≤20줄) — 오류 종류/위반 커널/접근 상세/스택 프레임/케이스별 rc.
#         이 출력 전체를 전달하면 수정 지점을 특정한다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$ROOT/kernel-kit/kernel-report" <<'EOF'
import json, os, re, sys
rd = sys.argv[1]
W = 76
def L(s): print(str(s).replace("\n", " ")[:W])

p = os.path.join(rd, "k05_marlin_moe_w4a16.json")
if not os.path.isfile(p):
    L("S0 k05 리포트 없음 — bash run/20-kernel-test.sh full k05_marlin_moe_w4a16 먼저")
    sys.exit(0)
d = json.load(open(p, encoding="utf-8"))
mt = d.get("meta", {}) or {}
L(f"S0 k05 overall={d.get('overall')} tier={mt.get('tier')} run={mt.get('run_id')}")
g = (d.get("gates") or {}).get("SANITIZER") or {}
L(f"S1 SANITIZER={g.get('status')} | {re.sub(r'[ ]+', ' ', str(g.get('reason','')))}")
ev = g.get("evidence") or {}
L(f"S2 bin={ev.get('sanitizer_bin')}")

n = 0
for run in (ev.get("runs") or []):
    n += 1
    L(f"S3.{n} case={run.get('case')} status={run.get('status')} rc={run.get('rc')} "
      f"class={run.get('class')} {run.get('duration_s')}s")
    rev = run.get("evidence") or {}
    tail = rev.get("san_excerpt") or rev.get("tail") or ""   # excerpt(오류 블록 원문) 우선
    if run.get("status") != "FAIL" or not tail:
        if run.get("status") == "FAIL":
            L(f"S4.{n} (오류 블록 미보존 — git pull 후 k05 재실행 필요: 구버전 리포트)")
        continue
    # 오류 종류
    kinds = {}
    for m in re.finditer(r"(Invalid|Illegal)[a-z_ ]*?(read|write|access)[a-z_ ]*(of size (\d+))?", tail):
        kinds[m.group(0).strip()] = kinds.get(m.group(0).strip(), 0) + 1
    for k, c in sorted(kinds.items(), key=lambda x: -x[1])[:3]:
        L(f"S4.{n} x{c} {k}")
    # 위반 커널 심볼 (memcheck 는 '=== at 0x... in <kernel>' 형태)
    syms = re.findall(r"in ([A-Za-z_][A-Za-z0-9_:<>,\. ]{4,70})", tail)
    seen = []
    for s in syms:
        s = s.strip()
        if s not in seen:
            seen.append(s)
    for s in seen[:4]:
        L(f"S5.{n} in {s}")
    # 접근 상세 (주소 종류 / 정렬 / 스레드)
    for pat in (r"Address 0x[0-9a-f]+ is [^\n]{0,60}",
                r"in (global|shared|local) memory",
                r"by thread \([^)]*\) in block \([^)]*\)"):
        mm = re.search(pat, tail)
        if mm:
            L(f"S6.{n} {mm.group(0)}")
    # 스택 상위 프레임 (호스트측 호출 지점 — 파일명 있으면 결정적)
    frames = re.findall(r"=========\s+#\d+\s+(.{0,66})", tail)
    for f in frames[:4]:
        L(f"S7.{n} {f}")
    L(f"S8.{n} err_count={((run.get('evidence') or {}).get('error_count'))}")

cases = (d.get("evidence") or {}).get("cases") or (d.get("evidence") or {}).get("runs") or {}
if isinstance(cases, dict) and cases:
    L(f"S9 케이스: {','.join(list(cases)[:8])}")
else:
    L("S9 (케이스 목록 evidence 키 상이 — 필요 시 kernel-report/k05_*.json 확인)")
EOF
