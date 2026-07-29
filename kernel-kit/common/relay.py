"""화면 전달용 압축 요약(RELAY) — 파일 반출이 불가한 사내 환경 전제의 회신 채널.

러너(Jimmy)는 tgz 를 보낼 수 없고 화면의 짧은 텍스트만 옮길 수 있다(2026-07-29 확정).
→ run_all 끝에 이 블록을 항상 출력한다. 규칙:
  - 전체 ≤ 12줄, 줄당 ≤ 76자 (타이핑/사진→전사 가능해야 함)
  - 커널당 1줄: 4게이트 축약(N/S/H/F = NUMERIC/SANITIZER/SHAPES/PERF, 값 P/F/S/I)
    + 실패 시 가장 진단적인 꼬리 1개 (실패 게이트 사유 압축)
  - 이 블록만으로 개발 호스트가 S3 다음 행동을 결정할 수 있어야 한다

사용: python3 relay.py <report-dir>     (실패해도 exit 0 — 러너 흐름을 깨지 않는다)
"""

from __future__ import annotations

import json
import os
import re
import sys

GATES = ("NUMERIC", "SANITIZER", "SHAPES", "PERF")
ABBR = {"PASS": "P", "FAIL": "F", "SKIP": "S", "INCOMPLETE": "I"}
MAXW = 76


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _diag(d: dict) -> str:
    """실패한 리포트에서 가장 진단적인 꼬리 1개를 뽑는다."""
    ev = d.get("evidence", {}) or {}
    # 1) 하네스 crash (k01~k03 대칭 처리 / k04 case_errors)
    hc = ev.get("harness_crash")
    if isinstance(hc, dict) and hc.get("error"):
        return "CRASH " + _short(hc["error"], 40)
    ce = ev.get("case_errors")
    if isinstance(ce, dict) and ce:
        k, v = next(iter(ce.items()))
        return f"ERR@{_short(k,14)} " + _short((v or {}).get("error", ""), 30)
    # 2) 첫 FAIL 게이트의 사유
    for g in GATES:
        gd = (d.get("gates") or {}).get(g) or {}
        if gd.get("status") == "FAIL":
            return f"{g[0]}: " + _short(gd.get("reason", ""), 44)
    return ""


def build(report_dir: str) -> list[str]:
    lines: list[str] = []
    # 환경 1줄
    envp = os.path.join(report_dir, "env.json")
    envs = ""
    try:
        with open(envp, encoding="utf-8") as f:
            e = (json.load(f) or {}).get("env", {})
        gpu = f"{e.get('device_count','?')}x{_short(e.get('device_name','nogpu'),16)}"
        drv = (e.get("nvidia_smi") or "").split(",")[-1].strip() if e.get("nvidia_smi") else "?"
        envs = f"R0 {gpu} drv={drv} torch={e.get('torch','-')} vllm={e.get('vllm') or '-'} git={e.get('repo_commit','?')}"
    except Exception:
        envs = "R0 env.json 읽기 실패"
    lines.append(_short(envs, MAXW))

    counts = {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0, "MISSING": 0}
    i = 0
    for fn in sorted(os.listdir(report_dir)) if os.path.isdir(report_dir) else []:
        if not (fn.startswith("k0") and fn.endswith(".json")):
            continue
        i += 1
        try:
            with open(os.path.join(report_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
            kid = d.get("id", fn[:-5])[:3]          # k01..k07
            gletter = {"NUMERIC": "N", "SANITIZER": "S", "SHAPES": "H", "PERF": "F"}
            gs = "".join(
                f"{gletter[g]}{ABBR.get(((d.get('gates') or {}).get(g) or {}).get('status',''), '?')}"
                for g in GATES)                      # 예: "NPSPHPFP" = N:P S:P H:P F:P
            overall = d.get("overall", "?")
            counts[overall] = counts.get(overall, 0) + 1
            tail = _diag(d) if overall != "PASS" else ""
            lines.append(_short(f"R{i} {kid} {gs} {ABBR.get(overall, overall)}"
                                + (f" | {tail}" if tail else ""), MAXW))
        except Exception as ex:
            lines.append(_short(f"R{i} {fn}: 파싱 실패 {type(ex).__name__}", MAXW))
    lines.append(_short(
        f"R9 sum PASS={counts.get('PASS',0)} FAIL={counts.get('FAIL',0)} "
        f"INC={counts.get('INCOMPLETE',0)}", MAXW))
    return lines


def main() -> int:
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel-report")
    try:
        lines = build(rd)
    except Exception as ex:  # 어떤 경우에도 러너 흐름을 깨지 않는다
        print(f"[relay] 요약 생성 실패: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 0
    print()
    print("==== RELAY — 아래 블록만 그대로 전달 (게이트 N=수치 S=sanitizer H=shape F=perf, 값 P/F/S/I) ====")
    for ln in lines:
        print(ln)
    print("==== RELAY 끝 (+ run_all 의 최종 exit 코드도 함께) ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
