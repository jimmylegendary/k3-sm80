"""화면 전달용 압축 요약(RELAY) — 파일 반출이 불가한 사내 환경 전제의 회신 채널.

러너(Jimmy)는 tgz 를 보낼 수 없고 화면의 짧은 텍스트만 옮길 수 있다(2026-07-29 확정).
→ run_all 끝에 이 블록을 항상 출력한다. 규칙:
  - 전체 ≤ 12줄, 줄당 ≤ 76자 (타이핑/사진→전사 가능해야 함)
  - 줄 구성: R0(환경 1줄) + 커널 줄 R1..R8 (≤ 8) + R9(합계 1줄) + R!(혼합실행 경고, 있을 때만 1줄) = 최대 11줄.
    커널 줄을 R8 까지로 막아 "R9 sum" 번호가 커널 줄과 겹치지 않게 한다(구 코드는 커널 9종에서 R9 중복).
    커널 11종(k01~k09·k11·k12)이 되면서 "커널당 1줄"은 13~14줄이 되어 ≤12줄 계약을 깨므로 아래 압축 규칙을 쓴다.
  - 게이트 축약: N/S/H/F = NUMERIC/SANITIZER/SHAPES/PERF, 값 P/F/S/I (예: "NPSPHPFP" = 전부 통과)
  - **커널 줄 압축 규칙(결정적)**:
      1) overall=PASS 인 커널은 4게이트가 모두 P 이므로 한 줄로 묶는다: `R1 PASS k01 k02 …` (id 3자만).
      2) PASS 가 아닌 커널은 한 줄씩 상세: `Rn kid 게이트8자 overall | 가장 진단적인 꼬리 1개`.
      3) (PASS 줄 + 상세 줄) 이 8줄을 넘으면, id 순서상 뒤쪽 상세 줄부터 **꼬리 없이 4개씩 한 줄로 묶는다**
         (`Rn kid 게이트8자 overall  kid …`). 앞쪽 커널의 꼬리는 끝까지 보존한다 — 최악(11종 전부
         비-PASS)에도 상세 7줄 + 묶음 1줄(4종) = 8줄이다.
  - 이 블록만으로 개발 호스트가 S3 다음 행동을 결정할 수 있어야 한다 (묶인 커널의 꼬리는 필요 시 재요청)
  - **KernelReport 만 센다**: report dir 에는 `kNN_*.json` 이름이어도 KernelReport(`schema_version == s2-kernel-report/v1`)가
    아닌 리포트가 섞인다 — k10(DSpark 통합 게이트, `k10-dspark-report/v1`)·c01(EP32 A2A, `c01-ep32-report/v1`) 등은 run_all 대상 커널이 아니고
    meta(run_id/tier)도 없다. 이를 커널 줄로 세면 합계가 틀리고 meta 없음이 `R!` 혼합실행 거짓 경고를 만든다 → 스키마가 다르면 무시한다
    (읽을 수 없는 JSON 은 스키마를 알 수 없으므로 종전대로 ERR 줄로 남긴다).

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
MAX_LINES = 12                  # 계약: build() 반환 줄 수 상한
MAX_KERNEL_LINES = 8            # 커널 줄 예산 (R1..R8) — R0·R9·R!(선택) 몫과 R9 번호 충돌 방지를 뺀 값
PACK = 4                        # 꼬리 없는 묶음 줄 하나에 넣는 커널 수 (4×15자 + 접두 ≤ 76자)
_REPORT_FILE = re.compile(r"^k\d\d_.*\.json$")   # k01..k99 (구 필터 startswith("k0")는 k10~ 을 조용히 탈락시켰다)
# 이 값만 KernelReport 다. common/report.py 의 SCHEMA_VERSION 과 같아야 한다(relay 는 stdlib 전용 단독 스크립트라 import 하지 않고,
# tests/kernel/test_relay_budget.py 가 두 값의 일치를 고정한다).
KERNEL_REPORT_SCHEMA = "s2-kernel-report/v1"


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


def _entry_line(e: dict, with_tail: bool = True) -> str:
    base = f"{e['kid']} {e['gs']} {e['ov']}" if e["gs"] else f"{e['kid']} {e['ov']}"
    return base + (f" | {e['tail']}" if with_tail and e["tail"] else "")


def compress_kernel_lines(entries: list[dict]) -> list[str]:
    """커널 entry 목록 → 줄 목록 (모듈 docstring 의 압축 규칙 1)~3) 구현). 접두 Rn 은 붙여서 반환."""
    entries = sorted(entries, key=lambda e: e["kid"])
    passes = [e for e in entries if e["ov"] == "P"]
    fails = [e for e in entries if e["ov"] != "P"]
    pass_lines = 1 if passes else 0
    n_detail = len(fails)
    while n_detail > 0 and pass_lines + n_detail + -(-(len(fails) - n_detail) // PACK) > MAX_KERNEL_LINES:
        n_detail -= 1
    body: list[str] = []
    if passes:
        body.append("PASS " + " ".join(e["kid"] for e in passes))
    body += [_entry_line(e) for e in fails[:n_detail]]
    rest = fails[n_detail:]
    for i in range(0, len(rest), PACK):
        body.append("  ".join(_entry_line(e, with_tail=False) for e in rest[i:i + PACK]))
    return [_short(f"R{i} {ln}", MAXW) for i, ln in enumerate(body, start=1)]


def scan_reports(report_dir: str) -> dict:
    """report dir 의 `kNN_*.json` 을 읽어 KernelReport 만 entry 로 모은다. 반환: entries / counts / stamps / ignored(스키마가 달라 무시한 파일명)."""
    counts = {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0, "MISSING": 0}
    stamps: dict[str, list[str]] = {}      # run_id/tier → 커널 목록 (stale 혼합 감지)
    entries: list[dict] = []
    ignored: list[str] = []
    for fn in sorted(os.listdir(report_dir)) if os.path.isdir(report_dir) else []:
        if not _REPORT_FILE.match(fn):
            continue
        try:
            with open(os.path.join(report_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict) or d.get("schema_version") != KERNEL_REPORT_SCHEMA:
                ignored.append(fn)                  # k10/c01 등 다른 게이트의 리포트 — 커널 줄·합계·혼합 경고에서 제외
                continue
            kid = d.get("id", fn[:-5])[:3]          # k01..k12
            gletter = {"NUMERIC": "N", "SANITIZER": "S", "SHAPES": "H", "PERF": "F"}
            gs = "".join(
                f"{gletter[g]}{ABBR.get(((d.get('gates') or {}).get(g) or {}).get('status',''), '?')}"
                for g in GATES)                      # 예: "NPSPHPFP" = N:P S:P H:P F:P
            overall = d.get("overall", "?")
            counts[overall] = counts.get(overall, 0) + 1
            mt = d.get("meta", {}) or {}
            stamps.setdefault(f"{mt.get('run_id','unset')}/{mt.get('tier','unset')}",
                              []).append(kid)
            entries.append({"kid": kid, "gs": gs, "ov": ABBR.get(overall, overall),
                            "tail": _diag(d) if overall != "PASS" else ""})
        except Exception as ex:
            entries.append({"kid": fn[:3], "gs": "", "ov": "ERR",
                            "tail": f"파싱 실패 {type(ex).__name__}"})
    return {"entries": entries, "counts": counts, "stamps": stamps, "ignored": ignored}


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

    scan = scan_reports(report_dir)
    counts, stamps, entries = scan["counts"], scan["stamps"], scan["entries"]
    lines += compress_kernel_lines(entries)
    lines.append(_short(
        f"R9 sum PASS={counts.get('PASS',0)} FAIL={counts.get('FAIL',0)} "
        f"INC={counts.get('INCOMPLETE',0)}", MAXW))
    if len(stamps) > 1:
        newest = max(stamps)                       # run_id 는 타임스탬프 → 사전순 = 시간순
        stale = [f"{k.split('/')[1]}:{','.join(v)}" for k, v in stamps.items() if k != newest]
        lines.append(_short(f"R! 혼합실행 경고 — 최신={newest}, 이전실행 결과 잔존: "
                            f"{'; '.join(stale)} (해당 커널 판정은 이 스윕 것이 아님)", MAXW))
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
