"""S2 커널 공통 판정 스키마 + JSON/md 리포트 emitter + 집계.

스키마 (kernel-report/<id>.json) — docs/PLAN.md §S2 완료판정의 기계 형식:
{
  "schema_version": "s2-kernel-report/v1",
  "id": "k01_kda_chunk",
  "overall": "PASS" | "FAIL" | "INCOMPLETE",
  "gates": {
    "NUMERIC":   {"status": "PASS|FAIL|SKIP", "reason": str, "evidence": {...}},
    "SANITIZER": {...}, "SHAPES": {...}, "PERF": {...}
  },
  "evidence": {...},   # 커널 공통 증거 (seed, 케이스 설정, 부가 수치)
  "env": {...},        # env_check.collect_env()
  "meta": {"created_utc", "dry_run", "duration_s", "pins"}
}

판정 규칙:
  - 게이트 3값: PASS / FAIL / SKIP(반드시 사유). 커널 종합(overall) = 4게이트 all-PASS.
  - FAIL 1개라도 있으면 overall=FAIL. FAIL 없이 SKIP이 있으면 INCOMPLETE —
    dry-run 은 원리상 INCOMPLETE 까지만 도달한다(실커널 미실행 = PASS 사칭 금지).

exit code 계약 (하네스·run_all.sh 공유):
  0 = overall PASS, 또는 dry-run 이고 FAIL 없음
  1 = FAIL 존재
  2 = 사전조건(GPU 등) 미충족 skip
  3 = 하네스 내부 오류
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, shapes  # noqa: E402

SCHEMA_VERSION = "s2-kernel-report/v1"
GATES = ("NUMERIC", "SANITIZER", "SHAPES", "PERF")
STATUSES = ("PASS", "FAIL", "SKIP")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PRECOND = 2
EXIT_ERR = 3


class KernelReport:
    def __init__(self, kernel_id: str, dry_run: bool = False, pins: dict | None = None):
        assert kernel_id in shapes.KERNEL_IDS or kernel_id.startswith("k00_"), \
            f"미등록 커널 id: {kernel_id} (허용: {shapes.KERNEL_IDS})"
        self.id = kernel_id
        self.dry_run = dry_run
        self.pins = pins or {}
        self._t0 = datetime.datetime.now(datetime.timezone.utc)
        self.gates: dict[str, dict] = {}
        self.evidence: dict = {}

    def set_gate(self, gate: str, status: str, reason: str = "", evidence: dict | None = None):
        assert gate in GATES, f"미정의 게이트: {gate} (허용: {GATES})"
        assert status in STATUSES, f"미정의 상태: {status}"
        if status == "SKIP":
            assert reason, f"{gate}: SKIP 은 사유(reason) 필수"
        self.gates[gate] = {"status": status, "reason": reason, "evidence": evidence or {}}

    def add_evidence(self, key: str, value):
        self.evidence[key] = value

    def overall(self) -> str:
        sts = [self.gates.get(g, {}).get("status", "SKIP") for g in GATES]
        if any(s == "FAIL" for s in sts):
            return "FAIL"
        if all(s == "PASS" for s in sts):
            return "PASS"
        return "INCOMPLETE"

    def as_dict(self) -> dict:
        for g in GATES:  # 미설정 게이트는 명시적 SKIP으로 채움 (스키마 완전성)
            if g not in self.gates:
                self.gates[g] = {"status": "SKIP", "reason": "하네스가 게이트를 설정하지 않음",
                                 "evidence": {}}
        dur = (datetime.datetime.now(datetime.timezone.utc) - self._t0).total_seconds()
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "overall": self.overall(),
            "gates": {g: self.gates[g] for g in GATES},
            "evidence": self.evidence,
            "env": env_check.collect_env(),
            "meta": {
                "created_utc": self._t0.isoformat(timespec="seconds"),
                "dry_run": self.dry_run,
                "duration_s": round(dur, 3),
                "pins": self.pins,
            },
        }

    def write(self, report_dir: str) -> tuple[str, str]:
        os.makedirs(report_dir, exist_ok=True)
        d = self.as_dict()
        jpath = os.path.join(report_dir, f"{self.id}.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        mpath = os.path.join(report_dir, f"{self.id}.md")
        with open(mpath, "w", encoding="utf-8") as f:
            f.write(render_md(d))
        return jpath, mpath

    def exit_code(self) -> int:
        ov = self.overall()
        if ov == "FAIL":
            return EXIT_FAIL
        if ov == "PASS" or self.dry_run:
            return EXIT_OK
        return EXIT_OK  # INCOMPLETE(GPU 모드 SKIP 잔존)도 러너 진행은 허용 — 집계가 드러냄


def validate_report(d: dict) -> list[str]:
    """스키마 검증 — 위반 목록 반환(빈 리스트 = 정상)."""
    errs = []
    for key in ("schema_version", "id", "overall", "gates", "evidence", "env", "meta"):
        if key not in d:
            errs.append(f"필수 키 없음: {key}")
    if d.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version 불일치: {d.get('schema_version')}")
    for g in GATES:
        gd = d.get("gates", {}).get(g)
        if gd is None:
            errs.append(f"게이트 없음: {g}")
            continue
        if gd.get("status") not in STATUSES:
            errs.append(f"{g}.status 이상: {gd.get('status')}")
        if gd.get("status") == "SKIP" and not gd.get("reason"):
            errs.append(f"{g}: SKIP 인데 사유 없음")
    if not errs:
        sts = [d["gates"][g]["status"] for g in GATES]
        expect = ("FAIL" if "FAIL" in sts
                  else "PASS" if all(s == "PASS" for s in sts) else "INCOMPLETE")
        if d["overall"] != expect:
            errs.append(f"overall {d['overall']} ≠ 게이트 유도값 {expect}")
    return errs


# ---------------------------------------------------------------------------
# SANITIZER 출력 파서 (compute-sanitizer memcheck 공통)
# ---------------------------------------------------------------------------
def sanitizer_gate_from_output(output: str, cmd: str = "",
                               returncode: int | None = None) -> tuple[str, str, dict]:
    """compute-sanitizer 출력 → SANITIZER 게이트.

    memcheck 표준 요약줄 "========= ERROR SUMMARY: N error(s)" 를 파싱.
    요약줄이 없으면(비정상 종료/포맷 변경) 판독 불가 — FAIL이 아닌 SKIP(사유)로 남겨
    가짜 clean 을 만들지 않는다. 단 returncode!=0 이면서 요약줄도 없으면 FAIL(crash).
    """
    ev = {"cmd": cmd, "returncode": returncode,
          "tail": output[-2000:] if output else ""}
    m = re.search(r"ERROR SUMMARY:\s*(\d+)\s*error", output or "")
    if m:
        n = int(m.group(1))
        ev["error_count"] = n
        if n == 0 and (returncode in (0, None)):
            return ("PASS", "compute-sanitizer memcheck clean (0 errors)", ev)
        if n == 0 and returncode not in (0, None):
            return ("FAIL", f"메모리 오류 0건이나 프로세스 비정상 종료(rc={returncode}) — crash 의심", ev)
        return ("FAIL", f"compute-sanitizer 오류 {n}건 (memcheck)", ev)
    if returncode not in (0, None):
        return ("FAIL", f"sanitizer 하 실행 crash (rc={returncode}, ERROR SUMMARY 없음) — "
                        "#35922 형 illegal access 의심, tail 확인", ev)
    return ("SKIP", "출력에서 ERROR SUMMARY 미검출 — 판독 불가(도구/포맷 확인)", ev)


# ---------------------------------------------------------------------------
# 렌더/집계
# ---------------------------------------------------------------------------
_STATUS_KO = {"PASS": "통과", "FAIL": "실패", "SKIP": "생략"}


def render_md(d: dict) -> str:
    lines = [f"# {d['id']} — {shapes.KERNEL_TITLES.get(d['id'], '')}",
             "",
             f"- 종합: **{d['overall']}**"
             + (" (cpu-dry-run — 실커널 미실행)" if d["meta"].get("dry_run") else ""),
             f"- 생성: {d['meta']['created_utc']} · 소요 {d['meta']['duration_s']}s"
             f" · repo {d['env'].get('repo_commit')}",
             "",
             "| 게이트 | 판정 | 사유 |",
             "|---|---|---|"]
    for g in GATES:
        gd = d["gates"][g]
        lines.append(f"| {g} | {gd['status']} ({_STATUS_KO[gd['status']]}) | "
                     f"{gd['reason'].replace('|', '\\|')} |")
    env = d["env"]
    lines += ["",
              f"환경: {env.get('device_name') or 'GPU 없음'}"
              f" · torch {env.get('torch')} · triton {env.get('triton')}"
              f" · vllm {env.get('vllm')}",
              f"핀: {json.dumps(d['meta'].get('pins', {}), ensure_ascii=False)}",
              ""]
    return "\n".join(lines)


def aggregate(report_dir: str, expected: list[str] | None = None) -> dict:
    expected = expected or list(shapes.KERNEL_IDS)
    found: dict[str, dict] = {}
    invalid: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(report_dir, "k[0-9][0-9]_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        errs = validate_report(d)
        if errs:
            invalid[os.path.basename(path)] = errs
        found[d.get("id", os.path.basename(path))] = d
    rows = []
    for kid in expected:
        d = found.get(kid)
        if d is None:
            rows.append({"id": kid, "overall": "MISSING",
                         "gates": {g: "-" for g in GATES}, "dry_run": None})
        else:
            rows.append({"id": kid, "overall": d["overall"],
                         "gates": {g: d["gates"][g]["status"] for g in GATES},
                         "dry_run": d["meta"].get("dry_run")})
    counts = {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0, "MISSING": 0}
    for r in rows:
        counts[r["overall"]] += 1
    return {"generated_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
            "report_dir": os.path.abspath(report_dir),
            "counts": counts, "kernels": rows, "invalid": invalid,
            "s2_complete": counts["PASS"] == len(expected)}


def render_summary_md(s: dict) -> str:
    lines = ["# S2 kernel-kit 집계",
             "",
             f"- 생성: {s['generated_utc']} · 대상 {len(s['kernels'])}커널",
             f"- 종합: PASS {s['counts']['PASS']} / FAIL {s['counts']['FAIL']} / "
             f"INCOMPLETE {s['counts']['INCOMPLETE']} / MISSING {s['counts']['MISSING']}"
             f" → S2 완료(전 커널 PASS): {'예' if s['s2_complete'] else '아니오'}",
             "",
             "| 커널 | 종합 | NUMERIC | SANITIZER | SHAPES | PERF |",
             "|---|---|---|---|---|---|"]
    for r in s["kernels"]:
        g = r["gates"]
        tag = " (dry)" if r.get("dry_run") else ""
        lines.append(f"| {r['id']}{tag} | **{r['overall']}** | "
                     f"{g['NUMERIC']} | {g['SANITIZER']} | {g['SHAPES']} | {g['PERF']} |")
    if s["invalid"]:
        lines += ["", "## 스키마 위반 리포트", ""]
        for name, errs in s["invalid"].items():
            lines.append(f"- {name}: {'; '.join(errs)}")
    lines += ["",
              "해석: MISSING=하네스 미저작/미실행 · INCOMPLETE=FAIL 없이 SKIP 잔존"
              "(dry-run 상한) · S4 진입은 해당 커널 overall=PASS 가 조건 (docs/PLAN.md).",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _selftest(report_dir: str | None) -> int:
    """리포트 경로 전체(생성→기록→검증→집계 로직)를 CPU에서 검증. 결정적."""
    import tempfile
    tmp = report_dir or tempfile.mkdtemp(prefix="s2-report-selftest-")
    r = KernelReport("k02_kda_recurrent", dry_run=True, pins={"selftest": True})
    st, rs, ev = shapes.shapes_gate("k02_kda_recurrent",
                                    list(shapes.REQUIRED_SHAPES["k02_kda_recurrent"]))
    r.set_gate("SHAPES", st, rs, ev)
    r.set_gate("NUMERIC", "SKIP", "selftest — 실비교 없음")
    r.set_gate("SANITIZER", *sanitizer_gate_from_output(
        "========= ERROR SUMMARY: 0 errors", cmd="selftest", returncode=0)[0:2])
    r.set_gate("PERF", "SKIP", "selftest — 측정 없음")
    d = r.as_dict()
    errs = validate_report(d)
    ok = not errs and d["overall"] == "INCOMPLETE" and r.exit_code() == EXIT_OK
    # FAIL 방향
    r2 = KernelReport("k05_marlin_moe_w4a16", dry_run=True)
    st2 = sanitizer_gate_from_output("boom", cmd="selftest", returncode=139)
    r2.set_gate("SANITIZER", *st2[0:2], st2[2])
    ok &= r2.overall() == "FAIL" and r2.exit_code() == EXIT_FAIL
    ok &= sanitizer_gate_from_output("========= ERROR SUMMARY: 3 errors")[0] == "FAIL"
    # 기록 경로 (집계에 안 섞이도록 별도 이름으로 저장)
    jpath = os.path.join(tmp, "selftest.sample.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(tmp, "selftest.sample.md"), "w", encoding="utf-8") as f:
        f.write(render_md(d))
    with open(jpath, encoding="utf-8") as f:
        ok &= not validate_report(json.load(f))
    agg = aggregate(tmp)   # 빈 디렉터리 집계 → 전부 MISSING 이어야 함
    ok &= agg["counts"]["MISSING"] == len(shapes.KERNEL_IDS)
    print(json.dumps({"report_selftest": "PASS" if ok else "FAIL",
                      "sample": jpath, "validate_errors": errs}, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="S2 리포트 스키마/집계 도구")
    ap.add_argument("--aggregate", metavar="DIR", help="kernel-report 디렉터리 집계")
    ap.add_argument("--expect", nargs="*", default=None, help="기대 커널 id 목록(기본: 7종 전부)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report-dir", default=None)
    a = ap.parse_args()
    if a.selftest:
        return _selftest(a.report_dir)
    if a.aggregate:
        s = aggregate(a.aggregate, a.expect)
        with open(os.path.join(a.aggregate, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        md = render_summary_md(s)
        with open(os.path.join(a.aggregate, "SUMMARY.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print(md)
        return EXIT_FAIL if s["counts"]["FAIL"] > 0 else EXIT_OK
    ap.print_help()
    return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
