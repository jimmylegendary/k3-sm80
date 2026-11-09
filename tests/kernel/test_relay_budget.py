"""RELAY(화면 전달 요약) 줄 예산 계약 — 커널 11종에서도 ≤12줄·줄당 ≤76자이고 모든 커널이 표시되는지.

relay.py 는 구 필터 `startswith("k0")` 때문에 k10~ 을 조용히 떨궜고, 커널 1줄 규칙은 11종에서 13줄을 넘었다.
압축 규칙(PASS 묶음 / 상세 / 꼬리 없는 4개 묶음)이 최악 입력에서도 예산을 지키는지 시나리오별로 확인한다.

반대 방향의 결함도 고정한다: `^k\\d\\d_` 이름 필터가 k10(DSpark 통합 게이트, 스키마 `k10-dspark-report/v1`)을 집으면 run_all 대상이 아닌 그
리포트(meta 없음)가 커널 줄·합계에 끼고 `R!` 혼합실행 거짓 경고를 만든다. RELAY·집계는 KernelReport(`s2-kernel-report/v1`)만 센다 —
깨끗한 dir 와 'k10·c01 리포트가 섞인 dir' 에서 `run_all.sh --cpu-dry-run` 의 RELAY 가 같아야 한다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common

relay = import_common("relay")
report = import_common("report")
shapes = import_common("shapes")

GATE_CODE = {"P": "PASS", "F": "FAIL", "S": "SKIP"}
GATES = ("NUMERIC", "SANITIZER", "SHAPES", "PERF")


def fake_report(kid: str, overall: str, codes: str = "PPPP", run_id: str = "20261109T000000Z") -> dict:
    return {"schema_version": report.SCHEMA_VERSION, "id": kid, "overall": overall,
            "gates": {g: {"status": GATE_CODE[c], "reason": "원인 " * 30} for g, c in zip(GATES, codes)},
            "evidence": {}, "meta": {"run_id": run_id, "tier": "full"}}


def write_dir(reports: list[dict]) -> str:
    tmp = tempfile.mkdtemp(prefix="relay-test-")
    unittest.addModuleCleanup(shutil.rmtree, tmp, True)         # 모듈 시험이 끝나면 임시 디렉터리 정리
    for r in reports:
        (Path(tmp) / f"{r['id']}.json").write_text(json.dumps(r))
    (Path(tmp) / "env.json").write_text(json.dumps({"env": {"device_count": 8, "device_name": "NVIDIA A100-SXM4-80GB",
                                                          "torch": "2.14.0", "repo_commit": "abc1234"}}))
    return tmp


class RelayBudgetTest(unittest.TestCase):
    ids = list(shapes.KERNEL_IDS)

    def check(self, lines: list[str]) -> None:
        self.assertLessEqual(len(lines), 12, "\n".join(lines))
        self.assertTrue(all(len(ln) <= relay.MAXW for ln in lines), [ln for ln in lines if len(ln) > relay.MAXW])
        body = "\n".join(lines)
        for kid in self.ids:
            self.assertIn(kid[:3], body, f"{kid} 가 RELAY 에 안 보임\n{body}")
        nums = re.findall(r"^R(\d|!)", body, flags=re.M)
        self.assertEqual(len(nums), len(set(nums)), f"R 번호 중복\n{body}")

    def test_eleven_kernels_present_in_registry(self) -> None:
        self.assertEqual(len(self.ids), 11)
        self.assertEqual(sum(1 for k in self.ids if k.startswith(("k11", "k12"))), 2)

    def test_all_pass_collapses_to_one_line(self) -> None:
        lines = relay.build(write_dir([fake_report(k, "PASS") for k in self.ids]))
        self.check(lines)
        self.assertEqual(len(lines), 3)                     # R0 + 'R1 PASS …' + R9
        self.assertIn("PASS=11", lines[-1])

    def test_all_incomplete_dry_run_shape(self) -> None:
        lines = relay.build(write_dir([fake_report(k, "INCOMPLETE", "SSSS") for k in self.ids]))
        self.check(lines)
        self.assertIn("INC=11", lines[-1])

    def test_worst_case_all_failing_with_stale_mixed_run(self) -> None:
        reports = [fake_report(k, "FAIL", "FPFS", run_id="20261109T000000Z") for k in self.ids]
        reports[3]["meta"]["run_id"] = "20261108T000000Z"  # 혼합 실행 → R! 경고 줄 추가
        lines = relay.build(write_dir(reports))
        self.check(lines)
        self.assertTrue(any(ln.startswith("R!") for ln in lines))

    def test_failing_tails_are_kept_for_the_first_kernels(self) -> None:
        reports = [fake_report(k, "FAIL", "FPPP") for k in self.ids]
        lines = relay.build(write_dir(reports))
        self.check(lines)
        detail = [ln for ln in lines if ln.startswith("R") and " | N:" in ln]
        self.assertGreaterEqual(len(detail), 7)             # 최악에도 앞쪽 7종은 꼬리 보존
        self.assertTrue(detail[0].startswith("R1 k01"))

    def test_mixed_pass_and_fail(self) -> None:
        reports = [fake_report(k, "PASS") if i % 2 else fake_report(k, "FAIL", "PFPP") for i, k in enumerate(self.ids)]
        lines = relay.build(write_dir(reports))
        self.check(lines)
        self.assertTrue(any(ln.startswith("R1 PASS") for ln in lines))

    def test_k10_plus_reports_are_not_dropped_by_the_filter(self) -> None:
        # 구 필터(startswith 'k0')였다면 k11/k12 가 통째로 사라졌다
        lines = relay.build(write_dir([fake_report("k11_dsa_indexer_sm80", "PASS"),
                                       fake_report("k12_sparse_mla_sm80", "FAIL", "FPPP")]))
        body = "\n".join(lines)
        self.assertIn("k11", body)
        self.assertIn("k12", body)


def foreign_report(kid: str, schema: str, overall: str = "INCOMPLETE") -> dict:
    """k10/c01 처럼 KernelReport 가 아닌 리포트 — meta(run_id/tier) 도 4게이트 이름도 없다."""
    return {"schema_version": schema, "id": kid, "overall": overall,
            "gates": {"STATE": {"status": "SKIP", "reason": "cpu-dry-run"}}, "evidence": {}}


def add_foreign(directory: str, *, overall: str = "INCOMPLETE") -> None:
    reports = {"k10_k3_dspark_sm80": foreign_report("k10_k3_dspark_sm80", "k10-dspark-report/v1", overall),
               "c01_ep32_a2a": foreign_report("c01_ep32_a2a", "c01-ep32-report/v1", overall),
               "k13_future_gate": foreign_report("k13_future_gate", "k13-future/v1", overall)}
    for name, payload in reports.items():
        (Path(directory) / f"{name}.json").write_text(json.dumps(payload))


class RelayIgnoresNonKernelReportsTest(unittest.TestCase):
    ids = list(shapes.KERNEL_IDS)

    def pair(self, reports: list[dict], **foreign_kwargs) -> tuple[list[str], list[str], str]:
        clean_dir, mixed_dir = write_dir(reports), write_dir(reports)
        add_foreign(mixed_dir, **foreign_kwargs)
        return relay.build(clean_dir), relay.build(mixed_dir), mixed_dir

    def test_schema_constant_matches_the_report_module(self) -> None:
        self.assertEqual(relay.KERNEL_REPORT_SCHEMA, report.SCHEMA_VERSION)       # relay 는 stdlib 단독이라 값을 복제한다 -> 드리프트 고정
        self.assertTrue(report.SCHEMA_VERSION.startswith(report.SCHEMA_FAMILY))

    def test_dry_run_shape_is_identical_with_and_without_foreign_reports(self) -> None:
        clean, mixed, _ = self.pair([fake_report(k, "INCOMPLETE", "SSSS") for k in self.ids])
        self.assertEqual(clean, mixed)
        self.assertLessEqual(len(mixed), 12)
        self.assertFalse(any(ln.startswith("R!") for ln in mixed), "\n".join(mixed))     # meta 없는 k10 이 혼합실행 거짓 경고를 만들던 결함
        self.assertIn("INC=11", mixed[-1])                                              # k10 이 12번째 INCOMPLETE 로 끼지 않는다
        self.assertNotIn("k10", "\n".join(mixed))

    def test_all_pass_stays_one_line_and_foreign_failures_do_not_leak(self) -> None:
        clean, mixed, _ = self.pair([fake_report(k, "PASS") for k in self.ids], overall="FAIL")
        self.assertEqual(clean, mixed)
        self.assertEqual(len(mixed), 3)
        self.assertIn("PASS=11 FAIL=0", mixed[-1])                                       # 다른 게이트의 FAIL 이 합계에 새지 않는다

    def test_worst_case_with_foreign_reports_keeps_the_budget(self) -> None:
        clean, mixed, _ = self.pair([fake_report(k, "FAIL", "FPFS") for k in self.ids])
        self.assertEqual(clean, mixed)
        self.assertLessEqual(len(mixed), 12)
        self.assertTrue(all(len(ln) <= relay.MAXW for ln in mixed))
        self.assertFalse(any(ln.startswith("R!") for ln in mixed))

    def test_a_genuine_mixed_run_of_kernel_reports_still_warns(self) -> None:
        """무시 규칙이 과해서 진짜 혼합실행(KernelReport 끼리 run_id 가 다름)까지 가리면 안 된다."""
        reports = [fake_report(k, "PASS") for k in self.ids]
        reports[3]["meta"]["run_id"] = "20261108T000000Z"
        clean, mixed, _ = self.pair(reports)
        self.assertEqual(clean, mixed)
        warn = [ln for ln in mixed if ln.startswith("R!")]
        self.assertEqual(len(warn), 1, "\n".join(mixed))
        self.assertIn("k04", warn[0])
        self.assertNotIn("k10", warn[0])

    def test_scan_reports_lists_the_ignored_files(self) -> None:
        _, _, mixed_dir = self.pair([fake_report(k, "PASS") for k in self.ids])
        scan = relay.scan_reports(mixed_dir)
        self.assertEqual(scan["ignored"], ["k10_k3_dspark_sm80.json", "k13_future_gate.json"])   # c01 은 이름(kNN_) 필터에서 먼저 빠진다
        self.assertEqual(len(scan["entries"]), 11)
        self.assertEqual(scan["counts"]["PASS"], 11)

    def test_fixture_has_teeth_naive_counting_would_warn(self) -> None:
        """위 시나리오가 공회전이 아님: 스키마를 보지 않고 세면(구 동작) k10 의 meta 없음이 stamp 를 갈라 R! 가 된다."""
        _, _, mixed_dir = self.pair([fake_report(k, "INCOMPLETE", "SSSS") for k in self.ids])
        stamps = set()
        for fn in os.listdir(mixed_dir):
            if relay._REPORT_FILE.match(fn):
                meta = json.loads((Path(mixed_dir) / fn).read_text()).get("meta", {}) or {}
                stamps.add(f"{meta.get('run_id', 'unset')}/{meta.get('tier', 'unset')}")
        self.assertGreater(len(stamps), 1)

    def test_unreadable_kernel_named_json_is_still_reported_as_err(self) -> None:
        directory = write_dir([fake_report(k, "PASS") for k in self.ids if k != "k11_dsa_indexer_sm80"])
        (Path(directory) / "k11_dsa_indexer_sm80.json").write_text("{ not json")
        body = "\n".join(relay.build(directory))
        self.assertIn("k11 ERR", body)                                                   # 스키마를 알 수 없는 파일은 조용히 지우지 않는다

    def test_non_dict_json_is_ignored_not_crashing(self) -> None:
        directory = write_dir([fake_report(k, "PASS") for k in self.ids])
        (Path(directory) / "k20_list.json").write_text("[1, 2, 3]")
        self.assertEqual(relay.build(directory), relay.build(write_dir([fake_report(k, "PASS") for k in self.ids])))


@unittest.skipUnless(shutil.which("bash"), "bash 필요")
class RunAllDryRunRelayTest(unittest.TestCase):
    """`bash run_all.sh --cpu-dry-run` 실제 실행 — 깨끗한 report dir 와 'k10·c01 리포트가 섞인 dir' 둘 다 RELAY ≤ 12줄 · `R!` 없음.

    킷을 임시 디렉터리로 복사해 돌린다(run_all 이 킷 폴더에 tgz 를 남기므로 리포 오염 방지).
    """

    @staticmethod
    def run_all(seed_foreign: bool) -> tuple[subprocess.CompletedProcess, list[str], dict]:
        with tempfile.TemporaryDirectory() as tmp:
            kit = Path(tmp) / "kernel-kit"
            shutil.copytree(KIT_ROOT, kit, ignore=shutil.ignore_patterns("__pycache__", "*.tgz", "kernel-report", "*.pyc"))
            out = Path(tmp) / "report"
            out.mkdir()
            if seed_foreign:        # 실제 k10·c01 하네스의 dry-run 리포트를 같은 dir 에 미리 둔다 (다른 게이트가 남긴 잔존물)
                for harness in ("k10_k3_dspark_sm80", "c01_ep32_a2a"):
                    seeded = subprocess.run([sys.executable, str(kit / harness / "harness.py"), "--cpu-dry-run",
                                             "--report-dir", str(out)], capture_output=True, text=True)
                    assert seeded.returncode == 0, seeded.stderr
                assert (out / "k10_k3_dspark_sm80.json").is_file() and (out / "c01_ep32_a2a.json").is_file()
            env = {**os.environ, "PYTHON": sys.executable, "KERNEL_REPORT_DIR": str(out),
                   "PHASE0_VERDICT": str(KIT_ROOT.parent / "phase0" / "verdict.json"),
                   "K3_RUN_ID": "20261109T000000Z", "K3_TIER": "test"}
            proc = subprocess.run(["bash", str(kit / "run_all.sh"), "--cpu-dry-run"], env=env, capture_output=True,
                                  text=True, timeout=600)
            lines = proc.stdout.splitlines()
            start = next(i for i, ln in enumerate(lines) if ln.startswith("==== RELAY —"))
            end = next(i for i, ln in enumerate(lines) if ln.startswith("==== RELAY 끝"))
            summary = json.loads((out / "summary.json").read_text())
            return proc, lines[start + 1:end], summary

    def check(self, proc: subprocess.CompletedProcess, block: list[str], summary: dict) -> None:
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        self.assertLessEqual(len(block), 12, "\n".join(block))
        self.assertTrue(all(len(ln) <= relay.MAXW for ln in block), block)
        self.assertFalse(any(ln.startswith("R!") for ln in block), "\n".join(block))
        self.assertIn("INC=11", block[-1])
        self.assertEqual(summary["invalid"], {})
        self.assertEqual(summary["counts"], {"PASS": 0, "FAIL": 0, "INCOMPLETE": 11, "MISSING": 0})
        for kid in shapes.KERNEL_IDS:
            self.assertIn(kid[:3], "\n".join(block))

    def test_clean_report_dir(self) -> None:
        proc, block, summary = self.run_all(seed_foreign=False)
        self.check(proc, block, summary)
        self.assertEqual(summary["foreign_schema_reports"], {})

    def test_report_dir_mixed_with_k10_and_c01_reports(self) -> None:
        clean = self.run_all(seed_foreign=False)
        proc, block, summary = self.run_all(seed_foreign=True)
        self.check(proc, block, summary)
        self.assertEqual(block, clean[1])                                                # RELAY 가 깨끗한 dir 와 줄 단위로 같다
        self.assertNotIn("k10", "\n".join(block))
        self.assertEqual(list(summary["foreign_schema_reports"]), ["k10_k3_dspark_sm80.json"])   # 집계 제외로 별도 기록(스키마 위반 아님)


class ReportSelftestMixedDirTest(unittest.TestCase):
    def test_report_selftest_covers_the_mixed_dir_and_is_green_in_a_dirty_report_dir(self) -> None:
        for seed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                if seed:
                    add_foreign(tmp)
                res = subprocess.run([sys.executable, str(KIT_ROOT / "common" / "report.py"), "--selftest", "--report-dir", tmp],
                                     capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                out = json.loads(res.stdout.strip().splitlines()[-1])
                self.assertEqual(out["report_selftest"], "PASS")
                self.assertTrue(all(out["mixed_dir"].values()), out["mixed_dir"])
                self.assertEqual(len(out["mixed_dir"]), 7)

    def test_aggregate_ignores_other_gate_schemas_but_still_flags_broken_kernel_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for kid in shapes.KERNEL_IDS:
                (Path(tmp) / f"{kid}.json").write_text(json.dumps(fake_report(kid, "PASS")))
            add_foreign(tmp)
            agg = report.aggregate(tmp)
            self.assertEqual(sorted(agg["foreign_schema_reports"]), ["k10_k3_dspark_sm80.json", "k13_future_gate.json"])
            # fake_report 는 env 가 없어 validate_report 가 위반을 보고하지만, 그건 KernelReport 로서의 결함이고 foreign 때문이 아니다
            self.assertFalse(any(name.startswith(("k10", "k13")) for name in agg["invalid"]))
            # 같은 계열(s2-kernel-report/*)의 버전 불일치는 여전히 스키마 위반으로 잡힌다 (과잉 무시 방지)
            (Path(tmp) / "k01_kda_chunk.json").write_text(json.dumps({**fake_report("k01_kda_chunk", "PASS"),
                                                                       "schema_version": "s2-kernel-report/v9"}))
            self.assertTrue(any("schema_version 불일치" in e for e in report.aggregate(tmp)["invalid"]["k01_kda_chunk.json"]))
            # schema_version 이 아예 없는 kNN_ 리포트도 종전대로 위반으로 잡힌다 (다른 게이트라고 선언하지 않았으므로 무시하지 않는다)
            schemaless = fake_report("k02_kda_recurrent", "PASS")
            del schemaless["schema_version"]
            (Path(tmp) / "k02_kda_recurrent.json").write_text(json.dumps(schemaless))
            errs = report.aggregate(tmp)["invalid"]["k02_kda_recurrent.json"]
            self.assertIn("필수 키 없음: schema_version", errs)


if __name__ == "__main__":
    unittest.main()
