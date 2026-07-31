from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_specs import LANES
from deploy.tooling.render_lanes import render_all


HEADINGS = (
    "## 1. 검증 범위",
    "## 2. 토폴로지와 rank",
    "## 3. 단일 `.env`",
    "## 4. 모델 cache",
    "## 5. 이미지",
    "## 6. Preflight",
    "## 7. Gate 실행",
    "## 8. Serving 실행",
    "## 9. PASS 기준",
    "## 10. 오류 코드",
    "## 11. 전달할 출력",
    "## 12. 정리",
    "## 13. 승격",
)


class RunbookTest(unittest.TestCase):
    def test_runbooks_have_ordered_operator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                text = (root / spec.id / "RUNBOOK.md").read_text()
                offsets = [text.index(heading) for heading in HEADINGS]
                self.assertEqual(offsets, sorted(offsets), spec.id)
                for command in (
                    "make preflight",
                    "make gates",
                    "make up",
                    "make gate-load",
                    "make gate-context",
                    "make smoke",
                    "make status",
                    "make down",
                ):
                    self.assertIn(command, text, spec.id)
                self.assertIn("최대 20줄", text)
                self.assertIn("tgz/tarball은 필요 없다", text)
                self.assertNotIn("tar -", text.lower())

    def test_feature_specific_operator_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                text = (root / spec.id / "RUNBOOK.md").read_text()
                if spec.dspark:
                    self.assertIn("make gate-dspark", text, spec.id)
                if spec.fp8_kv:
                    self.assertIn("make calibrate", text, spec.id)
                    for length in ("262144", "524288", "786432", "1048576"):
                        self.assertIn(length, text, spec.id)

    def test_smoke_script_has_real_openai_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                smoke = (root / spec.id / "scripts" / "smoke.py").read_text()
                self.assertIn("/v1/models", smoke)
                self.assertIn("/v1/completions", smoke)
                self.assertIn("K3-SMOKE-", smoke)
                self.assertNotIn("NOT-WIRED", smoke)

    def test_preflight_rejects_existing_gpu_work_and_k3_containers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                preflight = (
                    root / spec.id / "scripts" / "preflight.py"
                ).read_text()
                self.assertIn("--query-compute-apps=pid,process_name", preflight)
                self.assertIn("PREFLIGHT-GPU-BUSY", preflight)
                self.assertIn("PREFLIGHT-K3-CONTAINER-BUSY", preflight)


if __name__ == "__main__":
    unittest.main()
