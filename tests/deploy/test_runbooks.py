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

    def test_k3_runbooks_keep_k3_model_names_and_never_mention_k31(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                if spec.is_k31:
                    continue
                text = (root / spec.id / "RUNBOOK.md").read_text()
                self.assertIn("`moonshotai/Kimi-K3` MXFP4 checkpoint", text, spec.id)
                self.assertIn(
                    "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`다", text, spec.id
                )
                self.assertIn("models--moonshotai--Kimi-K3/snapshots", text, spec.id)
                self.assertIn("models--Inferact--Kimi-K3-DSpark/snapshots", text, spec.id)
                self.assertIn("kda-sm80-int64", text, spec.id)
                for needle in (
                    "K3.1",
                    "gate-k11",
                    "gate-k12",
                    "reasoning_effort",
                    "32-GPU cluster lock",
                ):
                    self.assertNotIn(needle, text, spec.id)

    def test_k31_runbook_describes_k31_gates_image_and_honest_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            text = (root / "11-k31-tp8-pp4-256k-safe" / "RUNBOOK.md").read_text()

        self.assertIn("TP8/PP4/DP1, EP=0", text)
        self.assertIn("DSpark=0", text)
        self.assertIn("`k08 kda k11 k12 pp`", text)
        self.assertIn("이 lane의 선행 gate는 `k08 kda k11 k12`다", text)
        self.assertIn("make gate-k11", text)
        self.assertIn("make gate-k12", text)
        self.assertIn("k11_dsa_indexer_sm80.json", text)
        self.assertIn("k12_sparse_mla_sm80.json", text)
        self.assertIn("`--kv-cache-dtype bfloat16`", text)
        self.assertIn("fp8_ds_mla", text)
        self.assertIn("`a3c14efb`", text)
        self.assertIn("docker/patches-k31/", text)
        self.assertIn("lane 11은 PP4 stamp가 현재 env와 일치", text)
        self.assertIn("32-GPU cluster lock", text)
        self.assertIn("단독 gate 실행은 preflight를 거치지 않아 막지 못한다", text)
        self.assertNotIn("lane 02–05는 DP4/EP32", text)
        # 정직한 상한: 속도 미보증, 256k만, DSpark 없음, 1M 미지원, Q fp8 미적용
        self.assertIn("속도는 보증하지 않으며", text)
        self.assertIn("1M 미지원", text)
        self.assertIn("top-k 경계가 달라질 수 있다", text)
        self.assertIn("`K3.1 P2`로 이동한다", text)
        self.assertIn(
            "[K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md)",
            text,
        )

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
