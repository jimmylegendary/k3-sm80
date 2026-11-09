from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.render_lanes import render_all
from deploy.tooling.validate_lanes import validate_repository


REPO_ROOT = Path(__file__).resolve().parents[2]


class StaticLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        render_all(self.root / "deploy")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_committed_repository_is_valid(self) -> None:
        self.assertEqual(validate_repository(REPO_ROOT), [])

    def test_fresh_render_is_valid_with_site_placeholders(self) -> None:
        self.assertEqual(validate_repository(self.root), [])

    def test_rejects_override_env_files(self) -> None:
        override = (
            self.root
            / "deploy"
            / "01-tp8-pp4-256k-safe"
            / ".env.worker1"
        )
        override.write_text("VLLM_MAX_MODEL_LEN=1\n", encoding="utf-8")

        errors = validate_repository(self.root)

        self.assertTrue(any("override env" in error for error in errors), errors)

    def test_ignores_python_bytecode_cache_only(self) -> None:
        cache = (
            self.root
            / "deploy"
            / "01-tp8-pp4-256k-safe"
            / "scripts"
            / "__pycache__"
        )
        cache.mkdir()
        (cache / "cluster.cpython-312.pyc").write_bytes(b"runtime cache")

        self.assertEqual(validate_repository(self.root), [])

    def test_rejects_duplicate_ports_between_lanes(self) -> None:
        first = self.root / "deploy" / "01-tp8-pp4-256k-safe" / ".env"
        second = self.root / "deploy" / "02-tp8-dp4-ep32-256k-safe" / ".env"
        first_port = next(
            line for line in first.read_text().splitlines() if line.startswith("VLLM_API_PORT=")
        )
        text = second.read_text()
        text = "\n".join(
            first_port if line.startswith("VLLM_API_PORT=") else line
            for line in text.splitlines()
        )
        second.write_text(text + "\n", encoding="utf-8")

        errors = validate_repository(self.root)

        self.assertTrue(any("duplicate VLLM_API_PORT" in error for error in errors), errors)

    def test_fresh_render_has_all_six_lanes_and_env_files(self) -> None:
        deploy = self.root / "deploy"

        self.assertEqual(len(list(deploy.rglob(".env"))), 6)
        self.assertTrue((deploy / "11-k31-tp8-pp4-256k-safe" / ".env").is_file())

    def test_missing_k31_lane_directory_is_reported(self) -> None:
        shutil.rmtree(self.root / "deploy" / "11-k31-tp8-pp4-256k-safe")

        errors = validate_repository(self.root)

        self.assertIn("missing lane: 11-k31-tp8-pp4-256k-safe", errors)
        self.assertTrue(any("expected exactly 6 .env files" in e for e in errors), errors)

    def test_unexpected_two_digit_lane_directory_is_reported(self) -> None:
        stray = self.root / "deploy" / "12-k31-tp8-dp4-ep32-256k-dspark"
        stray.mkdir()
        (stray / ".env").write_text("VLLM_MAX_MODEL_LEN=1\n", encoding="utf-8")

        errors = validate_repository(self.root)

        self.assertIn("unexpected lane: 12-k31-tp8-dp4-ep32-256k-dspark", errors)

    def test_k31_env_with_k3_attention_backend_is_rejected(self) -> None:
        env_path = self.root / "deploy" / "11-k31-tp8-pp4-256k-safe" / ".env"
        env_path.write_text(
            env_path.read_text().replace(
                "VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE\n",
                "VLLM_ATTENTION_BACKEND=TRITON_MLA\n",
            ),
            encoding="utf-8",
        )

        errors = validate_repository(self.root)

        self.assertTrue(
            any("11-k31" in e and "TRITON_MLA_SPARSE" in e for e in errors), errors
        )

    def test_k31_lane_port_collision_with_a_k3_lane_is_reported(self) -> None:
        env_path = self.root / "deploy" / "11-k31-tp8-pp4-256k-safe" / ".env"
        env_path.write_text(
            env_path.read_text().replace("VLLM_API_PORT=8111\n", "VLLM_API_PORT=8101\n"),
            encoding="utf-8",
        )

        errors = validate_repository(self.root)

        self.assertTrue(any("duplicate VLLM_API_PORT=8101" in e for e in errors), errors)

    def test_rejects_missing_compose_profile(self) -> None:
        compose = (
            self.root
            / "deploy"
            / "03-tp8-dp4-ep32-256k-dspark"
            / "compose.yaml"
        )
        compose.write_text(
            compose.read_text().replace('profiles: ["worker3"]', 'profiles: ["worker2"]'),
            encoding="utf-8",
        )

        errors = validate_repository(self.root)

        self.assertTrue(any("profiles" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
