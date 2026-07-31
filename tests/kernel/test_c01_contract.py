from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT


HARNESS = KIT_ROOT / "c01_ep32_a2a" / "harness.py"
LAUNCHER = KIT_ROOT / "c01_ep32_a2a" / "launch.sh"


def load_harness():
    spec = importlib.util.spec_from_file_location("c01_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load c01 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class C01ContractTest(unittest.TestCase):
    def test_rank_layout_covers_32_unique_ranks(self) -> None:
        module = load_harness()
        rows = [
            module.rank_layout(node, local)
            for node in range(4)
            for local in range(8)
        ]

        self.assertEqual(
            sorted(global_rank for global_rank, _, _ in rows), list(range(32))
        )
        self.assertEqual({tp for _, tp, _ in rows}, set(range(8)))
        self.assertEqual({dp for _, _, dp in rows}, set(range(4)))

    def test_sm90_backends_are_rejected(self) -> None:
        module = load_harness()

        module.validate_backend("allgather_reducescatter")
        with self.assertRaisesRegex(ValueError, "SM80 correctness backend"):
            module.validate_backend("deepep_v2")

    def test_fixtures_are_deterministic_and_include_skew(self) -> None:
        module = load_harness()

        for tokens in (1, 8, 32):
            self.assertEqual(
                module.routed_token_fixture(tokens, 7),
                module.routed_token_fixture(tokens, 7),
            )
            self.assertEqual(len(module.routed_token_fixture(tokens, 7)), tokens)
        skew = module.routed_token_fixture(32, 7, skew_expert=511)
        self.assertEqual({row["expert"] for row in skew}, {511})

    def test_timeout_is_not_classified_as_backend_failure(self) -> None:
        module = load_harness()

        classified = module.classify_launch(124, timed_out=True, stderr="")
        self.assertEqual(classified["classification"], "TIMEOUT")
        self.assertEqual(classified["status"], "INCOMPLETE")

    def test_dry_run_and_launcher_contract(self) -> None:
        module = load_harness()
        dry = module.build_dry_report()
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertEqual(dry["id"], "c01_ep32_a2a")
        self.assertEqual(dry["overall"], "INCOMPLETE")
        self.assertIn("--nnodes=4", launcher)
        self.assertIn("--nproc-per-node=8", launcher)
        self.assertIn("MASTER_ADDR", launcher)
        self.assertIn("NODE_RANK", launcher)

        result = subprocess.run(
            ["python3", str(HARNESS), "--cpu-dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("C01-OVERALL INCOMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
