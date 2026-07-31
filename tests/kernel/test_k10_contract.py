from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common


dspark_ref = import_common("dspark_ref")
shapes = import_common("shapes")
HARNESS = KIT_ROOT / "k10_k3_dspark_sm80" / "harness.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("k10_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k10 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class K10ContractTest(unittest.TestCase):
    def test_every_accept_count_commits_only_accepted_states(self) -> None:
        states = tuple(f"state-{index}" for index in range(8))
        for accepted in range(8):
            self.assertEqual(
                dspark_ref.accepted_state_prefix(states, accepted),
                states[:accepted],
            )

    def test_invalid_accept_count_is_rejected(self) -> None:
        for accepted in (-1, 8):
            with self.assertRaisesRegex(ValueError, r"\[0, 7\]"):
                dspark_ref.accepted_state_prefix((1, 2), accepted)

    def test_k3_dspark_contract(self) -> None:
        self.assertEqual(
            dspark_ref.TARGET_AUX_LAYER_IDS, (2, 23, 47, 71, 89)
        )
        self.assertEqual(dspark_ref.DSPARK_BLOCK_SIZE, 7)
        self.assertEqual(
            set(shapes.DSPARK_REQUIRED_CASES),
            {
                *(f"greedy_accept_{value}" for value in range(8)),
                "probabilistic_block",
                "ctx256k",
                "ctx1m",
            },
        )

    def test_block_rejection_stops_at_first_rejected_draft(self) -> None:
        self.assertEqual(
            dspark_ref.accepted_block_length(
                draft_probabilities=(0.9, 0.8, 0.7, 0.6),
                uniforms=(0.1, 0.2, 0.95, 0.0),
            ),
            2,
        )

    def test_dry_run_never_claims_gpu_pass(self) -> None:
        module = load_harness()
        payload = module.build_dry_report()

        self.assertEqual(payload["overall"], "INCOMPLETE")
        self.assertEqual(payload["evidence"]["attention_backend"], "TRITON_MLA")
        self.assertTrue(
            all(gate["status"] == "SKIP" for gate in payload["gates"].values())
        )
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("moonshotai/Kimi-K3", source)
        self.assertIn("Inferact/Kimi-K3-DSpark", source)

        result = subprocess.run(
            ["python3", str(HARNESS), "--cpu-dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("K10-OVERALL INCOMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
