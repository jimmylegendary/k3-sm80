from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_schema import ALLOWED_KEYS, LaneEnv
from deploy.tooling.lane_specs import LANES


class LaneSpecTest(unittest.TestCase):
    def test_exact_lane_contract(self) -> None:
        self.assertEqual(
            [
                (
                    lane.id,
                    lane.tp,
                    lane.pp,
                    lane.dp,
                    lane.ep,
                    lane.max_len,
                    lane.dspark,
                    lane.fp8_kv,
                )
                for lane in LANES
            ],
            [
                ("01-tp8-pp4-256k-safe", 8, 4, 1, False, 262144, False, False),
                (
                    "02-tp8-dp4-ep32-256k-safe",
                    8,
                    1,
                    4,
                    True,
                    262144,
                    False,
                    False,
                ),
                (
                    "03-tp8-dp4-ep32-256k-dspark",
                    8,
                    1,
                    4,
                    True,
                    262144,
                    True,
                    False,
                ),
                (
                    "04-tp8-dp4-ep32-1m-fp8kv",
                    8,
                    1,
                    4,
                    True,
                    1048576,
                    False,
                    True,
                ),
                (
                    "05-tp8-dp4-ep32-1m-fp8kv-dspark",
                    8,
                    1,
                    4,
                    True,
                    1048576,
                    True,
                    True,
                ),
            ],
        )

    def test_every_lane_uses_exactly_32_processes(self) -> None:
        self.assertEqual(
            [lane.tp * lane.pp * lane.dp for lane in LANES],
            [32, 32, 32, 32, 32],
        )


class LaneEnvParsingTest(unittest.TestCase):
    def test_rejects_unknown_and_duplicate_keys(self) -> None:
        env = LaneEnv.parse(
            "VLLM_MAX_MODEL_LEN=262144\n"
            "VLLM_MAX_MODEL_LEN=1048576\n"
            "BOGUS=1\n"
        )

        self.assertIn("duplicate key: VLLM_MAX_MODEL_LEN", env.errors)
        self.assertIn("unknown key: BOGUS", env.errors)

    def test_rejects_malformed_lines_without_executing_them(self) -> None:
        marker = Path(tempfile.gettempdir()) / "k3-sm80-lane-schema-must-not-exist"
        marker.unlink(missing_ok=True)

        env = LaneEnv.parse(f"touch {marker}\n")

        self.assertEqual(env.errors, ["line 1: expected KEY=VALUE"])
        self.assertFalse(marker.exists())

    def test_allowed_keys_cover_serving_controls(self) -> None:
        self.assertTrue(
            {
                "K3_HEAD_NODE_IP",
                "K3_WORKER3_SSH_HOST",
                "VLLM_TENSOR_PARALLEL_SIZE",
                "VLLM_DATA_PARALLEL_SIZE",
                "VLLM_ENABLE_EXPERT_PARALLEL",
                "VLLM_MAX_MODEL_LEN",
                "VLLM_MAX_NUM_SEQS",
                "VLLM_SPECULATIVE_CONFIG",
                "VLLM_EXTRA_ARGS",
            }.issubset(ALLOWED_KEYS)
        )


if __name__ == "__main__":
    unittest.main()
