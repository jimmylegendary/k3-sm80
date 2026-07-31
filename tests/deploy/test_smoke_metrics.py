from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = (
    REPO_ROOT
    / "deploy"
    / "03-tp8-dp4-ep32-256k-dspark"
    / "scripts"
    / "smoke.py"
)


def load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lane03_smoke", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmokeMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke = load_smoke()

    def test_metric_total_matches_full_prometheus_name_only(self) -> None:
        metrics = """
# HELP vllm:spec_decode_num_accepted_tokens accepted
vllm:spec_decode_num_accepted_tokens_total{model_name="kimi-k3"} 5
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 3
vllm:spec_decode_num_drafts_total 2
"""

        self.assertEqual(
            self.smoke._metric_total(
                metrics,
                "spec_decode_num_accepted_tokens",
            ),
            5.0,
        )
        self.assertEqual(
            self.smoke._metric_total(metrics, "spec_decode_num_drafts"),
            2.0,
        )

    def test_position_samples_are_summed_across_series(self) -> None:
        metrics = """
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 3
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 4
"""

        self.assertEqual(self.smoke._accepted_positions(metrics), {0: 5.0, 1: 4.0})


if __name__ == "__main__":
    unittest.main()
