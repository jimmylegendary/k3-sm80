from __future__ import annotations

import os
import runpy
import shlex
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.tooling.command import build_vllm_args
from deploy.tooling.lane_schema import LaneEnv
from deploy.tooling.lane_specs import LANES


REPO_ROOT = Path(__file__).resolve().parents[2]


class ClusterEntrypointTest(unittest.TestCase):
    def test_container_entrypoint_matches_central_builder_for_every_rank(self) -> None:
        for spec in LANES:
            lane = REPO_ROOT / "deploy" / spec.id
            parsed = LaneEnv.load(lane / ".env")
            process_env = os.environ.copy()
            process_env.update(parsed.values)
            for rank in range(4):
                with self.subTest(lane=spec.id, rank=rank):
                    result = subprocess.run(
                        [
                            "python3",
                            str(lane / "scripts" / "cluster.py"),
                            "serve",
                            "--node-rank",
                            str(rank),
                            "--dry-run",
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                        env=process_env,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    marker = " argv="
                    self.assertIn(marker, result.stdout)
                    actual = shlex.split(result.stdout.split(marker, 1)[1].strip())
                    expected = ["vllm", "serve", *build_vllm_args(parsed, rank)]
                    self.assertEqual(actual, expected)

    def test_worker_profiles_are_headless_but_head_profile_is_not(self) -> None:
        lane = REPO_ROOT / "deploy" / "02-tp8-dp4-ep32-256k-safe"
        env = LaneEnv.load(lane / ".env")
        namespace = runpy.run_path(str(lane / "scripts" / "cluster.py"))

        with patch.dict(os.environ, env.values, clear=True):
            self.assertNotIn("--headless", namespace["serve_args"](0))
            for node_rank in (1, 2, 3):
                with self.subTest(node_rank=node_rank):
                    self.assertIn("--headless", namespace["serve_args"](node_rank))


if __name__ == "__main__":
    unittest.main()
