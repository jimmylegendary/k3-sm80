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

    def test_k31_entrypoint_passes_the_sparse_indexer_flag_and_k3_does_not(self) -> None:
        for spec in LANES:
            lane = REPO_ROOT / "deploy" / spec.id
            env = LaneEnv.load(lane / ".env")
            namespace = runpy.run_path(str(lane / "scripts" / "cluster.py"))
            with self.subTest(lane=spec.id):
                with patch.dict(os.environ, env.values, clear=True):
                    args = namespace["serve_args"](0)
                if spec.is_k31:
                    index = args.index("--sparse-indexer-topk-backend")
                    self.assertEqual(args[index + 1], "persistent")
                    self.assertEqual(args.count("--sparse-indexer-topk-backend"), 1)
                    self.assertEqual(
                        args[args.index("--attention-backend") + 1],
                        "TRITON_MLA_SPARSE",
                    )
                else:
                    self.assertNotIn("--sparse-indexer-topk-backend", args)

    def test_k31_entrypoint_fails_closed_when_the_topk_key_is_missing(self) -> None:
        lane = REPO_ROOT / "deploy" / "11-k31-tp8-pp4-256k-safe"
        env = LaneEnv.load(lane / ".env")
        values = {
            key: value
            for key, value in env.values.items()
            if key != "VLLM_SPARSE_INDEXER_TOPK_BACKEND"
        }
        namespace = runpy.run_path(str(lane / "scripts" / "cluster.py"))

        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(KeyError):
                namespace["serve_args"](0)

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
