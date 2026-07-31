from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_specs import LANES
from deploy.tooling.render_lanes import render_all, render_lane


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RenderLanesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_rendered_lane_is_self_contained(self) -> None:
        render_lane(LANES[0], self.root)
        lane = self.root / LANES[0].id

        self.assertEqual(
            {path.name for path in lane.iterdir()},
            {".env", "compose.yaml", "Makefile", "RUNBOOK.md", "scripts"},
        )
        self.assertEqual(
            {path.name for path in (lane / "scripts").iterdir()},
            {"preflight.py", "cluster.py", "smoke.py"},
        )

    def test_render_all_produces_exactly_five_env_files(self) -> None:
        render_all(self.root)

        env_files = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob(".env"))
        self.assertEqual(
            env_files,
            [f"{lane.id}/.env" for lane in LANES],
        )

    def test_render_is_byte_deterministic(self) -> None:
        render_all(self.root)
        first = tree_hash(self.root)

        render_all(self.root)

        self.assertEqual(tree_hash(self.root), first)

    def test_lane_env_values_match_topology_and_features(self) -> None:
        render_all(self.root)

        lane_01 = (self.root / LANES[0].id / ".env").read_text()
        lane_05 = (self.root / LANES[4].id / ".env").read_text()
        self.assertIn("VLLM_PIPELINE_PARALLEL_SIZE=4\n", lane_01)
        self.assertIn("VLLM_DATA_PARALLEL_SIZE=1\n", lane_01)
        self.assertIn("VLLM_KV_CACHE_DTYPE=bfloat16\n", lane_01)
        self.assertIn("VLLM_PIPELINE_PARALLEL_SIZE=1\n", lane_05)
        self.assertIn("VLLM_DATA_PARALLEL_SIZE=4\n", lane_05)
        self.assertIn("VLLM_ENABLE_EXPERT_PARALLEL=1\n", lane_05)
        self.assertIn("VLLM_KV_CACHE_DTYPE=fp8\n", lane_05)
        self.assertIn('"method":"dspark"', lane_05)

    def test_compose_contains_four_distinct_profiles(self) -> None:
        render_lane(LANES[2], self.root)
        compose = (self.root / LANES[2].id / "compose.yaml").read_text()

        for profile in ("head", "worker1", "worker2", "worker3"):
            self.assertEqual(compose.count(f'profiles: ["{profile}"]'), 1)


if __name__ == "__main__":
    unittest.main()
