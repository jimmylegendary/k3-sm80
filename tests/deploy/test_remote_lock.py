from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_schema import LaneEnv
from deploy.tooling.remote import (
    _acquire_cluster_lock,
    _cluster_lock_path,
    _release_cluster_lock,
)


class ClusterLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.env = LaneEnv(values={"K3_SERVING_ROOT": str(root / "service")})
        self.lane = root / "01-tp8-pp4-256k-safe"
        self.other_lane = root / "02-tp8-dp4-ep32-256k-safe"
        self.lane.mkdir()
        self.other_lane.mkdir()

    def test_lock_is_global_and_records_owner(self) -> None:
        lock = _acquire_cluster_lock(self.lane, self.env, "run-a")

        self.assertEqual(lock, _cluster_lock_path(self.env))
        owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        self.assertEqual(owner["lane"], self.lane.name)
        self.assertEqual(owner["run_id"], "run-a")
        self.assertIsInstance(owner["pid"], int)
        with self.assertRaisesRegex(RuntimeError, "32-GPU cluster lock held"):
            _acquire_cluster_lock(self.other_lane, self.env, "run-b")

    def test_only_owning_lane_can_release_lock(self) -> None:
        lock = _acquire_cluster_lock(self.lane, self.env, "run-a")

        with self.assertRaisesRegex(RuntimeError, "belongs to"):
            _release_cluster_lock(self.other_lane, self.env)
        self.assertTrue(lock.is_dir())

        _release_cluster_lock(self.lane, self.env)
        self.assertFalse(lock.exists())

    def test_missing_or_corrupt_owner_fails_closed(self) -> None:
        lock = _cluster_lock_path(self.env)
        lock.mkdir(parents=True)

        with self.assertRaisesRegex(RuntimeError, "owner missing"):
            _release_cluster_lock(self.lane, self.env)
        (lock / "owner.json").write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "owner corrupt"):
            _release_cluster_lock(self.lane, self.env)


if __name__ == "__main__":
    unittest.main()
