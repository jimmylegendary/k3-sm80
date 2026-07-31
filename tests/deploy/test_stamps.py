from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.stamps import StampKey, validate_stamp, write_stamp


class StampTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "k08.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_round_trip_pass_stamp(self) -> None:
        key = StampKey("abc", "sha256:image", "sha256:env-a", "k08")

        write_stamp(self.path, key, {"overall": "PASS", "gpu": "A100"})

        self.assertEqual(validate_stamp(self.path, key), [])
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["evidence"]["overall"], "PASS")

    def test_env_change_invalidates_stamp(self) -> None:
        key = StampKey("abc", "sha256:image", "sha256:env-a", "k08")
        write_stamp(self.path, key, {"overall": "PASS"})
        changed = StampKey("abc", "sha256:image", "sha256:env-b", "k08")

        self.assertEqual(validate_stamp(self.path, changed), ["env_sha256 mismatch"])

    def test_each_key_field_is_bound(self) -> None:
        key = StampKey("abc", "sha256:image", "sha256:env", "k08")
        write_stamp(self.path, key, {"overall": "PASS"})

        variants = (
            (StampKey("def", key.image_digest, key.env_sha256, key.gate_id), "repo_commit"),
            (StampKey(key.repo_commit, "sha256:other", key.env_sha256, key.gate_id), "image_digest"),
            (StampKey(key.repo_commit, key.image_digest, "sha256:other", key.gate_id), "env_sha256"),
            (StampKey(key.repo_commit, key.image_digest, key.env_sha256, "k09"), "gate_id"),
        )
        for changed, field in variants:
            with self.subTest(field=field):
                self.assertEqual(validate_stamp(self.path, changed), [f"{field} mismatch"])

    def test_missing_corrupt_and_non_pass_stamps_fail_closed(self) -> None:
        key = StampKey("abc", "sha256:image", "sha256:env", "k08")
        self.assertEqual(validate_stamp(self.path, key), ["stamp missing"])

        self.path.write_text("{", encoding="utf-8")
        self.assertEqual(validate_stamp(self.path, key), ["stamp is not valid JSON"])

        write_stamp(self.path, key, {"overall": "FAIL"})
        self.assertEqual(validate_stamp(self.path, key), ["evidence overall is not PASS"])


if __name__ == "__main__":
    unittest.main()
