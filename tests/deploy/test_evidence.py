from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deploy.tooling.validate_lanes import validate_evidence


HEADER = (
    "| id | command | image digest | env hash | GPU/driver | result | artifact |\n"
    "|---|---|---|---|---|---|---|\n"
)


def row(identifier: str, result: str = "INCOMPLETE") -> str:
    return (
        f"| {identifier} | `not-run` | not-run | not-run | not-run | "
        f"{result} | not-run |\n"
    )


class EvidenceValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "evidence.md"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_complete_incomplete_table_is_valid(self) -> None:
        ids = ("k08", "c01", "k10", "k09", "lane01", "lane02", "lane03", "lane04", "lane05")
        self.path.write_text(HEADER + "".join(row(item) for item in ids))

        self.assertEqual(validate_evidence(self.path), [])

    def test_pass_cannot_use_not_run_fields(self) -> None:
        ids = ("k08", "c01", "k10", "k09", "lane01", "lane02", "lane03", "lane04", "lane05")
        self.path.write_text(
            HEADER + "".join(row(item, "PASS" if item == "k08" else "INCOMPLETE") for item in ids)
        )

        errors = validate_evidence(self.path)

        self.assertTrue(any("k08" in error and "not-run" in error for error in errors), errors)

    def test_missing_or_duplicate_rows_fail(self) -> None:
        self.path.write_text(HEADER + row("k08") + row("k08"))

        errors = validate_evidence(self.path)

        self.assertTrue(any("duplicate" in error for error in errors), errors)
        self.assertTrue(any("missing" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
