from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deploy.tooling.validate_lanes import (
    _EVIDENCE_IDS,
    _EVIDENCE_IDS_K31,
    validate_evidence,
    validate_evidence_k31,
)


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


class K31EvidenceValidationTest(unittest.TestCase):
    """docs/A100-K31-RESULTS.md 행 집합: k08 kda k11 k12 lane11."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "k31-evidence.md"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_id_sets_are_exactly_the_documented_ones(self) -> None:
        self.assertEqual(_EVIDENCE_IDS_K31, ("k08", "kda", "k11", "k12", "lane11"))
        self.assertEqual(
            _EVIDENCE_IDS,
            ("k08", "c01", "k10", "k09", "lane01", "lane02", "lane03", "lane04", "lane05"),
        )

    def test_complete_incomplete_table_is_valid(self) -> None:
        self.path.write_text(HEADER + "".join(row(item) for item in _EVIDENCE_IDS_K31))

        self.assertEqual(validate_evidence_k31(self.path), [])

    def test_missing_k11_or_k12_row_fails(self) -> None:
        for dropped in ("k11", "k12", "lane11", "kda"):
            with self.subTest(dropped=dropped):
                ids = [item for item in _EVIDENCE_IDS_K31 if item != dropped]
                self.path.write_text(HEADER + "".join(row(item) for item in ids))

                errors = validate_evidence_k31(self.path)

                self.assertTrue(
                    any("missing evidence rows" in e and dropped in e for e in errors),
                    errors,
                )

    def test_pass_cannot_use_not_run_fields(self) -> None:
        self.path.write_text(
            HEADER
            + "".join(
                row(item, "PASS" if item == "k11" else "INCOMPLETE")
                for item in _EVIDENCE_IDS_K31
            )
        )

        errors = validate_evidence_k31(self.path)

        self.assertTrue(any("k11" in e and "not-run" in e for e in errors), errors)

    def test_real_pass_row_needs_sha256_digests(self) -> None:
        good = (
            "| k12 | `make -C deploy/11-k31-tp8-pp4-256k-safe gate-k12` | "
            "sha256:" + "a" * 64 + " | sha256:" + "b" * 64 + " | 32xA100 driver 580 | "
            "PASS | /mnt/nas/runs/k12_sparse_mla_sm80.json |\n"
        )
        rest = "".join(row(item) for item in _EVIDENCE_IDS_K31 if item != "k12")
        self.path.write_text(HEADER + good + rest)
        self.assertEqual(validate_evidence_k31(self.path), [])

        self.path.write_text(HEADER + good.replace("sha256:" + "a" * 64, "latest") + rest)
        errors = validate_evidence_k31(self.path)
        self.assertTrue(any("image digest must start sha256" in e for e in errors), errors)

    def test_the_two_documents_validate_independently(self) -> None:
        self.path.write_text(HEADER + "".join(row(item) for item in _EVIDENCE_IDS_K31))
        k3_errors = validate_evidence(self.path)
        self.assertTrue(any("missing evidence rows" in e for e in k3_errors), k3_errors)
        self.assertFalse(any("k11" in e or "lane11" in e for e in k3_errors), k3_errors)

        self.path.write_text(HEADER + "".join(row(item) for item in _EVIDENCE_IDS))
        k31_errors = validate_evidence_k31(self.path)
        self.assertTrue(any("missing evidence rows" in e for e in k31_errors), k31_errors)
        self.assertEqual(validate_evidence(self.path), [])

    def test_missing_file_is_reported(self) -> None:
        errors = validate_evidence_k31(Path(self.tempdir.name) / "absent.md")

        self.assertEqual(len(errors), 1)
        self.assertIn("evidence missing", errors[0])


if __name__ == "__main__":
    unittest.main()
