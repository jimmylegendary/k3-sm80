from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common


shapes = import_common("shapes")
HARNESS = KIT_ROOT / "k09_mla_fp8_kv_sm80" / "harness.py"
KERNEL = KIT_ROOT / "vendor" / "mla_fp8_kv_sm80.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("k09_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k09 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class K09ContractTest(unittest.TestCase):
    def test_k3_mla_fp8_shapes_are_registered(self) -> None:
        required = shapes.REQUIRED_SHAPES["k09_mla_fp8_kv_sm80"]

        self.assertEqual(shapes.MLA_CACHE_D, 576)
        self.assertEqual(shapes.MLA_FP8_PAGE_SIZE, 64)
        self.assertEqual(
            set(required),
            {
                "contiguous_8k",
                "noncontiguous_256k",
                "partial_1m",
            },
        )
        self.assertEqual(required["partial_1m"]["tokens"], 1_048_576)

    def test_kernel_source_is_software_byte_storage_not_fp8_matmul(self) -> None:
        source = KERNEL.read_text(encoding="utf-8")

        self.assertIn("def e4m3_bits_to_fp32", source)
        self.assertIn("def pack_bf16_to_e4m3", source)
        self.assertIn("def load_e4m3_to_bf16", source)
        self.assertIn("tl.uint8", source)
        self.assertNotIn("tl.dot", source)
        self.assertNotIn("float8_e4m3fn", source)

    def test_dry_run_never_claims_gpu_pass(self) -> None:
        payload = load_harness().build_dry_report()

        self.assertEqual(payload["id"], "k09_mla_fp8_kv_sm80")
        self.assertEqual(payload["overall"], "INCOMPLETE")
        self.assertEqual(payload["evidence"]["storage_dtype"], "uint8/e4m3fn")
        self.assertTrue(
            all(gate["status"] == "SKIP" for gate in payload["gates"].values())
        )

        with tempfile.TemporaryDirectory() as tempdir:
            result = subprocess.run(
                [
                    "python3",
                    str(HARNESS),
                    "--cpu-dry-run",
                    "--report-dir",
                    tempdir,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report_payload = json.loads(
                (Path(tempdir) / "k09_mla_fp8_kv_sm80.json").read_text()
            )

        self.assertIn("K09-OVERALL INCOMPLETE", result.stdout)
        self.assertEqual(report_payload["overall"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
