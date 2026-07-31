from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common


shapes = import_common("shapes")
HARNESS_PATH = KIT_ROOT / "k08_mxfp4_marlin_moe" / "harness.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("k08_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k08 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class K08DryRunTest(unittest.TestCase):
    def test_k08_dry_run_requires_both_layout_arms(self) -> None:
        report = load_harness().build_dry_report()

        self.assertEqual(report["id"], "k08_mxfp4_marlin_moe")
        self.assertEqual(report["overall"], "INCOMPLETE")
        self.assertEqual(report["evidence"]["layout_arms"], ["tp8", "ep32"])
        self.assertTrue(
            all(gate["status"] == "SKIP" for gate in report["gates"].values())
        )

    def test_k3_shapes_are_registered(self) -> None:
        required = shapes.REQUIRED_SHAPES["k08_mxfp4_marlin_moe"]

        self.assertEqual(
            required,
            {
                "m1": {"m": 1, "k": 3584, "n": 3072, "e": 896, "topk": 16},
                "m8": {"m": 8, "k": 3584, "n": 3072, "e": 896, "topk": 16},
                "m32": {"m": 32, "k": 3584, "n": 3072, "e": 896, "topk": 16},
                "prefill": {"m": 8192, "k": 3584, "n": 3072, "e": 896, "topk": 16},
            },
        )

    def test_cli_writes_incomplete_report_without_gpu_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = subprocess.run(
                [
                    "python3",
                    str(HARNESS_PATH),
                    "--cpu-dry-run",
                    "--report-dir",
                    tempdir,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(
                (Path(tempdir) / "k08_mxfp4_marlin_moe.json").read_text()
            )

        self.assertEqual(report["overall"], "INCOMPLETE")
        self.assertNotIn("PASS", {gate["status"] for gate in report["gates"].values()})


if __name__ == "__main__":
    unittest.main()
