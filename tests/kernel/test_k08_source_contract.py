from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT


REPO_ROOT = KIT_ROOT.parent
HARNESS_PATH = KIT_ROOT / "k08_mxfp4_marlin_moe" / "harness.py"
PATCH_PATH = REPO_ROOT / "docker" / "patches" / "mxfp4-marlin-moe-sm80.patch"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile.openai"


def load_harness():
    spec = importlib.util.spec_from_file_location("k08_gpu_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k08 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class K08SourceContractTest(unittest.TestCase):
    def test_marlin_patch_and_harness_share_original_mxfp4_contract(self) -> None:
        self.assertTrue(PATCH_PATH.is_file(), "MXFP4 SM80 patch is missing")
        patch = PATCH_PATH.read_text(encoding="utf-8")
        harness = HARNESS_PATH.read_text(encoding="utf-8")

        for token in (
            "sm_80",
            "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
            "kFE2M1f",
            "kFE8M0fnu",
            "fused_marlin_moe",
            "device_max_shared_mem",
            "sh_cache_size",
            "max_c_tmp_size",
        ):
            self.assertIn(token, patch)
        for token in (
            "weight_packed",
            "weight_scale",
            "prepare_moe_mxfp4_layer_for_marlin",
            "fused_marlin_moe",
            "scalar_types.float4_e2m1f.id",
        ):
            self.assertIn(token, harness)

    def test_openai_image_rebuilds_the_patched_cuda_extension(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

        self.assertIn("mxfp4-marlin-moe-sm80.patch", dockerfile)
        self.assertIn("git apply --check", dockerfile)
        self.assertIn("pip install . --no-build-isolation", dockerfile)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.0", dockerfile)
        self.assertIn("cuda-nvrtc-dev-13-0", dockerfile)
        self.assertIn("sm80_kernel_bfloat16_fe2m1f_bfloat16.cu", dockerfile)

    def test_classify_child_keeps_allocator_oom_separate_from_memory_violation(
        self,
    ) -> None:
        harness = load_harness()

        oom = harness.classify_child(
            "m1",
            88,
            "/does/not/exist.json",
            "Program hit cudaErrorMemoryAllocation (error 2) due to out of memory",
        )
        self.assertEqual(oom["classification"], "OOM")
        self.assertTrue(oom["allocator_oom"])
        self.assertEqual(oom["memory_violation_count"], 0)

        illegal = harness.classify_child(
            "m1",
            88,
            "/does/not/exist.json",
            "CUDA error: an illegal memory access was encountered",
        )
        self.assertEqual(illegal["classification"], "MEMORY_VIOLATION")
        self.assertFalse(illegal["allocator_oom"])
        self.assertGreater(illegal["memory_violation_count"], 0)

    def test_classify_child_accepts_only_explicit_success_json(self) -> None:
        harness = load_harness()
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "child.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "k08-child/v1",
                        "status": "PASS",
                        "numeric": {"rmse_rel": 0.0, "max_abs": 0.0},
                        "canary_ok": True,
                    }
                ),
                encoding="utf-8",
            )
            record = harness.classify_child("m1", 0, str(path), "")

        self.assertEqual(record["classification"], "PASS")
        self.assertEqual(record["status"], "PASS")
        self.assertTrue(record["child_json_valid"])


if __name__ == "__main__":
    unittest.main()
