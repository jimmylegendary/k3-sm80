from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH = REPO_ROOT / "docker" / "patches" / "mla-fp8-kv-sm80.patch"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.openai"
VERIFY = REPO_ROOT / "docker" / "verify_image.py"


class Fp8PatchContractTest(unittest.TestCase):
    def test_fp8_patch_is_k3_sm80_opt_in(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")

        for token in (
            "VLLM_K3_SM80_FP8_KV",
            "uint8",
            "Kimi-K3",
            "k3_sm80_software_fp8",
            "software_fp8",
            "e4m3_bits_to_fp32",
            "software_fp8_cache_insert",
            "software_gather_fp8_cache",
        ):
            self.assertIn(token, patch)
        self.assertNotIn(
            '-    if self.kv_cache_dtype.startswith("fp8")',
            patch,
        )
        self.assertIn(
            "current_platform.has_device_capability(80)",
            patch,
        )
        self.assertIn("native_fp8_arithmetic=False", patch)

    def test_patch_preserves_bf16_queries_and_byte_storage(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn("mqa_q = torch.cat", patch)
        self.assertIn("preserves BF16 prefill queries", patch)
        self.assertIn("assert not fp8_prefill", patch)
        self.assertIn("cache.view(torch.uint8)", patch)
        self.assertIn("value = e4m3_bits_to_fp32(bits)", patch)
        self.assertNotIn("supports_fp8(cls)", added)
        self.assertNotIn("torch.float8_e4m3fn)", added)

    def test_image_applies_python_overlay_after_cuda_build(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        verify = VERIFY.read_text(encoding="utf-8")

        self.assertIn("mla-fp8-kv-sm80.patch", dockerfile)
        self.assertIn("AS python-overlay", dockerfile)
        self.assertIn("COPY --from=python-overlay /opt/venv /opt/venv", dockerfile)
        self.assertIn("mla-fp8-kv-sm80-v1", dockerfile)
        self.assertIn("mla-fp8-kv-sm80-v1", verify)
        self.assertIn("VLLM_K3_SM80_FP8_KV", verify)


if __name__ == "__main__":
    unittest.main()
