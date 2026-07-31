from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.openai"
PATCH = REPO_ROOT / "docker" / "patches" / "kda-sm80-int64.patch"


class OpenAIImageContractTest(unittest.TestCase):
    def test_openai_image_is_pinned_and_patched(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("0498dc7ea9204f1863b46515816f5d8d458e01e7", dockerfile)
        self.assertIn("sha256:d5fc27444ec6a9f52ac18a81acd88394f1840ce747c3eb989a5b4b0316079752", dockerfile)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.0", dockerfile)
        self.assertIn("kda-sm80-int64.patch", dockerfile)
        self.assertIn("mla-fp8-kv-sm80-dspark-bf16.patch", dockerfile)
        self.assertIn('ENTRYPOINT ["vllm", "serve"]', dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)

    def test_image_has_immutable_provenance_labels(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn("io.k3.cuda_arch", dockerfile)
        self.assertIn("io.k3.patchset", dockerfile)
        self.assertIn("io.k3.base_digest", dockerfile)

    def test_patch_targets_installed_vllm_sources(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")

        self.assertIn(
            "vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py",
            patch,
        )
        self.assertIn(
            "vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py",
            patch,
        )
        self.assertGreaterEqual(patch.count(".to(tl.int64)"), 6)
        self.assertNotIn("kernel-kit/vendor", patch)

    def test_build_release_and_verify_clis_are_syntax_valid(self) -> None:
        scripts = (
            REPO_ROOT / "docker" / "build-openai.sh",
            REPO_ROOT / "docker" / "release-openai.sh",
        )
        result = subprocess.run(
            ["bash", "-n", *(str(path) for path in scripts)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        verify = subprocess.run(
            ["python3", str(REPO_ROOT / "docker" / "verify_image.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("IMAGE", verify.stdout)


if __name__ == "__main__":
    unittest.main()
