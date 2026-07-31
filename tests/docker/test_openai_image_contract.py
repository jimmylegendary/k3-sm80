from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.openai"
PATCH = REPO_ROOT / "docker" / "patches" / "kda-sm80-int64.patch"
VERIFY = REPO_ROOT / "docker" / "verify_image.py"


def load_verify_module():
    spec = importlib.util.spec_from_file_location("k3_verify_image", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenAIImageContractTest(unittest.TestCase):
    def test_openai_image_is_pinned_and_patched(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("0498dc7ea9204f1863b46515816f5d8d458e01e7", dockerfile)
        self.assertIn("sha256:d5fc27444ec6a9f52ac18a81acd88394f1840ce747c3eb989a5b4b0316079752", dockerfile)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.0", dockerfile)
        self.assertIn("libcusparse-dev-13-0", dockerfile)
        self.assertIn("libcusolver-dev-13-0", dockerfile)
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

    def test_verify_requires_all_lane_cli_flags(self) -> None:
        module = load_verify_module()
        required = set(module.REQUIRED_SERVE_FLAGS)

        self.assertTrue(
            {
                "--headless",
                "--data-parallel-backend",
                "--distributed-executor-backend",
                "--nnodes",
                "--node-rank",
                "--kda-prefill-backend",
                "--moe-backend",
                "--kv-cache-dtype-skip-layers",
            }.issubset(required)
        )
        self.assertEqual(module.required_cli_errors("\n".join(required)), [])
        self.assertEqual(
            module.required_cli_errors("\n".join(required - {"--headless"})),
            ["vllm serve --help missing flag: --headless"],
        )

    def test_verify_cli_probe_is_cpu_host_safe(self) -> None:
        module = load_verify_module()

        self.assertIn("CpuPlatform", module.CLI_HELP_PROBE)
        self.assertIn(
            'sys.argv = ["vllm", "serve", "--help=all"]',
            module.CLI_HELP_PROBE,
        )
        self.assertIn("main()", module.CLI_HELP_PROBE)


if __name__ == "__main__":
    unittest.main()
