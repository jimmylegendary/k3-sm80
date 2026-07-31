#!/usr/bin/env python3
"""Verify immutable K3 OpenAI image metadata and CPU/GPU contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


COMMIT = "0498dc7ea9204f1863b46515816f5d8d458e01e7"
BASE_DIGEST = "sha256:d5fc27444ec6a9f52ac18a81acd88394f1840ce747c3eb989a5b4b0316079752"
EXPECTED_LABELS = {
    "org.opencontainers.image.revision": COMMIT,
    "io.k3.cuda_arch": "sm80",
    "io.k3.patchset": (
        "kda-sm80-int64+mxfp4-marlin-moe-sm80-v2+"
        "mla-fp8-kv-sm80-v1+dspark-bf16-v1"
    ),
    "io.k3.base_digest": BASE_DIGEST,
}


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, text=True, capture_output=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def inspect_image(image: str) -> tuple[dict[str, object] | None, list[str]]:
    result = run(["docker", "image", "inspect", image])
    if result.returncode:
        return None, [f"docker image inspect rc={result.returncode}: {result.stderr.strip()}"]
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, [f"docker inspect returned invalid JSON: {exc}"]
    if not isinstance(values, list) or len(values) != 1:
        return None, ["docker inspect must return exactly one image"]
    return values[0], []


def metadata_errors(metadata: dict[str, object]) -> list[str]:
    errors: list[str] = []
    config = metadata.get("Config")
    if not isinstance(config, dict):
        return ["image Config missing"]
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        labels = {}
    for key, expected in EXPECTED_LABELS.items():
        if labels.get(key) != expected:
            errors.append(f"label {key} mismatch")
    if config.get("Entrypoint") != ["vllm", "serve"]:
        errors.append("entrypoint mismatch")
    if not config.get("Healthcheck"):
        errors.append("healthcheck missing")
    return errors


CPU_PROBE = r"""
import importlib, pathlib, torch, vllm
from vllm import _custom_ops
expected = "0498dc7ea9204f1863b46515816f5d8d458e01e7"
assert pathlib.Path("/opt/vllm-build/VLLM_COMMIT.txt").read_text().strip() == expected
for module in (
    "vllm.entrypoints.openai.api_server",
    "vllm.tool_parsers.kimi_k3_tool_parser",
    "vllm.reasoning.kimi_k3_reasoning_parser",
):
    importlib.import_module(module)
root = pathlib.Path(vllm.__file__).parent
chunk = (root / "models/kimi_k3/nvidia/ops/third_party/kda/chunk.py").read_text()
delta = (root / "third_party/flash_linear_attention/ops/chunk_delta_h.py").read_text()
fp8_kv = (root / "models/kimi_k3/nvidia/ops/sm80_fp8_kv.py").read_text()
mla = (root / "models/kimi_k3/nvidia/mla.py").read_text()
envs_source = (root / "envs.py").read_text()
assert "(i_tg * H + i_h).to(tl.int64) * K * V" in chunk
assert "(boh * H + i_h).to(tl.int64) * V * K" in delta
assert 'PATCH_ID = "mla-fp8-kv-sm80-v1"' in fp8_kv
assert "native_fp8_arithmetic = False" in fp8_kv
assert "cache_config.kv_cache_dtype_skip_layers" in mla
assert 'self.kv_cache_dtype in ("fp8", "fp8_e4m3")' in mla
assert "VLLM_K3_SM80_FP8_KV" in envs_source
assert "sm_80" in torch.cuda.get_arch_list()
assert pathlib.Path("/opt/k3-patches/mxfp4-marlin-moe-sm80.patch").is_file()
assert pathlib.Path("/opt/k3-patches/mla-fp8-kv-sm80.patch").is_file()
assert pathlib.Path(
    "/opt/k3-patches/mla-fp8-kv-sm80-dspark-bf16.patch"
).is_file()
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
print("CPU_CONTRACT_PASS")
"""

GPU_PROBE = r"""
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
for index in range(8):
    name = torch.cuda.get_device_name(index)
    cap = torch.cuda.get_device_capability(index)
    assert "A100" in name, (index, name)
    assert cap == (8, 0), (index, cap)
print("GPU_CONTRACT_PASS devices=8 arch=sm80")
"""


def container_probe(image: str, *, gpu: bool) -> list[str]:
    argv = ["docker", "run", "--rm"]
    if gpu:
        argv.extend(["--gpus", "all"])
    argv.extend(["--entrypoint", "python3", image, "-c", GPU_PROBE if gpu else CPU_PROBE])
    result = run(argv)
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        return [f"{'GPU' if gpu else 'CPU'} probe rc={result.returncode}: {' | '.join(tail)}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", metavar="IMAGE")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also require exactly 8 A100 SM80 devices inside the container",
    )
    args = parser.parse_args()

    metadata, errors = inspect_image(args.image)
    if metadata is not None:
        errors.extend(metadata_errors(metadata))
        errors.extend(container_probe(args.image, gpu=False))
        if args.gpu:
            errors.extend(container_probe(args.image, gpu=True))
            help_result = run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "--entrypoint",
                    "vllm",
                    args.image,
                    "serve",
                    "--help",
                ]
            )
            if help_result.returncode:
                errors.append(f"vllm serve --help rc={help_result.returncode}")

    for error in errors:
        print(f"K3-OPENAI-VERIFY-FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    image_id = metadata.get("Id", "unknown") if metadata else "unknown"
    mode = "gpu" if args.gpu else "cpu"
    print(f"K3-OPENAI-VERIFY-PASS image={args.image} id={image_id} mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
