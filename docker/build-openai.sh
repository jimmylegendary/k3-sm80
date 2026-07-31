#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
image="${K3_OPENAI_IMAGE:-k3-sm80-openai:0498dc7e-sm80}"
base_image="${K3_OPENAI_BASE_IMAGE:-ghcr.io/jimmylegendary/k3-sm80-kernelkit@sha256:d5fc27444ec6a9f52ac18a81acd88394f1840ce747c3eb989a5b4b0316079752}"

command -v docker >/dev/null || {
  printf 'K3-OPENAI-BUILD-FAIL docker missing\n' >&2
  exit 2
}
if ! docker image inspect "$base_image" >/dev/null 2>&1; then
  printf 'K3-OPENAI-BUILD-FAIL base image missing: %s\n' "$base_image" >&2
  printf 'Load the kernel-kit full release first; GHCR access is not required after load.\n' >&2
  exit 2
fi

printf 'K3-OPENAI-BUILD image=%s base=%s\n' "$image" "$base_image"
DOCKER_BUILDKIT=1 docker build \
  --target contract \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "VLLM_COMMIT=0498dc7ea9204f1863b46515816f5d8d458e01e7" \
  -f "$script_dir/Dockerfile.openai" \
  -t "$image" \
  "$repo_dir"

python3 "$script_dir/verify_image.py" "$image"
