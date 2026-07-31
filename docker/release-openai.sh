#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${K3_OPENAI_IMAGE:-k3-sm80-openai:0498dc7e-sm80}"
output_dir="${K3_OPENAI_RELEASE_DIR:-$script_dir/releases/openai-0498dc7e-sm80}"
prefix="$output_dir/k3-sm80-openai-0498dc7e-sm80.tar.zst.part-"

command -v docker >/dev/null || { printf 'docker missing\n' >&2; exit 2; }
command -v zstd >/dev/null || { printf 'zstd missing\n' >&2; exit 2; }
command -v split >/dev/null || { printf 'split missing\n' >&2; exit 2; }
docker image inspect "$image" >/dev/null
mkdir -p "$output_dir"

printf 'K3-OPENAI-RELEASE image=%s output=%s\n' "$image" "$output_dir"
docker save "$image" | zstd -T0 -10 | split -b 1900m - "$prefix"
(
  cd "$output_dir"
  sha256sum k3-sm80-openai-0498dc7e-sm80.tar.zst.part-* \
    > SHA256SUMS
)
docker image inspect "$image" > "$output_dir/image-inspect.json"
printf '%s\n' \
  'Load:' \
  "  cd $output_dir" \
  '  sha256sum -c SHA256SUMS' \
  '  cat k3-sm80-openai-0498dc7e-sm80.tar.zst.part-* | zstd -d | docker load'
