#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${K3_OPENAI_IMAGE:-k3-sm80-openai:0498dc7e-sm80-v3}"
output_dir="${K3_OPENAI_RELEASE_DIR:-$script_dir/releases/openai-0498dc7e-sm80-v3}"
prefix="$output_dir/k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-"
upload="${K3_OPENAI_GITHUB_UPLOAD:-0}"
release_tag="${K3_OPENAI_RELEASE_TAG:-openai-0498dc7e-sm80-v3}"
github_repo="${K3_OPENAI_GITHUB_REPO:-jimmylegendary/k3-sm80}"

command -v docker >/dev/null || { printf 'docker missing\n' >&2; exit 2; }
command -v zstd >/dev/null || { printf 'zstd missing\n' >&2; exit 2; }
command -v split >/dev/null || { printf 'split missing\n' >&2; exit 2; }
[[ "$upload" == "0" || "$upload" == "1" ]] || {
  printf 'K3-OPENAI-RELEASE-FAIL K3_OPENAI_GITHUB_UPLOAD must be 0 or 1\n' >&2
  exit 2
}
docker image inspect "$image" >/dev/null
mkdir -p "$output_dir"

printf 'K3-OPENAI-RELEASE image=%s output=%s\n' "$image" "$output_dir"
docker save "$image" | zstd -T0 -10 | split -b 1900m - "$prefix"
(
  cd "$output_dir"
  sha256sum k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-* \
    > SHA256SUMS
)
docker image inspect "$image" > "$output_dir/image-inspect.json"
if [[ "$upload" == "1" ]]; then
  command -v gh >/dev/null || {
    printf 'K3-OPENAI-RELEASE-FAIL gh missing\n' >&2
    exit 2
  }
  if ! gh release view "$release_tag" -R "$github_repo" >/dev/null 2>&1; then
    gh release create "$release_tag" -R "$github_repo" \
      --title "Kimi-K3 SM80 OpenAI image v3" \
      --notes "Pinned vLLM 0498dc7e; original MXFP4; A100 GPU gates remain INCOMPLETE until run in-cluster."
  fi
  gh release upload "$release_tag" -R "$github_repo" --clobber \
    "$output_dir"/k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-* \
    "$output_dir/SHA256SUMS" \
    "$output_dir/image-inspect.json"
  printf 'K3-OPENAI-RELEASE-UPLOAD-PASS tag=%s repo=%s\n' \
    "$release_tag" "$github_repo"
fi
printf '%s\n' \
  'Load:' \
  "  cd $output_dir" \
  '  sha256sum -c SHA256SUMS' \
  '  cat k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-* | zstd -d | docker load'
