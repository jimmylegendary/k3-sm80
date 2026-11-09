#!/usr/bin/env bash
# K3.1 OpenAI 서빙 이미지를 GitHub Release 용으로 패키징(docker save | zstd | split 1900m)하고 선택적으로 업로드한다.
# K3 의 docker/release-openai.sh 와 별개다(태그·경로·환경변수 모두 K31_ 접두, 그 파일은 그대로).
#
# 사용:
#   bash docker/release-k31.sh                          # 패키징만 (docker/releases/openai-k31-a3c14efb-sm80-v1/)
#   K31_OPENAI_GITHUB_UPLOAD=1 bash docker/release-k31.sh   # 패키징 + gh release create/upload
#
# 환경변수: K31_OPENAI_IMAGE · K31_OPENAI_RELEASE_DIR · K31_OPENAI_GITHUB_UPLOAD(0|1) ·
#           K31_OPENAI_RELEASE_TAG · K31_OPENAI_GITHUB_REPO
# Release 업로드는 이미지의 CPU 계약만 의미한다 — A100 kernel/service PASS 가 아니다.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${K31_OPENAI_IMAGE:-k31-sm80-openai:a3c14efb-sm80-v1}"
asset="k31-sm80-openai-a3c14efb-sm80-v1"
output_dir="${K31_OPENAI_RELEASE_DIR:-$script_dir/releases/openai-k31-a3c14efb-sm80-v1}"
prefix="$output_dir/$asset.tar.zst.part-"
upload="${K31_OPENAI_GITHUB_UPLOAD:-0}"
release_tag="${K31_OPENAI_RELEASE_TAG:-openai-k31-a3c14efb-sm80-v1}"
github_repo="${K31_OPENAI_GITHUB_REPO:-jimmylegendary/k3-sm80}"

command -v docker >/dev/null || { printf 'docker missing\n' >&2; exit 2; }
command -v zstd >/dev/null || { printf 'zstd missing\n' >&2; exit 2; }
command -v split >/dev/null || { printf 'split missing\n' >&2; exit 2; }
[[ "$upload" == "0" || "$upload" == "1" ]] || {
  printf 'K31-OPENAI-RELEASE-FAIL K31_OPENAI_GITHUB_UPLOAD must be 0 or 1\n' >&2
  exit 2
}
docker image inspect "$image" >/dev/null
mkdir -p "$output_dir"

printf 'K31-OPENAI-RELEASE image=%s output=%s\n' "$image" "$output_dir"
docker save "$image" | zstd -T0 -10 | split -b 1900m - "$prefix"
(
  cd "$output_dir"
  sha256sum "$asset".tar.zst.part-* \
    > SHA256SUMS
)
docker image inspect "$image" > "$output_dir/image-inspect.json"
if [[ "$upload" == "1" ]]; then
  command -v gh >/dev/null || {
    printf 'K31-OPENAI-RELEASE-FAIL gh missing\n' >&2
    exit 2
  }
  if ! gh release view "$release_tag" -R "$github_repo" >/dev/null 2>&1; then
    gh release create "$release_tag" -R "$github_repo" \
      --title "Kimi-K3.1 SM80 OpenAI image v1" \
      --notes "Pinned vLLM a3c14efb; Kimi-K3.1-Pro SM80 patch set (marlin rebase + TRITON_MLA_SPARSE + DSA glue); A100 GPU gates remain INCOMPLETE until run in-cluster."
  fi
  gh release upload "$release_tag" -R "$github_repo" --clobber \
    "$output_dir/$asset".tar.zst.part-* \
    "$output_dir/SHA256SUMS" \
    "$output_dir/image-inspect.json"
  printf 'K31-OPENAI-RELEASE-UPLOAD-PASS tag=%s repo=%s\n' \
    "$release_tag" "$github_repo"
fi
printf '%s\n' \
  'Load:' \
  "  cd $output_dir" \
  '  sha256sum -c SHA256SUMS' \
  "  cat $asset.tar.zst.part-* | zstd -d | docker load" \
  '(사내 노드: bash run/11-load-k31-image.sh — 멱등, part 오염 검사 포함)'
