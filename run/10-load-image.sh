#!/usr/bin/env bash
# GitHub Release 에서 도커 이미지를 받아 load 한다 (ghcr 차단 환경용 폴백 경로).
#
# 사용:   bash run/10-load-image.sh [triton|full] [release-tag]
#         tier 생략 = triton. release-tag 생략 = docker/RELEASE-TAG.txt (repo 가 최신을 앎).
# 결과:   docker images 에 ghcr.io/jimmylegendary/k3-sm80-kernelkit:<tier> 등록.
#         이미 load 돼 있으면 아무것도 안 하고 성공(멱등).
#         exit 0 = load 완료 / 3 = 수동 조치 필요(안내 출력)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-triton}"
[[ $TIER == triton || $TIER == full ]] || { echo "[load] tier 는 triton|full" >&2; exit 3; }
TAG="${2:-$(cat "$ROOT/docker/RELEASE-TAG.txt" 2>/dev/null || echo kernelkit-20260729)}"
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit:$TIER"

if docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "[load] $IMG 이미 load 됨 — skip (강제 재로드: docker rmi $IMG 후 재실행)"
  exit 0
fi
command -v unzstd >/dev/null || { echo "[load] zstd 미설치 — apt install zstd" >&2; exit 3; }

DL="$ROOT/.dl-$TAG-$TIER"          # 티어·태그 전용 디렉터리 = part 결합 오염 방지(PULL.md §2 주의)
mkdir -p "$DL"; cd "$DL"
if ls k3-sm80-kernelkit-"$TIER"-*.part-* >/dev/null 2>&1; then
  echo "[load] 기존 다운로드 재사용: $DL"
elif command -v gh >/dev/null; then
  echo "[load] release $TAG 에서 $TIER 티어 다운로드 (수 GB)"
  gh release download "$TAG" -R jimmylegendary/k3-sm80 -p "k3-sm80-kernelkit-$TIER-*"
else
  echo "[load] gh 없음 — 브라우저에서 아래 URL 의 k3-sm80-kernelkit-$TIER-* 파일 전부를"
  echo "       $DL/ 에 받은 뒤(파일명 유지) 이 스크립트를 재실행:"
  echo "       https://github.com/jimmylegendary/k3-sm80/releases/tag/$TAG"
  exit 3
fi

sha256sum -c k3-sm80-kernelkit-"$TIER"-*.sha256
cat k3-sm80-kernelkit-"$TIER"-*.tar.zst.part-* | unzstd | docker load
echo "[load] 완료: $IMG"
echo "[load] 다운로드 캐시 $DL 는 재사용용 — 공간 회수는 rm -rf '$DL'"
echo "[load] 다음: bash run/20-kernel-test.sh $TIER"
