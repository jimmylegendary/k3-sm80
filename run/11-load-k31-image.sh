#!/usr/bin/env bash
# GitHub Release 에서 K3.1(Kimi-K3.1-Pro) OpenAI 서빙 이미지를 받아 load 한다 (ghcr 차단 환경용 폴백 경로).
# K3 의 run/10-load-image.sh 와 별개다 — 이미지명·Release 태그·다운로드 디렉터리가 모두 다르다(그 파일은 그대로).
#
# 사용:   bash run/11-load-k31-image.sh [release-tag]
#         release-tag 생략 = docker/RELEASE-TAG-K31.txt (repo 가 최신을 앎).
# 결과:   docker images 에 k31-sm80-openai:a3c14efb-sm80-v1 등록.
#         이미 load 돼 있으면 아무것도 안 하고 성공(멱등).
#         exit 0 = load 완료 또는 이미 있음 / 3 = 수동 조치 필요(안내 출력)
# 환경변수: K31_OPENAI_IMAGE(기본 k31-sm80-openai:a3c14efb-sm80-v1) · K31_OPENAI_GITHUB_REPO(기본 jimmylegendary/k3-sm80)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(cat "$ROOT/docker/RELEASE-TAG-K31.txt" 2>/dev/null || echo openai-k31-a3c14efb-sm80-v1)}"
IMG="${K31_OPENAI_IMAGE:-k31-sm80-openai:a3c14efb-sm80-v1}"
ASSET="k31-sm80-openai-a3c14efb-sm80-v1"
REPO="${K31_OPENAI_GITHUB_REPO:-jimmylegendary/k3-sm80}"

show_image() { # 이미지 ID 와 패치셋 라벨을 보여준다 (네 노드의 ID 가 같아야 한다 — lane preflight 가 다시 검증)
  docker image inspect "$IMG" --format '[load] id={{.Id}} patchset={{index .Config.Labels "io.k3.patchset"}}'
}

if docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "[load] $IMG 이미 load 됨 — skip (강제 재로드: docker rmi $IMG 후 재실행)"
  show_image
  exit 0
fi
command -v zstd >/dev/null || { echo "[load] zstd 미설치 — apt install zstd" >&2; exit 3; }

DL="$ROOT/.dl-k31-$TAG"             # K3.1 전용 다운로드 디렉터리 = part 결합 오염 방지(PULL-K31.md 주의)
mkdir -p "$DL"; cd "$DL"
if ls "$ASSET".tar.zst.part-* >/dev/null 2>&1 && [[ -f SHA256SUMS ]]; then
  echo "[load] 기존 다운로드 재사용: $DL"
elif command -v gh >/dev/null; then
  echo "[load] release $TAG 에서 K3.1 openai 이미지 다운로드 (수 GB)"
  gh release download "$TAG" -R "$REPO" --clobber \
    -p "$ASSET.tar.zst.part-*" -p SHA256SUMS -p image-inspect.json
else
  echo "[load] gh 없음 — 브라우저에서 아래 URL 의 $ASSET.tar.zst.part-* · SHA256SUMS · image-inspect.json 전부를"
  echo "       $DL/ 에 받은 뒤(파일명 유지) 이 스크립트를 재실행:"
  echo "       https://github.com/$REPO/releases/tag/$TAG"
  exit 3
fi

# 결합 오염 방지: `cat *.part-*` 는 글롭에 걸리는 모든 파일을 이어 붙인다 — SHA256SUMS 의 목록과 정확히 같아야 한다.
mapfile -t listed < <(awk '{print $2}' SHA256SUMS | sort)
mapfile -t present < <(ls -1 "$ASSET".tar.zst.part-* | sort)
if [[ "${listed[*]}" != "${present[*]}" ]]; then
  echo "[load] FAIL: 디렉터리의 part 목록이 SHA256SUMS 와 다르다 (이전 다운로드 잔재?)" >&2
  echo "  SHA256SUMS : ${listed[*]}" >&2
  echo "  디렉터리   : ${present[*]}" >&2
  echo "  조치: rm -rf '$DL' 후 재실행" >&2
  exit 3
fi
sha256sum -c SHA256SUMS
cat "$ASSET".tar.zst.part-* | zstd -d | docker load
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "[load] FAIL: load 후에도 $IMG 가 없다 — docker image ls 로 태그 확인" >&2
  exit 3
fi
echo "[load] 완료: $IMG"
show_image
if [[ -f image-inspect.json ]] && command -v python3 >/dev/null; then
  want="$(python3 -c 'import json,sys; print(json.load(open("image-inspect.json"))[0]["Id"])' 2>/dev/null || true)"
  got="$(docker image inspect "$IMG" --format '{{.Id}}')"
  [[ -z "$want" || "$want" == "$got" ]] \
    || echo "[load] 주의: image ID 가 릴리스의 image-inspect.json($want)과 다르다 — 다른 docker 스토리지 백엔드면 정상일 수 있다. 네 노드끼리의 ID 일치가 중요하다." >&2
fi
echo "[load] 다운로드 캐시 $DL 는 재사용용 — 공간 회수는 rm -rf '$DL'"
echo "[load] 다음: python3 docker/verify_k31_image.py --gpu  →  bash run/21-k31-kernel-test.sh"
