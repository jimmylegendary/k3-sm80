#!/usr/bin/env bash
# k3-sm80 kernel-kit 이미지 배포 — 2경로 (PLAN.md §S2 "반입 2경로") · 멱등
#
# 사용:
#   bash docker/push.sh                        # ① ghcr push (gh auth token 으로 login)
#   bash docker/push.sh --release              # ② GitHub Release 폴백: save|zstd|split → upload
#   TIERS="triton" bash docker/push.sh         # 일부 티어만
#   RELEASE_TAG=kernelkit-20260729 FORCE=1 bash docker/push.sh --release   # 재업로드 강제
#
# 멱등성: docker push 재실행 무해 / release 는 gh release view 로 존재 확인 후 생성,
#   동명 자산은 FORCE=1 + --clobber 로만 교체(기본은 skip).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit"
REPO_SLUG="jimmylegendary/k3-sm80"
TIERS="${TIERS:-triton full}"
DATE_TAG="$(date +%Y%m%d)"
MODE="ghcr"
for a in "$@"; do
  case "$a" in
    --release) MODE="release" ;;
    *) echo "[push] 알 수 없는 인자: $a" >&2; exit 3 ;;
  esac
done

command -v docker >/dev/null || { echo "[push] docker 없음" >&2; exit 3; }
command -v gh >/dev/null || { echo "[push] gh CLI 없음 (ghcr login/release 에 필요)" >&2; exit 3; }
gh auth status >/dev/null 2>&1 || { echo "[push] gh 미인증 — gh auth login 먼저" >&2; exit 3; }

for tier in $TIERS; do
  docker image inspect "$IMG:$tier" >/dev/null 2>&1 \
    || { echo "[push] $IMG:$tier 로컬에 없음 — 먼저 bash docker/build.sh" >&2; exit 3; }
done

# ---------------- ① ghcr push ----------------
if [[ "$MODE" == "ghcr" ]]; then
  GH_USER="$(gh api user -q .login 2>/dev/null || echo jimmylegendary)"
  echo "[push] ghcr login as $GH_USER"
  gh auth token | docker login ghcr.io -u "$GH_USER" --password-stdin
  for tier in $TIERS; do
    docker push "$IMG:$tier"
    # 날짜태그: 오늘 빌드분이 있으면 함께 push (없으면 안내만)
    if docker image inspect "$IMG:$tier-$DATE_TAG" >/dev/null 2>&1; then
      docker push "$IMG:$tier-$DATE_TAG"
    else
      echo "[push] 참고: $IMG:$tier-$DATE_TAG 없음 (오늘 빌드 아님) — :$tier 만 push"
    fi
  done
  echo "[push] ghcr 완료. 사내 반입 절차: docker/PULL.md §1"
  exit 0
fi

# ---------------- ② GitHub Release 폴백 (사내 registry 차단 대비) ----------------
# docker save | zstd | split -b 1900M (GitHub 자산 2GiB 한도 밑) → gh release upload
RELEASE_TAG="${RELEASE_TAG:-kernelkit-$DATE_TAG}"
WORK="${TMPDIR:-/tmp}/k3-sm80-push"
mkdir -p "$WORK"
command -v zstd >/dev/null || { echo "[push] zstd 없음 (apt install zstd)" >&2; exit 3; }

# 디스크 가드: 압축 이미지(최대 ~10GB/티어)를 임시로 둘 공간 (-P: 장치명이 길어도 1행 보장)
avail_gb=$(( $(df -Pk "$WORK" | awk 'NR==2{print $4}') / 1024 / 1024 ))
(( avail_gb < 15 )) && { echo "[push] $WORK 여유 ${avail_gb}GB < 15GB — 공간 확보 후 재시도" >&2; exit 3; }

if ! gh release view "$RELEASE_TAG" -R "$REPO_SLUG" >/dev/null 2>&1; then
  gh release create "$RELEASE_TAG" -R "$REPO_SLUG" \
    --title "kernel-kit docker images $DATE_TAG" \
    --notes "PLAN §S2 반입 경로②: docker save 분할 이미지. 결합/적재 절차는 docker/PULL.md §2. 이미지: $IMG:{triton,full}"
  echo "[push] release $RELEASE_TAG 생성"
else
  echo "[push] release $RELEASE_TAG 존재 — 자산만 갱신"
fi

existing_assets="$(gh release view "$RELEASE_TAG" -R "$REPO_SLUG" --json assets -q '.assets[].name' || true)"

for tier in $TIERS; do
  base="k3-sm80-kernelkit-$tier-$DATE_TAG.tar.zst"
  if grep -Fxq "$base.part-aa" <<<"$existing_assets" && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[push] $tier: $base.part-* 이미 업로드됨 — skip (재업로드는 FORCE=1)"
    continue
  fi
  # 스테일 자산 제거(릴리스당 tier 1세트 불변식): 재업로드에서 part 수가 줄거나
  # 날짜가 달라지면 이전 part 가 남아 PULL.md 의 'cat *.part-*' 결합을 손상시킨다 —
  # 같은 tier prefix 의 기존 자산(날짜 불문, part/sha256 전부)을 업로드 전에 지운다.
  while IFS= read -r stale; do
    [[ -z "$stale" ]] && continue
    echo "[push]   스테일 자산 제거: $stale"
    gh release delete-asset "$RELEASE_TAG" "$stale" -R "$REPO_SLUG" -y
  done < <(grep -E "^k3-sm80-kernelkit-$tier-" <<<"$existing_assets" || true)
  echo "[push] ── $tier: save|zstd|split (스트리밍, 중간 tar 미생성)"
  rm -f "$WORK/$base.part-"* "$WORK/$base.sha256"
  docker save "$IMG:$tier" | zstd -T0 -8 | split -b 1900M - "$WORK/$base.part-"
  ( cd "$WORK" && sha256sum "$base.part-"* > "$base.sha256" )
  ls -lh "$WORK/$base.part-"* | awk '{print "[push]   " $9 "  " $5}'
  gh release upload "$RELEASE_TAG" "$WORK/$base.part-"* "$WORK/$base.sha256" \
    -R "$REPO_SLUG" --clobber
  rm -f "$WORK/$base.part-"* "$WORK/$base.sha256"
  echo "[push] $tier 업로드 완료"
done

echo "[push] release 경로 완료: https://github.com/$REPO_SLUG/releases/tag/$RELEASE_TAG"
echo "[push] 사내 반입 절차: docker/PULL.md §2"
