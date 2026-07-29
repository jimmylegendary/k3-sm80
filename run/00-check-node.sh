#!/usr/bin/env bash
# 사내 A100 노드 사전점검 — 이 노드에서 킷이 돌 수 있는지 판정만 한다 (아무것도 설치 안 함).
#
# 사용:   bash run/00-check-node.sh
# 결과:   항목별 PASS/FAIL 표 + 종합 판정.
#         exit 0 = 진행 가능 / 1 = FAIL 있음(각 줄의 조치 안내 참조)
set -uo pipefail
FAIL=0
p() { printf '  %-4s %-22s %s\n' "$1" "$2" "$3"; [[ $1 == FAIL ]] && FAIL=1; return 0; }

echo "[check-node] 사내 노드 사전점검"
command -v docker >/dev/null && p PASS docker "$(docker --version)" \
  || p FAIL docker "docker 미설치 — 관리자에게 요청 (이미지 방식의 유일한 노드 전제)"
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q . \
  && p PASS gpu "$(nvidia-smi -L | head -1)" \
  || p FAIL gpu "GPU/드라이버 안 보임 — nvidia-smi 확인"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
[[ -n ${drv:-} ]] && p INFO driver "$drv (triton 티어 r560+ 권장 / full 티어 r580+ 필요 — docker/PULL.md 상단)"
command -v unzstd >/dev/null && p PASS zstd "$(zstd --version 2>/dev/null | head -c 40)" \
  || p FAIL zstd "zstd 미설치 — apt install zstd (release tar 해제에 필요)"
command -v gh >/dev/null && p PASS gh "$(gh --version | head -1)" \
  || p INFO gh "gh 없음 — 10-load-image.sh 가 웹 다운로드 경로를 안내함(치명 아님)"

# GPU 패스스루(nvidia-container-toolkit): load 된 킷 이미지가 있으면 실측, 없으면 보류
img=$(docker images -q ghcr.io/jimmylegendary/k3-sm80-kernelkit 2>/dev/null | head -1)
if [[ -n $img ]]; then
  docker run --rm --gpus all "$img" nvidia-smi -L >/dev/null 2>&1 \
    && p PASS gpus-passthrough "--gpus all 동작" \
    || p FAIL gpus-passthrough "nvidia-container-toolkit 미구성 — 관리자에게 요청"
else
  p INFO gpus-passthrough "킷 이미지 load 후 재실행하면 실측됨 (10-load-image.sh 다음에)"
fi

if [[ $FAIL -eq 1 ]]; then echo "[check-node] 종합: FAIL 있음 — 위 조치 후 재실행"; exit 1; fi
echo "[check-node] 종합: 진행 가능 → 다음: bash run/10-load-image.sh triton"
