#!/usr/bin/env bash
# k3-sm80 kernel-kit 도커 2티어 로컬 빌드 (PLAN.md §S2 전달 모델)
#
# 사용:
#   bash docker/build.sh                 # triton + full 순차 빌드 + triton 컨테이너 gate
#   bash docker/build.sh --triton-only   # triton 티어만 (분 단위)
#   bash docker/build.sh --full-only     # full 티어만 (vLLM SM80 소스빌드 — 수 시간)
#   bash docker/build.sh --skip-gate     # 빌드만 (컨테이너 gate 생략)
#   MAX_JOBS=4 bash docker/build.sh      # 병렬 수동 지정 (기본: RAM 가드가 산출, 상한 8)
#
# GPU 불필요 — full 티어의 nvcc 도 크로스컴파일이다 (Dockerfile.full 주석 참조).
# exit: 0 = 요청한 빌드·gate 전부 성공 / 그 외 = 실패 지점의 로그 파일 확인
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$DIR")"
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit"
DATE_TAG="$(date +%Y%m%d)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

BUILD_TRITON=1
BUILD_FULL=1
RUN_GATE=1
for a in "$@"; do
  case "$a" in
    --triton-only) BUILD_FULL=0 ;;
    --full-only)   BUILD_TRITON=0 ;;
    --skip-gate)   RUN_GATE=0 ;;
    *) echo "[build] 알 수 없는 인자: $a" >&2; exit 3 ;;
  esac
done

command -v docker >/dev/null || { echo "[build] docker 없음" >&2; exit 3; }
export DOCKER_BUILDKIT=1

# ---- RAM 가드: nvcc 잡당 ~2GB(RUNBOOK §6) + 시스템 여유 4GB, 상한 8 (호스트 계약) ----
if [[ -z "${MAX_JOBS:-}" ]]; then
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  MAX_JOBS=$(( (avail_gb - 4) / 2 ))
  (( MAX_JOBS > 8 )) && MAX_JOBS=8
  (( MAX_JOBS < 2 )) && MAX_JOBS=2
  echo "[build] RAM 가드: MemAvailable=${avail_gb}GB → MAX_JOBS=$MAX_JOBS (수동: MAX_JOBS=n bash $0)"
else
  echo "[build] MAX_JOBS=$MAX_JOBS (수동 지정)"
fi

build_one() { # tier
  local tier="$1" log="$LOG_DIR/build-$1-$(date +%Y%m%d-%H%M%S).log"
  echo "[build] ── $tier 티어 빌드 시작 (log: $log)"
  local t0=$SECONDS
  local args=( -f "$DIR/Dockerfile.$tier" -t "$IMG:$tier" -t "$IMG:$tier-$DATE_TAG" )
  [[ "$tier" == "full" ]] && args+=( --build-arg "MAX_JOBS=$MAX_JOBS" )
  # if-조건으로 감싸 pipefail+set -e 즉사 대신 안내를 남긴다
  if docker build "${args[@]}" "$DIR" 2>&1 | tee "$log"; then
    echo "[build] $tier OK ($(( SECONDS - t0 ))s) → $IMG:$tier , $IMG:$tier-$DATE_TAG"
  else
    echo "[build] $tier FAIL — $log 끝부분 확인" >&2
    return 1
  fi
}

gate_tier() { # tier — PLAN §S2 컨테이너 gate: 이미지 안에서 run_all.sh --cpu-dry-run green.
  # (vllm import 0건 검사는 run_all 이 vendor/check_no_vllm.py 로 수행 + Dockerfile.triton 의
  #  역방향 게이트가 triton 이미지 자체의 vllm 부재를 보증)
  local tier="$1" log="$LOG_DIR/gate-$1-$(date +%Y%m%d-%H%M%S).log"
  echo "[build] ── 컨테이너 gate: run_all.sh --cpu-dry-run in $IMG:$tier (log: $log)"
  if docker run --rm -v "$REPO_DIR":/work -u "$(id -u):$(id -g)" -e HOME=/tmp \
      "$IMG:$tier" bash /work/kernel-kit/run_all.sh --cpu-dry-run 2>&1 | tee "$log"; then
    echo "[build] $tier 컨테이너 gate green"
  else
    echo "[build] $tier 컨테이너 gate FAIL — 반입 금지 (PLAN §S2)" >&2
    return 1
  fi
}

if [[ $BUILD_TRITON -eq 1 ]]; then
  build_one triton
  if [[ $RUN_GATE -eq 1 ]]; then gate_tier triton; fi
fi
if [[ $BUILD_FULL -eq 1 ]]; then
  build_one full
  if [[ $RUN_GATE -eq 1 ]]; then gate_tier full; fi
fi

echo "[build] 완료. 이미지:"
docker image ls "$IMG" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}'
echo "[build] 다음: bash docker/push.sh (ghcr) / bash docker/push.sh --release (registry 차단 폴백)"
