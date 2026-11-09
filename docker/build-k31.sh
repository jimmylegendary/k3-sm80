#!/usr/bin/env bash
# k31-sm80 (Kimi K3.1-Pro) 도커 빌드: full 티어 → openai 티어 → verify. K3 이미지·태그·스크립트와 별개다.
#
# 사용:
#   bash docker/build-k31.sh                # full(수 시간) → openai → verify_k31_image.py
#   bash docker/build-k31.sh --full-only    # full 티어만 (vLLM main a3c14efb SM80 소스빌드)
#   bash docker/build-k31.sh --openai-only  # openai 티어만 (full 로컬 태그가 이미 있어야 함)
#   bash docker/build-k31.sh --skip-verify  # verify_k31_image.py 생략
#   MAX_JOBS=4 bash docker/build-k31.sh     # 병렬 수동 지정 (기본: RAM 가드가 산출, 상한 8)
#
# 환경변수 (K31_ 접두):
#   K31_FULL_IMAGE        기본 k31-sm80-kernelkit:full-a3c14efb
#   K31_OPENAI_IMAGE      기본 k31-sm80-openai:a3c14efb-sm80-v1
#   K31_PIP_EXTRA_INDEX_URL  full 빌드의 pip --extra-index-url (기본 없음; 예: https://download.pytorch.org/whl/cu130)
#   K31_VERIFY_GPU=1      verify 를 --gpu 로 실행 (GPU 가 있는 호스트에서만)
#   K31_SKIP_DISK_CHECK=1 디스크 여유 검사 생략
#
# 필요한 것: 인터넷(GitHub clone·PyPI·NVIDIA apt — 빌드 중 계속 접근), docker+buildx, 디스크 여유 ≥ 100GB 권장
# (full 빌드 캐시·레이어 ≈ 60–80GB 피크). GPU 불필요 — nvcc 크로스컴파일(TORCH_CUDA_ARCH_LIST=8.0).
# 예상 시간(추정): full 2–3h (K3 full 실측 6,282s@MAX_JOBS=8 + 핀 소스 증가분), openai 40–80min.
# exit: 0 = 요청한 빌드·verify 전부 성공 / 2 = 사전조건 미충족 / 3 = 인자 오류 / 그 외 = 실패 지점의 로그 확인
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$DIR")"
readonly VLLM_COMMIT="a3c14efbc77c018e8bf579935095a87721349e05"
FULL_IMAGE="${K31_FULL_IMAGE:-k31-sm80-kernelkit:full-a3c14efb}"
OPENAI_IMAGE="${K31_OPENAI_IMAGE:-k31-sm80-openai:a3c14efb-sm80-v1}"
LOG_DIR="$DIR/logs"
PATCH_DIR="$DIR/patches-k31"
PATCHES=(mxfp4-marlin-moe-sm80-main triton-mla-sparse-sm80 k31-sm80-dsa-glue)
mkdir -p "$LOG_DIR"

BUILD_FULL=1
BUILD_OPENAI=1
RUN_VERIFY=1
for a in "$@"; do
  case "$a" in
    --full-only)   BUILD_OPENAI=0; RUN_VERIFY=0 ;;
    --openai-only) BUILD_FULL=0 ;;
    --skip-verify) RUN_VERIFY=0 ;;
    -h|--help)     sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[build-k31] 알 수 없는 인자: $a" >&2; exit 3 ;;
  esac
done
if [[ $BUILD_FULL -eq 0 && $BUILD_OPENAI -eq 0 ]]; then
  echo "[build-k31] --full-only 와 --openai-only 는 함께 쓸 수 없다" >&2; exit 3
fi

command -v docker >/dev/null || { echo "[build-k31] docker 없음" >&2; exit 2; }
export DOCKER_BUILDKIT=1

# ---- 디스크 가드: 도커 루트 여유 (경고만이 아니라 중단 — 수 시간 뒤 ENOSPC 방지) ----
if [[ "${K31_SKIP_DISK_CHECK:-0}" != "1" ]]; then
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
  avail_gb="$(df --output=avail -BG "$docker_root" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
  avail_gb="${avail_gb:-0}"
  echo "[build-k31] 디스크 여유: ${avail_gb}GB ($docker_root)"
  if (( avail_gb < 60 )); then
    echo "[build-k31] 디스크 여유 ${avail_gb}GB < 60GB — 중단 (무시: K31_SKIP_DISK_CHECK=1)" >&2; exit 2
  elif (( avail_gb < 100 )); then
    echo "[build-k31] 경고: 여유 ${avail_gb}GB < 100GB — full 빌드 피크(≈60–80GB)에 빠듯하다" >&2
  fi
fi

# ---- RAM 가드: nvcc 잡당 ~2GB(RUNBOOK §6) + 시스템 여유 4GB, 상한 8 (호스트 계약) ----
if [[ -z "${MAX_JOBS:-}" ]]; then
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  avail_mem_gb=$(( avail_kb / 1024 / 1024 ))
  MAX_JOBS=$(( (avail_mem_gb - 4) / 2 ))
  (( MAX_JOBS > 8 )) && MAX_JOBS=8
  (( MAX_JOBS < 2 )) && MAX_JOBS=2
  echo "[build-k31] RAM 가드: MemAvailable=${avail_mem_gb}GB → MAX_JOBS=$MAX_JOBS (수동: MAX_JOBS=n bash $0)"
else
  echo "[build-k31] MAX_JOBS=$MAX_JOBS (수동 지정)"
fi
NVCC_THREADS="${NVCC_THREADS:-2}"

# docker build 1회 실행 + 로그. 실패 시 로그 위치를 안내하고 비-0 반환.
run_build() { # name log -- docker build args...
  local name="$1" log="$2"; shift 2
  echo "[build-k31] ── $name 빌드 시작 (log: $log)"
  local t0=$SECONDS
  if docker build --progress=plain "$@" 2>&1 | tee "$log"; then
    echo "[build-k31] $name OK ($(( SECONDS - t0 ))s)"
  else
    echo "[build-k31] $name FAIL ($(( SECONDS - t0 ))s) — $log 끝부분 확인" >&2
    return 1
  fi
}

build_full() {
  local extra=()
  [[ -n "${K31_PIP_EXTRA_INDEX_URL:-}" ]] && extra+=(--build-arg "PIP_EXTRA_INDEX_URL=$K31_PIP_EXTRA_INDEX_URL")
  run_build "full 티어($FULL_IMAGE)" "$LOG_DIR/k31-full-$(date +%Y%m%d-%H%M%S).log" \
    -f "$DIR/Dockerfile.k31-full" -t "$FULL_IMAGE" \
    --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
    --build-arg "MAX_JOBS=$MAX_JOBS" --build-arg "NVCC_THREADS=$NVCC_THREADS" \
    ${extra[@]+"${extra[@]}"} "$DIR"
}

build_openai() {
  local missing=0 p
  for p in "${PATCHES[@]}"; do
    if [[ ! -s "$PATCH_DIR/$p.patch" ]]; then
      echo "[build-k31] 패치 없음: docker/patches-k31/$p.patch" >&2; missing=1
    fi
  done
  (( missing == 0 )) || { echo "[build-k31] 패치 3개가 모두 있어야 openai 티어를 만들 수 있다" >&2; return 2; }
  docker image inspect "$FULL_IMAGE" >/dev/null 2>&1 \
    || { echo "[build-k31] full 이미지 없음: $FULL_IMAGE — 먼저 bash $0 --full-only" >&2; return 2; }
  local base_id
  base_id="$(docker image inspect --format '{{.Id}}' "$FULL_IMAGE")"
  echo "[build-k31] base=$FULL_IMAGE id=$base_id (io.k3.base_digest 라벨로 고정된다)"
  run_build "openai 티어($OPENAI_IMAGE)" "$LOG_DIR/k31-openai-$(date +%Y%m%d-%H%M%S).log" \
    --target contract -f "$DIR/Dockerfile.k31-openai" -t "$OPENAI_IMAGE" \
    --build-arg "BASE_IMAGE=$FULL_IMAGE" --build-arg "BASE_DIGEST=$base_id" \
    --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
    --build-arg "MAX_JOBS=$MAX_JOBS" --build-arg "NVCC_THREADS=$NVCC_THREADS" \
    "$REPO_DIR"
}

verify_openai() {
  local vlog="$LOG_DIR/k31-verify-$(date +%Y%m%d-%H%M%S).log" gpu=()
  [[ "${K31_VERIFY_GPU:-0}" == "1" ]] && gpu+=(--gpu)
  echo "[build-k31] ── verify_k31_image.py (log: $vlog)"
  python3 "$DIR/verify_k31_image.py" "$OPENAI_IMAGE" --base-image "$FULL_IMAGE" ${gpu[@]+"${gpu[@]}"} 2>&1 | tee "$vlog"
}

[[ $BUILD_FULL -eq 1 ]] && build_full
[[ $BUILD_OPENAI -eq 1 ]] && build_openai
[[ $RUN_VERIFY -eq 1 ]] && verify_openai

echo "[build-k31] 완료. 이미지 요약:"
for img in "$FULL_IMAGE" "$OPENAI_IMAGE"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    docker image inspect "$img" --format '  {{index .RepoTags 0}}  id={{.Id}}  size={{.Size}}  patchset={{index .Config.Labels "io.k3.patchset"}}'
  fi
done
echo "[build-k31] 다음: bash docker/release-k31.sh (K31_OPENAI_GITHUB_UPLOAD=1 로 Release 업로드), GPU 노드에서 verify_k31_image.py --gpu"
