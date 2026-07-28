#!/usr/bin/env bash
# 사내 A100 노드 원커맨드 부트스트랩 — RUNBOOK §2(Stage 1)/§3(Stage 2)의 자동화. 멱등.
#
# 사용:
#   bash bootstrap.sh --stage1 [--yes]   # venv-probe: vllm==0.26.0 wheel (수 GB DL, 분 단위)
#   bash bootstrap.sh --stage2 [--yes]   # venv-k3: kimi-k3 pin 소스빌드 (40~90분, kernel-kit 권장)
#   K3_WORK_DIR=<dir> 로 작업 위치 변경 (기본 ~/k3-sm80-work; run_all.sh 가 같은 규약으로 자동 감지)
#
# 정책: pip 로 해결되는 것만 자동 조치한다. GPU/드라이버 부재는 자동 조치 불가 항목이라
#       명확한 사유와 확인 명령을 출력하고 중단한다(RUNBOOK §1.1·§6.1).
# exit: 0 = 구성 완료 / 2 = GPU·드라이버 등 자동 조치 불가 / 3 = 인자·사용자 중단 / 1 = 기타 실패
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # a100-kit/
REPO="$(dirname "$HERE")"
WORK="${K3_WORK_DIR:-$HOME/k3-sm80-work}"
K3_PIN=0498dc7ea9204f1863b46515816f5d8d458e01e7           # RUNBOOK §0 버전 고정표
VLLM_WHEEL=0.26.0

STAGE=""; YES=0
for a in "$@"; do
  case "$a" in
    --stage1) STAGE=1 ;;
    --stage2) STAGE=2 ;;
    --yes)    YES=1 ;;
    *) echo "[bootstrap] 알 수 없는 인자: $a  (사용: --stage1|--stage2 [--yes])" >&2; exit 3 ;;
  esac
done
[[ -z $STAGE ]] && { echo "[bootstrap] 사용: bash bootstrap.sh --stage1|--stage2 [--yes]" >&2; exit 3; }

confirm() {
  [[ $YES -eq 1 ]] && return 0
  read -r -p "[bootstrap] $1 진행? [y/N] " ans \
    || { echo "[bootstrap] 비대화형 실행 감지(stdin EOF) — --yes 를 붙여 실행할 것" >&2; exit 3; }
  [[ ${ans,,} == y* ]] || { echo "[bootstrap] 사용자 중단" >&2; exit 3; }
}

# ---- 자동 조치 불가 항목: GPU / 드라이버 -----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] 자동 조치 불가: nvidia-smi 없음 — NVIDIA 드라이버 미설치/미로드." >&2
  echo "  확인: lsmod | grep nvidia ; ls /proc/driver/nvidia ; RUNBOOK §1.1" >&2
  exit 2
fi
if ! nvidia-smi -L 2>/dev/null | grep -q .; then
  echo "[bootstrap] 자동 조치 불가: nvidia-smi 는 있으나 GPU 가 안 보임 (드라이버/권한/컨테이너 가시성)." >&2
  echo "  확인: nvidia-smi ; echo \$CUDA_VISIBLE_DEVICES ; RUNBOOK §1.1·§6.1" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader \
  | sed 's/^/[bootstrap] GPU: /'

mkdir -p "$WORK"
cd "$WORK"

if [[ $STAGE -eq 1 ]]; then
  # ---- Stage 1: venv-probe + 릴리스 wheel (RUNBOOK §2) ----------------------------
  V="$WORK/venv-probe"
  if [[ -x "$V/bin/python3" ]] && "$V/bin/python3" -c "import vllm" >/dev/null 2>&1; then
    echo "[bootstrap] venv-probe 이미 구성됨 — 멱등 skip ($V)"
  else
    confirm "Stage 1: venv-probe 생성 + vllm==$VLLM_WHEEL wheel 설치 (torch 자동 수반, 수 GB 다운로드)"
    python3 -m venv "$V"
    "$V/bin/pip" install -U pip
    "$V/bin/pip" install "vllm==$VLLM_WHEEL"
  fi
  PYV="$V/bin/python3"
else
  # ---- Stage 2: venv-k3 + kimi-k3 pin 소스빌드 (RUNBOOK §3) -----------------------
  V="$WORK/venv-k3"
  if [[ -x "$V/bin/python3" ]] && "$V/bin/python3" -c "import vllm" >/dev/null 2>&1; then
    echo "[bootstrap] venv-k3 이미 구성됨 — 멱등 skip ($V)"
  else
    confirm "Stage 2: venv-k3 + vLLM kimi-k3(${K3_PIN:0:8}) TORCH_CUDA_ARCH_LIST=8.0 소스빌드 (40~90분)"
    python3 -m venv "$V"
    "$V/bin/pip" install -U pip
    if [[ ! -d "$WORK/vllm-k3/.git" ]]; then
      git clone https://github.com/vllm-project/vllm.git "$WORK/vllm-k3"
    fi
    cd "$WORK/vllm-k3"
    git fetch origin "pull/50000/head:kimi-k3-pr" 2>/dev/null || git fetch origin || true
    git checkout "$K3_PIN" \
      || { echo "[bootstrap] PIN 커밋 소실(force-push 의심) — 새 head 로 진행, 해시 기록(RUNBOOK §3)"; \
           git checkout kimi-k3-pr; git rev-parse HEAD | tee "$WORK/K3_BRANCH_ACTUAL_COMMIT.txt"; }
    echo "[bootstrap] vllm-k3 HEAD = $(git rev-parse HEAD)"
    export TORCH_CUDA_ARCH_LIST=8.0
    export MAX_JOBS="${MAX_JOBS:-32}"
    export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
    "$V/bin/pip" install -r requirements/build/cuda.txt
    "$V/bin/pip" install -r requirements/cuda.txt
    "$V/bin/pip" install -e . --no-build-isolation 2>&1 | tee "$WORK/build-k3.log"
    cd "$WORK"
  fi
  PYV="$V/bin/python3"
fi

# ---- 최종 재검사 + 다음 명령 안내 ------------------------------------------------
echo "[bootstrap] env_check 재검사:"
"$PYV" "$REPO/kernel-kit/common/env_check.py" || true
echo "[bootstrap] 완료. 다음:"
echo "  bash $REPO/kernel-kit/run_all.sh          # venv 자동 감지(K3_WORK_DIR=$WORK)"
echo "  # 또는 source $V/bin/activate 후 실행"
