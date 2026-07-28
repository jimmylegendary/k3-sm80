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
  # TTY 아닌 stdin(파이프/리다이렉트)은 EOF 가 아니어도 read 가 무기한 대기할 수 있다 → 차단
  [[ -t 0 ]] || { echo "[bootstrap] 비대화형 실행 감지(stdin≠TTY) — --yes 를 붙여 실행할 것" >&2; exit 3; }
  read -r -p "[bootstrap] $1 진행? [y/N] " ans \
    || { echo "[bootstrap] stdin EOF — --yes 를 붙여 실행할 것" >&2; exit 3; }
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
  # 멱등 판정: import 만으론 '실패한/절반 빌드'를 성공으로 오인할 수 있다(codex 리뷰 확인)
  # → 완료 마커(빌드된 커밋 기록) + import 둘 다 요구. 마커는 빌드 성공 후에만 기록된다.
  MARKER="$V/.bootstrap-ok-stage2"
  if [[ -x "$V/bin/python3" && -f "$MARKER" ]] \
     && "$V/bin/python3" -c "import vllm" >/dev/null 2>&1; then
    echo "[bootstrap] venv-k3 이미 구성됨 — 멱등 skip ($V, built=$(cat "$MARKER"))"
  else
    confirm "Stage 2: venv-k3 + vLLM kimi-k3(${K3_PIN:0:8}) TORCH_CUDA_ARCH_LIST=8.0 소스빌드 (40~90분)"
    python3 -m venv "$V"
    "$V/bin/pip" install -U pip
    if [[ ! -d "$WORK/vllm-k3/.git" ]]; then
      git clone https://github.com/vllm-project/vllm.git "$WORK/vllm-k3"
    fi
    cd "$WORK/vllm-k3"
    # fetch 실패를 삼키면 stale 로컬 브랜치로 폴백할 수 있다(codex 리뷰 확인) → 실패를 드러낸다
    if ! git fetch origin "pull/50000/head:kimi-k3-pr"; then
      echo "[bootstrap] WARN: PR ref fetch 실패 — pin 커밋 직접 fetch 시도" >&2
      git fetch origin "$K3_PIN" || echo "[bootstrap] WARN: pin 직접 fetch 도 실패 — 로컬 객체로 시도" >&2
    fi
    if ! git checkout "$K3_PIN"; then
      echo "[bootstrap] PIN 커밋 체크아웃 실패(force-push/네트워크 의심) — RUNBOOK §3" >&2
      if git rev-parse --verify -q kimi-k3-pr >/dev/null; then
        echo "[bootstrap] 새 PR head 로 진행, 실제 해시 기록" >&2
        git checkout kimi-k3-pr
        git rev-parse HEAD | tee "$WORK/K3_BRANCH_ACTUAL_COMMIT.txt"
      else
        echo "[bootstrap] 폴백 브랜치도 없음 — 네트워크/GitHub 접근을 확인하고 재실행할 것" >&2
        exit 1
      fi
    fi
    echo "[bootstrap] vllm-k3 HEAD = $(git rev-parse HEAD)"
    export TORCH_CUDA_ARCH_LIST=8.0
    export MAX_JOBS="${MAX_JOBS:-32}"
    export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
    "$V/bin/pip" install -r requirements/build/cuda.txt
    "$V/bin/pip" install -r requirements/cuda.txt
    "$V/bin/pip" install -e . --no-build-isolation 2>&1 | tee "$WORK/build-k3.log"
    git rev-parse HEAD > "$MARKER"          # 빌드 성공 시에만 도달(set -e·pipefail)
    cd "$WORK"
  fi
  PYV="$V/bin/python3"
fi

# ---- 최종 재검사 (결과를 삼키지 않는다 — 미충족이면 bootstrap 도 실패로 끝난다) ----
echo "[bootstrap] env_check 재검사:"
if ! "$PYV" "$REPO/kernel-kit/common/env_check.py"; then
  echo "[bootstrap] env_check 여전히 미충족 — 위 FAIL 항목의 '해결:' 을 따를 것" >&2
  exit 2
fi
echo "[bootstrap] 완료. 다음:"
echo "  bash $REPO/kernel-kit/run_all.sh          # venv 자동 감지(K3_WORK_DIR=$WORK)"
echo "  # 또는 source $V/bin/activate 후 실행"
