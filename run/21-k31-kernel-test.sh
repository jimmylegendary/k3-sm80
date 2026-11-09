#!/usr/bin/env bash
# K3.1 커널 단독검증(S2 kernel-kit) 실행 — vLLM 서빙이 아니라 커널을 하나씩 참조구현과 대조한다.
# K3 의 run/20-kernel-test.sh 와 별개다(그 파일은 그대로). 이미지·기본 커널 목록·K3_TIER 각인이 K3.1용이다.
#
# 사용:   bash run/21-k31-kernel-test.sh [auto|full|openai] [--cpu-dry-run] [커널id ...]
#         auto(기본)  = k31 full 티어 이미지가 로컬에 있으면 그것, 없으면 openai 이미지(run/11 이 load 한 것)
#         full        = k31-sm80-kernelkit:full-a3c14efb   패치 전 기준선 (k08 은 upstream Marlin 그대로 = 수정 전 crash 수집용)
#         openai      = k31-sm80-openai:a3c14efb-sm80-v1   패치 적용본 (k08 이 "수정 후 회귀 기준"; 서빙 lane gate 와 같은 이미지)
#                       openai 이미지는 full 티어의 상위집합(vLLM·compute-sanitizer 동일)이며 ENTRYPOINT 가 vllm serve 라
#                       --entrypoint bash 로 실행한다.
#         커널id 를 나열하면 그것만 부분 실행 (예: bash run/21-k31-kernel-test.sh auto k11_dsa_indexer_sm80)
#         KERNELS="k01_… k12_…" 환경변수도 같은 효과 (위치 인자가 우선).
#         --cpu-dry-run = GPU 없이 배선만 검증(스모크용 — GPU 판정은 안 나옴)
# 기본 커널: k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k07_situ_act
#            k08_mxfp4_marlin_moe k11_dsa_indexer_sm80 k12_sparse_mla_sm80
# 결과:   kernel-kit/kernel-report/ 에 커널별 4게이트(NUMERIC/SANITIZER/SHAPES/PERF) 판정 +
#         반출용 kernel-kit/k3-sm80-kernel-report-<ts>.tgz 생성.
#         exit 0 = FAIL 없음 / 1 = FAIL 존재(정상 — 이것도 유효한 결과) /
#         2 = 사전조건 미충족(성공 아님) / 3 = 이 스크립트의 사전점검 실패 / 그 외 = 프레임워크 오류
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_IMG="${K31_FULL_IMAGE:-k31-sm80-kernelkit:full-a3c14efb}"
OPENAI_IMG="${K31_OPENAI_IMAGE:-k31-sm80-openai:a3c14efb-sm80-v1}"
DEFAULT_KERNELS="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k07_situ_act k08_mxfp4_marlin_moe k11_dsa_indexer_sm80 k12_sparse_mla_sm80"

TIER="auto"
case "${1:-}" in auto|full|openai) TIER="$1"; shift ;; esac

have() { docker image inspect "$1" >/dev/null 2>&1; }
case $TIER in
  auto)
    if have "$FULL_IMG"; then TIER=full
    elif have "$OPENAI_IMG"; then TIER=openai
    else echo "[test] k31 이미지 없음 ($FULL_IMG / $OPENAI_IMG) — 먼저: bash run/11-load-k31-image.sh" >&2; exit 3; fi ;;
  full)   have "$FULL_IMG"   || { echo "[test] $FULL_IMG 없음 — 사내에는 openai 이미지만 load 돼 있으면: bash run/21-k31-kernel-test.sh openai" >&2; exit 3; } ;;
  openai) have "$OPENAI_IMG" || { echo "[test] $OPENAI_IMG 없음 — 먼저: bash run/11-load-k31-image.sh" >&2; exit 3; } ;;
esac
if [[ $TIER == full ]]; then IMG="$FULL_IMG"; else IMG="$OPENAI_IMG"; fi

DRY=""; ARGS=()
for a in "$@"; do
  [[ $a == --cpu-dry-run ]] && DRY="--cpu-dry-run" || ARGS+=("$a")
done
if [[ ${#ARGS[@]} -gt 0 ]]; then KERNELS="${ARGS[*]}"
elif [[ -n "${KERNELS:-}" ]]; then :
else KERNELS="$DEFAULT_KERNELS"
fi

# ---- preflight: 어느 repo 를 쓰는지 + 클론 완전성 + 컨테이너 마운트 가시성 -------
# (K3 사내 실측 2026-07-30: env_check.py 부재 → python rc=2 → 'GPU 사전조건 미충족' 오표시.
#  원인 3후보 — 다른 디렉터리에서 실행 / 클론 불완전 / 원격 docker 데몬의 빈 마운트 — 를 여기서 구분한다.)
echo "[test] repo=$ROOT commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '(git repo 아님!)')"
for f in kernel-kit/run_all.sh kernel-kit/common/env_check.py kernel-kit/vendor/__init__.py; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "[test] FAIL: $ROOT/$f 없음 — 이 디렉터리는 완전한 클론이 아니다." >&2
    echo "  조치: cd <재클론한 k3-sm80 루트> 에서 실행했는지 확인. git status / 재클론 권장." >&2
    exit 3
  fi
done
for k in $KERNELS; do
  if [[ ! -f "$ROOT/kernel-kit/$k/harness.py" ]]; then
    echo "[test] FAIL: kernel-kit/$k/harness.py 없음 — 이 클론에 $k 가 없다 (k11/k12 는 K3.1 갱신 이후 커밋)." >&2
    echo "  조치: git pull 로 최신 커밋을 받았는지 확인 (git log --oneline -1 로 위 commit 확인)." >&2
    exit 3
  fi
done
if ! docker run --rm --entrypoint bash -v "$ROOT":/work "$IMG" -c 'test -f /work/kernel-kit/common/env_check.py'; then
  echo "[test] FAIL: 호스트엔 파일이 있는데 컨테이너 /work 마운트에선 안 보인다." >&2
  echo "  1순위: docker 데몬이 원격(-v 는 데몬 쪽 경로를 마운트) — 확인: docker context ls; echo \$DOCKER_HOST" >&2
  echo "  2순위: 마운트 경로 권한(NFS)/SELinux — 확인: ls -la $ROOT/kernel-kit/common/" >&2
  exit 3
fi

# GPU 플래그: dry-run 스모크는 GPU 없는 곳에서도 돌 수 있게 --gpus 생략
GPUS="--gpus all"; [[ -n $DRY ]] && GPUS=""

# ★ -u + HOME=/tmp 필수 (컨테이너 gate 와 동일 조건):
#   - 컨테이너 기본 root 는 NFS root_squash/userns-remap 환경에서 마운트된 repo 에 쓰기 불가
#     → kernel-report/ 생성 실패 = "permission denied ... env_check 내부 오류(rc=1)" (사내 실측)
#   - 비root 로 돌면 $HOME 이 없으므로 torch/triton 캐시를 /tmp 로 보낸다
#     (Triton JIT 는 컨테이너 안 gcc + Python.h 가 필요 — 이미지에 포함됨)
echo "[test] tier=$TIER image=$IMG kernels=$KERNELS ${DRY:+(dry-run)}"
[[ $TIER == full ]] && echo "[test] 참고: full 티어의 k08 은 패치 전(upstream) Marlin 기준선이다 — 패치 후 회귀는 'openai' 티어에서 본다."
docker run --rm $GPUS --entrypoint bash -v "$ROOT":/work \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton \
  -e K3_TIER="k31-$TIER" -e K3_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)" \
  -e KERNELS="$KERNELS" \
  "$IMG" /work/kernel-kit/run_all.sh $DRY
rc=$?

echo
case $rc in
  0) echo "[test] 종합 exit 0 — FAIL 없음. 리포트: kernel-kit/kernel-report/SUMMARY.md" ;;
  1) echo "[test] 종합 exit 1 — FAIL/CRASH 존재. 이것도 유효한 결과다(실패 로그가 다음 수정의 입력)." ;;
  2) echo "[test] 종합 exit 2 — 사전조건 미충족: 커널이 실행되지 않았다. 성공으로 읽지 말 것." ;;
  *) echo "[test] 종합 exit $rc — 프레임워크 오류. kernel-kit/kernel-report/*.log 확인." ;;
esac
echo "[test] 회신: 위 RELAY 블록 + 'exit $rc' 를 그대로 전달 (파일 반출 가능하면 run/30 으로 tgz 회신)"
exit $rc
