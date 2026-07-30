#!/usr/bin/env bash
# 커널 단독검증(S2 kernel-kit) 실행 — vLLM 서빙이 아니라 커널을 하나씩 참조구현과 대조한다.
#
# 사용:   bash run/20-kernel-test.sh [triton|full] [--cpu-dry-run] [커널id ...]
#         triton = k05 제외 전부(vLLM 불필요) / full = k01~k07 전부(k05 Marlin 포함)
#         커널id 를 나열하면 그것만 부분 실행 (예: bash run/20-kernel-test.sh triton k01_kda_chunk)
#         --cpu-dry-run = GPU 없이 배선만 검증(스모크용 — GPU 판정은 안 나옴)
# 결과:   kernel-kit/kernel-report/ 에 커널별 4게이트(NUMERIC/SANITIZER/SHAPES/PERF) 판정 +
#         반출용 kernel-kit/k3-sm80-kernel-report-<ts>.tgz 생성.
#         exit 0 = FAIL 없음 / 1 = FAIL 존재(정상 — 특히 k05 crash 는 수집 목적) /
#         2 = 사전조건 미충족(성공 아님) / 그 외 = 프레임워크 오류
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-triton}"; shift || true
[[ $TIER == triton || $TIER == full ]] || { echo "[test] tier 는 triton|full" >&2; exit 3; }
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit:$TIER"
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "[test] $IMG 없음 — 먼저: bash run/10-load-image.sh $TIER" >&2; exit 3; }

# ---- preflight: 어느 repo 를 쓰는지 + 클론 완전성 + 컨테이너 마운트 가시성 -------
# (사내 실측 2026-07-30: env_check.py 부재 → python rc=2 → 'GPU 사전조건 미충족' 오표시.
#  원인 3후보 — 다른 디렉터리에서 실행 / 클론 불완전 / 원격 docker 데몬의 빈 마운트 —
#  를 여기서 구분해 정확한 메시지로 알린다.)
echo "[test] repo=$ROOT commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '(git repo 아님!)')"
for f in kernel-kit/run_all.sh kernel-kit/common/env_check.py kernel-kit/vendor/__init__.py; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "[test] FAIL: $ROOT/$f 없음 — 이 디렉터리는 완전한 클론이 아니다." >&2
    echo "  조치: cd <재클론한 k3-sm80 루트> 에서 실행했는지 확인. git status / 재클론 권장." >&2
    exit 3
  fi
done
if ! docker run --rm -v "$ROOT":/work "$IMG" test -f /work/kernel-kit/common/env_check.py; then
  echo "[test] FAIL: 호스트엔 파일이 있는데 컨테이너 /work 마운트에선 안 보인다." >&2
  echo "  1순위: docker 데몬이 원격(-v 는 데몬 쪽 경로를 마운트) — 확인: docker context ls; echo \$DOCKER_HOST" >&2
  echo "  2순위: 마운트 경로 권한(NFS)/SELinux — 확인: ls -la $ROOT/kernel-kit/common/" >&2
  exit 3
fi

DRY=""; ARGS=()
for a in "$@"; do
  [[ $a == --cpu-dry-run ]] && DRY="--cpu-dry-run" || ARGS+=("$a")
done
if [[ ${#ARGS[@]} -gt 0 ]]; then KERNELS="${ARGS[*]}"
elif [[ $TIER == triton ]]; then
  KERNELS="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k06_dequant_bench k07_situ_act"
else
  KERNELS=""   # 빈 값 = run_all 기본(전부, k05 포함)
fi

# GPU 플래그: dry-run 스모크는 GPU 없는 곳에서도 돌 수 있게 --gpus 생략
GPUS="--gpus all"; [[ -n $DRY ]] && GPUS=""

# ★ -u + HOME=/tmp 필수 (컨테이너 gate 와 동일 조건):
#   - 컨테이너 기본 root 는 NFS root_squash/userns-remap 환경에서 마운트된 repo 에 쓰기 불가
#     → kernel-report/ 생성 실패 = "permission denied ... env_check 내부 오류(rc=1)" (사내 실측)
#   - 비root 로 돌면 $HOME 이 없으므로 torch/triton 캐시를 /tmp 로 보낸다
echo "[test] tier=$TIER kernels=${KERNELS:-<전부>} ${DRY:+(dry-run)}"
docker run --rm $GPUS -v "$ROOT":/work \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton \
  -e K3_TIER="$TIER" -e K3_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)" \
  ${KERNELS:+-e KERNELS="$KERNELS"} \
  "$IMG" bash /work/kernel-kit/run_all.sh $DRY
rc=$?

echo
case $rc in
  0) echo "[test] 종합 exit 0 — FAIL 없음. 리포트: kernel-kit/kernel-report/SUMMARY.md" ;;
  1) echo "[test] 종합 exit 1 — FAIL/CRASH 존재. 이것도 유효한 결과다(특히 k05 는 crash 수집이 목적)." ;;
  2) echo "[test] 종합 exit 2 — 사전조건 미충족: 커널이 실행되지 않았다. 성공으로 읽지 말 것." ;;
  *) echo "[test] 종합 exit $rc — 프레임워크 오류. kernel-kit/kernel-report/*.log 확인." ;;
esac
echo "[test] 회신: 위 RELAY 블록 + 'exit $rc' 를 그대로 전달 (파일 반출 가능하면 run/30 으로 tgz 회신)"
exit $rc
