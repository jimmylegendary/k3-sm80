#!/usr/bin/env bash
# 커널 단독검증(S2 kernel-kit) 실행 — vLLM 서빙이 아니라 커널을 하나씩 참조구현과 대조한다.
#
# 사용:   bash run/20-kernel-test.sh [triton|full] [커널id ...]
#         triton = k05 제외 전부(vLLM 불필요) / full = k01~k07 전부(k05 Marlin 포함)
#         커널id 를 나열하면 그것만 부분 실행 (예: bash run/20-kernel-test.sh triton k01_kda_chunk)
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

if [[ $# -gt 0 ]]; then KERNELS="$*"
elif [[ $TIER == triton ]]; then
  KERNELS="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k06_dequant_bench k07_situ_act"
else
  KERNELS=""   # 빈 값 = run_all 기본(전부, k05 포함)
fi

echo "[test] tier=$TIER kernels=${KERNELS:-<전부>}"
docker run --rm --gpus all -v "$ROOT":/work ${KERNELS:+-e KERNELS="$KERNELS"} \
  "$IMG" bash /work/kernel-kit/run_all.sh
rc=$?

echo
case $rc in
  0) echo "[test] 종합 exit 0 — FAIL 없음. 리포트: kernel-kit/kernel-report/SUMMARY.md" ;;
  1) echo "[test] 종합 exit 1 — FAIL/CRASH 존재. 이것도 유효한 결과다(특히 k05 는 crash 수집이 목적)." ;;
  2) echo "[test] 종합 exit 2 — 사전조건 미충족: 커널이 실행되지 않았다. 성공으로 읽지 말 것." ;;
  *) echo "[test] 종합 exit $rc — 프레임워크 오류. kernel-kit/kernel-report/*.log 확인." ;;
esac
tgz=$(ls -t "$ROOT"/kernel-kit/k3-sm80-kernel-report-*.tgz 2>/dev/null | head -1)
[[ -n ${tgz:-} ]] && echo "[test] 반출 tar: ${tgz#$ROOT/}  → 다음: bash run/30-send-results.sh"
exit $rc
