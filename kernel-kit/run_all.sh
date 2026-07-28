#!/usr/bin/env bash
# S2 kernel-kit 러너 — k01..k07 순차 실행 + kernel-report/ 집계 + tar 반출.
#
# 사용:
#   bash run_all.sh                  # GPU 모드 (A100 노드; 사전조건 미충족 시 명확히 skip)
#   bash run_all.sh --cpu-dry-run    # 저작 호스트: 참조구현·판정로직·리포트 경로만 검증
#   KERNELS="k01_kda_chunk k02_kda_recurrent" bash run_all.sh    # 부분 실행
#   PHASE0_VERDICT=/path/verdict.json bash run_all.sh            # PERF 목표 원천 오버라이드
#
# exit: 0 = FAIL 없음 / 1 = FAIL 존재 / 2 = GPU 사전조건 미충족 skip / 3 = 프레임워크 오류
set -uo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${KERNEL_REPORT_DIR:-$KIT_DIR/kernel-report}"
VERDICT="${PHASE0_VERDICT:-$KIT_DIR/../phase0/verdict.json}"
KERNELS_DEFAULT="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k05_marlin_moe_w4a16 k06_dequant_bench k07_situ_act"
KERNELS="${KERNELS:-$KERNELS_DEFAULT}"

DRY=""
for a in "$@"; do
  case "$a" in
    --cpu-dry-run) DRY="--cpu-dry-run" ;;
    *) echo "[run_all] 알 수 없는 인자: $a" >&2; exit 3 ;;
  esac
done

mkdir -p "$REPORT_DIR"
echo "[run_all] kit=$KIT_DIR report=$REPORT_DIR verdict=$VERDICT mode=${DRY:-gpu}"

# ---- 0) 사전조건 (env_check가 판정·메시지 담당) --------------------------------
python3 "$KIT_DIR/common/env_check.py" --json $DRY > "$REPORT_DIR/env.json"
rc=$?
python3 "$KIT_DIR/common/env_check.py" $DRY --quiet >/dev/null 2>&1 || true
if [[ $rc -eq 2 ]]; then
  echo "[run_all] GPU 사전조건 미충족 — 커널 실행 skip (상세: $REPORT_DIR/env.json)." >&2
  echo "[run_all] GPU 없는 호스트에서는 'bash run_all.sh --cpu-dry-run' 을 쓸 것." >&2
  exit 2
elif [[ $rc -ne 0 ]]; then
  echo "[run_all] env_check 내부 오류 (rc=$rc)" >&2
  exit 3
fi

# ---- 1) 프레임워크 자체점검 (호스트 게이트: py_compile + selftest green) --------
echo "[run_all] 프레임워크 자체점검"
python3 -m py_compile "$KIT_DIR"/common/*.py || exit 3
python3 "$KIT_DIR/common/shapes.py" --check > "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] shapes selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
python3 "$KIT_DIR/common/tolerance.py" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] tolerance selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
python3 "$KIT_DIR/common/timing.py" --selftest --phase0-verdict "$VERDICT" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] timing selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
python3 "$KIT_DIR/common/report.py" --selftest --report-dir "$REPORT_DIR" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] report selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
echo "[run_all] 자체점검 green (framework-selftest.log)"

# ---- 2) 커널 하네스 순차 실행 ---------------------------------------------------
declare -A RC
MISSING=()
for k in $KERNELS; do
  H=""
  [[ -f "$KIT_DIR/$k/harness.py" ]] && H="$KIT_DIR/$k/harness.py"
  [[ -z $H && -f "$KIT_DIR/$k.py" ]] && H="$KIT_DIR/$k.py"
  if [[ -z $H ]]; then
    echo "[run_all] $k: 하네스 없음(미저작) — MISSING"
    MISSING+=("$k")
    continue
  fi
  echo "[run_all] ── $k 실행 ($H)"
  t0=$SECONDS
  python3 "$H" --phase0-verdict "$VERDICT" --report-dir "$REPORT_DIR" $DRY \
    2>&1 | tee "$REPORT_DIR/$k.log"
  RC[$k]=${PIPESTATUS[0]}
  echo "[run_all] $k 종료 rc=${RC[$k]} ($((SECONDS - t0))s)"
done

# ---- 3) 집계 + 반출 -------------------------------------------------------------
python3 "$KIT_DIR/common/report.py" --aggregate "$REPORT_DIR" --expect $KERNELS
AGG_RC=$?

STAMP=$(date +%Y%m%d-%H%M%S)
TAR="$KIT_DIR/k3-sm80-kernel-report-$STAMP.tgz"
tar czf "$TAR" -C "$(dirname "$REPORT_DIR")" "$(basename "$REPORT_DIR")"
echo "[run_all] 반출 tar: $TAR"
[[ ${#MISSING[@]} -gt 0 ]] && echo "[run_all] 미저작 하네스: ${MISSING[*]}"

FAIL=0
for k in "${!RC[@]}"; do
  [[ ${RC[$k]} -eq 1 ]] && FAIL=1
  [[ ${RC[$k]} -ge 3 ]] && FAIL=1
done
[[ $AGG_RC -eq 1 ]] && FAIL=1
if [[ $FAIL -eq 1 ]]; then
  echo "[run_all] 종합: FAIL 존재 — kernel-report/SUMMARY.md 확인 (FAIL/CRASH 로그도 S3 입력물이다)"
  exit 1
fi
echo "[run_all] 종합: FAIL 없음 (${DRY:+dry-run 상한=INCOMPLETE})"
exit 0
