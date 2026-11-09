#!/usr/bin/env bash
# S2 kernel-kit 러너 — k01..k09 + k11·k12 순차 실행 + kernel-report/ 집계 + tar 반출.
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
KERNELS_DEFAULT="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k05_marlin_moe_w4a16 k06_dequant_bench k07_situ_act k08_mxfp4_marlin_moe k09_mla_fp8_kv_sm80 k11_dsa_indexer_sm80 k12_sparse_mla_sm80"
KERNELS="${KERNELS:-$KERNELS_DEFAULT}"

DRY=""
for a in "$@"; do
  case "$a" in
    --cpu-dry-run) DRY="--cpu-dry-run" ;;
    *) echo "[run_all] 알 수 없는 인자: $a" >&2; exit 3 ;;
  esac
done

mkdir -p "$REPORT_DIR" 2>/dev/null
# 쓰기 가능성 선검사 — 컨테이너 root vs NFS root_squash/userns-remap 에서 여기가 첫 실패 지점.
# 이걸 안 잡으면 env.json 리다이렉트 실패(rc=1)가 "env_check 내부 오류"로 위장된다(사내 실측).
if ! touch "$REPORT_DIR/.write-probe" 2>/dev/null; then
  echo "[run_all] FAIL: 리포트 디렉터리 쓰기 불가 — $REPORT_DIR" >&2
  echo "[run_all]   원인 후보: 컨테이너를 root 로 실행 (NFS root_squash/userns-remap 환경)" >&2
  echo "[run_all]   해결: run/20-kernel-test.sh 를 쓰거나, docker run 에 다음을 추가:" >&2
  echo "[run_all]     -u \"\$(id -u):\$(id -g)\" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton" >&2
  exit 3
fi
rm -f "$REPORT_DIR/.write-probe"

# ---- python 선택: venv 자동 감지 (PYTHON > VIRTUAL_ENV > K3_WORK_DIR 의 venv) ----
# RUNBOOK 규약: 작업 디렉터리 ~/k3-sm80-work (K3_WORK_DIR 로 변경 가능)에 venv-k3(소스빌드,
# 우선) / venv-probe(Stage1 wheel) — activate 없이 실행해도 여기서 자동으로 잡는다.
# 실행 식별 — 리포트에 각인해 혼합/stale 판독을 막는다 (run/20 이 K3_TIER 를 넘긴다)
export K3_RUN_ID="${K3_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export K3_TIER="${K3_TIER:-unset}"

WORK="${K3_WORK_DIR:-$HOME/k3-sm80-work}"
if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
  # PATH 가 venv 를 안 가리키는 잔류 VIRTUAL_ENV 케이스 방지 — 경로로 직접 지정(codex 리뷰 확인)
  PY="$VIRTUAL_ENV/bin/python3"
elif [[ -x "$WORK/venv-k3/bin/python3" ]]; then
  PY="$WORK/venv-k3/bin/python3"; echo "[run_all] venv 자동선택: $WORK/venv-k3"
elif [[ -x "$WORK/venv-probe/bin/python3" ]]; then
  PY="$WORK/venv-probe/bin/python3"
  echo "[run_all] venv 자동선택: $WORK/venv-probe (Stage1 wheel — K3 브랜치 전용 경로는 venv-k3 권장)"
else PY="python3"; fi
echo "[run_all] kit=$KIT_DIR report=$REPORT_DIR verdict=$VERDICT mode=${DRY:-gpu} python=$PY"

# ---- 0) 사전조건 (env_check가 판정·메시지 담당) --------------------------------
# python 이 파일을 못 열면 rc=2 인데 이는 사전조건 미충족(rc=2)과 겹친다 — 먼저 실재 확인
if [[ ! -f "$KIT_DIR/common/env_check.py" ]]; then
  echo "[run_all] FAIL: $KIT_DIR/common/env_check.py 없음 — 마운트/클론 문제 (GPU 문제 아님)." >&2
  echo "[run_all]   run/20-kernel-test.sh 로 실행하면 preflight 가 원인을 구분해 준다." >&2
  exit 3
fi
"$PY" "$KIT_DIR/common/env_check.py" --json $DRY > "$REPORT_DIR/env.json"
rc=$?
if [[ $rc -eq 2 && "${AUTO_FIX:-0}" == "1" && -z "$DRY" ]] \
   && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  # GPU 는 있는데 python 환경이 문제 → bootstrap 으로 자동 조치 후 1회 재시도
  echo "[run_all] AUTO_FIX=1 — GPU 확인됨, bootstrap 으로 환경 자동 구성 시도" >&2
  bash "$(dirname "$KIT_DIR")/a100-kit/bootstrap.sh" --stage1 --yes >&2 || true
  [[ -x "$WORK/venv-probe/bin/python3" ]] && PY="$WORK/venv-probe/bin/python3"
  "$PY" "$KIT_DIR/common/env_check.py" --json $DRY > "$REPORT_DIR/env.json"
  rc=$?
fi
if [[ $rc -eq 2 ]]; then
  echo "[run_all] GPU 사전조건 미충족 — 항목별 판정 (FAIL 의 '해결:' 을 따를 것):" >&2
  "$PY" "$KIT_DIR/common/env_check.py" $DRY >&2 || true
  echo "[run_all] 자동 구성: bash a100-kit/bootstrap.sh --stage1|--stage2  (GPU/드라이버 문제만 수동)" >&2
  echo "[run_all] GPU 없는 호스트에서는 'bash run_all.sh --cpu-dry-run' 을 쓸 것. (상세: $REPORT_DIR/env.json)" >&2
  exit 2
elif [[ $rc -ne 0 ]]; then
  echo "[run_all] env_check 내부 오류 (rc=$rc)" >&2
  exit 3
fi

# ---- 1) 프레임워크 자체점검 (호스트 게이트: py_compile + selftest green) --------
echo "[run_all] 프레임워크 자체점검"
"$PY" -m py_compile "$KIT_DIR"/common/*.py || exit 3
# vendor 게이트: py_compile + AST(vllm import 0건) — PLAN §S2 컨테이너 gate 의 "vllm import 0건(k05 제외)"
"$PY" "$KIT_DIR/vendor/check_no_vllm.py" --quiet || { echo "[run_all] vendor check_no_vllm FAIL"; exit 3; }
"$PY" "$KIT_DIR/common/shapes.py" --check > "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] shapes selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
"$PY" "$KIT_DIR/common/tolerance.py" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] tolerance selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
"$PY" "$KIT_DIR/common/timing.py" --selftest --phase0-verdict "$VERDICT" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] timing selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
"$PY" "$KIT_DIR/common/report.py" --selftest --report-dir "$REPORT_DIR" >> "$REPORT_DIR/framework-selftest.log" 2>&1 || { echo "[run_all] report selftest FAIL"; tail -5 "$REPORT_DIR/framework-selftest.log"; exit 3; }
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
  "$PY" "$H" --phase0-verdict "$VERDICT" --report-dir "$REPORT_DIR" $DRY \
    2>&1 | tee "$REPORT_DIR/$k.log"
  RC[$k]=${PIPESTATUS[0]}
  echo "[run_all] $k 종료 rc=${RC[$k]} ($((SECONDS - t0))s)"
done

# ---- 3) 집계 + 반출 -------------------------------------------------------------
"$PY" "$KIT_DIR/common/report.py" --aggregate "$REPORT_DIR" --expect $KERNELS
AGG_RC=$?

STAMP=$(date +%Y%m%d-%H%M%S)
TAR="$KIT_DIR/k3-sm80-kernel-report-$STAMP.tgz"
tar czf "$TAR" -C "$(dirname "$REPORT_DIR")" "$(basename "$REPORT_DIR")"
echo "[run_all] 반출 tar: $TAR (파일 반출 불가 환경이면 아래 RELAY 블록으로 회신)"
[[ ${#MISSING[@]} -gt 0 ]] && echo "[run_all] 미저작 하네스: ${MISSING[*]}"

# ---- 3.5) 화면 전달용 압축 요약 (RELAY) — 항상 출력, 실패해도 rc 에 영향 없음 ------
"$PY" "$KIT_DIR/common/relay.py" "$REPORT_DIR" || true

FAIL=0
PRECOND=()
for k in "${!RC[@]}"; do
  [[ ${RC[$k]} -eq 1 ]] && FAIL=1
  [[ ${RC[$k]} -eq 2 ]] && PRECOND+=("$k")
  [[ ${RC[$k]} -ge 3 ]] && FAIL=1
done
[[ $AGG_RC -eq 1 ]] && FAIL=1
if [[ $FAIL -eq 1 ]]; then
  echo "[run_all] 종합: FAIL 존재 — kernel-report/SUMMARY.md 확인 (FAIL/CRASH 로그도 S3 입력물이다)"
  exit 1
fi
if [[ ${#PRECOND[@]} -gt 0 ]]; then
  echo "[run_all] 종합: FAIL 없음 — 단, 사전조건 미충족(rc=2)으로 실행되지 않은 커널 ${#PRECOND[@]}개: ${PRECOND[*]}"
  echo "[run_all]        이 결과를 성공으로 읽지 말 것 — 해당 커널 로그 확인 후 재실행 필요"
  exit 2
fi
echo "[run_all] 종합: FAIL 없음 (${DRY:+dry-run 상한=INCOMPLETE})"
exit 0
