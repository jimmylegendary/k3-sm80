#!/usr/bin/env bash
# =============================================================================
# probe_35922.sh — vLLM issue #35922 (SM80 Marlin MoE crash) 재현 + 원인 특정 프로브
#
# k3-sm80 Phase 1 · 2026-07-28 · 러너: 사내 A100 노드 (호스트에는 GPU 없음)
# 근거: recon/vllm-upstream.md (1)(3)(5)(6) — 커밋/체크포인트 고정은 RUNBOOK.md §0
#
# Arms:
#   A  W4A16 INT4 (RedHatAI/Qwen3-30B-A3B-quantized.w4a16) + 기본(auto→Marlin)
#      → 기대: cudaErrorIllegalAddress 재현 (issue #35922 본문과 동일)
#   B  arm A 동일 조건 + compute-sanitizer --tool memcheck
#      → 기대: marlin 커널의 invalid read/write 위치 특정 (#36889의 c_tmp OOB 가설 검증)
#   C  W4A16 INT4 + kernel_config.moe_backend="triton" (우회 — 스펙 Phase 2 1안)
#      → 기대: 정상 생성 (recon (6): TritonExperts capability>=7.5)
#   D  openai/gpt-oss-20b (MXFP4 MoE) + 기본(auto)
#      → SM80에서 auto가 고르는 MXFP4 백엔드 + 동작 여부 기록 (K3 원본 경로 대리)
#   E  openai/gpt-oss-20b + kernel_config.moe_backend="marlin" (강제)
#      → 기대 분기: 동작 / no-kernel-image / IMA — recon (5) "bf16-act FP4 sm80
#        인스턴스화됨" 가설의 실증. K3 MXFP4 직행 가능성 판정의 핵심 입력.
#
# 사용:
#   bash probe_35922.sh                    # 전 arm
#   PROBE_ARMS="A C" bash probe_35922.sh   # 선택 실행
#   PROBE_SKIP_DOWNLOAD=1 ...              # HF 캐시가 이미 있을 때
# 판정 기준(요약): SUMMARY.txt 의 PROBE35922| 라인 → DECISION-TREE.md 로.
# =============================================================================
set -u -o pipefail

# ----------------------------- 설정 (env 오버라이드 가능) --------------------
MODEL_W4A16="${PROBE_MODEL_W4A16:-RedHatAI/Qwen3-30B-A3B-quantized.w4a16}"
REV_W4A16="${PROBE_REV_W4A16:-1c842a5000febac4b335c94a04789a9d4756a18e}"
MODEL_MXFP4="${PROBE_MODEL_MXFP4:-openai/gpt-oss-20b}"
REV_MXFP4="${PROBE_REV_MXFP4:-6cee5e81ee83917806bbde320786a8fb61efebee}"
PROBE_ARMS="${PROBE_ARMS:-A B C D E}"
MAX_MODEL_LEN="${PROBE_MAX_MODEL_LEN:-16384}"     # issue 재현 조건 (본문이 16384 사용)
GPU_MEM_UTIL="${PROBE_GPU_MEM_UTIL:-0.85}"        # issue 본문 값
TIMEOUT_ARM="${PROBE_TIMEOUT_ARM:-3600}"          # arm당 초 (다운로드 제외)
TIMEOUT_SAN="${PROBE_TIMEOUT_SAN:-14400}"         # sanitizer arm (10~100x 느림)
RESULTS_ROOT="${PROBE_RESULTS_DIR:-$(pwd)/results}"

TS="$(date +%Y%m%d_%H%M%S)"
RES="$RESULTS_ROOT/probe_35922_$TS"
mkdir -p "$RES/scripts" "$RES/logs" || { echo "FATAL: 결과 디렉터리 생성 실패: $RES"; exit 1; }
SUMMARY="$RES/SUMMARY.txt"
: > "$SUMMARY"

log() { echo "[probe_35922] $*" | tee -a "$RES/logs/driver.log"; }
summary() { echo "$*" | tee -a "$SUMMARY"; }

# ----------------------------- 사전조건 체크 ---------------------------------
precheck() {
    local fail=0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "FATAL: nvidia-smi 없음 — GPU 노드가 아님"; exit 1
    fi
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    log "GPU compute_cap = ${cc:-unknown}"
    if [ "$cc" != "8.0" ]; then
        if [ "${PROBE_ALLOW_NON_SM80:-0}" != "1" ]; then
            log "FATAL: SM80(A100)이 아님(cc=$cc). 이 프로브는 SM80 전용 판정임."
            log "       강행하려면 PROBE_ALLOW_NON_SM80=1 (판정 비교 불가 주의)"; exit 1
        fi
        log "WARN: non-SM80에서 강행 (판정은 SM80 기준과 비교 불가)"
    fi
    if ! python -c "import vllm" >/dev/null 2>&1; then
        log "FATAL: vllm import 실패 — RUNBOOK §2 Stage 1 venv 활성화 여부 확인"; exit 1
    fi
    python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$RES/logs/driver.log"
    # 디스크: HF 캐시 대상 파티션 여유
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    mkdir -p "$hf_home" 2>/dev/null
    local free_gb
    free_gb=$(df -BG --output=avail "$hf_home" 2>/dev/null | tail -1 | tr -dc '0-9')
    log "HF_HOME=$hf_home 여유 ${free_gb:-?}GB (필요 ~35GB)"
    if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 35 ]; then
        log "WARN: 디스크 여유 35GB 미만 — 다운로드 실패 가능"
    fi
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
        SANITIZER_BIN="$(ls /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | head -1 || true)"
        if [ -z "${SANITIZER_BIN:-}" ]; then
            log "WARN: compute-sanitizer 없음 — arm B 자동 스킵"
            SANITIZER_BIN=""
        fi
    else
        SANITIZER_BIN="$(command -v compute-sanitizer)"
    fi
    log "compute-sanitizer: ${SANITIZER_BIN:-<none>}"
    return $fail
}

# ----------------------------- 환경 스냅샷 -----------------------------------
snapshot_env() {
    {
        echo "== date =="; date -Is
        echo "== nvidia-smi =="; nvidia-smi
        echo "== driver/cc =="; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
        echo "== nvcc =="; nvcc --version 2>&1 || echo "(nvcc 없음 — wheel 경로면 정상)"
        echo "== python =="; python --version
        echo "== pip freeze (핵심) =="; pip freeze 2>/dev/null | grep -Ei "vllm|torch|triton|flashinfer|flash-linear|compressed|numpy" || true
        echo "== torch cuda =="; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" 2>&1
    } > "$RES/logs/env.txt" 2>&1
    pip freeze > "$RES/logs/pip-freeze-full.txt" 2>/dev/null
}

# ----------------------------- 재현 스크립트 생성 ----------------------------
write_repro_py() {
    cat > "$RES/scripts/repro.py" <<'PYEOF'
# 자동 생성: probe_35922.sh — vLLM offline 1-shot 재현기
# 크래시 지점: issue #35922 기준 profile_run() (엔진 초기화 중) → 생성까지 가면 통과
import argparse
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 단일 프로세스: sanitizer 관측 보장
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--revision", required=True)
p.add_argument("--moe-backend", default="auto")
p.add_argument("--max-model-len", type=int, default=16384)
p.add_argument("--gpu-mem-util", type=float, default=0.85)
args = p.parse_args()

print(f"PROBE_ARGS model={args.model} rev={args.revision} moe_backend={args.moe_backend} "
      f"max_model_len={args.max_model_len}", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

kwargs = {}
if args.moe_backend != "auto":
    # KernelConfig.moe_backend — vllm/config/kernel.py (main d223c900에서 확인)
    kwargs["kernel_config"] = {"moe_backend": args.moe_backend}

llm = LLM(
    model=args.model,
    revision=args.revision,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_mem_util,
    enforce_eager=True,   # 크래시는 eager profile_run에서 발생(이슈 본문) + sanitizer 친화
    **kwargs,
)
print("PROBE_ENGINE_INIT_OK", flush=True)

out = llm.generate(
    ["The capital of France is", "1+1="],
    SamplingParams(max_tokens=32, temperature=0.0),
)
for i, o in enumerate(out):
    print(f"PROBE_GEN[{i}]: {o.outputs[0].text!r}", flush=True)
print("PROBE_INNER_OK", flush=True)
PYEOF
}

# ----------------------------- 다운로드 (프로브 타임아웃과 분리) -------------
prefetch() {
    local model="$1" rev="$2"
    log "prefetch: $model @ $rev"
    python - "$model" "$rev" <<'PYEOF' 2>&1 | tee -a "$RES/logs/prefetch.log"
import sys
from huggingface_hub import snapshot_download
model, rev = sys.argv[1], sys.argv[2]
path = snapshot_download(model, revision=rev)
print("PREFETCH_OK", model, "->", path)
PYEOF
    # PIPESTATUS[0] = python 종료코드
    return "${PIPESTATUS[0]}"
}

# ----------------------------- arm 실행 + 판정 -------------------------------
# classify <logfile> <rc>  → verdict 문자열 echo
classify() {
    local logf="$1" rc="$2"
    if grep -q "PROBE_INNER_OK" "$logf"; then echo "SUCCESS"; return; fi
    if grep -qiE "illegal memory access|cudaErrorIllegalAddress|an illegal instruction" "$logf"; then
        echo "CRASH_IMA"; return; fi
    if grep -qiE "no kernel image is available" "$logf"; then echo "CRASH_NO_KERNEL_IMAGE"; return; fi
    if grep -qiE "out of memory|CUDA out of memory" "$logf"; then echo "FAIL_OOM"; return; fi
    if [ "$rc" = "124" ]; then echo "TIMEOUT"; return; fi
    if grep -q "PROBE_ENGINE_INIT_OK" "$logf"; then echo "FAIL_AFTER_INIT_rc=$rc"; return; fi
    echo "FAIL_OTHER_rc=$rc"
}

# extract_backend <logfile> — 오라클이 고른 MoE 백엔드 라인 수집(정보용)
extract_backend() {
    grep -iE "moe.*backend|backend.*moe|marlin|Mxfp4MoeBackend|WNA16MoEBackend|using .* backend" "$1" \
        | head -20 || true
}

run_arm() {
    # $1=arm이름 $2=model $3=rev $4=moe_backend $5=timeout $6=sanitizer(0/1)
    local arm="$1" model="$2" rev="$3" moe="$4" tmo="$5" san="$6"
    local logf="$RES/logs/arm_${arm}.log"
    local t0 t1 rc
    log "=== ARM $arm 시작: model=$model moe_backend=$moe sanitizer=$san (timeout ${tmo}s) ==="
    t0=$(date +%s)
    if [ "$san" = "1" ]; then
        if [ -z "${SANITIZER_BIN:-}" ]; then
            summary "PROBE35922|arm=$arm|verdict=SKIPPED_NO_SANITIZER|model=$model"
            return 0
        fi
        CUDA_LAUNCH_BLOCKING=1 timeout "$tmo" \
            "$SANITIZER_BIN" --tool memcheck --launch-timeout 120 --error-exitcode 99 \
            --log-file "$RES/logs/sanitizer_${arm}.log" -- \
            python "$RES/scripts/repro.py" --model "$model" --revision "$rev" \
                --moe-backend "$moe" --max-model-len "$MAX_MODEL_LEN" \
                --gpu-mem-util "$GPU_MEM_UTIL" \
            > "$logf" 2>&1
        rc=$?
    else
        timeout "$tmo" \
            python "$RES/scripts/repro.py" --model "$model" --revision "$rev" \
                --moe-backend "$moe" --max-model-len "$MAX_MODEL_LEN" \
                --gpu-mem-util "$GPU_MEM_UTIL" \
            > "$logf" 2>&1
        rc=$?
    fi
    t1=$(date +%s)
    local verdict
    verdict="$(classify "$logf" "$rc")"
    # sanitizer 요약 추출 (에러가 잡혔는데 앱이 완주하면 verdict에 표식을 남긴다 —
    # memcheck는 invalid access를 보고만 하고 계속 진행할 수 있음)
    local san_note=""
    if [ "$san" = "1" ] && [ -f "$RES/logs/sanitizer_${arm}.log" ]; then
        local nerr
        nerr=$(grep -c "^========= Invalid" "$RES/logs/sanitizer_${arm}.log" 2>/dev/null)
        nerr=${nerr:-0}
        grep -E "^========= " "$RES/logs/sanitizer_${arm}.log" | head -40 \
            > "$RES/logs/sanitizer_${arm}_head.txt" 2>/dev/null || true
        san_note="|sanitizer_invalid=$nerr"
        if [ "$nerr" -gt 0 ] 2>/dev/null; then
            verdict="${verdict}_WITH_SAN_ERRORS"
        fi
    fi
    summary "PROBE35922|arm=$arm|verdict=$verdict|rc=$rc|sec=$((t1-t0))|model=$model|moe_backend=$moe$san_note|log=logs/arm_${arm}.log"
    {
        echo "--- ARM $arm backend 선택 로그 발췌 ---"
        extract_backend "$logf"
    } >> "$RES/logs/backend_lines.txt"
    log "ARM $arm 종료: $verdict (rc=$rc, $((t1-t0))s)"
    # GPU 상태가 오염됐을 수 있으므로(IMA 후) 다음 arm 전 5초 대기
    sleep 5
    return 0
}

# ----------------------------- 메인 -----------------------------------------
log "결과 디렉터리: $RES"
precheck
snapshot_env
write_repro_py

summary "PROBE35922|meta|date=$TS|w4a16=$MODEL_W4A16@$REV_W4A16|mxfp4=$MODEL_MXFP4@$REV_MXFP4|arms=$PROBE_ARMS"

if [ "${PROBE_SKIP_DOWNLOAD:-0}" != "1" ]; then
    case " $PROBE_ARMS " in *" A "*|*" B "*|*" C "*)
        prefetch "$MODEL_W4A16" "$REV_W4A16" || { summary "PROBE35922|meta|verdict=PREFETCH_FAIL_W4A16"; log "W4A16 prefetch 실패 — arm A/B/C 스킵됨"; PROBE_ARMS="${PROBE_ARMS//A/}"; PROBE_ARMS="${PROBE_ARMS//B/}"; PROBE_ARMS="${PROBE_ARMS//C/}"; } ;;
    esac
    case " $PROBE_ARMS " in *" D "*|*" E "*)
        prefetch "$MODEL_MXFP4" "$REV_MXFP4" || { summary "PROBE35922|meta|verdict=PREFETCH_FAIL_MXFP4"; log "MXFP4 prefetch 실패 — arm D/E 스킵됨"; PROBE_ARMS="${PROBE_ARMS//D/}"; PROBE_ARMS="${PROBE_ARMS//E/}"; } ;;
    esac
fi

for arm in $PROBE_ARMS; do
    case "$arm" in
        A) run_arm A "$MODEL_W4A16" "$REV_W4A16" auto   "$TIMEOUT_ARM" 0 ;;
        B) run_arm B "$MODEL_W4A16" "$REV_W4A16" auto   "$TIMEOUT_SAN" 1 ;;
        C) run_arm C "$MODEL_W4A16" "$REV_W4A16" triton "$TIMEOUT_ARM" 0 ;;
        D) run_arm D "$MODEL_MXFP4" "$REV_MXFP4" auto   "$TIMEOUT_ARM" 0 ;;
        E) run_arm E "$MODEL_MXFP4" "$REV_MXFP4" marlin "$TIMEOUT_ARM" 0 ;;
        *) log "알 수 없는 arm: $arm"
           summary "PROBE35922|arm=$arm|verdict=UNKNOWN_ARM|note=오타 의심 — 유효 arm은 A B C D E" ;;
    esac
done

# dmesg Xid (권한 없으면 무시)
dmesg 2>/dev/null | grep -i "xid" | tail -20 > "$RES/logs/dmesg_xid.txt" || true

log "=== 완료. 요약: ==="
cat "$SUMMARY" | tee -a "$RES/logs/driver.log"
log "판정 → DECISION-TREE.md §1. 반출: tar czf probe_35922_$TS.tgz -C $RESULTS_ROOT probe_35922_$TS"
