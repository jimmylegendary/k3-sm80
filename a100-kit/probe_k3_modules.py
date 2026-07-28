#!/usr/bin/env python3
# =============================================================================
# probe_k3_modules.py — Kimi K3 모듈(KDA / Gated-MLA+AttnRes / LatentMoE)의
#                       SM80 초기화·forward 경로 분리 검증 (vLLM kimi-k3 브랜치 전용)
#
# k3-sm80 Phase 1 · 2026-07-28 · 러너: 사내 A100 노드
#
# 전제 (RUNBOOK.md §0/§3):
#   - vLLM kimi-k3 브랜치 @ 0498dc7ea9204f1863b46515816f5d8d458e01e7 소스빌드 venv
#   - `inference-optimization/Kimi-K3-0.18B`는 HF에 존재하지 않음(2026-07-28 HF API 401 확인)
#     → 대안: moonshotai/Kimi-K3 @ 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 에서
#       config/tokenizer/remote-code(소형 파일)만 받고, text_config를 축소한 뒤
#       vLLM `--load-format dummy`(랜덤 가중치)로 e2e를 돌린다. 가중치 다운로드 0바이트.
#   - 축소 config는 실물 config.json을 로드해 "필드 구조는 그대로, 크기만 축소"
#     (구조 드리프트에 강함). per-head 기하(head_dim 128, MLA nope128/rope64/v128,
#     kv_lora 512, q_lora 1536)는 실물값 유지 — 커널이 실제로 타는 차원을 보존.
#   - quantization_config는 제거(bf16 dummy). MXFP4 Marlin 경로는 probe_35922의
#     소관이며, 여기 섞으면 #35922 크래시가 모듈 검증을 가린다.
#
# Stages:
#   S0  환경/브랜치 확인 (kimi_k3 패키지 import — wheel이면 여기서 FAIL이 정상)
#   S1  KDA 백엔드 해석 단위검증 (resolve_kda_prefill_backend 등, 엔진 불필요)
#   S2  ModelRegistry 등재 확인
#   S3  tiny-dummy e2e: 모델 파일 준비 → LLM 초기화 → greedy 생성 → 백엔드 로그 수집
#
# 종료코드: 0=전부 PASS · 10=S0 실패(브랜치 아님/임포트) · 11=S1 불일치
#           12=S3 엔진 초기화 실패 · 13=S3 생성 실패 · 2=기타 FAIL
# 출력: PROBE_K3M|stage=..|verdict=..|... (기계판독), 사람용 로그는 stderr/stdout 혼재
# =============================================================================

import argparse
import io
import json
import logging
import os
import shutil
import sys
import time
import traceback

# vLLM import 전에 걸어야 하는 env (엔진 인프로세스 = 로그/디버깅 관측 보장)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

HF_REPO = "moonshotai/Kimi-K3"
HF_REV = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"  # recon/k3-config.md 고정 커밋
PIN_BRANCH_COMMIT = "0498dc7ea9204f1863b46515816f5d8d458e01e7"
NONEXISTENT_TINY = "inference-optimization/Kimi-K3-0.18B"  # 존재하지 않음(문서화용)

RESULTS: list[str] = []


def emit(stage: str, verdict: str, **kv):
    extra = "".join(f"|{k}={v}" for k, v in kv.items())
    line = f"PROBE_K3M|stage={stage}|verdict={verdict}{extra}"
    RESULTS.append(line)
    print(line, flush=True)


def fail_exit(code: int, stage: str, verdict: str, **kv):
    emit(stage, verdict, **kv)
    print("\n==== PROBE_K3M SUMMARY ====")
    for r in RESULTS:
        print(r)
    sys.exit(code)


# ----------------------------------------------------------------------------
# S0 — 환경/브랜치
# ----------------------------------------------------------------------------
def stage_s0():
    try:
        import torch
    except Exception as e:
        fail_exit(10, "S0", "FAIL_TORCH_IMPORT", error=repr(e))
    if not torch.cuda.is_available():
        fail_exit(10, "S0", "FAIL_NO_CUDA")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    if cap != (8, 0) and os.environ.get("PROBE_ALLOW_NON_SM80") != "1":
        fail_exit(10, "S0", "FAIL_NOT_SM80", capability=str(cap).replace(" ", ""),
                  gpu=name.replace(" ", "_"))
    try:
        import vllm
    except Exception as e:
        fail_exit(10, "S0", "FAIL_VLLM_IMPORT", error=repr(e))
    try:
        import vllm.models.kimi_k3  # noqa: F401  (kimi-k3 브랜치에만 존재 — recon (4))
    except Exception as e:
        fail_exit(
            10, "S0", "FAIL_NOT_K3_BRANCH",
            vllm_version=vllm.__version__,
            hint="wheel(0.26.0)/main 빌드에는 K3 모델 코드가 없음 — RUNBOOK §3 Stage 2 venv에서 실행",
            error=repr(e),
        )
    emit("S0", "PASS", vllm_version=vllm.__version__, gpu=name.replace(" ", "_"),
         capability=str(cap).replace(" ", ""), torch=torch.__version__,
         pinned_branch=PIN_BRANCH_COMMIT[:12])


# ----------------------------------------------------------------------------
# S1 — KDA 백엔드 해석 (kda.py @ 브랜치: resolve_kda_prefill_backend L148,
#       is_flashkda_supported L85, is_fused_kda_decode_supported L58)
# ----------------------------------------------------------------------------
def stage_s1():
    import torch
    try:
        from vllm.models.kimi_k3.nvidia.kda import (
            is_flashkda_supported,
            is_fused_kda_decode_supported,
            resolve_kda_prefill_backend,
        )
    except Exception as e:
        fail_exit(11, "S1", "FAIL_IMPORT_KDA_FUNCS", error=repr(e),
                  hint="브랜치가 함수를 이동/개명했을 수 있음 — 커밋 재고정 필요(DECISION-TREE §3)")

    ok = True
    # K3 실제 파라미터: head_dim=128, bf16, gate_lower_bound=-5.0 (recon/k3-config.md)
    flashkda = is_flashkda_supported(128, torch.bfloat16, -5.0)
    if flashkda:
        ok = False
        emit("S1", "UNEXPECTED_FLASHKDA_SUPPORTED",
             note="SM80에서 True면 게이트 로직 변경 — 재조사 필요")
    resolved = None
    try:
        resolved = resolve_kda_prefill_backend("auto", 128, torch.bfloat16, -5.0)
    except Exception as e:
        ok = False
        emit("S1", "FAIL_RESOLVE_AUTO", error=repr(e))
    if resolved != "triton":
        ok = False
        emit("S1", "FAIL_AUTO_NOT_TRITON", resolved=resolved)

    # "flashkda" 명시 강제는 SM80에서 RuntimeError가 정상 (recon (2))
    forced_raises = False
    try:
        resolve_kda_prefill_backend("flashkda", 128, torch.bfloat16, -5.0)
    except RuntimeError:
        forced_raises = True
    except Exception as e:
        emit("S1", "WARN_FORCED_FLASHKDA_WRONG_EXC", error=repr(e))
    if not forced_raises:
        emit("S1", "WARN_FORCED_FLASHKDA_NO_RAISE",
             note="SM80에서 flashkda 강제가 조용히 통과 — 게이트 완화 여부 확인")

    # decode: KDA 96 heads/128 dim/conv4/bf16 — SM80은 False(→ Triton fused_recurrent 폴백)
    decode_cuda = is_fused_kda_decode_supported(96, 128, 4, 0, torch.bfloat16, torch.bfloat16)
    if decode_cuda:
        ok = False
        emit("S1", "UNEXPECTED_FUSED_DECODE_SUPPORTED",
             note="SM80 빌드에 fused_kda_decode가 있으면 안 됨(SM90+ 전용)")

    if ok:
        emit("S1", "PASS", prefill_auto=resolved, flashkda_supported=flashkda,
             fused_decode_supported=decode_cuda, forced_flashkda_raises=forced_raises)
    else:
        fail_exit(11, "S1", "FAIL", note="위 개별 라인 참조")


# ----------------------------------------------------------------------------
# S2 — 레지스트리
# ----------------------------------------------------------------------------
def stage_s2():
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        archs = ModelRegistry.get_supported_archs()
        present = "KimiK3ForConditionalGeneration" in archs
        emit("S2", "PASS" if present else "FAIL_NOT_REGISTERED",
             kimi_k3_registered=present,
             kimi_linear_registered="KimiLinearForCausalLM" in archs)
    except Exception as e:
        emit("S2", "WARN_REGISTRY_API_DRIFT", error=repr(e))  # 치명 아님 — S3가 실질 검증


# ----------------------------------------------------------------------------
# S3 — tiny dummy e2e
# ----------------------------------------------------------------------------
TINY_OVERRIDES_NOTE = """축소 규칙(실물 9f62e4e9 config 기반 mutate):
  hidden 7168→1024 · layers 93→8([KDA,KDA,KDA,MLA]x2 패턴 유지: kda=[1,2,3,5,6,7], full=[4,8])
  heads 96→8(KDA/MLA 공통) · experts 896→32(top16 유지) · moe_inter 3072→256 · latent 3584→512
  dense inter 33792→2048 · attn_res_block_size 12→4(8층 내 경계 2회 확보) · vocab 163840 유지(실물 tokenizer)
  유지: head_dim 128, qk_nope 128, qk_rope 64, v_head 128, kv_lora 512, q_lora 1536,
        situ betas, gate_lower_bound -5.0, use_full_rank_gate, noaux_tc/sigmoid router,
        num_shared_experts 2, first_k_dense_replace 1, mla_use_nope/output_gate, vision_config 원본 그대로
  제거: quantization_config (bf16 dummy — MXFP4 경로는 probe_35922 소관)"""


def build_tiny_model_dir(workdir: str) -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(workdir, exist_ok=True)
    # 가중치(*.safetensors*) 제외 전부: remote code(.py), tokenizer(tiktoken.model 2.8MB),
    # preprocessor/generation/config json — 2026-07-28 tree 확인 기준 소형 파일뿐.
    snapshot_download(
        HF_REPO,
        revision=HF_REV,
        local_dir=workdir,
        allow_patterns=["*.py", "*.json", "*.model", "LICENSE"],
    )
    cfg_path = os.path.join(workdir, "config.json")
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"config.json 다운로드 실패: {workdir}")
    shutil.copy(cfg_path, os.path.join(workdir, "config.full.json"))  # 원본 보존

    with open(cfg_path) as f:
        cfg = json.load(f)

    tc = cfg["text_config"]
    tc["hidden_size"] = 1024
    tc["num_hidden_layers"] = 8
    tc["num_attention_heads"] = 8
    tc["num_key_value_heads"] = 8
    tc["intermediate_size"] = 2048
    tc["moe_intermediate_size"] = 256
    tc["routed_expert_hidden_size"] = 512
    tc["num_experts"] = 32
    tc["num_experts_per_token"] = 16
    tc["attn_res_block_size"] = 4
    tc["max_position_embeddings"] = 8192
    la = tc["linear_attn_config"]
    la["num_heads"] = 8
    la["kda_layers"] = [1, 2, 3, 5, 6, 7]
    la["full_attn_layers"] = [4, 8]
    tc.pop("quantization_config", None)
    # vision projector는 text_hidden_size로 투영한다 — 축소한 text hidden과 반드시 동기화
    # (원본 7168인 채로 두면 projector shape 불일치로 엔진 초기화 실패)
    if "vision_config" in cfg and isinstance(cfg["vision_config"], dict):
        cfg["vision_config"]["text_hidden_size"] = tc["hidden_size"]

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return workdir


def stage_s3(args):
    t0 = time.time()
    try:
        model_dir = build_tiny_model_dir(args.workdir)
    except Exception as e:
        traceback.print_exc()
        fail_exit(2, "S3", "FAIL_MODEL_PREP", error=repr(e),
                  hint="HF 접근/디스크 확인. revision pin 소실 시 recon/k3-config.md 재확인")
    emit("S3", "MODEL_PREP_OK", dir=model_dir, sec=round(time.time() - t0, 1))
    print(TINY_OVERRIDES_NOTE)

    # vLLM 로그 캡처(백엔드 선택 라인 수집용)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logging.getLogger("vllm").addHandler(handler)

    llm = None
    t1 = time.time()
    try:
        from vllm import LLM, SamplingParams

        kwargs = {}
        if args.mla_backend:
            kwargs["attention_backend"] = args.mla_backend
        llm = LLM(
            model=model_dir,
            trust_remote_code=True,             # tokenizer/config remote code (tokenization_kimi.py)
            load_format="dummy",                # 가중치 랜덤 초기화 — 다운로드 0
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=1024,
            max_num_seqs=4,
            gpu_memory_utilization=float(os.environ.get("PROBE_GPU_MEM_UTIL", "0.6")),
            tensor_parallel_size=1,
            # KDA prefill을 Triton으로 명시 고정 (recon (2): auto도 SM80→triton이나 명시 권장)
            additional_config={"kda_prefill_backend": os.environ.get("PROBE_KDA_PREFILL", "triton")},
            **kwargs,
        )
    except Exception as e:
        traceback.print_exc()
        _dump_backend_lines(buf)
        fail_exit(12, "S3", "FAIL_ENGINE_INIT", error=repr(e).replace("|", "/")[:400],
                  sec=round(time.time() - t1, 1),
                  hint="트레이스가 quantization/MoE라면 K3 코드의 mxfp4 전제 가능성 — DECISION-TREE §3")
    emit("S3", "ENGINE_INIT_OK", sec=round(time.time() - t1, 1))

    try:
        prompts = ["The capital of France is", "서울에서 부산까지"]
        outs = llm.generate(prompts, SamplingParams(max_tokens=int(args.max_tokens), temperature=0.0))
        ntok = [len(o.outputs[0].token_ids) for o in outs]
        assert all(n > 0 for n in ntok), f"빈 생성: {ntok}"
        # dummy 가중치 → 텍스트는 무의미. 판정 = 크래시 없이 토큰 생산.
        emit("S3", "GENERATE_OK", tokens=str(ntok).replace(" ", ""),
             note="dummy-weights라 텍스트 품질 무의미(무한반복 정상)")
    except Exception as e:
        traceback.print_exc()
        _dump_backend_lines(buf)
        fail_exit(13, "S3", "FAIL_GENERATE", error=repr(e).replace("|", "/")[:400])

    _dump_backend_lines(buf)
    emit("S3", "PASS", total_sec=round(time.time() - t0, 1),
         mla_backend_forced=args.mla_backend or "auto")


def _dump_backend_lines(buf: io.StringIO):
    print("\n==== vLLM 백엔드 선택 로그 발췌 (S3_BACKEND_REPORT) ====")
    keys = ("backend", "Backend", "KDA", "MLA", "FlashKDA", "attention", "Using")
    seen = set()
    for line in buf.getvalue().splitlines():
        if any(k in line for k in keys):
            s = line.strip()[:240]
            if s not in seen:
                seen.add(s)
                print("S3_BACKEND_REPORT|", s)
    print("==== (끝) ====\n")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", default=os.path.join(os.getcwd(), "results", "k3_tiny_model"),
                   help="tiny 모델 파일 준비 디렉터리")
    p.add_argument("--stages", default="S0 S1 S2 S3", help='예: "S0 S1" 또는 "S3"')
    p.add_argument("--mla-backend", default=os.environ.get("PROBE_ATTN_BACKEND", ""),
                   help="attention_backend 강제 (예: TRITON_MLA). 빈값=auto")
    p.add_argument("--max-tokens", default=16)
    args = p.parse_args()

    stages = args.stages.split()
    if "S0" in stages:
        stage_s0()
    else:
        # S0를 건너뛰어도 브랜치 임포트는 필수 전제라 최소 확인
        try:
            import vllm.models.kimi_k3  # noqa: F401
        except Exception as e:
            fail_exit(10, "S0", "FAIL_NOT_K3_BRANCH", error=repr(e))
    if "S1" in stages:
        stage_s1()
    if "S2" in stages:
        stage_s2()
    if "S3" in stages:
        stage_s3(args)

    print("\n==== PROBE_K3M SUMMARY ====")
    for r in RESULTS:
        print(r)
    bad = [r for r in RESULTS if "|verdict=FAIL" in r or "|verdict=UNEXPECTED" in r]
    sys.exit(2 if bad else 0)


if __name__ == "__main__":
    main()
