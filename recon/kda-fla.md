# KDA 커널 생태계 정찰 — FLA / FlashKDA / vLLM 의존관계 (SM80 관점)

- 작성: 2026-07-28 (웹 스냅샷 기준일. vLLM은 주 단위로 변하므로 아래 커밋/버전 고정 참조)
- 스코프: Kimi K3의 **KDA(Kimi Delta Attention) 레이어**만. MoE MXFP4 경로·MLA는 별도 recon.

## 0. 판정 요약

**SM80(A100)에서 FLA-KDA 성립 가능성: 높음(조건부).**
vLLM main의 KDA 실행 경로는 prefill(`chunk_kda_with_fused_gate`)·decode(`fused_recurrent_kda`)·conv(`causal_conv1d_*`) 전부가 **벤더링된 순수 Triton 커널**로, 아치 게이팅 코드(`sm90`/`capability` 분기)가 0건이다. SM90 전용인 FlashKDA(CUTLASS)는 vLLM main에서 아예 사용되지 않고, FLA 직사용 시에도 자동 디스패치가 거부되어 Triton 경로로 폴백된다. 즉 **A100에서 컴파일·구조적 장벽은 없음**. 단, FLA 공식 CI는 H100 전용이라 A100 수치검증은 업스트림에 없다 → Phase 1 프로브에서 FLA 단위테스트 대조 필수.

**K3 KDA state 크기 (1 seq, TP=1, vLLM 저장 기준)**:
- per-layer recurrent state = 96 heads × 128 × 128 × **fp32(vLLM 하드코딩)** = **6,291,456 B = 6.0 MiB**
- 69 KDA layers 합 = **414.0 MiB** (+ conv state bf16 14.6 MiB) ≈ **428.6 MiB/seq ≈ 0.45 GB/seq**

---

## 1. FLA(flash-linear-attention)의 KDA 구현

저장소: https://github.com/fla-org/flash-linear-attention
고정: main HEAD `9c8e42e762fce087c27b673af4922795d9edb85e` (2026-07-27) · 최신 릴리스 **v0.5.1** (2026-06-18) · KDA 최초 도입 **v0.4.0** (2025-10-27, Kimi Linear 공개와 동시) ([releases](https://github.com/fla-org/flash-linear-attention/releases))

### 1.1 구성 (`fla/ops/kda/`)
| 파일 | 역할 |
|---|---|
| `chunk.py` | `chunk_kda` — **prefill(청크 병렬)**. autograd fn + `@dispatch('kda')` 백엔드 훅 |
| `chunk_fwd.py` / `chunk_intra.py` / `chunk_bwd.py` / `wy_fast.py` / `chunk_intra_token_parallel.py` | 청크 fwd/bwd Triton 커널 (intra/inter, WY representation) |
| `fused_recurrent.py` | `fused_recurrent_kda` — **decode(토큰 단위 recurrent)** |
| `gate.py` | `fused_kda_gate` (full-rank gate, `use_beta_sigmoid_in_kernel`) |
| `backends/` | `flash_kda.py`(CUTLASS), `tilelang/`, `triton_ascend/` — 기본은 Triton |
| `naive.py` | 수치 대조용 레퍼런스 |

### 1.2 순수 Triton 근거
- 커널 전부 `@triton.jit` + `triton.autotune`; C++/CUDA 확장 없음. vLLM에 복사된 동일 계열 코드([vllm kda.py](https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/kda.py))의 import는 `vllm.triton_utils`(tl, triton)뿐이며 `sm_90`/`capability` 문자열 **0건**(직접 grep 확인).
- FLA README: "All implementations are **platform-agnostic and verified on NVIDIA, AMD, and Intel hardware**" ([README](https://github.com/fla-org/flash-linear-attention/blob/main/README.md)).

### 1.3 SM80(A100) 실행 근거 (직접·간접)
- **직접**: [issue #625](https://github.com/fla-org/flash-linear-attention/issues/625) — 사용자가 **A100 + torch 2.9 + Triton 3.5**로 Kimi-Linear-48B를 FLA로 구동하다 `fla.kda.gate` crash 보고(2025-10-30). 추적 결과 **H200에서도 동일 재현** → 아치 문제가 아니라 gptqmodel 래퍼×Triton autotuner 상호작용으로 판명, closed. 즉 A100은 실제로 굴려지는 경로이고, 해당 crash는 SM80 고유 문제가 아님.
- **간접**: [PR #868](https://github.com/fla-org/flash-linear-attention/pull/868) ROCm용 KDA 커널 지원·튜닝 머지 — Hopper 전용 기능(TMA/WGMMA)에 묶여 있지 않다는 방증. Triton 커널의 chunk 크기 autotune 후보(BT∈{32,64,128})는 A100 smem 164KB 내에서 선택 가능.
- **한계**: FLA 메인테이너 발언 "our CI only have H100"(#625 코멘트) — **A100 CI 부재**. SM80 수치 정합은 업스트림이 보증하지 않는다.

### 1.4 prefill/decode 커버리지
- **prefill**: `chunk_kda(q,k,v,g,beta, initial_state=h0, output_final_state=True, cu_seqlens=...)` — varlen(cu_seqlens) + **initial_state 이어받기** 지원 → chunked prefill/프리픽스 캐시 이어쓰기 가능 ([chunk.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py)).
- **decode**: `fused_recurrent_kda(...)` — 토큰 단위 상태 갱신, initial_state in-place 갱신 지원 ([fused_recurrent.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py)).
- 양쪽 모두 학습 backward까지 있으나(서빙엔 fwd만 필요) 우리는 fwd만 사용.

---

## 2. MoonshotAI/FlashKDA — 컴파일 타깃과 제약

저장소: https://github.com/MoonshotAI/FlashKDA (MIT)
고정: master HEAD `d2ff19a6a0c82f39f796f637ebd1c36090b1268f` (2026-05-26)

- **컴파일 타깃**: `setup.py` L19 `SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]` — **SM80 없음**. README L10 "Requirements: **SM90 and above**". CUTLASS 기반(csrc + cutlass 서브모듈). → **A100에서는 빌드 자체가 불가**, 고려 대상 아님.
- **커널 제약** (README): "Currently requires **K = V = 128**" (K3는 head_dim 128이라 충족); q/k/v/gate **bf16 전용**; state `[B|N, H, V, K]` (bf16/fp32); **inference forward(청크 prefill)만** — backward·decode 없음; varlen은 `cu_seqlens`.
- **디스패치**: `flash-linear-attention >= 0.5.0`의 `chunk_kda`에서 자동 디스패치([FLA PR #852](https://github.com/fla-org/flash-linear-attention/pull/852)). 거부 조건은 [fla/ops/kda/backends/flash_kda.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/backends/flash_kda.py) verifier: `torch.inference_mode()` 밖 / dtype≠bf16 / K≠128 / V≠128 / `use_gate_in_kernel=False` / `import flash_kda` 실패(→A100은 여기서 항상 탈락) 시 **Triton 폴백**. 명시적 opt-out은 `FLA_FLASH_KDA=0`.
- 성능 참고: H20에서 FLA Triton 대비 prefill 1.72–2.22× ([MarkTechPost 2026-04-30](https://www.marktechpost.com/2026/04/30/moonshot-ai-open-sources-flashkda-cutlass-kernels-for-kimi-delta-attention-with-variable-length-batching-and-h20-benchmarks/), 저장소 `BENCHMARK_H20.md`/`BENCHMARK_GB200.md`). → SM80에서 못 쓰는 만큼의 prefill 상한 손실이 우리 기준선(순수 Triton)임을 의미.

---

## 3. K3의 KDA 수학 파라미터와 per-layer state 크기

출처: [moonshotai/Kimi-K3 `config.json`](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json) (main, 2026-07-28 조회; `text_config` = `KimiLinearForCausalLM`)

| 항목 | 값 |
|---|---|
| `num_hidden_layers` | 93 = **KDA 69** (`linear_attn_config.kda_layers`) + **MLA 24** (`full_attn_layers` = 4,8,…,88,92,**93**) |
| `linear_attn_config.num_heads` (H) | **96** |
| `linear_attn_config.head_dim` (d_k = d_v) | **128** |
| `linear_attn_config.short_conv_kernel_size` (K_conv) | 4 |
| gate | `use_full_rank_gate: true`, `gate_lower_bound: -5.0` |
| hidden_size | 7168 (KDA q/k/v proj는 96×128=12288로 확장) |
| 참고(MLA층) | heads 96, kv_lora 512, q_lora 1536, qk_nope 128, qk_rope 64, v_head 128 |
| 참고(MoE) | 896 routed(+2 shared), 16 active/tok, MXFP4 group 32 (routed experts만; self_attn·shared·lm_head는 bf16 유지) |

### state 크기 공식 (1 seq 기준)

vLLM main의 산정 코드 기준 ([mamba_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py) `kda_state_shape` L271, `kda_state_dtype` L131):

```
recurrent_state/layer = (H/TP, d_k, d_v)          dtype = float32 (vLLM 하드코딩: return (conv_dtype, torch.float32))
conv_state/layer      = (3·H·d_k/TP, K_conv−1)    dtype = mamba_cache_dtype(기본 model dtype = bf16)
```

**K3 수치 (TP=1, 1 seq):**

| 항목 | 원소 수 | bytes |
|---|---|---|
| recurrent /layer | 96×128×128 = 1,572,864 | fp32 **6,291,456 B = 6.00 MiB** (bf16 저장이었다면 3.0 MiB — vLLM은 fp32 고정) |
| conv /layer | 3×12288×3 = 110,592 | bf16 221,184 B = 216 KiB |
| **69 KDA layers 합** | — | recurrent **434,110,464 B = 414.0 MiB** + conv 15,261,696 B = 14.6 MiB → **총 ≈ 428.6 MiB/seq** |

- **TP=8 시 GPU당**: recurrent 0.75 MiB/layer → 전 레이어 51.8 MiB + conv 1.9 MiB ≈ **53.6 MiB/GPU/seq** (PP로 레이어 분할 시 stage 비례 감소).
- **배치 스케일**: batch 32 → 13.4 GiB(전체, fp32) / TP8 분산 시 GPU당 ~1.7 GiB. `mamba_cache_dtype`로는 conv만 바뀌고 recurrent fp32는 불변이므로 배치 상한 산정 시 fp32 기준 필수.
- **MLA 대비 감각치**: MLA KV = (512+64)×2B = 1,152 B/token/layer → KDA state 전체(≈449 MB)는 **24-layer MLA KV 약 16k 토큰 분량**. vLLM 블로그의 "KDA state ≫ 한 토큰 KV entry" 서술과 정합 — KDA 프리픽스 캐시가 물리 블록을 크게 잡고 fine-grained 매칭과 분리(copy-on-write)하는 이유 ([preview blog](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)).

---

## 4. vLLM ↔ FLA 의존 관계

기준: vLLM **v0.26.0** (2026-07-27 릴리스) / main 트리 2026-07-28 조회

- **vLLM은 fla pip 패키지에 의존하지 않는다.** FLA 커널을 **소스 벤더링**: `vllm/third_party/flash_linear_attention/ops/kda.py` (1,647줄; 파일 헤더에 "code copied from the flash-linear-attention project", MIT ⓒ Songlin Yang, Yu Zhang 명시). 구 경로 `vllm/model_executor/layers/fla/ops/kda`(v0.11.2 문서 기준)에서 이동됨.
- KDA 레이어 구현: [`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py) `KimiGatedDeltaNetAttention` — attention backend는 `gdn_attn`(GDNAttentionMetadata) 재사용:
  - **prefill**: `causal_conv1d_fn`×3(q,k,v; vLLM 자체 **Triton** 커널) → `chunk_kda_with_fused_gate`(initial_state gather + cu_seqlens varlen) → last state scatter
  - **decode**: `causal_conv1d_update`×3 → `fused_recurrent_kda`(`ssm_state_indices`로 paged state in-place 갱신)
- **FlashKDA는 vLLM main에서 미사용**: 트리 전체에 `flash_kda`/`flashkda` 경로·import 0건. 블로그(07-22)의 "FlashKDA and Triton paths are integrated; final backend selection … in progress"는 진행 중 작업 서술 — main 코드 기준 현재 KDA는 **100% Triton**. → SM80 관점에서 "우리가 쓸 경로 = 업스트림 기본 경로"라는 뜻이라 유리.
- 모델 등록: `KimiLinearForCausalLM`(= K3 `text_config`의 architecture)은 registry 등재. 단 **`KimiK3ForConditionalGeneration`(vision 포함 wrapper)은 2026-07-28 main 검색 0건** — day-0 발표(블로그 07-27)와 main 머지 사이 간극이 있어 Phase 1 시점에 재확인 필요. K3 text_config의 신규 요소(`hidden_act: situ`, `attn_res_block_size: 12`(AttnRes), `routed_expert_hidden_size: 3584`(LatentMoE), `mla_use_output_gate`)가 현 `kimi_linear.py`에 다 있는지도 `probe_k3_modules` 범위.
- 커널 테스트: `tests/kernels/test_kda.py` 존재(벤더링 커널 대상). vLLM 블로그의 "fused NVIDIA decode kernel(conv+recurrent+gate+norm 융합)"은 main 트리에서 미확인 — 상륙 시 아치 요구 재확인 필요.

## 5. SM80 잔여 리스크 (Phase 1 프로브로 전가)

1. **수치 검증 공백**: FLA CI=H100 전용, A100 결과는 업스트림 미보증 → `fla/ops/kda/naive.py` 및 vLLM `test_kda.py`를 A100에서 실행해 대조 (스펙의 "FLA 단위테스트 대조" 항목).
2. Triton autotune이 A100 smem 164KB에서 고르는 config의 성능(H100 대비 열화 폭) 미지 — 기능 리스크는 낮고 성능 리스크만.
3. int32 오버플로류 잠복 버그([PR #818](https://github.com/fla-org/flash-linear-attention/pull/818) open) — 256k 급 장문 prefill 시 varlen 인덱스 주의.
4. vLLM 벤더링 사본과 FLA 업스트림의 드리프트: vLLM 사본에는 paged-state용 `ssm_state_indices` 등 vLLM 전용 확장이 있음 — 검증은 **vLLM 사본** 기준으로 할 것.
5. K3 전체 서빙의 Hopper+ 요구(블로그 "NVIDIA (Hopper and Blackwell) and AMD (MI355X) support at launch")는 **KDA가 아니라 MXFP4 MoE 경로(TRTLLM-Gen/DeepGEMM)에서 옴** — KDA recon 결론과 혼동 금지 (MoE는 별도 recon).

## 6. 출처 (버전/커밋 고정)

| 대상 | 고정 | URL |
|---|---|---|
| FLA repo | HEAD `9c8e42e` (2026-07-27) / v0.5.1 (2026-06-18) / KDA 도입 v0.4.0 (2025-10-27) / FlashKDA backend v0.5.0 (2026-04-21) | https://github.com/fla-org/flash-linear-attention · [releases](https://github.com/fla-org/flash-linear-attention/releases) |
| FLA KDA ops | 상동 | https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda |
| FLA FlashKDA 디스패처 | 상동 | https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/backends/flash_kda.py |
| FLA A100 이슈 | #625 (closed) | https://github.com/fla-org/flash-linear-attention/issues/625 |
| FLA ROCm KDA | PR #868 (merged) | https://github.com/fla-org/flash-linear-attention/pull/868 |
| FlashKDA | HEAD `d2ff19a` (2026-05-26), `setup.py` L19 | https://github.com/MoonshotAI/FlashKDA · [README](https://github.com/MoonshotAI/FlashKDA/blob/master/README.md) |
| FLA×FlashKDA 통합 PR | #852 | https://github.com/fla-org/flash-linear-attention/pull/852 |
| vLLM | v0.26.0 (2026-07-27) / main 2026-07-28 조회 | https://github.com/vllm-project/vllm |
| vLLM 벤더링 KDA | main | https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/kda.py |
| vLLM KDA 레이어 | main | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py |
| vLLM state 산정 | main `kda_state_shape`/`kda_state_dtype` | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py |
| K3 config | HF main 2026-07-28 조회 | https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json |
| vLLM K3 블로그 | 2026-07-22 preview / 2026-07-27 day-0 | https://vllm.ai/blog/2026-07-22-kimi-k3-preview · https://vllm.ai/blog/2026-07-27-k3 |
| FlashKDA H20 벤치 보도 | 2026-04-30 | https://www.marktechpost.com/2026/04/30/moonshot-ai-open-sources-flashkda-cutlass-kernels-for-kimi-delta-attention-with-variable-length-batching-and-h20-benchmarks/ |
| Kimi Linear 논문 (KDA 수식) | arXiv:2510.26692 | https://arxiv.org/abs/2510.26692 |
