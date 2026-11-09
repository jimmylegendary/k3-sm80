# Kimi K3.1-Pro 지원 검토 — k3-sm80 (2026-11-09, 독립 검증 반영판)

> 상태: **검토(구현 전)**. 1차 출처(HF 체크포인트·vLLM main 소스)를 직접 읽어 확정한 사실, 그로부터 도출한 "무엇을 바꿔야
> 하는가", 그리고 읽기 전용 독립 검증(ultracode 워커 3명, vLLM main `a3c14efb` 원문 대조·리포 전수 grep)의 판정을 합친 판이다.
> 모든 SM80 판정은 **소스 판독**이며 A100에서 실행한 것은 없다. 구현 계획은 §5, Jimmy 결정 요청은 §6, 검증 판정은 §8, 앵커 전수표는 부록.

## 0. 한 줄 결론

K3.1-Pro는 K3와 **가중치 레이아웃이 사실상 동일**하다 — 차이는 24개 Gated MLA 층에 붙은 **DSA(DeepSeek Sparse Attention) indexer 텐서
120개(+357 MB)** 뿐이고 KDA 69층·MoE(MXFP4 expert 896)·토크나이저는 그대로다. 그러나 **SM80(A100)에서는 vLLM의 DSA 경로가 두 지점에서 막힌다.**

| # | 블록 지점 | 근거(vLLM main @ `a3c14efb`) | 함의 |
|---|---|---|---|
| B2 | **sparse MLA 백엔드가 SM80에 하나도 없다** — 첫 DSA 층의 `MLAAttention.__init__` → `get_attn_backend(use_sparse=True)`에서 **ValueError**(전체의 첫 실패) | `cuda.py:136-152` SM80 분기 후보 7개 전부 탈락(dense 4종 `sparse not supported`, sparse 3종 `compute capability not supported`; `backend.py:325-333`), 명시 선택 시 `cuda.py:441-445`, 자동 시 `:470-474`. 2차 방어선 `mla_attention.py:641-647` RuntimeError. dense 강제 스위치 없음 | **SM80용 sparse MLA 백엔드 신규** — 단, 기존 Triton `decode_attention_fwd`를 재사용하는 **신규 커널 0개·약 230줄 Python 백엔드**로 충분(§2 B2) |
| B1 | **DSA indexer 커널이 SM80에서 돌지 않는다** — Triton `tl.float8e4nv`(SM89+) 3곳 + DeepGEMM 호출 3곳·metadata 2곳(SM90+) | `ops/dsa_indexer.py:69,201-202,354`(fp8e4nv), `sparse_attn_indexer.py:882,1091,408`(`fp8_fp4_*mqa_logits`), `indexer.py:1837,1919`(metadata); DeepGEMM 자체가 `attention.hpp:147,197,232`에서 비 SM90/100/120 거부, 없으면 `deep_gemm.py:205-211 _missing()` | **SM80용 indexer logits 경로 신규(k11)** — bf16 Q × uint8(E4M3) K → Triton MQA logits(decode paged + prefill). **top-k는 교체 불필요**(`auto`→`persistent`, 전 arch 빌드), **index-K 캐시 쓰기는 stock CUDA op 재사용**(§2 B1) |

따라서 K3.1 지원 = **(a) base pin을 main(≥ split 3/6 #66037, 11-09)으로 올리고 SM80 패치 4개를 각각 삭제·rebase·재설계** + **(b) k11 SM80 indexer logits** +
**(c) k12 SM80 sparse MLA 백엔드** + **(d) 리포 앵커 갱신**(핀 손작성 59파일·lane·DSpark draft·shape 상수·gate 등록). 커널 신규 저작은 **k11 하나**로 줄었고,
pin 상향이 생각보다 무겁다(base 이미지 전체 소스빌드 재수행 + torch 2.13→2.14 등 의존성 드리프트). K3 자체의 A100 증거가 전부 `INCOMPLETE`이므로
([A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md)) **새 pin 위에서 K3 회귀(k01~k08)와 K3.1 신규 gate(k11·k12)를 같은 A100 라운드에서 돌리는 것**이 P1 종료 조건이다.

## 1. 사실관계 (1차 출처, 핀 고정)

### 1.1 체크포인트 — `moonshotai/Kimi-K3.1-Pro` @ `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` (lastModified 2026-11-05)

K3 핀(`9f62e4e9`, [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md))과 `config.json`·`model.safetensors.index.json`을 직접 diff 했다.

| 항목 | K3 (`9f62e4e9`) | K3.1-Pro (`d9e99b07`) | 비고 |
|---|---|---|---|
| architectures / model_type | `KimiK3ForConditionalGeneration` / `kimi_k3` | `KimiK31ForConditionalGeneration` / `kimi_k31` | vLLM registry는 둘 다 같은 클래스 `KimiK3ForConditionalGeneration`(`registry.py:485-491`); 모델 코드는 `config.is_dsa_layer()`로 분기(`model.py:754-755`) |
| HF remote modeling | `modeling_kimi_k3.py` auto_map 있음 | **AutoModel auto_map 제거**(configuration만) | HF transformers로는 직접 못 돌림 — vLLM/SGLang 전용 |
| `dsa_config` | 없음 | `dsa_layers=[4,8,…,92,93]`(24개, 1-indexed = 기존 Gated MLA 층 전부) · `index_num_heads=32` · `index_head_dim=128` · `index_topk=2048` · `index_rope_dim=0` · `index_k_norm_type=layernorm` · `index_skip_topk_offset=1`(vLLM 소비처 없음, parity 필드) · `index_topk_freq=1`(≠1 거부) | 모델카드: "replace the full-attention Gated MLA layers with Gated DSA" |
| 텐서 패턴 차이 | — | **추가 5패턴 × 24층 = 120개**: `self_attn.indexer.{wq_b, wk, weights_proj, k_norm.weight, k_norm.bias}` (0-indexed 층 3,7,…,91,92) | 그 외 497,220개 텐서 패턴·개수 **완전 동일**, 96 shard 동일 |
| total_size | 1,560,860,324,864 B | 1,561,217,377,280 B (**+357,052,416 B**) | indexer bf16 ≈ 178.5M param(층당 7.44M), 바이트 단위 폐쇄 일치 |
| MXFP4 범위 | routed expert w1/w2/w3만 | 동일(ignore 목록에 `block_sparse_moe.gate`·`routed_expert_(up|down)_proj` 정규식이 **명시**됐을 뿐) | 바이트 변화 없음 |
| MTP / draft 층 | 없음(`num_nextn_predict_layers=0`) | 없음 | DSpark draft는 별도 리포 |
| tokenizer / generation_config | — | 동일(`tokenizer_config.json`의 빈 `extra_special_tokens` 키 1개 삭제뿐) | `tiktoken.model` 동일, 렌더러 `encoding_k3.py`(Python) |
| transformers_version | 4.56.2 | 4.57.6 | |
| 1M context / vocab 163840 / KDA·MoE·AttnRes 상수 | — | 동일 | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §1~§3 전부 그대로 유효 |

### 1.2 vLLM upstream

| 항목 | 상태(2026-11-09) | 근거 |
|---|---|---|
| K3 코드 | PR #50000 **2026-07-30 main 병합**(merge `aeeb36b1`) → `vllm/models/kimi_k3/{nvidia,amd,common}` | 리포 핀 `0498dc7e`(07-28)은 병합 전 브랜치 — 브랜치 삭제, `refs/pull/50000/head`(`5a3eba03`)로만 도달. main은 merge 대비 +5,609 커밋 |
| K3.1 분할(**6개**) | 1/6 #66030 모델 파일(11-06 병합) · 2/6 #66028 Rust renderer(병합) · **3/6 #66037 KV cache·attention — 11-09 04:31Z 병합** · 4/6 #66532 AMD 모델 파일(open) · 5/6 #65685 ROCm·P/D·indexer top-k(open) · 6/6 #66027 DCP(open) | "Until [3/6] lands, K3.1 imports but does not serve" — **main은 오늘부터 K3.1을 서빙**(TP8, Blackwell). 남은 3개는 ROCm·P/D·DCP·SM100 top-k라 SM80 동작과 무관 |
| 최소 참조 구현 | PR #66145(mgoin, do-not-merge) ≈ 2.5k 줄 | SM80 포팅 범위 판단 기준점 |
| 릴리스 | v0.32.0(10-18), v0.33.0rc3(11-02). recipe: "**Requires vLLM >= 0.34.0 (not released yet — nightly wheel)**" | 공개 브랜치 없이 Docker 이미지 `vllm/vllm-openai:kimi-k31`만 — 사내는 registry 차단이라 **소스빌드 필수**(기존과 동일) |
| 지원 HW | B300/GB300/VR200(preview)/MI355X/TPU v7x/T-HEAD. **A100·Ampere·SM80 언급 0** | 블로그·recipe·모델카드·제3자 분석 전부 |
| 기본 커널(NVIDIA) | sparse MLA = FlashInfer sparse(trtllm-gen/PrimsTS) 또는 FlashMLA sparse; indexer = **DeepGEMM** `fp8_fp4_paged_mqa_logits`; 저지연 decode = CuTe DSL(SM100) | 전부 SM90+/SM100 |
| K3.1 KV 레이아웃 | split 3/6: **packed hybrid KV row**(MLA latent + indexer key + KDA state 페이지를 한 block row에, `get_packed_hybrid_block_size`); indexer K cache는 **fp8(132 B/token/층) 또는 mxfp4(SM100)만**(`dsa_cache.py:33-65`, `indexer.py:62-74`), bf16 옵션은 DSA에 없음(MiniMax M3 전용) | SM80은 FP8 텐서코어가 없지만 **E4M3 변환 자체는 CUDA·소프트웨어로 가능** |
| DSpark draft | `Inferact/Kimi-K3.1-Pro-DSpark-preview` @ `6a2ef20a`(11-05, "preview") — 5 dense layer 2.7B, block 7, `target_layer_ids=[75,79,83,87,91]`(1-indexed), MLA 96 heads·RoPE 64·`rope_theta=1e6` yarn, `markov_rank=512` | K3 draft(`Inferact/Kimi-K3-DSpark`: ids `[2,23,47,71,89]`, 64 heads, θ=5e4)와 **다름**. 블로그 acceptance length 4.597(@7 tokens, GSM8K) |

## 2. SM80 갭 분석 (vLLM main 원문 기준, 독립 검증 반영)

### B2. sparse MLA attention — SM80 백엔드 부재 (전체의 첫 실패 지점)
- 선택 흐름: `layers/mla.py:136 use_sparse=self.is_sparse` → `mla_attention.py:528-536 get_attn_backend(..., use_mla=True, use_sparse=True)` → `cuda.py`. SM80은 `cuda.py:136 else` 분기로 후보 `[FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA, FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE_SM90]`을 받고 **7개 전부 탈락**한다.
  리포 lane처럼 `--attention-backend TRITON_MLA`를 명시하면 `cuda.py:441-445` `ValueError: Selected backend ... Reason: ['sparse not supported']`가 **먼저** 난다(`mla_attention.py:641-647`의 RuntimeError는 selector를 우회할 때만 닿는 2차 방어선).
- dense 폴백 스위치는 upstream에 없다(`sparse_mla_force_mqa`는 반대 방향, `_sparse_mla_force_dense_mha`는 벤치마크 전용). `dsa_config`를 지워 dense MLA로 만들면 indexer 가중치 로드에서 KeyError가 날 것으로 보인다(`model.py:1442,1466-1471` rename이 `dsa_config is not None`일 때만). 또한 `prefill_max_seq_len ≤ 2048`이면 upstream도 dense prefill을 쓰지만(`sparse_mla_attention.py:138-146`), 그 이상에서 dense는 **학습 분포 밖**이라 제품 경로로 쓰지 않는다.
- **필요 작업(k12) — 규모는 당초 추정보다 작다(독립 검증 V4)**: `AttentionBackend`를 직접 상속하는 신규 `mla/triton_mla_sparse.py`(가칭). 템플릿은 가장 단순한 SM90 `flashattn_mla_sparse.py`(422줄).
  - Backend ~80줄: `get_name/get_impl_cls/get_builder_cls` + `supported_dtypes`, `supported_kv_cache_dtypes`(**bf16 필수** — `auto`/`fp8`은 §B3 훅이 `fp8_ds_mla`로 바꿔 selector에서 거부됨), `get_supported_head_sizes=[576]`, `get_supported_kernel_block_sizes=[MultipleOf(64)]`(indexer 백엔드가 64·BLHNC 요구, `dsa_cache.py:111-135`), `customize_spec=align_blocks_to_rows`, `is_mla/is_sparse=True`, `supports_compute_capability: major==8`, `supports_combination`.
  - Metadata 3줄(`SparseMLACommonMetadata` 상속) · MetadataBuilder ~35줄(`SparseMLACommonMetadataBuilder` 308줄 상속, `_init_reorder_batch_threshold(..., supports_spec_as_decode=True)`) · Impl ~100줄(`SparseMLACommonImpl` 644줄 상속, 유일한 추상 훅 `_forward_mqa_kernel(q, kv_cache, topk_indices, valid_counts, *, layer, block_size, is_decode)`).
  - **A안(권장, 신규 Triton 커널 0개)**: `_forward_mqa_kernel`이 기존 `triton_decode_attention.decode_attention_fwd(page_size=1, req_to_token=physical_topk, b_seq_len=valid_counts)`를 호출. top-k 논리 인덱스 → 물리 slot 변환은 upstream `triton_convert_req_index_to_global_index`(`sparse_utils.py:492-636`, 유효 항목 앞쪽 압축 + `valid_counts`)가 `forward_mqa`에서 이미 해 준다. prefill도 같은 `forward_mqa` 경로(쿼리별 top-k)라 **별도 prefill 커널 불필요**; ≤2048 구간 dense MHA prefill은 FA2가 자동 재사용된다. `reserve_workspace()`로 `attn_logits` 예약(`triton_mla.py:256-267` 패턴). 배선: `registry.py` +3줄, `cuda.py` 우선순위 +1~2줄. **P1 소계 약 210~260줄, 테스트 150~250줄.**
  - **B안(코드 0줄 bring-up, 정적 분석·미실행)**: capability 제한이 없는 in-tree Triton sparse 백엔드 `XPU_MLA_SPARSE`(`xpu_mla_sparse.py`, bf16 KV만)를 `--attention-backend XPU_MLA_SPARSE --kv-cache-dtype bfloat16 --enforce-eager`로 명시하면 selector를 통과한다. 단 `ops/xpu_mla_sparse.py:99-102,132-136`의 `indices * stride_k_token`이 int32라 packed row id(토큰 위치의 ≈26.75배)가 **약 139k 토큰에서 오버플로** → `indices.to(tl.int64)` 2줄 패치 필요. k11이 준비되면 lane 01 첫 smoke에 쓸 수 있다.
  - P3(성능/FP8): fused gather-MQA 커널 ~250줄(1개), `fp8_ds_mla` 또는 리포 576 B software-FP8 읽기 +80~150줄. **K3.1 DSA 층은 K3 `mla.py`가 아니라 generic `MLAAttention`을 쓰므로** 리포 software-FP8 패치는 새 Impl의 `do_kv_cache_update` override로 재설계해야 한다.

### B1. DSA indexer — 커널이 SM80에서 돌지 않음 (백엔드 문제를 풀었다고 가정했을 때의 다음 실패)
- `KimiK3DSAIndexer`(`dsa.py:108-412`)는 **객체 생성은 SM80에서 예외 없이 끝난다**. upstream `SparseAttnIndexer` 클래스는 K3.1이 인스턴스화하지 않으므로 그 `__init__`의 DeepGEMM RuntimeError는 해당 없음(초판 B1 정정). 실패 순서(소스 판독):
  1. profile run: `dsa.py:380-383` → `indexer_q_quant_fold_weights`의 `tl.float8e4nv`(`ops/dsa_indexer.py:201-202`) — Triton native fp8e4nv는 SM89+(`triton_mla.py:233-234`, `qsa.py:28` 주석) → 컴파일 실패.
  2. DeepGEMM import 가능 시 `indexer.py:1836-1842` metadata → DeepGEMM `attention.hpp:197` "Unsupported architecture"; 불가 시(리포 SM80-only 빌드는 `deepgemm.cmake:93-96` arch 교집합이 비어 **미빌드 가능성 높음**) 첫 logits에서 `deep_gemm.py:205-211 _missing()` RuntimeError.
  3. prefill `sparse_attn_indexer.py:882 fp8_fp4_mqa_logits`, decode `:1091 fp8_fp4_paged_mqa_logits` 동일.
- **교체 대상 6심볼**: `fp8_fp4_mqa_logits`(prefill dense logits) · `fp8_fp4_paged_mqa_logits`(decode) · `get_paged_mqa_logits_metadata`(`has_deep_gemm()`일 때만 호출) · Triton `indexer_q_quant_fold_weights` · `indexer_k_norm_quant_insert`(fp8e4nv 포인터 `u8_cache.view(_FP8_DTYPE)`) · `dsa_fused_prologue`(`kv_cache_dtype == "fp8_ds_mla"`일 때만 호출되므로 bf16 lane에서는 무관).
  **패치 지점**은 `sparse_attn_indexer()` 본문의 prefill 분기(`:822-943`)·decode 분기(`:1085-1153`)에 XPU(`:859-873, :1069-1083`)처럼 SM80 분기를 추가하는 것과 `indexer.py:1836-1844, :1918-1926`의 metadata 호출, `ops/dsa_indexer.py` 커널 3종이다. `VLLM_USE_DEEP_GEMM=0`은 indexer에 효력이 없다.
- **교체하지 않아도 되는 것(소스상 arch 가드 없음, 미실행)**: top-k — `auto`는 SM80에서 `cooperative` 탈락 후 **`persistent`**(`indexer_topk.py:264-281`; `csrc/topk.cu`는 전 arch 빌드, `__CUDA_ARCH__>=800` 가드, smem 128 KiB ≤ A100 optin 163 KiB), prefill top-k는 `ops.top_k_per_row_prefill`(CUDA, 백엔드 무관). `torch` 백엔드는 decode 전용 디버그 참조라 쓰지 않는다. index-K 캐시 **쓰기**: `use_fused_k_insert=False`면 fp32 LayerNorm(`dsa.py:352`) 후 stock CUDA `ops.indexer_k_quant_and_cache`(`cache_kernels.cu:705-773`, 변환 `quant_utils.cuh:232-241` SM80 OK) → **software-FP8 쓰기 커널 불필요**(단 현재 그 플래그를 끄는 env/flag가 없어 `dsa.py:221-225`에 스위치 패치 필요). prefill 읽기: `cp_gather_indexer_k_quant_cache` 후 `.to(bf16)`(ROCm 참조 `rocm_aiter_mla_sparse.py:929`).
- **k11 설계 요지**: Q는 **bf16 그대로**(fp8 양자화 생략, `weights × scale`만 접음 — DeepGEMM fp8 Q와 bit-exact하지 않아 top-k 경계가 달라질 수 있음 → NUMERIC 게이트는 logits 오차 + top-k 집합 일치율로 정의), K는 paged uint8 E4M3를 Triton 안에서 소프트웨어 디코드(리포 [mla_fp8_kv_sm80.py](/home/jimmy/repos/k3-sm80/kernel-kit/vendor/mla_fp8_kv_sm80.py)의 `e4m3_bits_to_fp32` 또는 upstream `qwen4_exp/.../qsa.py:25-64 _maybe_dequant`(sm_80 겨냥 inline PTX) 재사용), 레이아웃 `[block_size×128 fp8 | block_size×4B fp32 scale]`(`cache_kernels.cu:755-764`) + packed row의 `stride(0)`. fp32 참조 오라클은 upstream `rocm_aiter_mla_sparse.py`의 `fp8_mqa_logits_torch`(`:902-951`)·`fp8_paged_mqa_logits_torch`(`:471-569`). spec decode `next_n≥3`은 평탄화 경로(`indexer.py:951-957`).

### B3. KV cache dtype 재작성 훅 — lane 설정과의 충돌 (확인됨)
- `KimiK31ForConditionalGeneration`도 `KimiK3ForConditionalGenerationConfig`(`models/config.py:744,1565`)를 쓴다. `verify_and_update_config`(`:777-807`)는 `dsa_config`가 있으면 kv-cache dtype을 **재작성**한다: 백엔드가 `FLASHINFER_MLA_SPARSE`이거나(미지정 + SM100)이면 `auto`→`fp8_e4m3`, **그 밖의 모든 경우(SM80 포함) `auto` 또는 `fp8*` → `fp8_ds_mla`**(FlashMLA sparse 전용 656 B 레이아웃). `bfloat16`/`float16` 명시만 그대로다. K3(dsa_config 없음)는 `:785-786`에서 바로 return.
- 리포 lane 01~03은 이미 `--kv-cache-dtype bfloat16`을 명시([cluster.py.tmpl](/home/jimmy/repos/k3-sm80/deploy/tooling/templates/cluster.py.tmpl) `:39`)하므로 P1은 안전. **lane 04/05의 `fp8`(software-FP8 패치 트리거)은 모델 생성 전에 `fp8_ds_mla`로 바뀌어 패치 트리거가 소멸**하고, 애초에 K3.1 target 층은 패치 대상 클래스(`kimi_k3/nvidia/mla.py`)를 쓰지 않는다(draft만 해당) → P3에서 새 Impl 기준 재설계.
- speculative_config가 있으면 `configure_dense_draft`(`dense_draft.py:36-40`)가 draft에 **`attention_backend=FLASHINFER_MLA`, `kv_cache_dtype="fp8"` 기본값을 주입**(사용자 JSON이 안 정했을 때). 현 lane JSON은 `attention_backend`만 `TRITON_MLA`로 고정 → draft가 fp8이 되어 `TritonMLAImpl.__init__`(`triton_mla.py:225-236`)이 SM80에서 ValueError. **lane 03/05 speculative JSON에 `"kv_cache_dtype":"bfloat16"` 추가 필수**(P2).

### B4. base pin 드리프트 (0498dc7e → main, 3.5개월) — 패치 4개의 운명 (독립 검증 V5, `patch --dry-run` 실측)

| 패치 | main 상태 | 판정 |
|---|---|---|
| `kda-sm80-int64.patch` | `chunk_delta_h.py:98-103`에 **동일한 cast-before-multiply 수정 반영**(+`:109,:111,:143`), `kda/chunk.py:252`는 `make_block_ptr`→포인터 연산으로 재작성되고 `i_t`가 program_id에서 int64 승격(`:217`) → 전체 인덱스 int64 | **삭제**(Dockerfile.openai `:34-42` 검증, `verify_image.py:21`, tests 3곳 제거). crash 소멸은 k01 long-T 재실행으로만 확정 |
| `mxfp4-marlin-moe-sm80.patch` | 파일 이동: `csrc/libtorch_stable/moe/…` → `csrc/moe/marlin_moe_wna16/{ops.cu,kernel.h,marlin_template.h}`, generator는 `csrc/quantization/marlin/generate_kernels.py`로 통합(SM80 MXFP4 인스턴스 `sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`는 **upstream이 이미 생성**, CMake `:1445,:1493`). `ops.cu`의 버그 로직(`:329-330,:365-366,:369,:504-506` `max_shared_mem`·`c_tmp` 축소)은 **그대로 남아 있음**, 함수 시그니처는 단순화(`has_act_order`·`is_a_8bit`·`a_s_ptr` 없음) | **구조적 rebase**: ops.cu hunk 5개 중 2개 재작성, generator 가드는 Dockerfile `test -s`/`grep -Fq kFE8M0fnu`로 대체(generator 수정 시 dense Marlin까지 재생성됨). Dockerfile 경로 4곳(`:44,:46,:76,:78`) 변경 |
| `mla-fp8-kv-sm80.patch`(22 KB, hunk 25개) | 14 적용(fuzz)/11 실패. `envs.py` 이웃 변경, K3 `mla.py` 청크 컨텍스트 gather가 `impl._compute_prefill_context` 위임 → K3 자체 `_gather_context_latent`(`:956-1001`)로 이동, `mla_attention.py` 훅은 fuzz로 **엉뚱한 자리에 붙음**(`i`·필드 없음), `triton_decode_attention.py` 818→1575줄(`sliding_window` 인자 plumbing으로 hunk 7개 실패, 신규 "folded causal verify" 커널군은 별도 `is_fp8()` 분기) | **K3.1 target 기준 재설계**(target은 generic `MLAAttention`을 써서 이 패치를 우회; draft에만 걸림). K3 dense/draft 용도로만 쓸 거면 구조적 rebase. P3 |
| `mla-fp8-kv-sm80-dspark-bf16.patch` | 앵커 유지(패치 2 선행 시 offset +59 적용). 기능은 upstream `SpeculativeConfig.kv_cache_dtype`(draft 전용 override, `speculative.py:461-468`)로 **대체 가능** | **퇴역 후보** — draft JSON `kv_cache_dtype` 사용으로 대체 시 skip-layers `93 94 95 96 97` 계약도 불필요(P2 판정) |

- **pin 상향은 `VLLM_COMMIT` 한 줄이 아니다**: [Dockerfile.openai](/home/jimmy/repos/k3-sm80/docker/Dockerfile.openai)는 base `k3-sm80-kernelkit@sha256:d5fc…`에 `.so` 1개 + Python 8파일만 overlay하고, base는 [Dockerfile.full](/home/jimmy/repos/k3-sm80/docker/Dockerfile.full) `:84 pip install .` **전체 소스빌드("수 시간")** → main sha용 base 재빌드 필수. 의존성 드리프트: torch 2.13.0→**2.14.0**, torchvision 0.28→0.29, flashinfer-python 0.6.15.post1→**0.7.3**(`--extra-index-url`·`flashinfer-cubin` 줄 삭제), setuptools ≥78.1.1; `Dockerfile.full:142`의 torch 2.13 assert·[Dockerfile.triton](/home/jimmy/repos/k3-sm80/docker/Dockerfile.triton) 계약·[PULL.md](/home/jimmy/repos/k3-sm80/docker/PULL.md) 티어 표·[RUNBOOK.md](/home/jimmy/repos/k3-sm80/a100-kit/RUNBOOK.md) `:13` 수정. [build.sh](/home/jimmy/repos/k3-sm80/docker/build.sh) `:53-54`는 `VLLM_COMMIT` build-arg를 안 넘긴다. 빌드 대상 `--target _moe_C_stable_libtorch`·`_moe_C` 네임스페이스는 유지. 옛 main pin `d223c900`도 73행(손작성 44행) 남아 있어 함께 정리.
- main 신규 SM100 전용 경로는 전부 soft fallback 확인: fused KDA decode(`FUSED_KDA_DECODE_ARCHS="9.0a;10.0f;12.0f"` → `kda.py` `"triton"` `:367,:532`), AttnRes CUDA(`"10.0f"` → `ops/attn_res.py:199` Python), latent-MoE tail/runner(`latent_moe_runner.py:74-78`), low-latency GEMM(`:76-80,:701-705` → None), specialized decode(opt-in). `fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`는 일반 소스 목록(`CMakeLists.txt:484`)이라 SM80 빌드에 포함(동작은 미검증).
- **재vendor 영향**: kernel-kit k01/k02/k04의 시험 대상 자체가 바뀐다(KDA 커널 재작성, decode attention 2배 확장). `_vllm_shims`·`STUBS.md` 근거 재대조.

### B5. DSpark / 렌더러 / 기타
- DSpark draft 교체 실제 지점은 셋뿐: [dspark_ref.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/dspark_ref.py) `:10-11` `TARGET_AUX_LAYER_IDS=(2,23,47,71,89)`→`(75,79,83,87,91)`(1-indexed, `model.py:1261` `(layer_idx+1) in aux_hidden_state_layers`), [test_k10_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k10_contract.py) `:41,:43,:73-74`(`:74`는 교체만 하면 실패), 설계 문서. lane JSON에는 layer ids가 없다(draft config가 가짐). `K3_DRAFT_MODEL`은 5 lane 전부 필수 키라 텍스트는 다 바뀌지만 기능 영향은 03/05뿐. 블로그 권장 `num_speculative_tokens` 5 vs draft README 7.
- 렌더러: `reasoning_effort ∈ {low, high, max}`(기본 max; 렌더러는 medium도 수용) top-level 요청 필드(split 1/6 Python renderer, 2/6 Rust). 파서 `kimi_k3` 그대로. lane smoke에 `reasoning_effort` 케이스 추가.
- `torch.compile`: "K3.1 has no torch.compile submodule → FULL_DECODE_ONLY"(recipe). lane은 `enforce_eager=1`이라 P1 무관.
- Vision: MoonViT-V2 동일, `--language-model-only` 유지. 이슈 #66524(148 SM)는 SM100 specialized decode 전용 → A100 무관.

## 3. 수치 영향 (리포 코스트 모델 입력)

| 항목 | K3 | K3.1-Pro | 산식·출처 |
|---|---|---|---|
| 가중치 총량 | 1,560.86 GB | 1,561.22 GB (+0.36 GB) | index.json total_size |
| 토큰당 활성 가중치 bytes(원본 ckpt) | 136.96 GB | **≈137.32 GB** (+0.26%) | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §6 + indexer 178.5M param bf16 |
| 24층 attention cache, 토큰당(모든 TP rank 복제) | bf16 latent 27,648 B / FP8 13,824 B | + indexer K **3,168 B**(132×24) → bf16 30,816 B / FP8 16,992 B | vLLM 블로그 FP8 수치 13,824→16,992(+22.9%)와 일치 |
| seq당 @256k / @1M (bf16 latent) | 7.25 GB / 28.99 GB | **8.08 GB / 32.31 GB** | |
| decode 1토큰당 attention 읽기, 층당 @256k | latent 전체 302 MB(bf16) | index-K 34.6 MB + top-2048 latent 2.4 MB ≈ **37 MB (≈1/8)** | @1M: 1.21 GB → 141 MB |
| top-k 버퍼(컨텍스트 무관) | — | `max_num_batched_tokens × 2048 × 4 B`(8 KiB/행; 8192행 = 64 MiB) + index group 물리 버퍼 층당 `[rows+1, 2048] int32` × 24층 | `model.py:1041-1048`, `index_group.py:373-386` |
| indexer 연산 @1M | — | 32×128×1M ≈ 4.1 GMAC/층/토큰, 24층 ≈ 0.2 TFLOP/토큰 | 메모리 바운드(132 B/token 스트림) |

- **Phase 0 판정(TP8PP4, ctx 8k, 46.8 tok/s real)은 유지**: 병목은 가중치 스트리밍(137 GB/token). DSA는 장문맥(≥128k)에서 attention 바이트를 4~8× 줄인다.
- [k3_model.py](/home/jimmy/repos/k3-sm80/phase0/k3_model.py) `attn_mla`에 DSA 변형(indexer 스트림 + top-k + 2048행 MQA)과 indexer cache 바이트를 추가하고 `LAYERS` 패턴을 `dsa_config.dsa_layers`로 읽게 바꾼다(P3, k11/k12 실측 후 재보정).

## 4. 리포 변경 지도 (요약 — 전수표는 부록 A, 게이트 등록 지점은 부록 B)

독립 검증 grep: K3 고정값 히트 **595행**(tracked 152행, 84파일; 나머지는 gitignored 로그·리포트). 핀 `0498dc7e` 손작성 117행(kernel-kit 49, docker 26, docs 13, a100-kit 11 …), `moonshotai/Kimi-K3` 52행, `kimi-k3` 77행, DSpark 332행.

| 영역 | 핵심 파일 | 변경 |
|---|---|---|
| docker | [Dockerfile.openai](/home/jimmy/repos/k3-sm80/docker/Dockerfile.openai)(`:11-12,:34-46,:73-78,:88-156,:167-197`) · [Dockerfile.full](/home/jimmy/repos/k3-sm80/docker/Dockerfile.full)(`:33,:63,:142`) · [build.sh](/home/jimmy/repos/k3-sm80/docker/build.sh) · [build-openai.sh](/home/jimmy/repos/k3-sm80/docker/build-openai.sh) · [release-openai.sh](/home/jimmy/repos/k3-sm80/docker/release-openai.sh) · [verify_image.py](/home/jimmy/repos/k3-sm80/docker/verify_image.py) · [PULL.md](/home/jimmy/repos/k3-sm80/docker/PULL.md) · `patches/` | 새 pin + base 재빌드(torch 2.14·flashinfer 0.7.3) · 패치 집합 {marlin(rebase)} + 신규 {sm80-dsa-indexer, triton-mla-sparse, xpu-sparse-int64(B안 시)} · kda 삭제 · csrc 경로 변경 · 새 태그 |
| deploy | [render_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/render_lanes.py)(`:44-61,:91,:99-103,:115-118,:203`) · [lane_specs.py](/home/jimmy/repos/k3-sm80/deploy/tooling/lane_specs.py) · [lane_schema.py](/home/jimmy/repos/k3-sm80/deploy/tooling/lane_schema.py)(`:16,:221,:232`) · [gates.py](/home/jimmy/repos/k3-sm80/deploy/tooling/gates.py) · 템플릿 · 생성 lane 5개 | `K3_TARGET_MODEL`/`K3_DRAFT_MODEL`/`K3_IMAGE`/`K3_SERVING_ROOT` 파라미터화(LaneSpec model 필드), `VLLM_ATTENTION_BACKEND=TRITON_MLA`→SM80 sparse 백엔드, `VLLM_KV_CACHE_DTYPE=bfloat16` 명시 유지, speculative JSON에 `"kv_cache_dtype":"bfloat16"`, `VLLM_SERVED_MODEL_NAME=kimi-k31`, k11/k12 gate 행(GATE_MATRIX·Makefile·validate_lanes·A100-5LANE-RESULTS·tests), `reasoning_effort` smoke, `make render` |
| kernel-kit | [shapes.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/shapes.py) · [timing.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/timing.py) `PERF_POLICY` · [tolerance.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/tolerance.py) · [relay.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/relay.py) `:68` · [run_all.sh](/home/jimmy/repos/k3-sm80/kernel-kit/run_all.sh) `:16` · `k11_dsa_indexer_sm80/`·`k12_sparse_mla_sm80/`(신규, 템플릿 k09) · `common/dsa_ref.py`(신규) · [vendor/](/home/jimmy/repos/k3-sm80/kernel-kit/vendor) 재vendor · [dspark_ref.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/dspark_ref.py) | DSA 상수(`DSA_INDEX_HEADS=32, DSA_INDEX_HEAD_DIM=128, DSA_TOPK=2048, DSA_LAYERS=24, DSA_INDEX_K_BYTES=132`) + 케이스·perf 모델 + `PERF_POLICY`(**먼저 안 넣으면 전체 스윕 exit 3**) + `TOL`(top-k는 집합 일치 판정 별도) · **relay 필터 `startswith("k0")` → `^k\d\d_`**(현재 k10~k12 조용히 탈락) + RELAY 12줄 예산 압축 규칙 · 게이트 이름 **`k12_sparse_mla_sm80`로 확정**(KERNEL_IDS·디렉터리·JSON·PERF_POLICY 키 동일 문자열) |
| tests | [test_openai_image_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_openai_image_contract.py) `:27` · [test_fp8_patch_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_fp8_patch_contract.py) · [test_k10_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k10_contract.py) `:41,:43,:73-74` · [test_commands.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_commands.py) · [test_gate_matrix.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_gate_matrix.py) · 신규 `test_k11_*`·`test_k12_*`·`test_dsa_ref.py` | 핀·id 교체, 신규 contract(템플릿 [test_k09_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k09_contract.py)) |
| phase0 | [k3_model.py](/home/jimmy/repos/k3-sm80/phase0/k3_model.py) `:76-91,:293-300,:433-441,:519-524,:645-646` | DSA 항 + indexer cache 바이트 + `dsa_layers` 패턴(P3) |
| recon/docs | `recon/k31-config.md`(신규, §1.1 원문) · [SUMMARY.md](/home/jimmy/repos/k3-sm80/SUMMARY.md) · [PLAN.md](/home/jimmy/repos/k3-sm80/docs/PLAN.md) · [README.md](/home/jimmy/repos/k3-sm80/README.md) · `docs/superpowers/plans/…five-lane…md:794,811` | K3.1 핀·단계·draft 값 |

## 5. 구현 계획 — 점진적 해상도 (phase 종료 = 실제로 돌아가는 전체 제품)

| capability | P1 (walking skeleton) | P2 (넓이+깊이) | P3 (장문맥·성능) | 미룸 |
|---|---|---|---|---|
| vLLM base pin | main 커밋(≥ #66037) base 이미지 재빌드(torch 2.14·flashinfer 0.7.3), 패치 {marlin 구조적 rebase}, kda 삭제, fp8-kv·dspark-bf16 제외 | v0.34.0 출시 시 재핀 | — | — |
| K3 회귀(k01~k08) | 재vendor 후 새 pin에서 **A100 PASS**(k01 long-T로 int64 퇴역 확정) | c01·k10 포함 | k09 재설계분 | — |
| DSA indexer(k11) | Triton MQA logits(decode paged + prefill), bf16 Q, stock CUDA K-insert 스위치, 소프트웨어 E4M3 읽기; NUMERIC(logits 오차 + top-k 집합 일치)/SANITIZER/SHAPES **PASS**, PERF 측정 전용 | PERF gate, decode logits 청크화 | 1M prefill logits 버퍼 사이징 | mxfp4 index-K, DeepSelect, fp8 Q 동일성 |
| sparse MLA(k12) | **A안** Python 백엔드(기존 `decode_attention_fwd` 재사용, 신규 커널 0) **bf16 latent**; bring-up smoke는 **B안** `XPU_MLA_SPARSE`+int64 2줄 패치; NUMERIC/SANITIZER/SHAPES PASS, PERF 측정 전용 | PERF gate, fused gather-MQA 커널(~250줄), CUDA graph 캡처, batch>1 | software-FP8 latent를 새 Impl `do_kv_cache_update`로 재설계 | fp8_ds_mla 네이티브, DCP/ShardQ, HiSparse |
| 서빙 lane | **lane 01 K3.1(TP8/PP4, 256k, `--kv-cache-dtype bfloat16` 명시, `--attention-backend <SM80 sparse>`, DSpark 없음)** `gates up gate-load gate-context smoke` RELAY | lane 02/03 K3.1(EP32, DSpark preview + draft JSON `kv_cache_dtype: bfloat16`, dspark-bf16 패치 퇴역) | lane 04/05 K3.1(1M, software-FP8 KV) | specialized decode, P/D |
| 렌더러·API | `reasoning_effort` smoke 1케이스, 파서 kimi_k3 | tool-call·multi-turn(reasoning_content 회신) smoke | — | vision |
| 코스트 모델 | 변경 없음(판정 유지 근거 §3) | — | DSA 항 추가 + k11/k12 실측 재보정 | — |
| 문서 | recon/k31-config.md, PLAN S2' 등록, README 현황, pin 표기 59파일 | RUNBOOK K3.1 | SUMMARY 재작성 | — |

**P1 종료 gate(실제 인터페이스·실제 의존성)**: ① 호스트 — `python3 -m unittest` green, `shapes.py --check`·`timing.py --selftest`·`report.py --selftest`·`run_all.sh --cpu-dry-run` green, 컨테이너 gate(vllm import 0건) ② A100 — `run/20-kernel-test.sh full`에서 k01~k04·k07·k08·**k11·k12** NUMERIC/SANITIZER/SHAPES PASS(RELAY) ③ A100 — `make -C deploy/01-… gates up gate-load gate-context smoke`가 **K3.1-Pro 실체크포인트**로 응답 생성(greedy 재현성 + reasoning 채널) ④ 회귀 — K3 shape 케이스 전부 유지.
**P1 정직한 상한**: 속도 미보증(PERF 측정만 공개), 256k·bf16 KV·`max_num_seqs=1`·`enforce_eager`, DSpark 없음, 1M 미지원, Q fp8 미적용(DeepGEMM 대비 top-k 경계 차이 가능).
**P1 작업 단위(규모)**: pin 상향+base 재빌드+패치 정리(**M~L**, 빌드 수 시간) · k11 커널+하네스(M) · k12 백엔드+배선+하네스(**M**, 당초 L) · lane/테스트/문서 앵커 갱신(S~M, 59파일) · A100 1라운드(러너 Jimmy, RELAY 회신).

## 6. Jimmy 결정 요청

| # | 질문 | 선택지 | 추천 |
|---|---|---|---|
| D1 | base pin | (a) 지금 main 커밋 고정(`a3c14efb` 11-09 또는 #66037 병합 커밋 `4088c283`) → v0.34.0 출시 시 재핀 / (b) v0.34.0 대기 / (c) 4/6~6/6 병합 대기 | **(a)** — 3/6까지로 NVIDIA 기본 경로가 서빙 가능하며, 남은 분할은 AMD·ROCm·P/D·DCP·SM100 top-k라 SM80 무관. base 재빌드는 어차피 1회 |
| D2 | K3 lane 처우 | (a) 5 lane 모두 K3.1로 전환, K3는 kernel-kit 회귀(k01~k10)와 `.env` 한 줄(`K3_TARGET_MODEL`)로만 유지 / (b) K3·K3.1 lane 병행(디렉터리 10개, 이미지 2개) | **(a)** — K3 A100 증거가 없어 잃는 것이 없고, 같은 모델 클래스라 이미지 1개로 둘 다 가능 |
| D3 | P1 범위 | 위 §5대로(bf16 KV, DSpark·1M 제외, k12는 A안+B안 smoke) / DSpark를 P1에 포함 | **§5대로** — draft가 "preview"이고 draft 기본값 주입(§B3) 처리가 필요해 P2가 안전 |
| D4 | A100 1라운드 구성 | K3 회귀 + K3.1 k11/k12 + lane 01 K3.1을 한 번에 / 커널 라운드와 lane 라운드 분리 | **한 번에**(회신 채널이 RELAY 하나라 왕복 최소화), 단 kernel-kit PASS가 lane 진입 조건(PLAN 원칙) |

## 7. 미확인·리스크

- 모든 SM80 판정은 소스 판독이다: `persistent_topk`·`indexer_k_quant_and_cache`·CUDA E4M3 변환·`fused_kimi_k3_mla_*`의 SM80 동작, Triton fp8e4nv 에러 문자열, kda crash 소멸은 A100 1회차에서 판정.
- `XPU_MLA_SPARSE` B안은 정적 분석(selector 통과·int32 오버플로 위치)이며 미실행.
- main CMake를 `TORCH_CUDA_ARCH_LIST=8.0` 단일 arch + `--target` 1개로 configure하는 조합, torch 2.14.0 cu130 휠·flashinfer 0.7.3 가용성 미검증.
- split 3/6 packed hybrid block row와 리포 software-FP8 latent 레이아웃의 결합(P3); indexer prefill logits 버퍼(`get_max_prefill_buffer_size`)의 1M 메모리(P3).
- 5/6 #65685가 `sparse_attn_indexer.py:932-943`(prefill top-k)을 `SparseIndexerTopk.forward_prefill`로 옮긴다 → 우리 SM80 분기가 그 근처를 건드리면 재핀 시 conflict.
- DSpark preview draft는 "stronger version soon" → P2 시점 재확인. K3.1 draft config 값은 HF `6a2ef20a` 직접 조회값.
- 사내 환경 제약(ghcr 차단·NFS root_squash·RELAY 회신)은 K3와 동일 — [README.md](/home/jimmy/repos/k3-sm80/README.md).

## 8. 독립 검증 판정 (ultracode 워커 3명, 읽기 전용, vLLM main `a3c14efb` 원문 + DeepGEMM `82ca617e` + 리포 전수 grep)

| 항목 | 판정 | 요지 |
|---|---|---|
| V1 indexer가 SM80에서 init 실패 | **부분** | "SM80 indexer 신규 필요"는 확인. 그러나 K3.1은 `SparseAttnIndexer`를 쓰지 않아 init RuntimeError는 해당 없음; 실패는 Triton fp8e4nv(3곳)·DeepGEMM 호출(5곳)에서. `auto` top-k는 `persistent`로 가고, index-K 쓰기는 stock CUDA op 가능. 초판 B1 정정 반영 |
| V2 sparse 백엔드 부재·dense raise | **확인** | 후보 7개 전부 탈락, dense 강제 스위치 없음. 순서 정정: `cuda.py:441-445` ValueError가 먼저, `mla_attention.py:641-647`은 2차 방어선. 단서: `XPU_MLA_SPARSE`는 capability 제한이 없어 명시 지정 시 통과(정적) |
| V3 kv dtype 재작성 훅 | **확인(정밀화)** | `auto`/`fp8*`→`fp8_ds_mla`(SM100+FlashInfer만 `fp8_e4m3`), `bfloat16`/`float16` 명시만 우회. 리포 software-FP8 패치와 레이아웃(576 B vs 656 B)·트리거·대상 클래스 모두 비호환. 분할은 6개 |
| V4 SM80 sparse 백엔드 규모 | **부분(축소)** | 상속 4클래스, 추상 훅 1개(`_forward_mqa_kernel`); 기존 `decode_attention_fwd(page_size=1)` 재사용으로 신규 커널 0개·약 210~260줄. prefill 전용 커널 불필요. `index_skip_topk_offset`은 소비처 없음 |
| V5 pin 드리프트 | **패치별 상이** | kda 삭제(upstream 반영) · marlin 구조적 rebase(경로 이동, ops.cu 2 hunk 재작성) · mla-fp8 재설계(K3.1 target 우회, 11/25 hunk 실패) · dspark-bf16 퇴역 후보. pin 상향 = base 재빌드 + 의존성 드리프트 |
| A1 앵커 전수 | 완료 | 595행(tracked 152, 84파일) — 부록 A |
| A2 게이트 등록 | 완료 | 18지점 — 부록 B. `PERF_POLICY` 누락 시 스윕 exit 3, relay 필터 `k0` 버그 |

초판 대비 정정: B1 메커니즘·패치 지점, B2 에러 순서·prefill 커널 불필요·k12 규모 L→M, "5분할"→6분할, `index_skip_topk_offset` 의미 부여 삭제, draft `kv_cache_dtype` 기본값 주입 추가, 게이트 이름 `k12_sparse_mla_sm80` 통일, pin 상향 규모 M→M~L.

## 9. 출처

- HF `moonshotai/Kimi-K3.1-Pro` @ `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`: config.json · model.safetensors.index.json · configuration_kimi_k3.py · README.md (2026-11-09 fetch, K3 `9f62e4e9`와 diff)
- HF `Inferact/Kimi-K3.1-Pro-DSpark-preview` @ `6a2ef20a`, `Inferact/Kimi-K3-DSpark` @ `cf6b8244`: config.json · README.md
- vLLM main @ `a3c14efbc77c018e8bf579935095a87721349e05` (2026-11-09): `vllm/models/kimi_k3/**`, `vllm/model_executor/layers/{sparse_attn_indexer.py, indexer_topk.py, mla.py, attention/{mla_attention,sparse_mla_attention}.py}`, `vllm/v1/attention/backends/mla/**`, `vllm/platforms/cuda.py`, `vllm/config/{kernel,attention,speculative}.py`, `vllm/utils/deep_gemm.py`, `vllm/model_executor/models/{registry,config}.py`, `CMakeLists.txt`, `cmake/external_projects/deepgemm.cmake`, `csrc/{moe/marlin_moe_wna16,quantization/marlin,cache_kernels.cu,topk.cu}`; DeepGEMM `vllm-project/DeepGEMM@82ca617e` `csrc/apis/attention.hpp`
- vLLM PR #50000(병합 07-30, merge `aeeb36b1`) · #66030 · #66028 · #66037 · #66532 · #65685 · #66027 · #66145 · 이슈 #66524 (GitHub API, 2026-11-09)
- vLLM 블로그 https://vllm.ai/blog/2026-11-04-kimi-k31-pro · recipe https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro (Updated 2026-11-08) · 제3자 https://www.orcarouter.ai/blog/kimi-k3-1-vs-kimi-k3


## 부록 A — 리포 안의 K3 고정 지점 전수 (독립 검증 산출 원문, HEAD `9077457` 기준)

#### 2.1 표

| 영역 | 파일 | 행:내용 | 변경 종류 |
|---|---|---|---|
| docker | [Dockerfile.openai](/home/jimmy/repos/k3-sm80/docker/Dockerfile.openai) | `:11` `ARG BASE_IMAGE=ghcr.io/jimmylegendary/k3-sm80-kernelkit@sha256:d5fc27…` · `:12` `ARG VLLM_COMMIT=0498dc7ea92…` · `:167` `org.opencontainers.image.revision="0498dc7e…"` · 외 `:171,:182` | 값 교체 + base 재빌드 |
| docker | 〃 | `:30-33` `git clone … && git fetch origin pull/50000/head:kimi-k3-pr … checkout --detach "${VLLM_COMMIT}"` | 변경 불필요 (main sha 가능, fetch 줄은 불필요) |
| docker | 〃 | `:36-46` `git apply --check kda…`, `grep -Fq '(i_tg * H + i_h).to(tl.int64) * K * V'`, `…csrc/libtorch_stable/moe/…/ops.cu` ×2 | 값 교체 (kda 삭제, csrc 경로 변경) |
| docker | 〃 | `:73-78` `test -s csrc/libtorch_stable/moe/…/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu` | 값 교체 (`csrc/moe/…`) |
| docker | 〃 | `:88-117` python-overlay (patch 2개 + py_compile 6파일), `:132-156` `COPY --from=…` 8파일, `:128,:170` `K3_PATCHSET=kda-sm80-int64+…` | 값 교체 (패치 집합 변경) |
| docker | 〃 | `:196-197` `assert "(i_tg * H + i_h).to(tl.int64)…" in chunk` · `:211` `hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")` | `:196` 삭제, 나머지 유지 |
| docker | [Dockerfile.full](/home/jimmy/repos/k3-sm80/docker/Dockerfile.full) (PIN 3, kimi-k3 5) | `:33` `ARG VLLM_COMMIT=0498dc7e…` · `:63` `git fetch origin pull/50000/head:kimi-k3-pr` · `:142` `assert torch.__version__.startswith("2.13.0")` | 값 교체 (torch 2.14.0) |
| docker | [build-openai.sh](/home/jimmy/repos/k3-sm80/docker/build-openai.sh) | `:6` `image="${K3_OPENAI_IMAGE:-k3-sm80-openai:0498dc7e-sm80-v3}"` · `:23` `--build-arg "VLLM_COMMIT=0498dc7e…"` | 값 교체 |
| docker | [release-openai.sh](/home/jimmy/repos/k3-sm80/docker/release-openai.sh) (PIN 8, 이미지 태그 1) | `:5` 이미지 태그 · `:6` `…/openai-0498dc7e-sm80-v3` 외 6행 | 값 교체 |
| docker | [verify_image.py](/home/jimmy/repos/k3-sm80/docker/verify_image.py) (PIN 2, kimi_k3 5) | `:12` `COMMIT = "0498dc7e…"` · `:17-19` patchset 라벨 · `:90` `expected = "0498dc7e…"` · `:94-95` `kimi_k3_tool_parser` | 값 교체 |
| docker | [PULL.md](/home/jimmy/repos/k3-sm80/docker/PULL.md) (PIN 7, 이미지 태그 2) | `:9` `…kernelkit:full … kimi-k3 0498dc7e` · `:62` `k3-sm80-openai:0498dc7e-sm80-v3` 외 7행 | 값 교체 (문서) |
| docker | [GATE-RESULT.md](/home/jimmy/repos/k3-sm80/docker/GATE-RESULT.md) | `:53` `VENDOR_PIN=0498dc7e…` | 변경 불필요 (이력) |
| docker | [patches](/home/jimmy/repos/k3-sm80/docker/patches) 4개 (kimi_k3 14, skip.*layer 2, `indexer=None` 1) | `kda…:1` `diff --git a/vllm/models/kimi_k3/…/kda/chunk.py` 등 | kda 퇴역, marlin rebase, mla-fp8 재설계, dspark-bf16 rebase 경미 |
| docker | docker/logs/ 3파일 | PIN 410행, kimi-k3 4행 | 생성물 (gitignored) |
| deploy | [render_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/render_lanes.py) | `:57` `"K3_IMAGE": "k3-sm80-openai:0498dc7e-sm80-v3"` · `:58` `"K3_TARGET_MODEL": "moonshotai/Kimi-K3"` · `:59` `"K3_DRAFT_MODEL": "Inferact/Kimi-K3-DSpark"` | 파라미터화 (LaneSpec에 model 필드) |
| deploy | 〃 | `:48` `"model": "Inferact/Kimi-K3-DSpark"` · `:50` `"attention_backend": "TRITON_MLA"`(`kv_cache_dtype` 키 없음) | 값 교체 + 키 신규 |
| deploy | 〃 | `:61` `K3_SERVING_ROOT …/Kimi-K3` · `:91` `VLLM_ATTENTION_BACKEND: TRITON_MLA` · `:101` `"93 94 95 96 97" if spec.fp8_kv and spec.dspark` · `:115-117` `kimi-k3`/`kimi_k3` ×2 · `:203` "pinned 93-layer target" 문구 | 값 교체 / 파라미터화 |
| deploy | [lane_schema.py](/home/jimmy/repos/k3-sm80/deploy/tooling/lane_schema.py) | `:215` `VLLM_KV_CACHE_DTYPE …` · `:221` `"93 94 95 96 97" if …` · `:231-233` spec 필수/금지 규칙 | 값 교체 |
| deploy | [lane_specs.py](/home/jimmy/repos/k3-sm80/deploy/tooling/lane_specs.py) | `:29-43` LaneSpec 5개 (model 필드 없음) | 파라미터화 |
| deploy | [gates.py](/home/jimmy/repos/k3-sm80/deploy/tooling/gates.py) · [validate_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/validate_lanes.py) | `gates.py:27-32` GATE_MATRIX · `:37-42` PREREQUISITE_GATES · `:45-49` `_REPORT_NAMES` · `:193-199` `_gate_timeout` · `validate_lanes.py:19-29` `_EVIDENCE_IDS` | 신규 (k11/k12 행) |
| deploy | [command.py](/home/jimmy/repos/k3-sm80/deploy/tooling/command.py) | `:34,:182` `--attention-backend` · `:53-55,:192-196` served-name/parser 플래그 · `:212-220` spec/skip-layers (12행) | 변경 불필요 (키 기반) |
| deploy | [templates](/home/jimmy/repos/k3-sm80/deploy/tooling/templates) 6개 | `RUNBOOK.md.tmpl:5` `moonshotai/Kimi-K3` · `:43` `…/Kimi-K3` · `:47-48` HF 캐시 `find`(target·draft) · `Makefile.tmpl:16,56-66,106-108` gate 타깃 · `cluster.py.tmpl:44,49-51` | RUNBOOK 값 교체, Makefile 신규 |
| deploy | `deploy/01-…`~`05-…` ×5 (`.env`·RUNBOOK·Makefile·compose·scripts) | `.env:5-9` `K3_IMAGE=…0498dc7e-sm80-v3`, `K3_TARGET_MODEL=moonshotai/Kimi-K3`, `K3_DRAFT_MODEL=Inferact/Kimi-K3-DSpark`, `K3_SERVING_ROOT=…/Kimi-K3` · `:43` `TRITON_MLA` · `:62-64` served-name/parsers · lane05 `:49` skip `'93 94 95 96 97'` · lane03/05 `:61` spec JSON | 생성물 (커밋되는 파일; `python3 -m deploy.tooling.render_lanes` 재생성, drift test 있음) |
| kernel-kit | [shapes.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/shapes.py) | `:3` `HF moonshotai/Kimi-K3 @ 9f62e4e9fff…` · `:31-33` `N_LAYERS=93, KDA_LAYERS=69, MLA_LAYERS=24` · `:63` `MLA_CACHE_D = 576` · `:155` 주석 pin (MLA_* 32행, 재사용 가능) | 변경 불필요 + 출처 주석 갱신 |
| kernel-kit | 〃 | `:80-88` `DSPARK_REQUIRED_CASES` · `:116-126` `KERNEL_IDS`(9종) · `:128-138` `KERNEL_TITLES` · `:333-359` `_CROSSCHECK` | 신규 (A2) |
| kernel-kit | [dspark_ref.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/dspark_ref.py) | `:10` `TARGET_AUX_LAYER_IDS = (2, 23, 47, 71, 89)` · `:11` `DSPARK_BLOCK_SIZE = 7` | 값 교체 |
| kernel-kit | [k10 harness](/home/jimmy/repos/k3-sm80/kernel-kit/k10_k3_dspark_sm80/harness.py) | `:29,:80` `TARGET_AUX_LAYER_IDS` · `:37` `TARGET_DEFAULT="moonshotai/Kimi-K3"` · `:38` `DRAFT_DEFAULT="Inferact/Kimi-K3-DSpark"` · `:202` `default="kimi-k3"` | 파라미터화 |
| kernel-kit | [run_all.sh](/home/jimmy/repos/k3-sm80/kernel-kit/run_all.sh) | `:16` `KERNELS_DEFAULT="k01_kda_chunk … k09_mla_fp8_kv_sm80"` · `:17` | 신규 |
| kernel-kit | k01~k09 harness 9개 | `PINS` dict 한 줄씩: k01 `:43-44`, k02 `:47-48`, k03 `:50-51`, k04 `:86`, k05 `:66-67`, k06 `:63-64`, k07 `:72`, k08 `:36-37`, k09 `:34-36`, 본문 주석 pin 다수 (PIN 20행) | 값 교체 |
| kernel-kit | [vendor](/home/jimmy/repos/k3-sm80/kernel-kit/vendor) 20파일 (PIN 21행, kimi_k3 헤더 포함) | `fla_kda/chunk.py:1` `# vendored from vllm-project/vllm@0498dc7e… …/kda/chunk.py` · `__init__.py:19` `VENDOR_PIN = "0498dc7e…"` · `STUBS.md:3` | 재vendor |
| kernel-kit | [kda_ref.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/kda_ref.py) · [env_check.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/env_check.py) · [RUNNER.md](/home/jimmy/repos/k3-sm80/kernel-kit/RUNNER.md) · [VERIFICATION.md](/home/jimmy/repos/k3-sm80/kernel-kit/VERIFICATION.md) | `kda_ref.py:4,:11` pin 주석 · `env_check.py:127,:156` "kimi-k3 소스빌드" 안내 · `RUNNER.md:14,:39` pin 표기 | 값 교체 (문구) |
| kernel-kit | kernel-report/ 22파일 (PIN 20, HF sha 22, MLA_* 8, kimi_k3 24 등) | `k08….md:14` `핀: {"vllm_kimi_k3": "0498dc7e…"}` 등 | 생성물 (gitignored) |
| tests | [test_openai_image_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_openai_image_contract.py) · [test_fp8_patch_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_fp8_patch_contract.py) | `:27` pin assertIn · `:28` base digest · `:32-33` patch 이름 · `:49,:56` kda 패치 토큰 `.count(".to(tl.int64)") >= 6` · fp8 `:8-12,:79-83` | 값 교체 (kda 삭제) |
| tests | [test_k10_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k10_contract.py) | `:41` `(2, 23, 47, 71, 89)` · `:43` block 7 · `:73` `"moonshotai/Kimi-K3"` · `:74` `"Inferact/Kimi-K3-DSpark"` | 값 교체 |
| tests | [test_commands.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_commands.py) · [test_gate_matrix.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_gate_matrix.py) · [test_smoke_metrics.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_smoke_metrics.py) | `test_commands.py:36` `"moonshotai/Kimi-K3"` · `:59` skip `["93".."97"]` · `test_gate_matrix.py:20-26` `EXPECTED` · `:75` skip env · `test_smoke_metrics.py:35` `model_name="kimi-k3"` | 값 교체 / 신규 행 |
| tests | [test_k09_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k09_contract.py) | `:31-32` `MLA_CACHE_D==576`, `PAGE_SIZE==64` | 변경 불필요 |
| tests | 그 외 deploy 구조 테스트 (lane 키워드 약 40행) | lane 구조 검증 | 변경 불필요 |
| docs | [SUMMARY.md](/home/jimmy/repos/k3-sm80/SUMMARY.md) · [README.md](/home/jimmy/repos/k3-sm80/README.md) | `SUMMARY.md:5` `moonshotai/Kimi-K3` · `:30` HF 표 · `:32` `0498dc7ea92…` · `:88`, `:113` · `README.md:63` "k01~k07" | 값 교체 (문서) |
| docs | [PLAN.md](/home/jimmy/repos/k3-sm80/docs/PLAN.md) · [A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md) | `PLAN.md:32,:53,:59` pin · `A100-5LANE-RESULTS.md:9-17` gate/lane 행 | 문구 갱신 / 신규 행 |
| docs | docs/superpowers/ plans·specs 3개 (lane 키워드 94, `0498dc7e` 11행) | `…five-lane-serving.md:222` `K3_IMAGE=k3-sm80-openai:0498dc7e-sm80` | 변경 불필요 (이력) |
| docs | [K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md) | REV 101행 (dsa/indexer 29, `0498dc7e`·`9f62e4e9`·모델명 등 다수) | 변경 불필요 (본 검토 문서) |
| phase0 | [k3_model.py](/home/jimmy/repos/k3-sm80/phase0/k3_model.py) (36행) | `:43-44,:69,:772-774,:1039-1040` pin · `:76-91` `MLA_*` 상수 · `:521-522` 층 패턴 `i % 4 == 0 or i == N_LAYERS` | pin 값 교체 + DSA 항 신규 |
| phase0 | [verdict.json](/home/jimmy/repos/k3-sm80/phase0/verdict.json) · [REPORT.md](/home/jimmy/repos/k3-sm80/phase0/REPORT.md) · [VERIFICATION.md](/home/jimmy/repos/k3-sm80/phase0/VERIFICATION.md) | `verdict.json:37,39` · `REPORT.md:22,24` | 변경 불필요 (결과 기록) |
| recon | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) · [vllm-upstream.md](/home/jimmy/repos/k3-sm80/recon/vllm-upstream.md) · [kda-fla.md](/home/jimmy/repos/k3-sm80/recon/kda-fla.md) | `k3-config.md:7,:11-15` HF sha · `vllm-upstream.md:7,:61,:65` pin·클래스 · `kda-fla.md:62,:133` | 변경 불필요 (K3 사실) + `recon/k31-config.md` 신규 |
| a100-kit | [bootstrap.sh](/home/jimmy/repos/k3-sm80/a100-kit/bootstrap.sh) · [RUNBOOK.md](/home/jimmy/repos/k3-sm80/a100-kit/RUNBOOK.md) · [probe_k3_modules.py](/home/jimmy/repos/k3-sm80/a100-kit/probe_k3_modules.py) · [DECISION-TREE.md](/home/jimmy/repos/k3-sm80/a100-kit/DECISION-TREE.md) | `bootstrap.sh:17` `K3_PIN=0498dc7e…` · `RUNBOOK.md:11,:155` · `:13` torch==2.13.0, flashinfer 0.6.15.post1 · `probe_k3_modules.py:45-47` · `DECISION-TREE.md:151` `git diff 0498dc7..HEAD … csrc/libtorch_stable/moe/` | 값 교체 |
| run | [20-kernel-test.sh](/home/jimmy/repos/k3-sm80/run/20-kernel-test.sh) | 요청 패턴 히트 0. `:16` triton 티어 커널 목록이 하드코딩 | k08/k09 선례상 변경 불필요 |
| root | `kernel_kit_import.py`, `.gitignore` | 히트 0 | 변경 불필요 |

#### 2.2 패턴별 히트 수 (대소문자 구분)

| 패턴 | 전체 | GEN | REV | 손작성 |
|---|---|---|---|---|
| `0498dc7e` | 551 | 430 | 4 | 117 |
| `9f62e4e9` | 57 | 22 | 3 | 32 |
| `moonshotai/Kimi-K3` | 70 | 14 | 4 | 52 |
| `Inferact/Kimi-K3-DSpark` | 23 | 1 | 3 | 19 |
| `k3-sm80-openai:` | 16 | 0 | 1 | 15 |
| `kimi-k3` | 84 | 4 | 3 | 77 |
| `93 94 95 96 97` 리터럴 | 9 | 0 | 2 | 7 |
| `skip.*layer` | 37 | 0 | 4 | 33 |
| `TARGET_AUX_LAYER_IDS` | 8 | 0 | 2 | 6 |
| `MLA_*`/`N_LAYERS`/`KDA_LAYERS` | 87 | 8 | 3 | 76 |
| `KERNELS_DEFAULT` | 3 | 0 | 1 | 2 |
| lane 키워드류 (`lane`·`deploy/0N-`·`--max-model-len`·`--kv-cache-dtype`·`--speculative`) | 641 | 0 | 14 | 627 |
| `dspark`/`DSpark` | 352 | 6 | 14 | 332 |
| `kimi_k3` | 155 | 24 | 10 | 121 |
| `KimiK3` | 12 | 0 | 4 | 8 |
| `dsa`/`indexer` | 30 | 0 | 29 | 1 (패치 context `indexer=None`) |

- 대소문자 무시로 늘어나는 행:
  - `kimi-k3` +156 (대문자 `Kimi-K3` 경로형)
  - `dspark` +49
  - `skip.*layer` +27 (`SKIP_LAYERS` 환경변수)
  - `MLA_*` +97
  - `TARGET_AUX` +1 (GEN)
  - `dsa|indexer` +22 (`IndexError` 오탐 12 + 리뷰 문서)
- 손작성 영역별 분포 (0498dc7e 117행): a100-kit 11, deploy 6, docker 26, docs 13, kernel-kit 49, phase0 5, recon 2, root 4, tests 1.
- deploy 히트의 대부분은 생성 lane 디렉터리 `deploy/0N-*`입니다. 예: `0498dc7e`는 6행 중 5, HF 12행 중 10, 건너뛰는 층 22행 중 15, DSpark 218행 중 168행입니다.
- 요청 패턴 밖의 추가 앵커: 이전 main pin `d223c900`이 73행(손작성 44행)입니다. 예: `kernel-kit/k0N/harness.py`의 `PINS`, `phase0`, `RUNNER.md`. 재pin 때 함께 정리해야 합니다.

#### 2.3 다음 검증용 숫자

- 지시문의 명령 출력은 **595**입니다 (HEAD `9077457`의 working tree).
- 분해:
  - 생성물(gitignored) 443 = docker/logs 410 + kernel-report 33
  - tracked 152 = 리뷰 문서 7 + 손작성 145
  - `git grep` tracked-only 결과 152와 일치
- 히트가 있는 파일은 84개입니다.
- 리뷰 문서를 수정하면 이 값이 달라지고, fresh clone이라면 152 근처입니다.

---


## 부록 B — kernel-kit 신규 게이트(k11, k12) 등록 지점 (독립 검증 산출 원문)

#### 3.1 등록 지점

| # | 파일:행 | 함수/상수 | k11·k12 추가 시 변경 |
|---|---|---|---|
| 1 | [shapes.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/shapes.py):116-126 | `KERNEL_IDS` | `k11_dsa_indexer_sm80`, `k12_sparse_mla_sm80` 추가. 누락 시 [report.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/report.py):56 assert, [shapes.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/shapes.py):217 "미등록" |
| 2 | shapes.py:128-138 | `KERNEL_TITLES` | 제목 2개 (report.py:205는 `.get`이라 누락해도 에러는 없고 md 제목만 빔) |
| 3 | shapes.py:164-206 (패턴 `:92-108`, `:205`) | `REQUIRED_SHAPES` | 케이스 dict 2개. k09처럼 모듈 상수 dict를 만들고 `dict(...)`로 연결. **`:383`과 timing `_selftest`가 각 커널의 첫 케이스를 `perf_bytes_model(**case)`에 넣으므로 첫 케이스가 모델 입력 키를 가져야 함** |
| 4 | shapes.py:242-327 | `perf_bytes_model` | k11: indexer K 스트림(132 B/tok/층) + q + top-k, k12: 2048행 × 576 gather + q/o. else 분기(`:325`)가 KeyError |
| 5 | shapes.py:53-65 (MLA 블록), `:333-370` (`_CROSSCHECK`, `self_check`) | 상수 | `DSA_INDEX_HEADS=32`, `DSA_INDEX_HEAD_DIM=128`, `DSA_TOPK=2048`, `DSA_LAYERS=24`, `DSA_INDEX_K_BYTES=132`, `DSA_INDEX_SKIP_TOPK_OFFSET=1`. 대조표에 `recon/k31-config.md`(신규) 출처 추가, `DSA_LAYERS == MLA_LAYERS` assert |
| 6 | shapes.py:80-88 | `DSPARK_REQUIRED_CASES` | k10 전용이라 해당 없음. 주석 `:80-81`: k10은 KERNEL_IDS/run_all에 의도적으로 비등록 |
| 7 | [timing.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/timing.py):109-147 | `PERF_POLICY` | 키 필수. 없으면 `:266` KeyError라 run_all.sh:95 selftest가 FAIL해 **전체 스윕이 exit 3**. P1 "측정 전용"은 `mode="measure_only", cv_max=…` (k06 `:130-133`, k09 `:142-146` 선례) |
| 8 | [tolerance.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/tolerance.py):44-84 | `TOL` | `compare()` 사용 시 필수(`:159`). k09는 비트 비교라 항목이 없음. top-k 인덱스 비교는 `TolPolicy`로 못 표현하므로 집합 일치 판정을 별도로 둬야 함 |
| 9 | report.py:56-57, 205, 232, 235, 325 | `KernelReport.__init__` assert, title, aggregate | 코드 변경 불필요(자동 확장), glob `k[0-9][0-9]_*.json`은 k11/k12 매칭. **하드코딩된 "N/N" 없음.** 낡은 문구는 `:334` help "기본: 7종 전부" 하나 |
| 10 | [relay.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/relay.py):68, `:5`, `:74` | `build` 파일 필터 | `fn.startswith("k0")` 때문에 **k10·k11·k12가 RELAY에서 조용히 탈락**. 정규식(`^k\d\d_`)으로 변경 필요. 줄 수 계약(`:5` ≤12줄): 현재 R0 + 9 + R9 = 11줄. 11커널이면 13줄로 초과하므로 압축 규칙이 필요 |
| 11 | [env_check.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/env_check.py) | 커널 레지스트리 없음 | 변경 불필요. vendored torch+triton-only면 `:148-156` 문구에 한 줄 추가 |
| 12 | [run_all.sh](/home/jimmy/repos/k3-sm80/kernel-kit/run_all.sh):16, `:2`, `:92-96`, `:104-105`, `:120`, `:130` | `KERNELS_DEFAULT` | 목록에 추가. `:104-105`는 `kXX/harness.py` 탐색, `:93-96`의 selftest 4종은 KERNEL_IDS를 일괄 순회, `:120`은 `--expect $KERNELS` |
| 13 | `kernel-kit/k11_dsa_indexer_sm80/harness.py`, `k12_sparse_mla_sm80/harness.py` (신규) | 하네스 | 템플릿은 **k09**: [harness](/home/jimmy/repos/k3-sm80/kernel-kit/k09_mla_fp8_kv_sm80/harness.py)의 `build_dry_report`(`:45`), `KernelReport` 4게이트(`:472`), parent/child + compute-sanitizer, `--cpu-dry-run/--report-dir/--phase0-verdict` |
| 14 | `kernel-kit/common/dsa_ref.py` (신규) | fp32 참조 | `fp8_ref.py`, `mxfp4_ref.py`, `dspark_ref.py` 패턴. 테스트의 import는 [kernel_kit_import.py](/home/jimmy/repos/k3-sm80/kernel_kit_import.py):13-17의 `import_common`이라 수정 불필요 |
| 15 | [vendor](/home/jimmy/repos/k3-sm80/kernel-kit/vendor) 신규 커널 모듈 | vendored 커널 | [check_no_vllm.py](/home/jimmy/repos/k3-sm80/kernel-kit/vendor/check_no_vllm.py):60-63이 vendor 전체를 자동 스캔(허용 `{torch, triton, numpy, packaging}`, `:25`). `vendor/__init__.py`의 `_EXPORTS`(`:22`)는 k09 선례상 불필요(`harness:165`가 직접 import). `STUBS.md` 표만 갱신 |
| 16 | `tests/kernel/test_k11_contract.py`, `test_k12_contract.py`, `test_dsa_ref.py` (신규) | 테스트 | [test_k09_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k09_contract.py) 패턴: REQUIRED_SHAPES 키 집합, 상수, vendor 소스 계약, dry-run INCOMPLETE와 CLI JSON. 기존 테스트 중 KERNEL_IDS/KERNEL_TITLES의 내용·길이를 단언하는 것은 없음(전수 grep) |
| 17 | [build.sh](/home/jimmy/repos/k3-sm80/docker/build.sh):64-76 (`gate_tier`), [20-kernel-test.sh](/home/jimmy/repos/k3-sm80/run/20-kernel-test.sh):16 | 컨테이너 gate, triton 티어 목록 | 레지스트리 누락은 테스트가 아니라 `shapes.py --check`, `timing.py --selftest`, `report.py --selftest`, `run_all.sh --cpu-dry-run`에서 잡힘. triton 티어 목록은 k08/k09 미포함이므로 변경 불필요 |
| 18 | (lane 연동 시에만) [gates.py](/home/jimmy/repos/k3-sm80/deploy/tooling/gates.py):27-33, 37-43, 45-52, 193-201, 204-288 | `GATE_MATRIX`, `PREREQUISITE_GATES`, `_REPORT_NAMES`, `_gate_timeout`, `_generic_gate_commands` | `_gate_timeout`은 미등록 gate에 KeyError, `_generic_gate_commands`는 ValueError(`:288`). 같이 바꿀 곳: [Makefile.tmpl](/home/jimmy/repos/k3-sm80/deploy/tooling/templates/Makefile.tmpl):15-17, 56-66, [validate_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/validate_lanes.py):19-29, [A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md):9-17, [test_gate_matrix.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_gate_matrix.py):20-26, [test_evidence.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_evidence.py):32,38. 이후 lane 5개 재생성 |

#### 3.2 k10이 등록된 곳 (역추적)

- k10은 S2 커널이 아니라 서비스 통합 게이트입니다.
- 등록된 곳:
  - 디렉터리 `kernel-kit/k10_k3_dspark_sm80/`
  - `shapes.DSPARK_REQUIRED_CASES`
  - `tests/kernel/test_k10_contract.py`
  - lane 쪽 `gates.py` GATE_MATRIX(lane 03/05)와 `_REPORT_NAMES["k10"]`
  - Makefile `gate-dspark`
  - `validate_lanes._EVIDENCE_IDS`와 `docs/A100-5LANE-RESULTS.md`
- 등록되지 않은 곳: `KERNEL_IDS`, `run_all.sh`, `timing`, `tolerance`, RELAY.
- 자체 스키마 `k10-dspark-report/v1`을 쓰기 때문에 같은 report 디렉터리에서 `report.py aggregate`가 "스키마 위반 리포트"로 분류합니다. 기존 [SUMMARY.md](/home/jimmy/repos/k3-sm80/kernel-kit/kernel-report/SUMMARY.md)에 실제로 나와 있습니다.
- 4게이트(NUMERIC/SANITIZER/SHAPES/PERF) 커널의 템플릿으로는 부적합하고 k09가 맞습니다.

#### 3.3 신규 게이트 1개 추가 체크리스트

1. `shapes.py`: 상수, 케이스 dict, `KERNEL_IDS`, `KERNEL_TITLES`, `REQUIRED_SHAPES`, `perf_bytes_model`, `_CROSSCHECK`.
2. `timing.PERF_POLICY`를 추가합니다. 먼저 하지 않으면 전체 스윕이 exit 3입니다.
3. NUMERIC에 `compare()`를 쓰면 `tolerance.TOL`도 추가합니다.
4. `common/<x>_ref.py`와 `vendor/<x>.py`를 만들고 vllm import는 0건으로 둡니다.
5. `kXX_<name>/harness.py`를 만듭니다. 템플릿은 k09이고 `KernelReport`를 씁니다.
6. `run_all.sh` `KERNELS_DEFAULT`에 추가하고, `relay.py:68` 필터와 줄 수 계약을 고칩니다.
7. `tests/kernel/test_kXX_contract.py`를 추가합니다.
8. 확인 명령: `python3 kernel-kit/common/shapes.py --check`, `timing.py --selftest --phase0-verdict phase0/verdict.json`, `report.py --selftest`, `bash kernel-kit/run_all.sh --cpu-dry-run`, `python3 -m unittest`.
9. lane 연동이 필요하면 3.1의 #18과 생성 lane 재렌더, 문서 문구를 함께 바꿉니다.

---

