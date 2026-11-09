# Kimi K3.1-Pro 지원 검토 — k3-sm80 (2026-11-09)

> 상태: **검토(구현 전)**. 1차 출처(HF 체크포인트·vLLM main 소스)를 직접 읽어 확정한 사실과,
> 그로부터 도출한 "무엇을 바꿔야 하는가"를 적는다. 구현 계획은 §5, Jimmy 결정 요청은 §6.
> 독립 검증(ultracode 워크플로, 읽기 전용)의 판정은 §8에 반영한다.

## 0. 한 줄 결론

K3.1-Pro는 K3와 **가중치 레이아웃이 사실상 동일**하다 — 차이는 24개 Gated MLA 층에 붙은 **DSA(DeepSeek
Sparse Attention) indexer 텐서 120개(+357 MB)** 뿐이고 KDA 69층·MoE(MXFP4 expert 896)·토크나이저는 그대로다.
그러나 **SM80(A100)에서는 vLLM의 DSA 경로가 두 지점에서 하드 블록**된다.

| # | 블록 지점 | 근거(vLLM main @ `a3c14efb`, 2026-11-09) | 함의 |
|---|---|---|---|
| B1 | DSA indexer 초기화가 **DeepGEMM 필수** | `sparse_attn_indexer.py:1285` `if is_cuda() and not has_deep_gemm(): raise RuntimeError` · `platforms/cuda.py:693-695` `support_deep_gemm()` = SM90/SM100/SM120만 | A100에서 엔진이 뜨지 않음 → **SM80용 indexer 경로 신규**(Triton MQA logits + top-k + index-K cache) |
| B2 | **sparse MLA 백엔드가 SM80에 하나도 없음** + dense 백엔드로 DSA 층을 돌리면 raise | `flashmla_sparse.py:203-204` major∈[9,10] · `flashinfer_mla_sparse.py:117-118` major==10 · `:216-217` major==12 · `flashattn_mla_sparse.py:82-83` major==9 · `cuda.py:137-151` SM80 분기의 sparse tail 전부 SM90 전용 · `mla_attention.py:642-644` "sparse (DSA) attention layer requires a sparse backend" | **SM80용 Triton sparse MLA 백엔드 신규**(decode gather-MQA + prefill) |

따라서 K3.1 지원 = **(a) base pin을 main(≥ split 3/5 #66037, 11-09)으로 올리고 SM80 패치 4개를 rebase·퇴역**
+ **(b) SM80 DSA indexer 경로(k11)** + **(c) SM80 Triton sparse MLA 백엔드(k12)** + **(d) 리포 앵커 갱신**
(핀·lane·DSpark draft·shape 상수·gate 등록). 새 커널 2종이 핵심 작업량이고 나머지는 리베이스·파라미터화다.
K3 자체의 A100 증거가 아직 전부 `INCOMPLETE`이므로([A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md)),
**새 pin 위에서 K3 회귀(k01~k08)와 K3.1 신규 gate(k11·k12)를 같은 A100 라운드에서 돌리는 것**이 P1의 종료 조건이다.

## 1. 사실관계 (1차 출처, 핀 고정)

### 1.1 체크포인트 — `moonshotai/Kimi-K3.1-Pro` @ `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` (lastModified 2026-11-05)

K3 핀(`9f62e4e9`, [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md))과 `config.json`·`model.safetensors.index.json`을 직접 diff 했다.

| 항목 | K3 (`9f62e4e9`) | K3.1-Pro (`d9e99b07`) | 비고 |
|---|---|---|---|
| architectures / model_type | `KimiK3ForConditionalGeneration` / `kimi_k3` | `KimiK31ForConditionalGeneration` / `kimi_k31` | vLLM registry는 둘 다 같은 클래스 `KimiK3ForConditionalGeneration`으로 매핑(`registry.py:485-491`) |
| HF remote modeling | `modeling_kimi_k3.py` auto_map 있음 | **AutoModel auto_map 제거**(configuration만) | HF transformers로는 직접 못 돌림 — vLLM/SGLang 전용 |
| `dsa_config` | 없음 | `dsa_layers=[4,8,…,92,93]`(24개, 1-indexed = 기존 Gated MLA 층 전부) · `index_num_heads=32` · `index_head_dim=128` · `index_topk=2048` · `index_rope_dim=0` · `index_k_norm_type=layernorm` · `index_skip_topk_offset=1` · `index_topk_freq=1` | 모델카드: "replace the full-attention Gated MLA layers with Gated DSA" |
| 텐서 패턴 차이 | — | **추가 5패턴 × 24층 = 120개**: `self_attn.indexer.{wq_b, wk, weights_proj, k_norm.weight, k_norm.bias}` (0-indexed 층 3,7,…,91,92) | 그 외 497,220개 텐서 패턴·개수 **완전 동일**, 96 shard 동일 |
| total_size | 1,560,860,324,864 B | 1,561,217,377,280 B (**+357,052,416 B**) | indexer bf16 ≈ 178.5M param(층당 7.44M) |
| MXFP4 범위 | routed expert w1/w2/w3만 | 동일(ignore 목록에 `block_sparse_moe.gate`·`routed_expert_(up|down)_proj` 정규식이 **명시**됐을 뿐, K3에서도 bf16이었음) | 바이트 변화 없음 |
| MTP / draft 층 | 없음(`num_nextn_predict_layers=0`) | 없음 | DSpark draft는 별도 리포 |
| tokenizer / generation_config | — | 동일(`tokenizer_config.json`의 빈 `extra_special_tokens` 키 1개 삭제뿐) | `tiktoken.model` 동일, 렌더러 `encoding_k3.py`(Python) |
| transformers_version | 4.56.2 | 4.57.6 | |
| 1M context / vocab 163840 / KDA·MoE·AttnRes 상수 | — | 동일 | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §1~§3 전부 그대로 유효 |

### 1.2 vLLM upstream

| 항목 | 상태(2026-11-09) | 근거 |
|---|---|---|
| K3 코드 | PR #50000 **2026-07-30 main 병합** → `vllm/models/kimi_k3/{nvidia,amd,common}` | 리포 핀 `0498dc7e`(07-28)은 병합 전 브랜치 — 브랜치 삭제됨 |
| K3.1 5분할 | 1/5 #66030 모델 파일(11-06 병합) · 2/5 #66028 Rust renderer(병합) · **3/5 #66037 KV cache·attention — 11-09 04:31Z 병합** · 4/5 #65685 ROCm·P/D·indexer top-k(open) · 5/5 #66027 DCP(open) | "Until [3/5] lands, K3.1 imports but does not serve" — 즉 **main은 오늘부터 K3.1을 서빙**(TP8, Blackwell) |
| 최소 참조 구현 | PR #66145(mgoin, do-not-merge) ≈ 2.5k 줄: K3.1-Pro를 TP4 B300에서 돌린 최소 변경 집합 | SM80 포팅 범위 판단의 좋은 기준점 |
| 릴리스 | v0.32.0(10-18), v0.33.0rc3(11-02). recipe: "**Requires vLLM >= 0.34.0 (not released yet — nightly wheel)**" | 공개 브랜치 없이 Docker 이미지 `vllm/vllm-openai:kimi-k31`만 — 사내는 registry 차단이라 **소스빌드 필수**(기존과 동일) |
| 지원 HW | B300/GB300/VR200(preview)/MI355X/TPU v7x/T-HEAD. **A100·Ampere·SM80 언급 0**(블로그·recipe·모델카드·제3자 분석 전부) | |
| 기본 커널(NVIDIA) | sparse MLA = FlashInfer sparse(trtllm-gen/PrimsTS) 또는 FlashMLA sparse; indexer = **DeepGEMM** `fp8_fp4_paged_mqa_logits`; 저지연 decode = CuTe DSL(SM100) | 전부 SM90+/SM100 |
| K3.1 KV 레이아웃 | split 3/5: **packed hybrid KV row**(MLA latent + indexer key + KDA state 페이지를 한 block row에) `get_packed_hybrid_block_size`; indexer K cache는 **fp8(132 B/token/층) 또는 mxfp4(SM100)만** (`dsa_cache.py:33-65`, `indexer.py:62-64`) | SM80에는 FP8 텐서코어가 없음 → 리포의 software-E4M3FN 방식 재사용 대상 |
| DSpark draft | `Inferact/Kimi-K3.1-Pro-DSpark-preview` @ `6a2ef20a`(11-05, "preview") — 5 dense layer 2.7B, block 7, `target_layer_ids=[75,79,83,87,91]`, MLA 96 heads·RoPE 64·`rope_theta=1e6` yarn, `markov_rank=512` | K3 draft(`Inferact/Kimi-K3-DSpark`: ids `[2,23,47,71,89]`, 64 heads, θ=5e4)와 **다름** → 리포 상수 갱신 필요. 블로그 acceptance length 4.597(@7 tokens, GSM8K) |

## 2. SM80 갭 분석 (vLLM main 원문 기준)

### B1. DSA indexer — DeepGEMM 하드 의존 → 엔진 기동 실패
- `KimiK3DSAIndexer`(`dsa.py:108-412`)는 upstream `torch.ops.vllm.sparse_attn_indexer`를 그대로 호출한다(`dsa.py:388-412`).
  그 op의 CUDA 구현은 logits를 전부 DeepGEMM `fp8_fp4_paged_mqa_logits`/`fp8_fp4_mqa_logits`로 계산한다.
  `SparseAttnIndexer.__init__`은 `has_deep_gemm()`(패키지 또는 vLLM wheel 동봉 사본의 **존재** 검사이지 capability 검사가 아님)이 거짓이면
  RuntimeError(`sparse_attn_indexer.py:1285-1289`), 참이면 첫 호출에서 DeepGEMM JIT로 넘어가는데 DeepGEMM 자체가 SM90/SM100/SM120 전용이다
  (`cuda.py:689-695` `support_deep_gemm()`, `deep_gemm.py:151-156`). **어느 쪽이든 SM80 경로 없음** — 실패 지점(init vs 첫 forward)만 환경에 따라 다르다.
  (XPU는 `xpu_fp8_mqa_logits`(`:864,:1075`), ROCm은 aiter로 플랫폼 분기 — **SM80 Triton 분기를 같은 자리에 추가**하는 것이 자연스러운 설계.)
- top-k: `SparseIndexerTopkBackend = {auto, deep_select, cooperative, persistent, per_row, flashinfer, torch, aiter}`(`config/kernel.py:172-181`).
  `deep_select`는 SM100, `cooperative`는 SM90+(`indexer_topk.py:156-163`). **`torch` 백엔드는 capability 제약이 없어 SM80에서 top-k 자체는 가능**(성능은 P2 과제).
- indexer K cache: 항상 양자화(`fp8` 기본 132 B/token/층, `mxfp4`는 SM100 opt-in). 삽입 커널 `indexer_k_norm_quant_insert`는 Triton이지만
  `tl.float8e4nv` 변환을 쓴다(`dsa_indexer.py:69-70,201-202`) — SM80에서 Triton 네이티브 fp8 변환 가용성은 **A100 실측 전까지 미확인** →
  리포가 이미 쓰는 **software E4M3FN 비트 변환**([mla_fp8_kv_sm80.py](/home/jimmy/repos/k3-sm80/kernel-kit/vendor/mla_fp8_kv_sm80.py))을 재사용하면 의존성이 없다.
- **필요 작업(k11)**: ① LayerNorm + software-FP8 quant + cache insert(Triton) ② paged MQA logits: `q[T,32,128]`(bf16 또는 fp8-emul) × `K[L,128]` → ReLU → head-weight 합 → `logits[T,L]`,
  decode/prefill(varlen·causal) 두 변형 ③ `torch.topk` 연결(`index_skip_topk_offset=1` 의미 반영) ④ kernel-kit fp32 참조 대조(NUMERIC) + sanitizer + K3.1 shape(32×128, top-2048, ctx 8k/128k/256k).

### B2. sparse MLA attention — SM80 백엔드 부재 + dense 폴백 금지
- `cuda.py:_get_backend_priorities` SM80 분기(`else`, `:137-151`)의 sparse tail = `FLASH_ATTN_MLA_SPARSE`(SM90) · `FLASHMLA_SPARSE`(SM90/100) · `FLASHINFER_MLA_SPARSE_SM90`(SM90) → **전부 `supports_compute_capability` 탈락**.
- 설령 `TRITON_MLA`를 강제해도 `MLAAttention.__init__`이 `use_sparse and not impl.is_sparse` → RuntimeError(`mla_attention.py:642-644`, split 3/5에서 "silently running full attention" 방지로 추가).
- dense로 돌리는 것은 올바른 우회도 아니다: K3.1은 top-2048 sparse로 학습된 모델이라 ctx > 2048+offset에서 dense attention은 **학습 분포 밖**(정확도 변화). 디버그 용도 외 제품 경로로 쓰지 않는다.
- **필요 작업(k12 + 백엔드 배선)**: `TritonMLASparseBackend`(가칭) — `is_sparse=True`, `supports_compute_capability: major==8`, `SparseMLACommon{Metadata,Impl}` 상속(가장 단순한 SM90 `flashattn_mla_sparse.py` 15.6 KB가 템플릿),
  top-k 인덱스 → 물리 slot 변환은 upstream Triton 유틸 재사용(`sparse_utils.py: triton_convert_req_index_to_global_index`, `index_group.py`),
  커널: ① decode gather-MQA(쿼리 heads_local × 선택 2048행 latent 576(bf16 / 리포 software-FP8), online softmax) ② prefill(쿼리별 상이한 top-k 집합, varlen) ③ bf16 latent 우선, software-FP8 latent는 P3.
  `fp8_ds_mla`(656 B 레이아웃)는 FlashMLA 전용이라 채택하지 않는다.

### B3. KV cache 기본값·레이아웃 — split 3/5 변화와 리포 software-FP8 패치의 충돌
- `KimiK31ForConditionalGeneration`도 `KimiK3ForConditionalGenerationConfig`(`models/config.py:744,1565`)를 쓴다. 그 `verify_and_update_config`(`:775-805`)는 `dsa_config`가 있으면
  kv-cache dtype을 **재작성**한다: 백엔드가 `FLASHINFER_MLA_SPARSE`이거나(미지정 + SM100)이면 `auto`→`fp8_e4m3`, **그 밖의 모든 경우(SM80 포함) `auto` 또는 `fp8*` → `fp8_ds_mla`**(FlashMLA sparse 전용 656 B 레이아웃).
  `bfloat16`을 명시하면 건드리지 않는다. 리포 lane 01~03은 이미 `--kv-cache-dtype bfloat16`을 명시하므로 P1은 안전하지만, **lane 04/05의 `fp8`(software-FP8 패치 트리거)은 `fp8_ds_mla`로 바뀌어 패치와 충돌**한다 → P3에서 이 훅을 포함해 재설계.
  speculative_config가 있으면 `configure_dense_draft`(`dense_draft.py`)가 draft 설정을 추가로 조정한다(P2 DSpark 때 검토).
- lane이 명시하는 `--attention-backend TRITON_MLA`([cluster.py.tmpl](/home/jimmy/repos/k3-sm80/deploy/tooling/templates/cluster.py.tmpl) `:44`)는 K3.1에서 그대로 B2의 raise 원인이 된다 → 신규 SM80 sparse 백엔드 이름으로 교체(draft의 `attention_backend: TRITON_MLA`는 dense MLA라 유지).
- packed hybrid block row 안에 indexer K cache spec(`KimiK3IndexerCache`, entry 132 B)이 함께 배치되므로 `mla-fp8-kv-sm80.patch`(`mla.py`·`triton_mla.py`·`triton_decode_attention.py` 수정)는 **P3에서 sparse 백엔드 위에 재설계**한다.

### B4. base pin 드리프트 (0498dc7e → main, 3.5개월) — 패치 4개의 운명

| 패치 | 대상 | main 상태 | 판정 |
|---|---|---|---|
| `kda-sm80-int64.patch` | `kda/chunk.py` h 오프셋, `chunk_delta_h.py` | `chunk_delta_h.py:98-103`에 **동일한 cast-before-multiply 수정 반영됨**; `chunk.py`는 `make_block_ptr`가 포인터 연산으로 재작성(`:252`, `i_t`를 program_id에서 int64로 승격 `:217,:360`) | **퇴역 후보** — k01 long-T(≥87,424) 케이스를 새 pin에서 재실행해 crash 소멸을 확인하면 삭제 |
| `mxfp4-marlin-moe-sm80.patch` | `csrc/libtorch_stable/moe/marlin_moe_wna16/{generate_kernels.py, ops.cu}` | 대조 필요(§8 V5) | rebase |
| `mla-fp8-kv-sm80.patch` (22 KB) | K3 `mla.py`, `envs.py`, `mla_attention.py`, `triton_mla.py`, `triton_decode_attention.py`, 신규 `ops/sm80_fp8_kv.py` | K3 `mla.py`는 1134줄로 재편(fp8_ds_mla·fused fp8 insert·`fused_kimi_k3_mla_*` CUDA op 사용) | **P3로 이연**, sparse 백엔드 위에 재설계 |
| `mla-fp8-kv-sm80-dspark-bf16.patch` | `mla.py` skip-layers → draft BF16 | split 3/5: speculator draft 층이 **자체 attention group**을 가짐(`maps_slot_mapping`) | 재설계 또는 불필요 — P2에서 판정 |

- 그 밖의 main 신규 SM100 전용 경로는 전부 soft fallback임을 확인: fused KDA decode(CMake `FUSED_KDA_DECODE_ARCHS="9.0a;10.0f;12.0f"` → `kda.py` `"triton"` 폴백 `:367,:532`),
  AttnRes CUDA(`KIMI_K3_ATTN_RES_ARCHS="10.0f"` → `ops/attn_res.py:199` Python 경로), latent-MoE tail CuTe(`latent_moe_runner.py:74-78` 비SM100은 기본 경로), low-latency GEMM(`low_latency_gemm.py:76-80,701-705` 비해당 capability → None), specialized decode(opt-in `decode_backend=latency_specialized`, SM100).
  **미확인**: `fused_mla_key_concat_kv_cache.py`가 호출하는 `torch.ops._C.fused_kimi_k3_mla_*`(CMake `:484` 일반 소스 목록)가 SM80 빌드에 포함되고 bf16 경로에서 동작하는지 — 소스빌드 + k04 재실행으로 판정.
- 결론: **pin 상향은 K3 lane 전체의 재검증을 수반**한다. 리포의 K3 A100 증거가 전무하므로 잃는 것은 없지만, "K3는 되는데 K3.1만 추가" 구도가 아니라 **"새 pin에서 K3·K3.1 둘 다 처음 검증"** 구도임을 명시한다.

### B5. DSpark / 렌더러 / 기타
- DSpark: draft를 `Inferact/Kimi-K3.1-Pro-DSpark-preview`로 교체 — `kernel-kit/common/dspark_ref.py` `TARGET_AUX_LAYER_IDS=(2,23,47,71,89)` → `(75,79,83,87,91)`(0/1-index 규약 확인), `DSPARK_BLOCK_SIZE=7` 유지, `render_lanes.py` `K3_DRAFT_MODEL`·speculative JSON(`num_speculative_tokens` 7 또는 블로그 권장 5), `tests/kernel/test_k10_contract.py`, skip-layers `93 94 95 96 97` 계약은 5-layer draft라 형식상 유지되나 split 3/5의 draft attention group 변화로 **재검증**.
- 렌더러: K3.1은 `reasoning_effort ∈ {low, high, max}`(기본 max; 렌더러 코드는 medium도 받음)를 top-level 요청 필드로 받는다(split 1/5 Python renderer, 2/5 Rust renderer). 파서는 `--tool-call-parser kimi_k3 --reasoning-parser kimi_k3` 그대로. lane smoke에 `reasoning_effort` 케이스 추가.
- `torch.compile`: "K3.1 has no torch.compile submodule → FULL_DECODE_ONLY"(recipe AMD 노트). 리포 lane은 `enforce_eager=1`이라 P1 무관, P2 CUDA graph 단계에서 고려.
- Vision: MoonViT-V2 동일, lane은 `--language-model-only` 유지.
- 이슈 #66524(148 SM GPU에서 specialized decode 기동 실패)는 SM100 전용 경로라 **A100(108 SM)과 무관**.

## 3. 수치 영향 (리포 코스트 모델 입력)

| 항목 | K3 | K3.1-Pro | 산식·출처 |
|---|---|---|---|
| 가중치 총량 | 1,560.86 GB | 1,561.22 GB (+0.36 GB) | index.json total_size |
| 토큰당 활성 가중치 bytes(원본 ckpt) | 136.96 GB | **≈137.32 GB** (+178.5M param bf16 = +0.36 GB, +0.26%) | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §6 + indexer |
| 24층 attention cache, 토큰당(모든 TP rank 복제) | bf16 latent 27,648 B / software-FP8 13,824 B | + indexer K **3,168 B**(132×24) → bf16 30,816 B / FP8 16,992 B | vLLM 블로그 FP8 수치 13,824→16,992(+22.9%)와 일치 |
| seq당 @256k / @1M (bf16 latent) | 7.25 GB / 28.99 GB | **8.08 GB / 32.31 GB** | |
| decode 1토큰당 attention 읽기, 층당 @256k | latent 전체 302 MB(bf16) | index-K 34.6 MB + top-2048 latent 2.4 MB ≈ **37 MB (≈1/8)** | @1M: 1.21 GB → 141 MB |
| indexer 연산 @1M | — | 32×128×1M ≈ 4.1 GMAC/층/토큰, 24층 ≈ 0.2 TFLOP/토큰 | 메모리 바운드(132 B/token 스트림) |

- **Phase 0 판정(TP8PP4, ctx 8k, 46.8 tok/s real)은 유지**된다: 병목은 가중치 스트리밍(137 GB/token)이고 attention 항은 ctx 8k에서 작다. DSA는 **장문맥(≥128k)에서 attention 바이트를 4~8× 줄여** K3보다 유리하다.
- [k3_model.py](/home/jimmy/repos/k3-sm80/phase0/k3_model.py) `attn_mla`에 DSA 변형(indexer 스트림 + top-k + 2048행 MQA)과 indexer cache 바이트를 추가하고, `LAYERS` 패턴을 `dsa_config.dsa_layers`로 읽게 바꾼다(P3, k11/k12 실측 후 재보정).

## 4. 리포 변경 지도

| 영역 | 파일 | 현재(K3) | K3.1에서 | 종류 |
|---|---|---|---|---|
| docker | [Dockerfile.openai](/home/jimmy/repos/k3-sm80/docker/Dockerfile.openai) `:11-12,:34-39,:88-95,:167,:182-183` · [Dockerfile.full](/home/jimmy/repos/k3-sm80/docker/Dockerfile.full) · [build-openai.sh](/home/jimmy/repos/k3-sm80/docker/build-openai.sh) `:6,:23` | `VLLM_COMMIT=0498dc7e…`, 패치 4개 적용, 태그 `k3-sm80-openai:0498dc7e-sm80-v3` | 새 pin(main 커밋), 패치 집합 {mxfp4-marlin(rebase)} + 신규 {sm80-dsa-indexer, triton-mla-sparse}; P3에 fp8-kv 재합류; 새 태그 | 값 교체 + 패치 교체 |
| docker | `docker/patches/*.patch` | 4개 | §2 B4 표대로 퇴역/rebase/이연 + 신규 2개 | 신규 |
| deploy | [render_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/render_lanes.py) `:48,:57-61,:93,:101,:118` · [lane_specs.py](/home/jimmy/repos/k3-sm80/deploy/tooling/lane_specs.py) · `deploy/0*/.env`·RUNBOOK·compose·Makefile(생성물) | `K3_TARGET_MODEL=moonshotai/Kimi-K3`, `K3_DRAFT_MODEL=Inferact/Kimi-K3-DSpark`, `K3_IMAGE`, `K3_SERVING_ROOT=…/Kimi-K3`, `VLLM_SERVED_MODEL_NAME=kimi-k3`, skip-layers `93 94 95 96 97`, `VLLM_ATTENTION_BACKEND=TRITON_MLA` | 모델·draft·이미지·루트 교체, `VLLM_ATTENTION_BACKEND=TRITON_MLA` → 신규 SM80 sparse 백엔드 이름(dense로 두면 `mla_attention.py:642` raise), `VLLM_KV_CACHE_DTYPE=bfloat16` **명시 유지**(`auto` 금지 — §2 B3 재작성 훅), `--kernel-config`/`additional-config`로 `sparse_indexer_topk_backend=torch`, `reasoning_effort` smoke, `make render` 재생성 | 파라미터화(LaneSpec에 model 필드) |
| deploy | [docs/A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md) · [validate_lanes.py](/home/jimmy/repos/k3-sm80/deploy/tooling/validate_lanes.py) · [gates.py](/home/jimmy/repos/k3-sm80/deploy/tooling/gates.py) | gate 행 k08/c01/k10/k09 + lane01~05 | k11·k12 행 추가, lane 행을 K3.1 기준으로 | 신규 행 |
| kernel-kit | [shapes.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/shapes.py) `KERNEL_IDS`·`KERNEL_TITLES`·`REQUIRED_SHAPES`·`perf_bytes_model`·`_DECODE_CASES/_PREFILL_CASES` | K3 상수(MLA_LAYERS 24 등) | `DSA_INDEX_HEADS=32`, `DSA_INDEX_HEAD_DIM=128`, `DSA_TOPK=2048`, `DSA_INDEX_K_BYTES=132`, `DSA_LAYERS=24`(=MLA_LAYERS) + k11/k12 케이스·perf 모델 | 신규 상수·케이스 |
| kernel-kit | `k11_dsa_indexer_sm80/harness.py`(신규) · `k12_mla_sparse_sm80/harness.py`(신규) · `common/dsa_ref.py`(fp32 참조: layernorm·logits·top-k·gather-attention) | — | 4게이트(NUMERIC/SANITIZER/SHAPES/PERF) 하네스; P1은 PERF **측정 전용** | 신규 |
| kernel-kit | [run_all.sh](/home/jimmy/repos/k3-sm80/kernel-kit/run_all.sh) `:16 KERNELS_DEFAULT` · [report.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/report.py) · [relay.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/relay.py) | k01~k09 | + k11 k12 (RELAY ≤12줄 유지: 커널당 4게이트 1줄) | 등록 |
| kernel-kit | [vendor/](/home/jimmy/repos/k3-sm80/kernel-kit/vendor) (`fla_kda`, `attn_ops`, `mamba_ops`, `_vllm_shims`, 헤더 `vendored from vllm@0498dc7e`) · [dspark_ref.py](/home/jimmy/repos/k3-sm80/kernel-kit/common/dspark_ref.py) `:10-11` | 핀 0498dc7e 사본 | 새 pin에서 재vendor(출처 헤더 갱신) + `TARGET_AUX_LAYER_IDS=(75,79,83,87,91)` | 재vendor + 값 교체 |
| tests | [test_openai_image_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_openai_image_contract.py) `:27` · [test_fp8_patch_contract.py](/home/jimmy/repos/k3-sm80/tests/docker/test_fp8_patch_contract.py) `:23` · [test_k10_contract.py](/home/jimmy/repos/k3-sm80/tests/kernel/test_k10_contract.py) `:73-74` · [test_commands.py](/home/jimmy/repos/k3-sm80/tests/deploy/test_commands.py) `:36` · `tests/kernel/test_k11_*`·`test_k12_*`(신규) | 핀·모델 id 문자열 assert | 새 핀·id, 신규 contract 테스트 | 값 교체 + 신규 |
| phase0 | [k3_model.py](/home/jimmy/repos/k3-sm80/phase0/k3_model.py) `:76-91,:293-300,:433-441,:519-524,:645-646` | MLA dense 항, 24 MLA 층 | DSA 항(indexer 스트림 + top-k 2048 MQA) + indexer cache 바이트 + `dsa_layers` 패턴 | 모델 확장(P3) |
| recon/docs | `recon/k31-config.md`(신규 — §1.1 diff 원문) · [SUMMARY.md](/home/jimmy/repos/k3-sm80/SUMMARY.md) 핀 표 · [PLAN.md](/home/jimmy/repos/k3-sm80/docs/PLAN.md) S2 대상·S4 · [README.md](/home/jimmy/repos/k3-sm80/README.md) 지도·현황 | K3 | K3.1 핀·단계 추가 | 문서 |

## 5. 구현 계획 — 점진적 해상도 (phase 종료 = 실제로 돌아가는 전체 제품)

| capability | P1 (walking skeleton) | P2 (넓이+깊이) | P3 (장문맥·성능) | 미룸 |
|---|---|---|---|---|
| vLLM base pin | main 커밋(≥ #66037) 소스빌드, 패치 {marlin rebase}, int64 퇴역 판정 | v0.34.0 출시 시 재핀 | — | — |
| K3 회귀(k01~k08) | 새 pin에서 **A100 재실행 PASS** | c01·k10 포함 | k09 | — |
| DSA indexer(k11) | Triton logits + `torch` top-k + software-FP8 index-K; NUMERIC/SANITIZER/SHAPES **PASS**, PERF 측정 전용 | PERF gate(루프라인 대비), Triton top-k | 1M prefill 버퍼 사이징 | mxfp4 index-K, DeepSelect |
| sparse MLA(k12) | Triton decode+prefill, **bf16 latent**, 4게이트 중 PERF 측정 전용 | PERF gate, CUDA graph 캡처(enforce_eager 해제), batch>1 | software-FP8 latent 재설계(P3 fp8-kv) | fp8_ds_mla, DCP/ShardQ, HiSparse |
| 서빙 lane | **lane 01 K3.1(TP8/PP4, 256k, `--kv-cache-dtype bfloat16` 명시, `--attention-backend <SM80 sparse>`, top-k `torch`, DSpark 없음)** `gates up gate-load gate-context smoke` RELAY | lane 02/03 K3.1(EP32, DSpark preview) | lane 04/05 K3.1(1M, software-FP8 KV+index-K) | specialized decode, P/D |
| 렌더러·API | `reasoning_effort` smoke 1케이스, 파서 kimi_k3 | tool-call·multi-turn(reasoning_content 회신) smoke | — | vision |
| 코스트 모델 | 변경 없음(판정 유지 근거 §3) | — | DSA 항 추가 + k11/k12 실측 재보정 | — |
| 문서 | recon/k31-config.md, PLAN S2' 등록, README 현황 | RUNBOOK K3.1 | SUMMARY 재작성 | — |

**P1 종료 gate(실제 인터페이스·실제 의존성)**: ① 호스트 — `python -m unittest`(tests/) green, `run_all.sh --cpu-dry-run` green, 컨테이너 gate(vllm import 0건, k05/k08 제외) ② A100 — `run/20-kernel-test.sh full`에서 k01~k04·k07·k08·**k11·k12** NUMERIC/SANITIZER/SHAPES PASS(RELAY 블록) ③ A100 — `make -C deploy/01-… gates up gate-load gate-context smoke`가 **K3.1-Pro 실체크포인트**로 응답 생성(greedy 재현성 + reasoning 채널 확인) ④ 회귀 — K3 shape 케이스 전부 유지.
**P1 정직한 상한**: 속도 미보증(PERF 측정만 공개), 256k·bf16 KV·`max_num_seqs=1`·`enforce_eager`, DSpark 없음, 1M 미지원. 문구·등급은 이 상한을 넘지 않는다.
**P1 작업 단위(규모)**: pin 상향+패치 정리(M) · k11 커널+하네스(M) · k12 커널+백엔드 배선(**L**) · lane/테스트/문서 앵커 갱신(S) · A100 1라운드(러너 Jimmy, RELAY 회신).

## 6. Jimmy 결정 요청

| # | 질문 | 선택지 | 추천 |
|---|---|---|---|
| D1 | base pin | (a) 지금 main 커밋 고정(`a3c14efb` 11-09, 또는 #66037 병합 커밋 `4088c283`) → v0.34.0 출시 시 재핀 / (b) v0.34.0 출시까지 대기 / (c) split 4/5·5/5 병합까지 대기 | **(a)** — 3/5까지로 NVIDIA 기본 경로가 서빙 가능하며, 4/5·5/5는 ROCm·P/D·DCP라 SM80 무관 |
| D2 | K3 lane 처우 | (a) 5 lane 모두 K3.1로 전환, K3는 kernel-kit 회귀(k01~k10)와 `.env` 한 줄(`K3_TARGET_MODEL`)로만 유지 / (b) K3·K3.1 lane 병행(디렉터리 10개, 이미지 2개) | **(a)** — K3 A100 증거가 없어 잃는 것이 없고, 같은 모델 클래스라 이미지 1개로 둘 다 가능 |
| D3 | P1 범위 | 위 §5대로(bf16 KV, DSpark·1M 제외) / DSpark를 P1에 포함 | **§5대로** — draft가 "preview"이고 split 3/5 draft group 변화 재검증이 필요해 P2가 안전 |
| D4 | A100 1라운드 구성 | K3 회귀 + K3.1 k11/k12 + lane 01 K3.1을 한 번에 / 커널 라운드와 lane 라운드 분리 | **한 번에**(회신 채널이 RELAY 하나라 왕복 최소화), 단 kernel-kit이 먼저 PASS여야 lane 진입(PLAN 원칙) |

## 7. 미확인·리스크

- Triton `tl.float8e4nv` 변환의 SM80 가용성 — software 비트 변환으로 우회하면 무관. A100 실측 전 가정.
- DeepGEMM: vLLM wheel 동봉 사본 때문에 `has_deep_gemm()`이 참일 수 있어 증상이 "init 실패"가 아니라 "첫 forward 실패"로 나타날 수 있음 — 판정에는 영향 없음, 진단 시 혼동 주의.
- lane 04/05의 `--kv-cache-dtype fp8`이 K3.1에서 `fp8_ds_mla`로 재작성되는 훅(§2 B3)과 리포 software-FP8 패치의 충돌 — P3 설계 항목.
- `torch.ops._C.fused_kimi_k3_mla_*` CUDA op의 SM80 빌드·동작(§2 B4). 소스빌드 1회차에서 판정.
- split 3/5의 packed hybrid block row와 리포 software-FP8 latent 레이아웃의 결합(P3).
- indexer prefill 버퍼(`get_max_prefill_buffer_size`)의 1M 메모리(P3).
- split 4/5 `SparseIndexerTopk.forward_prefill` 미병합 — main의 비SM100 prefill top-k 경로가 완전한지 소스 확인 필요(k11 설계 시).
- `tl.float8e4nv` 외에 upstream indexer Triton 커널이 쓰는 `pack_ue8m0_x4` 등 SM100 전제 유틸 — k11은 독립 커널로 작성해 의존 제거.
- DSpark preview draft 자체가 "stronger version soon" — P2 시점에 재확인.
- 사내 환경 제약(ghcr 차단·NFS root_squash·RELAY 회신)은 K3와 동일 — [README.md](/home/jimmy/repos/k3-sm80/README.md).

## 8. 독립 검증 판정 (ultracode, 읽기 전용)

(워크플로 결과 수신 후 기입)

## 9. 출처

- HF `moonshotai/Kimi-K3.1-Pro` @ `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`: config.json · model.safetensors.index.json · configuration_kimi_k3.py · README.md (2026-11-09 fetch, K3 `9f62e4e9` 와 diff)
- HF `Inferact/Kimi-K3.1-Pro-DSpark-preview` @ `6a2ef20a`, `Inferact/Kimi-K3-DSpark` @ `cf6b8244`: config.json · README.md
- vLLM main @ `a3c14efbc77c018e8bf579935095a87721349e05` (2026-11-09): `vllm/models/kimi_k3/{nvidia/dsa.py, nvidia/ops/dsa_indexer.py, nvidia/ops/dsa_indexer_mxfp4.py, common/dsa_cache.py, nvidia/kda.py, nvidia/model.py, nvidia/low_latency_gemm.py, nvidia/ops/*.py}`, `vllm/model_executor/layers/{sparse_attn_indexer.py, indexer_topk.py, mla.py, attention/mla_attention.py}`, `vllm/v1/attention/backends/mla/{indexer.py, flashmla_sparse.py, flashinfer_mla_sparse.py, flashattn_mla_sparse.py, sparse_utils.py, index_group.py}`, `vllm/platforms/cuda.py`, `vllm/config/kernel.py`, `vllm/utils/deep_gemm.py`, `vllm/model_executor/models/{registry.py, config.py}`, `CMakeLists.txt`
- vLLM PR #50000(병합 07-30) · #66030 · #66028 · #66037 · #65685 · #66027 · #66145 · 이슈 #66524 (GitHub API, 2026-11-09)
- vLLM 블로그 https://vllm.ai/blog/2026-11-04-kimi-k31-pro · recipe https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro (Updated 2026-11-08) · 제3자 https://www.orcarouter.ai/blog/kimi-k3-1-vs-kimi-k3
