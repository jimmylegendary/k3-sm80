# Kimi K3.1-Pro 체크포인트 사실관계 — K3 대비 diff (코스트 모델·kernel-kit 상수 입력)

작성: 2026-11-09 · k3-sm80 K3.1 P1 recon. 모든 값은 1차 출처(HF 리포 파일)를 직접 받아 K3 핀과 기계적으로 diff 한 것이다.
K3 쪽 수치·모듈 shape는 [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md)가 그대로 유효하다(아래에 적은 것 외에는 변화 없음).

## 0. 출처 및 버전 고정

| 대상 | 핀 | URL |
|---|---|---|
| `moonshotai/Kimi-K3.1-Pro` | **`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`** (lastModified 2026-11-05T07:11:43Z; 커밋 2개: `6d0a0202` Initial commit ≈11-02, `d9e99b07` "Update encoding_k3.py") | https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e |
| 대조군 `moonshotai/Kimi-K3` | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §0 |
| diff 한 파일 | `config.json`, `model.safetensors.index.json`, `configuration_kimi_k3.py`, `generation_config.json`, `tokenizer_config.json`, `README.md` | `resolve/<sha>/<file>` |

리포 루트 파일(2026-11-09): `.gitattributes LICENSE README.md assets/kimi-logo.png config.json configuration_kimi_k3.py encoding_k3.py generation_config.json kimi_k3_processor.py kimi_k3_vision_processing.py media_utils.py model.safetensors.index.json preprocessor_config.json tiktoken.model tokenization_kimi.py tokenizer_config.json` + `model-000NN-of-000096.safetensors` 96개. **`modeling_*.py` 없음**(K3에 있던 HF remote modeling 제거 → `auto_map`에 `AutoConfig`만 남음; vLLM/SGLang 전용 배포).

## 1. `config.json` diff (K3 → K3.1-Pro) — 전부

| 필드 | K3 | K3.1-Pro |
|---|---|---|
| `architectures` | `KimiK3ForConditionalGeneration` | **`KimiK31ForConditionalGeneration`** |
| `model_type` | `kimi_k3` | **`kimi_k31`** |
| `auto_map` (최상위) | AutoConfig·AutoModel·AutoModelForCausalLM | **`AutoConfig: configuration_kimi_k3.KimiK3Config`만** |
| `text_config.auto_map` | `KimiLinearConfig`·`KimiLinearModel`·`KimiLinearForCausalLM` | **삭제** |
| `text_config.dsa_config` | 없음 | **신규** — 아래 §2 |
| `quantization_config.ignore` | `re:.*self_attn.*`, `re:.*shared_experts.*`, `re:.*mlp\.(gate|up|gate_up|down)_proj.*`, `re:.*lm_head.*`, `re:.*vision_tower.*`, `re:.*mm_projector.*` | + **`re:.*block_sparse_moe\.gate$`**, **`re:.*block_sparse_moe\.routed_expert_(up|down)_proj.*`** (K3에서도 bf16이었던 텐서를 명시한 것 — 바이트 변화 없음) |
| `transformers_version` | 4.56.2 | 4.57.6 |

그 외 모든 필드(hidden 7168, latent 3584, experts 896/top-16, MoE inter 3072, shared 2, 93 layers, KDA 69층 목록, kv_lora 512, q_lora 1536, qk_nope 128, qk_rope 64, v 128, heads 96, `mla_use_nope: true`, `mla_use_output_gate: true`, vocab 163840, `max_position_embeddings` 1,048,576, `hidden_act situ`, `attn_res_block_size 12`, `num_nextn_predict_layers 0`, MXFP4 `mxfp4-pack-quantized` group 32)는 **동일**.

## 2. `dsa_config` (text_config)

```json
"dsa_config": {
  "dsa_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
  "index_head_dim": 128,
  "index_k_norm_type": "layernorm",
  "index_num_heads": 32,
  "index_rope_dim": 0,
  "index_skip_topk_offset": 1,
  "index_topk": 2048,
  "index_topk_freq": 1
}
```

- `dsa_layers`는 **1-indexed**(`configuration_kimi_k3.py` `is_kda_layer`가 `(layer_idx + 1) in kda_layers`를 쓰는 규약과 동일; vLLM `kimi_linear.py:231-234` `is_dsa_layer`도 1-indexed 판정). 24개 = K3의 Gated MLA 24층 **전부**(4,8,…,92 + 마지막 93). 따라서 **K3.1에는 dense MLA 층이 없다** — 24층 모두 Gated DSA.
- `index_rope_dim: 0` → indexer에 RoPE 없음(메인 MLA도 NoPE). vLLM `index_rope.py`는 호출되지 않는다.
- `index_skip_topk_offset: 1`, `index_topk_freq: 1` → vLLM 쪽 소비처 없음(parity 필드; `index_topk_freq != 1`은 거부).
- 모델카드 표: "69 KDA + 24 Gated DSA", "DSA Index Head Dimension 128 / Number of Heads 32 / Top-K 2048", 각주 "replace the full-attention Gated MLA layers with Gated DeepSeek Sparse Attention (Gated DSA) layers for more efficient long-context modeling".

## 3. `model.safetensors.index.json` diff

| 항목 | K3 | K3.1-Pro |
|---|---|---|
| `metadata.total_size` | 1,560,860,324,864 B | **1,561,217,377,280 B** (+357,052,416 B) |
| 텐서 수 | 497,220 | **497,340** (+120) |
| shard 수 | 96 | 96 |
| 패턴(층 번호 정규화) 차이 | — | **추가 5패턴 × 24층**, 삭제 0, 개수 변동 0 |

추가 텐서(0-indexed 층 `3,7,11,…,91,92`):

| 체크포인트 이름 | shape(유도) | param/층 | vLLM 모듈 이름(`model.py` `_DSA_RENAME_PARAMS_MAPPING`/`_DSA_STACKED_PARAMS_MAPPING`) |
|---|---|---|---|
| `language_model.model.layers.N.self_attn.indexer.wq_b.weight` | `[32×128=4096, 1536]` (q_lora_rank 입력) | 6,291,456 | `self_attn.dsa_indexer.wq_index` |
| `…self_attn.indexer.wk.weight` | `[128, 7168]` | 917,504 | `self_attn.dsa_indexer.wkv_index` shard 0 |
| `…self_attn.indexer.weights_proj.weight` | `[32, 7168]` | 229,376 | `self_attn.dsa_indexer.wkv_index` shard 1 |
| `…self_attn.indexer.k_norm.weight` / `.bias` | `[128]` ×2 | 256 | `self_attn.dsa_indexer.k_index_norm` (bias 있는 fp32 LayerNorm) |
| **합계** | | **7,438,592 /층 → 178,526,208 (24층)** | bf16 2 B → **357,052,416 B = total_size 증분과 정확히 일치** |

MXFP4(`weight_packed`+`weight_scale`) 텐서 집합은 K3와 동일(routed expert w1/w2/w3만). indexer 텐서는 bf16(ignore 정규식 `self_attn.*`에 걸림).

## 4. 파생 수치 (kernel-kit·phase0 입력)

| 항목 | 값 | 비고 |
|---|---|---|
| 토큰당 활성 param | 104.19B + 0.179B = **≈104.36B** | indexer는 매 토큰 전 24층 실행 |
| 토큰당 가중치 bytes(원본 ckpt) | 136.96 GB + 0.357 GB = **≈137.32 GB** (+0.26%) | [k3-config.md](/home/jimmy/repos/k3-sm80/recon/k3-config.md) §6 |
| indexer K cache(vLLM 기본 fp8) | **132 B/token/층**(128 B E4M3 + 4 B fp32 scale) × 24 = **3,168 B/token** | `dsa_cache.py: indexer_cache_entry_bytes`; TP rank마다 복제 |
| MLA latent cache(bf16) | 576 × 2 B × 24 = 27,648 B/token | K3와 동일 |
| 합계 attention cache(bf16 latent) | **30,816 B/token** → @256k 8.08 GB, @1M 32.31 GB (replica당) | PP4면 층이 4분할 → /4 |
| indexer logits 연산 | 32 × 128 × L MAC/층/토큰 | @1M ≈ 4.1 GMAC/층 |
| top-k 버퍼 | `max_num_batched_tokens × 2048 × 4 B` | 컨텍스트 길이 무관 |

`kernel-kit/common/shapes.py` 신규 상수(권장 이름): `DSA_LAYERS = 24`(= `MLA_LAYERS`), `DSA_INDEX_HEADS = 32`, `DSA_INDEX_HEAD_DIM = 128`, `DSA_TOPK = 2048`, `DSA_INDEX_ROPE_DIM = 0`, `DSA_INDEX_K_BYTES = 132`, `DSA_WQ_IN = 1536`.

## 5. 그 밖의 파일

- `generation_config.json`: diff 없음.
- `tokenizer_config.json`: 빈 `extra_special_tokens: {}` 키 1개 삭제뿐. `tiktoken.model`·`tokenization_kimi.py` 동일.
- `configuration_kimi_k3.py`: `dsa_config: Optional[dict] = None` 인자·속성 추가, `model_type = "kimi_k31"`, 나머지는 포매팅 차이.
- `encoding_k3.py`(Python 렌더러): K3.1은 `reasoning_effort ∈ {low, high, max}`(기본 max; 문구에는 medium도 언급) top-level 요청 필드. 멀티턴·tool-call은 `reasoning_content`·`tool_calls`를 그대로 되돌려 줘야 함(모델카드).

## 6. DSpark draft (P2 참고)

| 항목 | `Inferact/Kimi-K3-DSpark` @ `cf6b8244` (K3용) | `Inferact/Kimi-K3.1-Pro-DSpark-preview` @ `6a2ef20a` (11-05, "preview") |
|---|---|---|
| 층 수 / params | 5 / 3.56B | 5 / 2.69B |
| `target_layer_ids`(1-indexed) | `[2, 23, 47, 71, 89]` | **`[75, 79, 83, 87, 91]`** |
| block_size | 7 | 7 |
| attention | MLA, heads 64, kv_lora 512, rope 64, `rope_theta 5e4` yarn ×32 | MLA, **heads 96**, kv_lora 512, rope 64, **`rope_theta 1e6`** yarn ×32, `rope_interleave true` |
| markov_rank / mask_token | 256 / 163837 | 512 / 163839 |
| README 성능 | — | GSM8K acceptance length 4.597 @7 tokens; "stronger version soon" |

## 7. 미확인

- indexer `wq_b`가 q_lora(1536) 입력인 것은 vLLM `dsa.py`의 `wq_index` 정의와 텐서 크기(6,291,456 = 4096×1536)로 유도한 것이며 HF modeling 코드가 없어 원문 확인은 불가.
- MXFP8 activation(QAT)은 K3와 같이 저장 포맷이 아님(`input_activations: null`).
