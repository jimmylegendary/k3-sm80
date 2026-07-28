# Kimi K3 아키텍처 수치 확정 (코스트 모델 입력)

작성: 2026-07-28 · k3-sm80 recon 하위작업

## 0. 출처 및 버전 고정

모든 1차 수치는 HF `moonshotai/Kimi-K3` 리포에서 **커밋 `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`** (lastModified 2026-07-27T16:29:18Z, 2026-07-28 fetch)로 고정해 직접 확보했다. 2차 출처(vLLM/SGLang/AMD 문서)는 **사용하지 않았다** — 아래 표의 모든 값이 1차 출처다.

| 파일 | URL (커밋 고정) |
|---|---|
| `config.json` | https://huggingface.co/moonshotai/Kimi-K3/resolve/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json |
| `configuration_kimi_k3.py` | https://huggingface.co/moonshotai/Kimi-K3/resolve/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/configuration_kimi_k3.py |
| `modeling_kimi_linear.py` (모듈 shape 근거) | https://huggingface.co/moonshotai/Kimi-K3/resolve/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py |
| `model.safetensors.index.json` (텐서 목록·총 바이트) | https://huggingface.co/moonshotai/Kimi-K3/resolve/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/model.safetensors.index.json |
| `README.md` (공식 모델카드 표) | https://huggingface.co/moonshotai/Kimi-K3/resolve/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/README.md |

검증 앵커: `model.safetensors.index.json` `metadata.total_size` = **1,560,860,324,864 B** (96 shard). 아래 §9에서 본 문서의 파라미터 분해가 이 값에 **바이트 단위로 폐쇄**됨을 보인다.

## 1. 요청 필드 원시값 (`config.json.text_config`)

| 필드 | 값 | 비고 |
|---|---|---|
| `hidden_size` | **7168** | |
| LatentMoE latent dim = `routed_expert_hidden_size` | **3584** | README 표 "Latent MoE Dimension 3584"로 교차확인 |
| `num_experts` | **896** | 예상(896?) 확정 |
| `num_experts_per_token` | **16** | 예상(16?) 확정 |
| `moe_intermediate_size` | **3072** | expert당 FFN 폭. 단, expert 입출력은 hidden이 아니라 **latent 3584** (§4) |
| shared expert | **있음, `num_shared_experts`=2** | 1개 fused `KimiMLP`(intermediate 2×3072=6144)로 구현, **hidden 7168 공간**에서 동작, **비양자화(bf16)** |
| `num_hidden_layers` | **93** | 예상(93?) 확정 |
| KDA:MLA 패턴 | **69 KDA : 24 MLA** — `[KDA,KDA,KDA,MLA]×23 + MLA` | §2 |
| `kv_lora_rank` | **512** | |
| `q_lora_rank` | **1536** | (요청 외 추가 — MLA q도 LoRA 경로) |
| `qk_rope_head_dim` | **64** | 단 `mla_use_nope: true` → **RoPE 미적용**(NoPE). 차원은 할당·캐시됨 (§3) |
| `qk_nope_head_dim` | **128** | q/k head dim = 128+64 = **192** |
| `v_head_dim` | **128** | |
| `num_attention_heads` | **96** (= `num_key_value_heads`) | |
| KDA heads / head_dim | **96 / 128** (`linear_attn_config`) | 투영폭 96×128 = **12288** |
| `vocab_size` | **163840** | `tie_word_embeddings: false` (embed·lm_head 별도) |
| dense 레이어 | **1개** (`first_k_dense_replace: 1` → 레이어 0) | `intermediate_size` 33792, bf16 |
| `max_position_embeddings` | 1,048,576 | 1M 컨텍스트 |
| `hidden_act` | `situ` (SiTU-GLU; β=4.0, linear-β=25.0) | |
| `attn_res_block_size` | 12 | AttnRes: 블록 경계마다 스칼라 게이트 residual (§3) |
| MoE 라우터 | sigmoid + `noaux_tc` + `e_score_correction_bias`, group 1/1 | DeepSeek-V3 계열 |
| `num_nextn_predict_layers` | **0** | MTP 없음 |
| `rope_theta` | 명시 없음 → 기본값 10000.0 | NoPE라 실사용 무관. **[기본값 추정]** |

비전 파트(코스트 모델 제외 대상): MoonViT-V2 27블록, vt_hidden 1024 — README 공식 401M, 체크포인트 폐쇄상 vision+projector 합계 458.5M(bf16 0.92 GB).

## 2. 레이어 구성 패턴 (1-indexed, `linear_attn_config` 명시 리스트)

- **KDA 69층**: 4의 배수와 93을 제외한 1..91 (즉 1,2,3, 5,6,7, …, 89,90,91)
- **Gated MLA 24층**: 4,8,12,…,92 (23층) + **93 (마지막 층, 예외적 연속 MLA)**
- **FFN**: 레이어 1 = dense MLP(33792), 레이어 2..93 = MoE (92층, `moe_layer_freq` 1)
- 반복 단위 = `[KDA,KDA,KDA,MLA]×23 + [MLA]`. PP 분할 시 92·93층이 MLA 2연속임에 유의.

## 3. 모듈별 정확한 shape (`modeling_kimi_linear.py` 정의 기반)

### KDA층 (`KimiDeltaAttention`) — 층당 443,740,384 param, bf16
| 텐서 | shape | param |
|---|---|---|
| `q_proj`/`k_proj`/`v_proj` | 7168→12288 ×3 | 264,241,152 |
| `q/k/v_conv1d` (depthwise, kernel 4, silu) | 12288×4 ×3 | 147,456 |
| `f_a_proj`+`f_b_proj` (forget gate, **저랭크 rank 128**) | 7168→128→12288 | 2,490,368 |
| `b_proj` (beta) | 7168→96 | 688,128 |
| `g_proj` (output gate, **full-rank** — `use_full_rank_gate: true`) | 7168→12288 | 88,080,384 |
| `o_proj` | 12288→7168 | 88,080,384 |
| `A_log`(96)+`dt_bias`(12288)+`o_norm`(128, gated RMSNorm) | | 12,512 |

주의: `use_full_rank_gate`는 **출력 게이트**(g_proj)에만 해당. forget gate는 항상 저랭크(f_a/f_b).

### Gated MLA층 (`KimiMLAAttention`) — 층당 232,196,096 param, bf16
| 텐서 | shape | param |
|---|---|---|
| `q_a_proj` | 7168→1536 | 11,010,048 |
| `q_b_proj` | 1536→96×192 | 28,311,552 |
| `kv_a_proj_with_mqa` | 7168→(512+64) | 4,128,768 |
| `kv_b_proj` | 512→96×(128+128) | 12,582,912 |
| `g_proj` (output gate, sigmoid — `mla_use_output_gate: true`) | 7168→12288 | 88,080,384 |
| `o_proj` | 12288→7168 | 88,080,384 |
| `q_a/kv_a_layernorm` | 1536+512 | 2,048 |

NoPE 확정: `assert self.use_nope`, `rotary_emb=None`, forward에 RoPE 호출 없음. rope 몫 64차원은 **위치인코딩 없이 그대로 사용·캐시**(k_rot는 head간 공유 MQA형 64차원).

### MoE층 (`KimiSparseMoeBlock`, 92층)
| 구성 | shape | 층당 param | dtype |
|---|---|---|---|
| routed expert ×896 (`KimiBlockSparseMLP`) | w1,w3: 3584→3072 / w2: 3072→3584 (expert당 33,030,144) | 29,595,009,024 | **MXFP4** |
| `routed_expert_down_proj` (hidden→latent, 전 토큰) | 7168→3584 | 25,690,112 | bf16 |
| `routed_expert_up_proj` (latent→hidden) | 3584→7168 | 25,690,112 | bf16 |
| `routed_expert_norm` | 3584 | 3,584 | bf16 |
| `gate` (라우터, nn.Parameter — hidden 입력) | 896×7168 (+bias 896) | 6,423,424 | bf16 |
| `shared_experts` (fused 2개, hidden 공간) | 7168↔6144 ×3 | 132,120,576 | bf16 |

LatentMoE 구조 확정: latent 투영은 **층당 1쌍 공유**(expert별 아님). 라우팅 게이트는 down_proj **이전의 hidden 7168**에서 계산.

### AttnRes (`attn_res_block_size: 12`)
층마다 `self_attention_res_proj`/`mlp_res_proj` = **7168→1 스칼라 게이트** (+RMSNorm), 12층 블록 경계에서 attention-residual 스트림 리셋, 모델 말단 `output_attn_res_proj`(7168→1). 층당 ~43K param — **코스트 모델에서 무시 가능**, 단 residual 스트림이 hidden 1개분 추가 활성 메모리/통신량을 만든다(PP 경계에서 hidden 2배 전송 여부는 구현 확인 필요 **[미확인]**).

## 4. 양자화 사실관계 (체크포인트 실물 기준)

- `quantization_config`: compressed-tensors `mxfp4-pack-quantized`, group 32, FP4(E2M1) symmetric, scale uint8(E8M0) → **0.53125 B/param**.
- **MXFP4 적용 범위 = routed expert w1/w2/w3 만** (index에서 `weight_packed`+`weight_scale` 보유 텐서는 92×896×3 = 247,296개의 expert 텐서가 전부).
- ignore 리스트(전부 bf16 저장): `self_attn.*`(KDA+MLA 전체), `shared_experts.*`, dense `mlp.*_proj`, `lm_head`, vision. latent 투영·라우터도 정규식 미매치로 bf16 (index로 확인).
- **MXFP8은 저장 포맷이 아니라 activation QAT 사양** (README §4 "MXFP4 weights / MXFP8 activations (quantization-aware training)"). 체크포인트에 FP8 텐서 없음. HF의 "8-bit" 태그는 자동 태그로 무시.
- SM80 함의: A100은 FP4 하드웨어 미지원 → 서빙은 (i) MXFP4를 Marlin류 dequant 에뮬레이션으로 읽거나 (ii) Phase 3 W4A16(INT4) 재양자화. bf16 업캐스트는 routed만 5.45 TB라 불가.

## 5. (a) 토큰당 활성 파라미터 분해 (decode 1토큰, 텍스트 경로)

**합계 104.19B** (README 공식 "Activated Parameters 104B"와 일치; 자체 유도값, embed 룩업 행 제외·lm_head 포함)

| 구성 | 활성 param | 비중 | 저장 dtype(원본) |
|---|---:|---:|---|
| KDA 69층 | 30.618B | 29.4% | bf16 |
| MLA 24층 | 5.573B | 5.3% | bf16 |
| MoE routed (16/896 ×92층) | 48.620B | 46.7% | **MXFP4** |
| MoE shared (92층) | 12.155B | 11.7% | bf16 |
| MoE latent 투영 (92층) | 4.727B | 4.5% | bf16 |
| MoE 라우터 (92층) | 0.591B | 0.6% | bf16 |
| dense FFN (1층) | 0.727B | 0.7% | bf16 |
| lm_head | 1.174B | 1.1% | bf16 |
| **합계** | **104.186B** | 100% | |

핵심: **활성분의 53.3%(55.57B)가 bf16 저장** — "104B 전부 4-bit"라는 가정은 이 체크포인트에 성립하지 않는다.

## 6. (b) 토큰당 가중치 읽기 바이트 (decode, batch=1, weight-only)

| 시나리오 | bytes/token | 분해 | 32×A100(HBM 2039GB/s, 완전병렬·완전중첩) 이론 하한 |
|---|---:|---|---:|
| **원본 그대로** (MXFP4 expert + bf16 나머지) | **136.96 GB** | MXFP4 25.83 + bf16 111.13 | **2.10 ms/token (~476 tok/s 상한)** |
| **전량 W4A16** (INT4 g128, 라우터·lm_head bf16 유지) | **56.36 GB** | INT4 52.80 + bf16 3.55 | **0.864 ms/token (~1158 tok/s 상한)** |
| (참고) expert만 W4A16 | 136.2 GB | 원본과 사실상 동일 | 2.09 ms/token |

- MXFP4 0.53125 B/param(4bit+scale/32), W4A16 g128 0.515625 B/param(4bit+fp16 scale/128) 기준. Marlin 에뮬레이션이든 native든 **읽기 바이트는 저장 포맷으로 결정**되므로 커널과 무관하게 성립.
- **스펙 문서의 "52GB/token(104B×0.5B)" 가정은 원본 체크포인트 기준 2.6× 과소** — 병목은 bf16 어텐션+shared. 대역폭 관점에서 **Phase 3 재양자화의 가치는 expert가 아니라 bf16 잔여부(어텐션·shared·latent·dense) 양자화에 있다.** 이 사실은 Phase 3를 "조건부"에서 사실상 필수로 격상시킨다(단독 판단은 Phase 0 종합에서).
- batch>1이면 expert 읽기는 유니크 expert 수(≤min(16·B, 896)/layer)로 초선형 재사용, bf16 부분은 완전 재사용.

## 7. (c) 총 체크포인트 크기 (재양자화 시나리오)

| 시나리오 | 총 크기 | GPU당(/32) | 비고 |
|---|---:|---:|---|
| 원본 MXFP4 (실측) | **1560.9 GB** | 48.78 GB | `total_size` 실측값 |
| W4A16 전 linear (g128; embed/lm_head/라우터/norm bf16) | **~1438.5 GB** | ~44.95 GB | 유도 추정 |
| W4A16 expert만 (어텐션 등 bf16 유지) | **~1518.3 GB** | ~47.45 GB | 유도 추정 |

셋 다 80GB×32 = 2560GB에 적재 가능(잔여 ≈31~35GB/GPU를 KV·state·activation·CUDA graph에 배분). g64/g32 선택 시 scale 오버헤드 소폭 증가(g64: 0.53125 B/param).

## 8. (부가) KV·state 메모리 (hat-kv-manager 입력)

| 항목 | 산식 | 값 |
|---|---|---:|
| MLA latent 캐시 (vLLM absorbed decode: 512+64=576/층) | 576×2B×24층 | **27,648 B/token** |
| — @256k / @1M (seq당, bf16) | | **7.25 GB / 28.99 GB** |
| (반례) HF 레퍼런스 구현의 전개형 K/V | 96×(192+128)×2B×24층 | 1.47 MB/token → @256k 386 GB, 서빙 불가. vLLM MLA 캐시 필수 근거 |
| KDA recurrent state (seq당 고정) | 69층×96×128×128 | 108.5M elem → **fp32 434 MB / bf16 217 MB** |
| KDA conv state (seq당) | 69층×3×12288×4 | ~20 MB (bf16) |

batch 32 기준 KDA state만 fp32 ~13.9 GB(전 GPU 합) — `max_num_seqs` 탐색 시 state가 KV보다 먼저 제약이 될 수 있음(층 배치에 따라 GPU별 편중).

## 9. 검증: 체크포인트 바이트 폐쇄

| 파트 | param | 저장 | bytes |
|---|---:|---|---:|
| routed experts 92×896 | 2,722,740,830,208 | MXFP4 (17/32 B) | 1,446,456,066,048 |
| 텍스트 bf16 (어텐션+shared+latent+라우터+dense+embed+lm_head+norm) | 56,743,645,792 | bf16 | 113,487,291,584 |
| vision+projector | 458,483,616 | bf16 | 916,967,232 |
| **합계** | **2,779,942,959,616 (2.780T)** | | **1,560,860,324,864 = `total_size` 정확 일치** |

총 2.780T ≈ 공식 "2.8T", 활성 104.19B ≈ 공식 "104B". 모듈 shape은 modeling 코드 정의와 index 텐서 목록(개수·이름) 이중 확인.

## 10. [미확인] / 주의

- **[미확인]** AttnRes 스트림의 PP 경계 통신량(hidden 2벡터 전송 여부) — modeling forward 정밀 추적 또는 vLLM 구현 확인 필요.
- **[미확인]** vLLM/SGLang의 K3 구현이 latent 투영·shared expert를 어느 dtype/커널로 태우는지(본 문서는 체크포인트 사실만 기술).
- **[미확인]** W4A16 재양자화의 품질 영향(특히 QAT-MXFP4 expert의 INT4 재매핑) — Phase 3 실험 항목.
- **[기본값 추정]** `rope_theta` 10000.0 (config 미명시, NoPE라 무영향).
- KDA state dtype(fp32 vs bf16)은 FLA 커널 구현 의존 — 위 표는 두 값 병기.
