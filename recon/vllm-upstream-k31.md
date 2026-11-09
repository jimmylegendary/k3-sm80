# vLLM upstream 재확인 — K3.1 P1 핀 `a3c14efb` (2026-11-09)

[vllm-upstream.md](/home/jimmy/repos/k3-sm80/recon/vllm-upstream.md)(2026-07-28, 핀 `0498dc7e`)의 후속. 라벨: **[확인됨]** 원문/API 직접 확인 · **[추정]** 코드 구조상 유일 경로이나 미실행 · **[미확인]**.

## 0. 핀

| 대상 | 값 |
|---|---|
| vLLM main | **`a3c14efbc77c018e8bf579935095a87721349e05`** (2026-11-09T09:47:12Z, "[CI][ROCm] Mirror BF16x3 Router GEMM Kernels on MI355 (#66308)") |
| 포함 | K3 PR #50000(merge `aeeb36b1`, 2026-07-30) · K3.1 split 1/6 #66030(11-06) · 2/6 #66028 · **3/6 #66037(11-09 04:31Z)** |
| 미포함(open) | 4/6 #66532(AMD 모델 파일·ROCm TP8) · 5/6 #65685(ROCm·P/D·indexer top-k 백엔드) · 6/6 #66027(DCP) — SM80 동작과 무관 |
| 릴리스 | v0.32.0(10-18) · v0.33.0rc3(11-02); K3.1 recipe "Requires vLLM >= 0.34.0 (not released yet — nightly wheel)" |
| 옛 핀 | kimi-k3 브랜치 `0498dc7e`는 병합 후 삭제; `refs/pull/50000/head`(`5a3eba03`)로만 도달. main은 merge 대비 +5,609 커밋 |
| 의존성 드리프트(핀 `requirements/`) | torch 2.13.0→**2.14.0**, torchvision 0.28→0.29, flashinfer-python 0.6.15.post1→**0.7.3**, setuptools ≥78.1.1 |

## 1. K3.1 모델 코드 [확인됨]

- registry: `KimiK31ForConditionalGeneration` → `vllm.models.kimi_k3.KimiK3ForConditionalGeneration`(같은 클래스, `model_executor/models/registry.py:485-491`). 층 분기 `model.py:754-755` `if config.is_dsa_layer(layer_idx): KimiK3DSAAttention`; 배선 assert `:1066-1079`.
- `KimiK3DSAAttention`(`kimi_k3/nvidia/dsa.py`) = generic `MLAModules(is_sparse=True)` + `KimiK3DSAIndexer`가 upstream `torch.ops.vllm.sparse_attn_indexer`를 직접 호출(`dsa.py:388-413`). K3 전용 `kimi_k3/nvidia/mla.py`는 **DSA 층에서 쓰이지 않는다**(DSpark draft만).
- indexer 가중치 rename/stack: `model.py:961-972`.
- `models/config.py:744,1565` `KimiK3ForConditionalGenerationConfig.verify_and_update_config`(`:777-807`): `dsa_config`가 있으면 kv dtype 재작성 — `FLASHINFER_MLA_SPARSE` 또는 (미지정+SM100)이면 `auto`→`fp8_e4m3`, **그 밖(SM80 포함) `auto`/`fp8*`→`fp8_ds_mla`**; `bfloat16`/`float16` 명시는 유지. speculative_config 있으면 `configure_dense_draft`가 draft에 `attention_backend=FLASHINFER_MLA`, `kv_cache_dtype="fp8"` 기본값 주입.
- indexer K cache dtype: `fp8`(132 B/token) 기본, `mxfp4`(SM100) opt-in. bf16 옵션 없음(`v1/attention/backends/mla/indexer.py:62-74`).

## 2. SM80 게이트 [확인됨] (상세는 [K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md) §2)

| 지점 | 파일:행 | SM80 결과 |
|---|---|---|
| sparse MLA 백엔드 선택 | `platforms/cuda.py:136-152`(SM8x/9x 분기 후보 7개), `flashmla_sparse.py:203-204` [9,10], `flashinfer_mla_sparse.py:117-118` 10, `:216-217` 12, `flashinfer_mla_sparse_sm90.py:184-190` 9, `flashattn_mla_sparse.py:82-83` 9; 판정 `v1/attention/backend.py:325-333` | 전부 탈락 → `cuda.py:441-445`(명시) / `:470-474`(자동) ValueError. 2차 방어 `mla_attention.py:641-647` |
| indexer logits | `sparse_attn_indexer.py:882`(prefill), `:1091`(decode), `:408`(SM100 chunk) → DeepGEMM `fp8_fp4_*mqa_logits`; metadata `indexer.py:1836-1842, :1918-1926`(`has_deep_gemm()` 가드) | DeepGEMM은 SM90/100/120 전용(`cuda.py:689-695`, DeepGEMM `csrc/apis/attention.hpp:147,197,232`); SM80-only 빌드는 DeepGEMM 자체가 없음(`deepgemm.cmake:93-96`) → `deep_gemm.py:205-211 _missing()` |
| K3.1 fused prologue | `kimi_k3/nvidia/ops/dsa_indexer.py:69,201-202,354` `tl.float8e4nv` | Triton native fp8e4nv는 SM89+ → 컴파일 실패 |
| top-k | `indexer_topk.py:264-281` `auto`: cooperative(SM90+) 탈락 → **persistent**(`csrc/topk.cu` 전 arch, `__CUDA_ARCH__>=800` 가드, smem 128 KiB) | 교체 불필요 [추정: 미실행] |
| index-K cache 쓰기 | fused Triton(`indexer_k_norm_quant_insert`, fp8e4nv) 또는 stock CUDA `ops.indexer_k_quant_and_cache`(`csrc/cache_kernels.cu:705-773`) | `use_fused_k_insert=False`면 SM80 가능 [추정] |
| 그 외 SM100 전용 경로 | fused KDA decode(CMake `9.0a;10.0f;12.0f`), AttnRes CUDA(`10.0f`), latent-MoE tail, low-latency GEMM, specialized decode | 전부 soft fallback(Triton/Python) |
| `fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu` | `CMakeLists.txt:484` 일반 소스 | SM80 빌드 포함 [동작 미확인] |

## 3. SM80 DSA 경로 조달원 — PR #47629 [확인됨, 미병합]

| 항목 | 값 |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/47629 "[Attention] TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA (rebase & takeover of #38476)" |
| 작성자 / head | thomaslwang / `thomaslwang/vllm:triton-mla-sparse-sm80` @ `064801dd` (원본 #38476 haosdent @ `3740c02b`, 이슈 #38006) |
| 상태 | open, `needs-rebase`(mergeable dirty), 최종 갱신 2026-09-29; 라벨 documentation·v1·deepseek·nvidia |
| 변경 | 12파일 +1,808: `vllm/v1/attention/ops/mqa_logits_triton.py`(+458: prefill·paged-decode MQA logits, FP8 K uint8 LUT in-kernel 디코드) · `vllm/v1/attention/ops/triton_mla_sparse_kernel.py`(+550: split-KV sparse decode, N-way online-softmax merge) · `vllm/v1/attention/backends/mla/triton_mla_sparse.py`(+174) · `sparse_attn_indexer.py`(+46−7: `is_xpu → DeepGEMM → Triton fallback`) · `indexer.py`(+2−2: `is_deep_gemm_supported()` 가드) · `cuda.py`(+5: SM8x/9x 우선순위 끝) · `registry.py`(+3) · `deepseek_v2.py`(+4) · `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu`(+34: MXFP4 link stub) · tests 2개(+515) · docs |
| 검증(PR 본문) | 8×A800-80GB PCIe, driver 580.95.05: `test_mqa_logits_triton.py` 41 passed, `test_triton_mla_sparse_kernel.py` 53 passed; GLM-5.2 NVFP4 TP8/EP bf16 KV 131k e2e(백엔드 선택 로그, CUDA graph FULL_AND_PIECEWISE 캡처 정상, 32.3 tok/s 단일, 100.3 tok/s 4동시), TP2×PP4 79k needle 정확 |
| 핀과의 간극 | base가 split 3/6 이전 → packed hybrid KV row(`flat_kv_row_view`·`BLOCK_STRIDE_ROWS`·block 64), `SparseMLACommon*`(`sparse_mla_attention.py`) 재편, `sparse_attn_indexer.py` chunk-decode/DCP 변경, `csrc/libtorch_stable/` 디렉터리 이동 → **백엔드 골격은 핀의 `flashattn_mla_sparse.py` 템플릿으로 다시 쓰고 커널 파일만 이식**(P1 패치 `triton-mla-sparse-sm80.patch`) |
| 검증된 모델 vs K3.1 | GLM-5.2(index heads 64, indexer rope 있음) vs K3.1(32 heads, `index_rope_dim 0`) — 커널은 파라미터화돼 있으나 K3.1 shape는 kernel-kit k11/k12에서 확인 |

관련(미병합): PR #66497(yaoyaoding, 11-09) DeepSeek-V4.1 on 8×A100/H100 — 정확도 GB300 동등(GSM8K 0.9757 등), 모델 전용 CuTe DSL 커널 · PR #66275(JimiZhou) DSv4.1 Ampere specialized decode(Triton sparse decode 150줄) · PR #64783(Z-Ghostshell) dense `TRITON_MLA` `fp8_ds_mla` software-dequant(SM80+, 520줄; 리포 `mla-fp8-kv-sm80.patch`의 upstream 대체 후보) · RFC #48374(sparse fp8 KV, sm_86 검증) · RFC #59016(통합 요청; GLM-5.3-Flash 4×A800 1M 운영 사례, fork `wtdcode/vllm-backport`) · PR #53522(metadata 가드) · PR #62344(Qwen4Exp QSA SM80 indexer prefill, in-tree uint8 E4M3 inline-PTX 디코드 `qsa.py`).

## 4. P1 패치 세트(`docker/patches-k31/`)가 무엇을 어디서 가져오는가

| 패치 | 원천 | 비고 |
|---|---|---|
| `triton-mla-sparse-sm80.patch` | #47629 커널 2파일 + 디스패치/가드/우선순위/registry, 백엔드는 핀 템플릿으로 재작성 | bf16 KV만 |
| `k31-sm80-dsa-glue.patch` | 신규(리포) — `kimi_k3/nvidia/dsa.py` SM80 분기: `use_fused_k_insert=False`, q 양자화 `per_token_group_quant_fp8` 경로 | K3.1 고유 |
| `mxfp4-marlin-moe-sm80-main.patch` | 리포 기존 [mxfp4-marlin-moe-sm80.patch](/home/jimmy/repos/k3-sm80/docker/patches/mxfp4-marlin-moe-sm80.patch)의 `ops.cu` hunk를 `csrc/moe/marlin_moe_wna16/ops.cu`로 rebase | generator 가드는 Dockerfile 검사로 대체 |
| (제외) `kda-sm80-int64` | upstream 반영(`chunk_delta_h.py:98-103`, `kda/chunk.py:217,252`) | k01 long-T 케이스로 소멸 확인 |
| (제외, P3) `mla-fp8-kv-sm80*` | K3.1 DSA 층은 대상 클래스를 쓰지 않음; PR #64783/RFC #48374 계열로 재설계 | K3 1M lane·draft 전용이면 구조적 rebase |

## 5. 운영 참고

- Docker 이미지 `vllm/vllm-openai:kimi-k31`(day-0)은 공개 브랜치가 없고 사내 registry 차단 → 소스빌드(기존 방식) 유지.
- `--trust-remote-code`, `--tool-call-parser kimi_k3 --reasoning-parser kimi_k3`, `--language-model-only` 그대로. `reasoning_effort` 요청 필드.
- 이슈 #66524(148 SM GPU에서 specialized decode 기동 실패)는 SM100 전용 경로 → A100 무관.

## 6. SM80 패치 지점 지도 (핀 `a3c14efb` 행 번호, P1 패치 작성 중 소스 판독)

| 지점 | 파일:행 | 사실 | 패치 처리 |
|---|---|---|---|
| 백엔드 선택 | `platforms/cuda.py:81` `_get_backend_priorities`, SM8x/9x `else` `:136-152` (`sparse_tail :137`, `return :146`) | sparse 후보 3종이 `supports_compute_capability`에서 탈락 | `TRITON_MLA_SPARSE`를 맨 뒤에 추가. SM12x 분기에는 넣지 않음(백엔드가 major==8만 허용) |
| registry | `v1/attention/backends/registry.py:94` | | `TRITON_MLA` 뒤에 항목 추가 |
| 공용 골격 | `layers/attention/sparse_mla_attention.py`: `SparseMLACommonMetadata :155`, `...Builder :210`(`_init_reorder_batch_threshold :302`), `SparseMLACommonImpl :683`(`group_conversion_decode_only :689`, `forward_mqa :809`, 유일한 추상 훅 `_forward_mqa_kernel :958`) | 이 클래스들은 `backends/mla/`가 아니라 `layers/attention/`에 있다 | 상속 |
| packed row | `backends/mla/sparse_utils.py`: `align_blocks_to_rows :43`, `flat_kv_row_view :49`, `out_val = base * BLOCK_STRIDE_ROWS + inblock_off :180`, `triton_convert_req_index_to_global_index :492` | row id는 int32 범위이나 `row*576` 오프셋은 약 3.73M row(약 139k 토큰)에서 int32 초과 | 커널 int64 승격 |
| 템플릿 | `backends/mla/flashattn_mla_sparse.py`: Backend `:41`, Builder `:124`, Impl `:160`, `_forward_mqa_kernel :215` | | 골격 재사용 |
| indexer op | `layers/sparse_attn_indexer.py:494`. key 삽입 `:679-696`(`skip_k_cache_insert=False`일 때 stock CUDA `ops.indexer_k_quant_and_cache`). top-k 버퍼 -1 fill `:722`. prefill XPU `:859` / DeepGEMM `:882` / `ops.top_k_per_row_prefill :932`. decode XPU `:1069` / DeepGEMM `:1091` / `get_indexer_topk :1146`. `chunk_decode :551`(SM100 전용). `SparseAttnIndexer.__init__` 가드 `:1285` | `top_k_per_row_prefill`은 rowStart를 빼 요청-로컬 인덱스를 냄(`csrc/sampler.cu:567`) → 절대 열 logits + `row_starts` 규약 가능 | XPU와 같은 자리에 Triton 분기. `:1285`는 경고로 완화(PR 동일). K3.1은 `SparseAttnIndexer`를 쓰지 않음 |
| metadata 가드 | `backends/mla/indexer.py:918/:926/:946`(`_supports_*`, `:922/:930/:951`는 `has_deep_gemm()`), `:1836`/`:1918` schedule metadata 호출 | `:922/:930`은 capability 가드가 SM80을 이미 거름. `:951`은 SM80에서 `next_n in (1,2)`로 동일 귀결. 기존 테스트가 `indexer.has_deep_gemm`을 monkeypatch | `:1836`/`:1918`만 `has_deep_gemm() and is_deep_gemm_supported()`로 변경 |
| DeepGEMM 판정 | `utils/deep_gemm.py:151 is_deep_gemm_supported`(`platforms/cuda.py:690 support_deep_gemm` = SM90 or family 100/120), `utils/import_utils.py:512 has_deep_gemm`(모듈 존재만) | SM80은 설치와 무관하게 False | |
| fused Q | `models/deepseek_v2.py:755 use_fused_indexer_q` | | `has_device_capability(89)` 게이트 |
| K3.1 indexer | `kimi_k3/nvidia/dsa.py`: `use_fused_k_insert :221-225`, `score :364`(Q 양자화 `:380`, op `:388-412`), `pack_projections :696`(fp8_ds_mla 조건 `:712`), `_fused_attention :768` | | glue |
| top-k | `layers/indexer_topk.py`(`auto`→persistent, SM80), `csrc/topk.cu`(stride 정렬 불필요) | | 변경 없음 |
| env 검증 | `platforms/interface.py:1503 validate_environ` | 미등록 `VLLM_*`는 경고(`--fail-on-environ-validation`이면 실패). `VLLM_ATTENTION_BACKEND`는 핀 `envs.py`에 없음 | `VLLM_K31_SM80_DSA` 등록 |
| csrc | `torch_bindings.cpp:841`, `CMakeLists.txt:1047`, `_custom_ops.py:714` | `mxfp4_experts_quant` 바인딩이 SM100 전용 소스로 이동, wrapper는 AttributeError→False | PR csrc hunk 불필요 |

호스트 검증으로 확인한 것: Triton 3.9 sm_80 컴파일 시 K3.1 `indexer_q_quant_fold_weights`는 `fp8e4nv not supported in this architecture`로 실패한다(sm_90은 OK). 패치된 sparse/logits 커널 전 config는 sm_80에서 컴파일되며 smem은 최대 52 KiB다.
