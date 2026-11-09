# docker/patches-k31 — K3.1 vLLM 패치 세트 (핀 `a3c14efb`)

K3.1 이미지(vLLM main `a3c14efbc77c018e8bf579935095a87721349e05`, 2026-11-09)에만 쓰는 패치다. 사내 K3 이미지·lane이 쓰는 [patches](/home/jimmy/repos/k3-sm80/docker/patches) 4개는 건드리지 않는다(병행 원칙). 설계·근거는 [K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md) §0.5·§2·§5, 핀 사실은 [vllm-upstream-k31.md](/home/jimmy/repos/k3-sm80/recon/vllm-upstream-k31.md)에 있다.

## 적용 순서 (계약)

| 순서 | 패치 | 내용 | 대상 |
|---|---|---|---|
| 1 | `mxfp4-marlin-moe-sm80-main.patch` | MXFP4 Marlin MoE SM80 (K3·K3.1 공통 MoE) | csrc |
| 2 | `triton-mla-sparse-sm80.patch` | TRITON_MLA_SPARSE + SM80 indexer Triton logits (PR #47629 rebase) | Python 8 + 테스트 3 |
| 3 | `k31-sm80-dsa-glue.patch` | K3.1 DSA의 SM80 glue | Python 2 + 테스트 2 |

세 패치의 파일 집합은 겹치지 않는다. 새 핀 worktree에서 1→2→3 순서로 `git apply --check`·적용·`git diff --check`를 통과했고, 2와 3은 단독으로도 적용된다. 순서는 계약이다: 이후 glue가 2의 파일을 건드리면 2 적용 후 상태를 base로 삼는다. 패치 머리글(첫 `diff --git` 앞)은 `git apply`가 무시한다.

    git apply --check docker/patches-k31/<patch> && git apply docker/patches-k31/<patch>

## 1. mxfp4-marlin-moe-sm80-main.patch

[mxfp4-marlin-moe-sm80-main.patch](/home/jimmy/repos/k3-sm80/docker/patches-k31/mxfp4-marlin-moe-sm80-main.patch)는 K3 패치 [mxfp4-marlin-moe-sm80.patch](/home/jimmy/repos/k3-sm80/docker/patches/mxfp4-marlin-moe-sm80.patch)(사내 동작 중, 수정 금지)를 핀 `a3c14efbc77c018e8bf579935095a87721349e05` 의 행에 맞춰 다시 쓴 신규 파일이다. 의도는 K3 패치와 같다(vLLM PR #36889 계열). 적용 순서는 marlin, triton-mla-sparse, dsa-glue 이다. csrc 변경이므로 이미지에는 `_moe_C_stable_libtorch.abi3.so` 1개 재빌드로만 반영된다. 소스 수준 rebase 이며 A100 실행 검증은 아직 없다.

#### hunk 5개 (모두 `csrc/moe/marlin_moe_wna16/ops.cu`, 핀 행)
| # | 행 | 무엇 | 왜 |
|---|---|---|---|
| H1 | 286-289 | `max_shared_mem` → `device_max_shared_mem` | 장치 능력치(OptIn 최대 동적 smem)를 불변으로 남긴다. 원 코드는 같은 변수를 뒤에서 per-block 몫으로 덮어썼다 |
| H2 | 318-321 | `determine_exec_config` 에 device 값 전달 | H1 의 이름 변경 추종, 동작 동일 |
| H3 | 328-330 | `int max_shared_mem = device_max_shared_mem;` 재도입 | 몫(device/blocks_per_sm-1024)은 `is_valid_config` 판정 전용으로 유지 |
| H4 | 365-369 | `cudaFuncSetAttribute` 는 device 값, launch 동적 smem 은 `sh_cache_size` | 함수 단위 상한을 호출마다 몫으로 덮어쓰지 않고, 커널이 실제 쓰는 양만 요청 |
| H5 | 501-510 | `c_tmp` 크기에서 `min((long)size_n*sorted_token_ids.size(0), …)` 의 batch 의존 항 제거 | 커널 global reduce 가 `locks_off*c_size`(`marlin_template.h:1321`)로 최대 sms*4 슬롯을 인덱싱하므로 항상 상한 크기로 할당 |

핀에서 달라진 것: 경로 이동(`csrc/libtorch_stable/moe/…` → `csrc/moe/…`), generator 통합, 시그니처 단순화(has_act_order/is_a_8bit/a_s_ptr 제거, vLLM #65542). 버그 로직 자체는 핀에도 남아 있어 패치가 필요하다.

#### generator 가드 hunk 폐기 → Dockerfile gate
K3 패치의 `generate_kernels.py` 가드는 버렸다. 핀의 `csrc/quantization/marlin/generate_kernels.py` 가 MXFP4(a=BF16, b=FE2M1f, s=FE8M0fnu, group_blocks=[2])를 이미 생성한다(`CMakeLists.txt:1445,1493`). 공용 generator 를 고치면 dense Marlin 까지 재생성되므로 건드리지 않는다. 대체 gate 는 [Dockerfile.k31-openai](/home/jimmy/repos/k3-sm80/docker/Dockerfile.k31-openai) 의 `test -s …sm80_kernel_bfloat16_fe2m1f_bfloat16.cu && grep -Fq kFE8M0fnu` 이다. 핀 generator 를 8.0 으로 직접 돌려 같은 파일에 `kFE8M0fnu` 인스턴스가 15개 생성됨을 확인했다.

#### nvfp4 링크 스텁(PR #47629 의 `nvfp4_quant_entry.cu` hunk): 불필요
- `csrc/torch_bindings.cpp:242,249` 는 `ops.def` 스키마만 있고 함수 포인터 참조가 없다.
- `csrc/torch_bindings.cpp:841` 주석: "mxfp4_experts_quant: registered in mxfp4_experts_quant.cu (SM100 only)". 무조건 `&mxfp4_experts_quant` 참조가 사라졌다.
- `csrc/quantization/fp4/mxfp4_experts_quant.cu:504-518`: 등록이 SM100 전용 파일 안에 있다. `CMakeLists.txt:1040-1059` 에서 이 파일은 `FP4_SM100_SRCS` 에만 들어간다.
- `vllm/_custom_ops.py:714-720`: `mxfp4_experts_quant_supported` 는 AttributeError 를 False 로 처리한다.
- PR 이 고친 경로 `csrc/libtorch_stable/quantization/fp4/` 는 핀에 없다(`csrc/quantization/fp4/`).
- `torch_bindings.cpp` 의 `TORCH_BOX` 참조 120개를 스캔했고, arch 한정 심볼은 `#ifdef VLLM_ENABLE_FUSED_KDA_*` 로 가드되거나 해당 .cu 안에서 등록된다.

#### .so 1개 overlay 모델이 핀에서 성립하는 근거
Marlin MoE 소스를 컴파일하는 타깃은 `_moe_C_stable_libtorch` 하나뿐이다(`CMakeLists.txt:1471-1516`), 다른 확장은 이 경로를 참조하지 않는다. 핀에서 이 타깃은 MoE 확장 전체(topk·align·permute·marlin)로 커졌다. 같은 커밋·같은 아치로 다시 만든 것이라 기능은 같다.

## 2. triton-mla-sparse-sm80.patch — PR #47629 TRITON_MLA_SPARSE rebase

**목적**: SM80에 sparse MLA 백엔드가 하나도 없어 첫 DSA 층에서 ValueError가 나는 문제(B2)와, indexer logits가 DeepGEMM(SM90+)·Triton fp8e4nv(SM89+)에만 있는 문제(B1)를 함께 푼다.

**출처**: vllm-project/vllm PR #47629 @ `064801dd`(원본 #38476 haosdent). 미병합이고 base가 K3.1 split 3/6 이전이라 그대로는 적용되지 않는다.

| 파일 | 처리 |
|---|---|
| `v1/attention/ops/mqa_logits_triton.py`, `triton_mla_sparse_kernel.py` | PR 파일 이식(출처 헤더 1줄 추가) + 아래 변경 |
| `v1/attention/backends/mla/triton_mla_sparse.py` | **신규 작성**: 핀 `flashattn_mla_sparse.py` 골격(`SparseMLACommon*`). PR의 `XPUMLASparse*` 상속은 쓰지 않는다 |
| `registry.py`, `platforms/cuda.py` | `TRITON_MLA_SPARSE` 등록, SM8x/9x sparse tail 맨 뒤 |
| `model_executor/layers/sparse_attn_indexer.py` | XPU 분기와 같은 자리(prefill/decode)에 `use_triton_logits`(CUDA이고 DeepGEMM 미지원) 분기 |
| `v1/attention/backends/mla/indexer.py` | DeepGEMM schedule metadata 호출 2곳에 `is_deep_gemm_supported()` 가드 |
| `model_executor/models/deepseek_v2.py` | fused indexer Q(Triton fp8e4nv)를 `has_device_capability(89)`로 게이트(PR 동일) |
| 테스트 3 | PR 커널 테스트 2(확장) + 새 백엔드 테스트 |

**PR 대비 변경점**
1. **int64**: packed hybrid KV row에서는 물리 row id가 토큰 위치의 약 26.75배라 `row * 576` 오프셋이 약 139k 토큰(풀 블록 약 2.2k개 이후)에서 int32를 넘는다. sparse 커널의 gathered row id·`program_id(0)`와 logits 커널의 token/row 오프셋을 int64로 올렸다. PR 원본은 큰 row id에서 잘못된 주소를 읽는다(호스트 Triton 인터프리터에서 segfault로 재현).
2. split-KV 스크래치(`mid_out`)를 공용 workspace에서 받도록 `mid_out_getter`를 추가했다. 예약 상한 `max_split_mid_out_numel`은 커널의 `_choose_num_kv_splits`로 계산한다.
3. 백엔드 신규: 모델 dtype은 bf16만, KV는 `auto/float16/bfloat16`만(FP8·`fp8_ds_mla`는 `--kv-cache-dtype bfloat16` 안내와 함께 거부), `supports_compute_capability: major == 8`, CUDA graph `NEVER`(P2에서 검증 후 상향), DCP/PCP 거부.
4. PR의 `_warmup_autotune`은 이식하지 않았다(첫 요청에서 autotune; P2 CUDA graph 도입 전에 warmup을 넣는다).
5. 디스패치는 핀 규약에 맞췄다. prefill은 XPU와 같은 absolute-column logits + `row_starts`. decode는 `seq_lens[:, -1]`, logits 너비는 `max_model_len`(DeepGEMM 출력 모양 유지). FP4 캐시·DCP/PCP·candidate_blocks 조합은 `NotImplementedError`.
6. PR의 csrc hunk(`nvfp4_quant_entry.cu` SM8x 링크 스텁)는 제외했다. 핀에서는 `mxfp4_experts_quant` 바인딩이 SM100 전용 소스로 옮겨져(`torch_bindings.cpp:841`) 불필요하다.
7. 핀의 ruff 설정(D410/D411/D413)에 맞춰 PR 파일 docstring 7곳을 정리했다.

**한계(P1)**: bf16/fp16 KV, `enforce_eager`, `max_num_seqs 1`. FP8 KV·CUDA graph·DCP는 미보증이다.

## 3. k31-sm80-dsa-glue.patch — K3.1 DSA의 SM80 glue

`kimi_k3/nvidia/dsa.py`: `use_sm80_dsa()`가 GPU capability < 89면 켜진다(환경변수 `VLLM_K31_SM80_DSA=1/0`은 강제 켬/끔 디버그 override, `envs.py`에 등록).
- index-K 삽입: `use_fused_k_insert=False`로 두어 op가 stock CUDA `indexer_k_quant_and_cache`로 쓴다. fp32 LayerNorm(`k_index_norm`)은 그대로 선행한다.
- Q 양자화: `indexer_q_quant_fold_weights`(Triton fp8e4nv) 대신 CUDA `per_token_group_quant_fp8(use_ue8m0=True)`로 양자화하고 head별 scale을 weights에 접는다. fused 경로와 같은 수식이라 Q를 bf16으로 두지 않고 fp8을 유지한다.
- MXFP4 index 캐시와 DCP/PCP 조합은 명시적 `NotImplementedError`.
- `dsa_fused_prologue`는 `fp8_ds_mla` KV에서만 도달한다. bf16 lane에서는 `pack_projections`가 걸러내며 `_fused_attention`에 assert를 뒀다.
- K3.1은 `index_rope_dim == 0`이라 indexer RoPE를 건너뛴다.

## `kda-sm80-int64`를 포함하지 않는 이유

핀에 이미 같은 수정이 있다: `chunk_delta_h.py:98-103`(cast-before-multiply), `kda/chunk.py:217,252`(`i_t` int64 승격과 포인터 연산화). crash 소멸 확정은 A100 k01 long-T 재실행으로 한다.

## 이미지에 올릴 런타임 파일 (overlay 대상 10)

triton 패치: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/cuda.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/backends/mla/triton_mla_sparse.py`(신규), `vllm/v1/attention/backends/registry.py`, `vllm/v1/attention/ops/mqa_logits_triton.py`(신규), `vllm/v1/attention/ops/triton_mla_sparse_kernel.py`(신규).
glue 패치: `vllm/envs.py`, `vllm/models/kimi_k3/nvidia/dsa.py`.
테스트 5개는 설치 트리에 없으니 소스 checkout에서 `pytest --noconftest`로 돌린다.

## 필수 lane 설정

`--kv-cache-dtype bfloat16`(K3.1 기본값은 `auto`/`fp8*`를 `fp8_ds_mla`로 바꿔 SM80에서 선택 실패)와 `--attention-backend TRITON_MLA_SPARSE`.

## 검증 현황 (2026-11-09, GPU 없음)

핀 worktree 적용, py_compile, AST 심볼 대조, ruff, unittest, Triton 인터프리터 기반 CPU 실행, sm_80 컴파일을 통과했다. GPU 수치와 e2e는 사내 A100 라운드가 판정한다.
