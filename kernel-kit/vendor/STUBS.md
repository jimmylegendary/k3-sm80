# vendor/ — vLLM 커널 vendored 사본: 대응표·스텁 목록·근거

핀: **vLLM main `a3c14efbc77c018e8bf579935095a87721349e05`** (2026-11-09 — K3 코드가 `vllm/models/kimi_k3/` 로 병합된 뒤, K3.1 split 3/6 포함).
이전 핀 `0498dc7ea9204f1863b46515816f5d8d458e01e7`(kimi-k3 브랜치 = PR #50000 head)은 퇴역.
예외: `dsa_ops/` 는 **vLLM PR #47629 head `064801dd2bc6ac2e265dc3fa1f5d803d71bde25d`**(원본 #38476 haosdent) — 병합 전 SM80 DSA 커널(§4).
모든 파일은 핀 클론에서 `git show <핀>:<경로>` 로 복사했다(2026-11-09). 목적: k01~k04·k07·k11·k12 하네스를 **vLLM 설치 없이 torch+triton 만으로** 실행(PLAN §S2 전달 모델).
게이트: `python3 vendor/check_no_vllm.py` = py_compile + AST(vllm import 0건 + 외부 의존 allowlist {torch, triton, numpy, packaging}) — run_all.sh 자체점검에 포함(33파일 green).
"본문 무수정" 증명: `tests/kernel/test_vendor_pin_contract.py::VendorBodyEqualsPinTest` — 핀 원본과 사본의 difflib diff 가 import 행(+ conv1d CPU 블록)뿐임을 검사(핀 클론 `~/repos/vllm-pin-k31` 필요).

## 1. vendored 파일 대응표 (원본 → vendor)

변경점은 전 파일 공통 (a) 출처 헤더 2줄 prepend (b) `from vllm.*` import 행의 상대화 재배선뿐 — 본문 무수정. 예외는 §3. 줄 수 = 핀 원본 → 사본(괄호 = 이전 핀 사본).

| 원본 (vllm@a3c14efb) | vendor | 줄 수 | 소비자 · 이전 핀 대비 변경 |
|---|---|---|---|
| `vllm/models/kimi_k3/nvidia/ops/third_party/kda/__init__.py` | `fla_kda/__init__.py` | 28→30 (30) | k01·k02 · 공개 심볼 9개 — `_EXPORTS` 재대조 일치 |
| `.../kda/chunk.py` | `fla_kda/chunk.py` | 830→832 (943) | k01 · `make_block_ptr`→포인터 연산, `i_t` int64 승격, `out=` 인자 추가 |
| `.../kda/chunk_intra.py` | `fla_kda/chunk_intra.py` | 597→599 (561) | chunk 내부 |
| `.../kda/chunk_intra_token_parallel.py` | `fla_kda/chunk_intra_token_parallel.py` | 194→196 (180) | chunk_intra 내부 |
| `.../kda/fused_recurrent.py` | `fla_kda/fused_recurrent.py` | 671→673 (673) | k02 · `fused_recurrent_kda_packed_decode(..., out=None)` 추가 |
| `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py` | `fla_common/chunk_delta_h.py` | 343→345 (387) | fla_kda/chunk · cast-before-multiply int64 수정이 원본에 반영 |
| `.../ops/cumsum.py` | `fla_common/cumsum.py` | 255→257 (284) | fla_kda/chunk |
| `.../ops/index.py` | `fla_common/index.py` | 40→42 (39) | fla_kda/* · `gpu_sync_allowed`·`async_tensor_h2d` 신규 의존 |
| `.../ops/l2norm.py` | `fla_common/l2norm.py` | 184→186 (153) | fla_kda/chunk · warmup dispatcher(`l2norm_fwd_block`) 신규 의존 |
| `.../ops/op.py` | `fla_common/op.py` | 62→64 (64) | fla_kda/* |
| `.../ops/utils.py` | `fla_common/utils.py` | 213→215 (202) | fla 전체 · `get_device_name`·`get_device_capability(0).major`·`cuda_get_device_properties` 신규 의존 |
| `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | `mamba_ops/causal_conv1d.py` | 1305→1303 (1287) | k03 · APC 인자 5개 제거(`block_idx_*`·`initial_state_idx`·`num_computed_tokens`·`block_size_to_align`) |
| `vllm/v1/attention/ops/triton_decode_attention.py` | `attn_ops/triton_decode_attention.py` | 1575→1577 (820) | k04 · `sliding_window` 인자(기본 None=무창) + folded causal verify 커널군 추가 |
| `vllm/utils/math_utils.py` | `_vllm_shims/math_utils.py` | 45→47 (34) | fla_kda (RCP_LN2, cdiv, next_power_of_2) — 전문 vendored |
| `vllm/model_executor/layers/activation.py` L184-191 발췌 | `situ.py` | — | k07 — §3 |

k04 sha 핀: 원본 `4ce0db08ddf0c978485c9b6f967d5a9221a3bdf61159d956773d3c26ea815300`, 사본 `6053f265f348b4ee17c5925f84599a8e93722119b1bdb33c1d06569f183118de`
(`k04_mla_decode/harness.py` `PINNED_KERNEL_SHA256`/`VENDORED_KERNEL_SHA256`; 테스트가 사본 해시를 대조).
kda 의존 체인은 위에서 닫힌다: fla_kda → fla_common + `_vllm_shims` → torch/triton 표준만(전 파일 import AST 전수 스캔).

## 2. 스텁 목록 (`_vllm_shims/` — vLLM 코드가 아닌 이 킷의 저작물)

스텁이 vendored 코드의 사용 심볼(`from .._vllm_shims.X import …`)과 `current_platform.<attr>` 를 전부 덮는지는
`test_vendor_pin_contract.py::ShimCoverageTest` 가 AST 로 대조한다(재vendor 때 가장 먼저 깨지는 곳).

| 스텁 | 대체 대상 | 근거 (핀 원본 원문 대조) |
|---|---|---|
| <a name="platforms"></a>`platforms.py` | `vllm.platforms.current_platform` | 원본은 nvml/ROCm/XPU/CPU 전 플랫폼 해석기. 실제 호출 메서드만 torch 빌드 기준으로 구현: `is_cuda`/`is_rocm`/`is_cpu`(interface.py L196/204/213; `is_cpu` 는 상수 False — CUDA 티어 전용, §3 cpu-block 과 짝), `is_cuda_alike`(L232), `has_device_capability`(L535-555: int `<major><minor>`·tuple), `get_device_capability`(cuda.py L1009-1011 — `DeviceCapability(major, minor)` NamedTuple 반환; 핀 `fla_common/utils.py` 가 `.major` 를 읽음), `get_device_name`(cuda.py L1014-1015), `is_arch_support_pdl`(cuda.py L737-743: major ≥ 9, 예외 시 False — SM80 은 항상 False). 이전 핀 대비 신규: `get_device_name`, NamedTuple 반환. |
| <a name="triton-utils"></a>`triton_utils.py` | `vllm.triton_utils` | 원본은 triton 부재 시 Placeholder 치환 + gluon/aggregate/use_tensor_descriptor 재수출 + `VLLM_TRITON_FORCE_FIRST_CONFIG` 설치. vendored 커널이 쓰는 것은 `tl`/`triton`/`tldevice` 와 상수 `LOG2E`/`LOGE2`(원본 L40-41 — dsa_ops 가 사용)뿐 → 직수입 3줄 + 상수 2개. triton 미설치면 import 실패가 곧 올바른 신호. |
| <a name="attn-backend-utils"></a>`attn_backend_utils.py` | `vllm.v1.attention.backends.utils` 상수 2개 | causal_conv1d.py 가 쓰는 `PAD_SLOT_ID = -1`, `NULL_BLOCK_ID = 0` (원본 L54-55) 값 발췌. |
| <a name="gpu-sync-debug"></a>`gpu_sync_debug.py` (신규) | `vllm.utils.gpu_sync_debug.gpu_sync_allowed` | 원본 403줄은 GPU↔CPU 동기화 감시기. 감시가 꺼져 있으면 `gpu_sync_allowed()` 는 no-op 컨텍스트(`_NOOP_CM = nullcontext()`, L145·L369-383). 이 킷은 감시기를 켜지 않으므로 그 경로만 구현. 호출처: `fla_common/index.py prepare_chunk_indices`. |
| <a name="torch-utils"></a>`torch_utils.py` (신규) | `vllm.utils.torch_utils.async_tensor_h2d` | 원본(L722-748)은 PIN_MEMORY 일 때 page-locked 버퍼에 스테이징해 비동기 복사. pinned 스테이징만 생략(값·의미 동일, 지연만 다름). 호출처: `index.py`(varlen chunk 인덱스 H2D). |
| <a name="platform-utils"></a>`platform_utils.py` (신규) | `vllm.utils.platform_utils.cuda_get_device_properties` + `Platform.num_compute_units` | `cuda_get_device_properties`(원본 L26-37)는 CUDA 미초기화 시 fork 서브프로세스로 질의 — 하네스는 어차피 CUDA 를 초기화하므로 직접 질의(AMD 분기에서만 호출, NVIDIA 경로 무관). `num_compute_units`(cuda.py L703-704) = `multi_processor_count` — `dsa_ops/triton_mla_sparse_kernel.py` 가 split 수 산정에 사용(`sm_count` 미지정 시). |
| <a name="jit-warmup"></a>`jit_warmup.py` (신규) | `vllm.model_executor.warmup.jit_warmup` 의 `WarmupChoices` | 원본 1145줄은 서버 기동 시 JIT 커널 사전 컴파일 레지스트리/AST 추적기. `l2norm.py` 는 warmup 입력 정의 함수에서 이 클래스를 참조할 뿐 그 함수는 런타임에 호출되지 않는다 → 값 마커 dataclass 만 발췌(L66-73). |
| <a name="jit-warmup-triton-helper"></a>`jit_warmup_triton_helper.py` (신규) | 같은 패키지의 `DispatchSpec`·`TritonWarmupTensor`·`triton_kernel_dispatcher_with_warmup` | 원본 795줄 = (a) warmup 키 도출·FakeTensor 실체화·병렬 컴파일 풀 + (b) "dispatch → 입력 이름을 커널 인자에 바인딩 → `kernel[grid](**kwargs)`" 런처. warmup 은 컴파일만 앞당길 뿐 결과를 바꾸지 않으므로 (b) 만 이식: `_launch_binding_plan`(L37-83)·`_prepare_launch_kwargs`(L312-325)·`launch`(L327-371 비-warming 분기)·`__call__`(L599-619). 입력→인자 바인딩 규칙(`x`→`X`/`x_ptr`, `*_stride{dim}` 자동 바인딩)까지 동일. kernel 인자 없는 dispatcher 형태는 `NotImplementedError`. 검증: 인터프리터에서 `l2norm_fwd`(bf16)가 fp32 참조와 일치(최대오차 9.7e-4 = bf16 출력 반올림). |

`_vllm_shims/math_utils.py` 는 스텁이 아니라 **원본 전문 vendored**(§1 표).

## 3. 본문 수정 예외 (재배선 외 변경)

- <a name="cpu-block"></a>**`mamba_ops/causal_conv1d.py`**: 원본 말미 L1298-1305 의 CPU 플랫폼 오버라이드 블록(`if current_platform.is_cpu():` 에서 fn/update 를 `ops.cpu.causal_conv1d` 구현으로 바꿔치기)을 제거. 그 구현은 vLLM C-확장(`_custom_ops`)에 의존해 vendor 불가이고 이 사본은 CUDA(triton) 티어 전용이라 죽은 분기다. 제거 자리에 주석 4줄 유지.
- <a name="fla-common-init"></a>**`fla_common/__init__.py`**: 원본 `ops/__init__.py` 를 미러하지 않고 빈 패키지로 대체. 원본 init 은 GDN 계열 커널 5종을 eager import 하는데 KDA 는 쓰지 않는다. 하위 모듈 직접 import.
- <a name="situ"></a>**`situ.py`**: 파일 전문이 아닌 발췌 — 원본 `activation.py` L184-191 `SituAndMul.forward_native` 수식을 함수 `situ_and_mul_native` 로 추출(이전 핀 L183-190 과 본문 동일, 줄 번호만 +1). 퓨즈드 CUDA op(`torch.ops._C.situ_and_mul`)는 C-확장이라 vendor 불가 — k07 fused arm 은 vLLM 존재 시에만 실행.
- **퇴역 — 인라인 SM80-FIX 2건**: 이전 사본의 `fla_kda/chunk.py`(P1: `h + (i_tg*H+i_h).to(tl.int64)*K*V`)와 `fla_common/chunk_delta_h.py`(P2: cast-before-multiply)는 본문 수정이었다. 핀 a3c14efb 는 두 수정을 upstream 에 포함(`chunk_delta_h.py` L98-103, `i_t` int64 승격, `make_block_ptr`→포인터 연산 재작성)하므로 사본은 다시 무수정 원본이다. 퇴역 확정은 A100 k01 `b1_T128k`(131,072 tok ≥ 87,424) NUMERIC PASS — `evidence.int64_retirement_probe.all_passed`. 실패(IMA·NUMERIC FAIL)하면 `docker/patches` 의 kda-sm80-int64 를 복원한다.

## 4. dsa_ops — SM80 DSA 커널 (PR #47629 head `064801dd`)

| 원본 (PR head) | vendor | 줄 수 | 소비자 |
|---|---|---|---|
| `vllm/v1/attention/ops/mqa_logits_triton.py` | `dsa_ops/mqa_logits_triton.py` | 458→461 | k11 (`fp8_paged_mqa_logits_triton` decode, `fp8_mqa_logits_triton` prefill) |
| `vllm/v1/attention/ops/triton_mla_sparse_kernel.py` | `dsa_ops/triton_mla_sparse_kernel.py` | 550→553 | k12 (`triton_mla_sparse_attention`) |
| — | `dsa_ops/__init__.py` (PEP 562 lazy 6심볼) | 36 | — |

- 출처 헤더 3줄(`PR #47629 head 064801dd…, origin #38476 haosdent`, Apache-2.0·SPDX 원문 유지). 본문 무수정.
- 쉼 재배선: `vllm.triton_utils`(tl, triton, LOG2E, LOGE2) → `triton_utils` 스텁, `vllm.utils.platform_utils.num_compute_units` → `platform_utils` 스텁. vllm import 0건.
- 두 파일은 main(a3c14efb)에 없고 PR 에만 있다(merge-base 62a86318 이후 main 변경 없음) → PR 사본이 곧 최신.
- 주의: 병합 전 사본(split 3/6 이전 base)이다. 배포 패치(rebase)가 커널 본문을 바꾸면 그 결과로 재vendor 할 것 — kernel-kit 은 이 사본만 검증한다.
- 입력 규약: k11 q 는 E4M3 바이트(fp8 view) + 접힌 fp32 가중치, K 캐시는 `[block×128 fp8 | block×4B fp32 scale]`(132 B/tok). k12 kv 는 `(S,1,576)` bf16 물리 행 풀, indices 는 `(T,1,topk)` int32 이며 <0 이면 무효.

## 5. `common/triton_interp.py` — CPU 인터프리터 보정 (k11/k12 `--interpret` 전용)

`TRITON_INTERPRET=1` 로 vendored DSA 커널을 CPU 에서 실제 실행하기 위한 보정 3종(triton 3.9.0 `runtime/interpreter.py` 실측).
결과는 evidence.interpret_cpu 에만 기록하고 게이트는 SKIP 유지 — CPU 대조는 PASS 근거가 아니다.
1. bf16 `tl.dot`: 인터프리터는 bf16 을 uint16 비트로 저장하는데 `create_dot` 은 fp8 만 변환 → bf16 비트끼리 matmul(오차 ≈ 4.5e10). 비트를 fp32 로 올려 matmul.
2. fp32→bf16 캐스트: `cast_impl` 이 rounding_mode=None 이라 버림(RZ). GPU 는 cvt.rn(RNE) → RNE 로 보정(보정 전 sparse rmse_rel 5.3e-3 편향, 후 1.8e-3).
3. `@triton.autotune`: GPU 드라이버 벤치마크를 요구("0 active drivers") → vLLM `VLLM_TRITON_FORCE_FIRST_CONFIG` 와 같은 방식으로 첫 config 만 사용.
`tests/kernel/test_triton_interp.py` 가 각 보정의 "보정 전 오답 → 후 정답"을 서브프로세스로 증명한다.

## 6. 런타임 요구 (docker triton 티어 이미지 체크리스트)

- **torch + triton**(필수; 핀 main 은 torch 2.14 동반 triton 전제). 커널이 `launch_pdl` launch kwarg 와 `tl.gather`(`is_gather_supported` 로 자동 폴백)를 사용한다.
- **numpy**(`mamba_ops/causal_conv1d.py`, `_vllm_shims/torch_utils.py`), **packaging**(`attn_ops/triton_decode_attention.py` 버전 비교) — 이미지에 명시 설치.
- GPU 없이도 import 가능(모듈 로드시 GPU 프로브는 전부 가드). 컨테이너 `run_all.sh --cpu-dry-run` gate 는 GPU 없는 빌드 호스트에서 성립.

## 7. 하네스 폴백 규약

k01~k04·k07 은 **vendor 우선 → 실패 시 vllm 폴백**(try/except). 어느 쪽을 썼는지 리포트 evidence(`kernel_import`/`kernel.source`/`native_arm_source`)에 기록된다.
k11·k12 는 `vendor.dsa_ops` 를 직접 import 하며 폴백이 없다(PR 사본이 대상). k05(Marlin)는 vLLM C-확장이라 vendor 대상 아님.
