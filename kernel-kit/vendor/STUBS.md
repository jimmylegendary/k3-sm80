# vendor/ — vLLM 커널 vendored 사본: 대응표·스텁 목록·근거

핀: **vLLM kimi-k3 브랜치 `0498dc7ea9204f1863b46515816f5d8d458e01e7`** (PR #50000 head).
모든 파일은 `raw.githubusercontent.com/vllm-project/vllm/<핀>/<경로>` 에서 직접 fetch (2026-07-29).
목적: k01~k04·k07 하네스가 **vLLM 설치 없이 torch+triton 만으로** 실행 (PLAN §S2 전달 모델).
게이트: `python3 vendor/check_no_vllm.py` = py_compile + AST(vllm import 0건 + 외부 의존
allowlist {torch, triton, numpy, packaging}) — 컨테이너 gate 겸용, run_all.sh 자체점검에 포함.

## 1. vendored 파일 대응표 (원본 경로 → vendor 경로)

변경점은 전 파일 공통 (a) 출처 헤더 2줄 prepend (b) `from vllm.*` import 행의 상대화
재배선뿐 — 본문 무수정. 예외는 §3 에 개별 기재.

| 원본 (vllm@0498dc7e) | vendor | 소비자 |
|---|---|---|
| `vllm/models/kimi_k3/nvidia/ops/third_party/kda/__init__.py` | `fla_kda/__init__.py` | k01·k02 |
| `.../kda/chunk.py` | `fla_kda/chunk.py` | k01 (`chunk_kda_with_fused_gate`) |
| `.../kda/chunk_intra.py` | `fla_kda/chunk_intra.py` | (chunk 내부) |
| `.../kda/chunk_intra_token_parallel.py` | `fla_kda/chunk_intra_token_parallel.py` | (chunk_intra 내부) |
| `.../kda/fused_recurrent.py` | `fla_kda/fused_recurrent.py` | k02 (`fused_recurrent_kda_fwd/_packed_decode`) |
| `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py` | `fla_common/chunk_delta_h.py` | fla_kda/chunk |
| `.../ops/cumsum.py` | `fla_common/cumsum.py` | fla_kda/chunk |
| `.../ops/index.py` | `fla_common/index.py` | fla_kda/* |
| `.../ops/l2norm.py` | `fla_common/l2norm.py` | fla_kda/chunk |
| `.../ops/op.py` | `fla_common/op.py` | fla_kda/* |
| `.../ops/utils.py` | `fla_common/utils.py` | fla 전체 (FLA_CHUNK_SIZE, is_amd, …) |
| `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | `mamba_ops/causal_conv1d.py` | k03 (`causal_conv1d_fn/update`) |
| `vllm/v1/attention/ops/triton_decode_attention.py` | `attn_ops/triton_decode_attention.py` | k04 (`decode_attention_fwd`) |
| `vllm/utils/math_utils.py` | `_vllm_shims/math_utils.py` | fla_kda (RCP_LN2, cdiv, next_power_of_2) — 32줄 순수 python이라 전문 vendored |
| `vllm/model_executor/layers/activation.py` **L183-190 발췌** | `situ.py` | k07 (`situ_and_mul_native`) — §3-situ |

kda 의존 체인은 위에서 닫힌다: fla_kda → fla_common + `_vllm_shims` → torch/triton 표준만.
(재귀 확인: 전 파일 import 전수 스캔 + check_no_vllm.py AST 게이트.)

## 2. 스텁 목록 (`_vllm_shims/` — vLLM 코드가 아닌 이 킷의 저작물)

| 스텁 | 대체 대상 | 근거 (핀 원본 원문 대조) |
|---|---|---|
| <a name="platforms"></a>`platforms.py` | `vllm.platforms.current_platform` | 원본은 nvml/ROCm/XPU/CPU 전 플랫폼 해석기(수천 줄, pynvml 의존). vendored 커널이 실제 호출하는 것은 6개뿐: `is_cuda_alike`(interface.py L220-222 = CUDA\|ROCM), `is_cuda`/`is_rocm`/`is_cpu`, `has_device_capability`(interface.py L433-455: int `<major><minor>`·tuple, 현재 cap ≥ 요구), `is_arch_support_pdl`(cuda.py L708-714: `torch.cuda.get_device_capability` major ≥ 9, 예외 시 False — SM80은 항상 False). torch 빌드 기준으로 동일 의미 구현. `is_cpu()` 는 상수 False (CUDA 티어 전용 — §3-cpu-block 과 짝). |
| <a name="triton-utils"></a>`triton_utils.py` | `vllm.triton_utils` | 원본은 triton 부재 시 TritonPlaceholder 치환 가드 + gluon/aggregate/use_tensor_descriptor 재수출. vendored 커널 사용분은 `tl`/`triton`/`tldevice`(+상수 LOG2E/LOGE2)뿐 → 직수입 3줄로 대체. triton 미설치면 import 실패가 곧 올바른 신호(트리톤 티어 전제). |
| <a name="attn-backend-utils"></a>`attn_backend_utils.py` | `vllm.v1.attention.backends.utils` 의 상수 2개 | 원본은 attention 백엔드 인프라 전체(config/kv_cache 연쇄 의존). causal_conv1d.py 가 쓰는 것은 `PAD_SLOT_ID = -1`, `NULL_BLOCK_ID = 0` (원본 L45-46 원문 확인) 상수뿐 → 값 발췌. |

`_vllm_shims/math_utils.py` 는 스텁이 아니라 **원본 전문 vendored** (§1 표 참조).

## 3. 본문 수정 예외 (재배선 외 변경 — 파일별)

- <a name="cpu-block"></a>**`mamba_ops/causal_conv1d.py`**: 원본 말미 L1282-1289 의
  CPU 플랫폼 오버라이드 블록(`if current_platform.is_cpu():` 에서 fn/update 를
  `ops.cpu.causal_conv1d` 구현으로 바꿔치기) **제거**. 그 CPU 구현은
  `vllm._custom_ops.causal_conv1d_update_cpu_vec`(C-확장)에 의존해 vendor 불가이고,
  이 사본은 CUDA(triton) 티어 전용이라 죽은 분기다. 제거 지점에 주석 유지.
- <a name="fla-common-init"></a>**`fla_common/__init__.py`**: 원본
  `flash_linear_attention/ops/__init__.py` 를 미러하지 않고 빈 패키지로 대체.
  원본 init 은 GDN 계열 커널 5종(chunk, fused_gdn_prefill_post_conv,
  fused_recurrent[GDN], fused_sigmoid_gating, layernorm_guard)을 eager import —
  전부 KDA 미사용이라 미러 시 vendor 범위만 5파일+ 불어난다. 하위 모듈 직접 import 로 사용.
- <a name="situ"></a>**`situ.py`**: 파일 전문이 아닌 **발췌**(원본 activation.py
  L183-190 `SituAndMul.forward_native` 수식을 함수 `situ_and_mul_native` 로 추출,
  수식·변수명 원문 유지). 원본 파일은 CustomOp/distributed/platform 인프라를 eager
  import 하는 레이어 파일이라 전문 vendor 가 부적절. 퓨즈드 CUDA op
  (`torch.ops._C.situ_and_mul`)는 C-확장이라 vendor 불가 — k07 의 fused arm 은
  vLLM 존재 시에만 실행(기존 semantics 유지).

## 4. 런타임 요구 (docker triton 티어 이미지 체크리스트)

- **torch + triton** (필수). triton 은 vLLM 핀이 쓰는 버전대(torch 2.9.x 동반 triton)
  권장 — 커널이 `launch_pdl` launch kwarg(신형 triton)와 `tl.gather`
  (`is_gather_supported` 로 자동 폴백)를 사용한다.
- **numpy** (`mamba_ops/causal_conv1d.py`), **packaging**
  (`attn_ops/triton_decode_attention.py` 버전 비교) — pip 기본 동반이 아닐 수 있으니
  이미지에 명시 설치.
- GPU 없이도 import 는 가능(모듈 로드시 GPU 프로브는 전부 가드됨) — 컨테이너
  `run_all.sh --cpu-dry-run` gate 는 GPU 없는 빌드 호스트에서 성립.

## 5. 하네스 폴백 규약

k01~k04·k07 은 **vendor 우선 → 실패 시 vllm 폴백** (try/except). 어느 쪽을 썼는지
리포트 evidence(`kernel_import`/`kernel.source`/`native_arm_source`)에 기록된다.
k05(Marlin)는 vLLM C-확장이라 vendor 대상 아님 — vLLM(소스빌드) 필수 (하네스 docstring).
