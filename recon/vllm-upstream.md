# vLLM upstream 재확인 (K3-SM80 Phase 1 선행)

- 조사일: **2026-07-28** (fetch 시각 KST 13:44–13:55)
- 기준 커밋 고정:
  - vLLM `main` HEAD = **`d223c900d85224c02f2162ee2c757a769e99f519`** (2026-07-28T04:36:45Z)
    — 파일은 전부 `raw.githubusercontent.com/vllm-project/vllm/main/...`에서 직접 fetch, 트리는 GitHub git/trees API(`truncated: false`).
  - Kimi K3 브랜치 = PR **#50000** head **`vllm-project/vllm@kimi-k3` @ `0498dc7ea9204f1863b46515816f5d8d458e01e7`**
    (PR 생성 2026-07-27T15:19Z, 최종 갱신 2026-07-28T04:47Z, open/non-draft)
- 라벨 규약: **[확인됨]** = 원본 파일/API 응답 직접 확인 · **[추정]** = 코드 구조상 유일 경로이나 실행 미검증 · **[미확인]** = 확인 수단 없음.

---

## (1) issue #35922 — SM80 Marlin MoE illegal memory access

**판정: [확인됨] 미수정. 수정 PR 없음. stale-bot이 `not_planned`로 닫음.**

- 제목: "[Bug]: Marlin MoE kernels crash on A100 (SM80) with compressed-tensors quantized Qwen3-30B-A3B"
  — 생성 2026-03-03, **closed 2026-07-03T02:20Z, `state_reason: not_planned`**, 라벨 `stale`, 코멘트 2건(전부 stale-bot).
  W4A16 INT4 → `cudaErrorIllegalAddress`, FP4 계열(NVFP4A16/MXFP4A16/NVFP4)도 별도 오류로 전부 crash (`profile_run()` 중).
  출처: https://github.com/vllm-project/vllm/issues/35922 (API fetch 2026-07-28)
- 타임라인 cross-reference는 단 1건: PR **#36889** "[WIP][Bugfix] Fix CUDA illegal memory access for GPTQ Marlin MoE with CUDA graphs"
  — 근본원인을 `csrc/moe/marlin_moe_wna16/ops.cu`의 **`c_tmp` reduction buffer out-of-bounds**로 지목. 그러나 **merged=false, 2026-07-14 closed(미병합)**. 대상 이슈 #36811("GPTQ Marlin IMA")은 **여전히 open(reopened)**.
  출처: https://github.com/vllm-project/vllm/pull/36889 · https://github.com/vllm-project/vllm/issues/36811
- 커널 소스는 2026-06-11 `csrc/libtorch_stable/moe/marlin_moe_wna16/`로 ABI 이전(#44565, `6e64c1bab`)된 뒤 **버그픽스 커밋 0건** (commits API, path 필터).
- **함의**: Phase 1 `probe_35922` 그대로 유효. compute-sanitizer로 원인 특정 시 #36889의 `c_tmp` 사이징 가설이 첫 후보. Phase 2 우회(Triton) 필요성 불변.

## (2) KDA prefill Triton 경로의 백엔드 노출

**판정: [확인됨] main = vendored-FLA Triton이 유일 경로(선택지 개념 없음) · K3 브랜치 = `kda_prefill_backend = auto|triton|flashkda` 명시 노출, SM80은 auto→triton 자동 폴백.**

- main: FLA KDA 커널이 **in-tree vendored** — `vllm/third_party/flash_linear_attention/ops/kda.py` (전부 `@triton.jit`, 2026-07-16 #48500 `626c90b2d`로 이동).
  `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`가 prefill=`chunk_kda_with_fused_gate`, decode=`fused_recurrent_kda` 호출.
  메타데이터 백엔드는 `MambaAttentionBackendEnum.GDN_ATTN` (`vllm/v1/attention/backends/registry.py` L176-180 → `gdn_attn.GDNAttentionBackend`).
- 수치검증: main에 **`tests/kernels/test_kda.py` 존재** — "Compares chunk_kda against a naive recurrent reference (float32)" (FLA naive.py 포팅). → 설계 스펙의 "KDA prefill Triton 수치검증 미완" 리스크는 **완화됨**(단위 레벨 레퍼런스 대조 있음). K3 브랜치는 이를 `tests/models/kimi_k3/test_kda.py`(757줄) + `test_kda_metadata.py`(411줄)로 대체·확장.
- K3 브랜치(`0498dc7`): K3 전용 KDA 레이어 `vllm/models/kimi_k3/nvidia/kda.py`.
  - `resolve_kda_prefill_backend(backend, ...)` L148-166: 허용값 `("auto","triton","flashkda")`. `is_flashkda_supported` = `capability.major in (9,10,12)` + bf16 + head_dim 128 + bounded gate → **SM80(major 8)은 불충족** → auto면 `"triton"` 반환. `"flashkda"` 강제 시에만 RuntimeError.
  - 설정 경로: `additional_config.get("kda_prefill_backend", "auto")` (L350).
  - Triton 구현은 K3 전용 vendored 사본 `vllm/models/kimi_k3/nvidia/ops/third_party/kda/{chunk,chunk_intra,chunk_intra_token_parallel,fused_recurrent}.py`.
  - decode도 동일 구조: CUDA `fused_kda_decode`는 `is_fused_kda_decode_supported`가 SM90/SM10x/SM12x + head 수 (12,24,48,96) 등으로 게이트 → SM80은 Triton `fused_recurrent_kda(_packed_decode)` 폴백.
  - FlashKDA 자체는 외부 프로젝트: https://github.com/vllm-project/FlashKDA @ `a3e42bbbece3bb38f7c426b880315294a336e82f` (`cmake/external_projects/flashkda.cmake`), 지원 아치 `9.0a / 10.0f·10.0a·10.3a / 12.0f·12.0a`만. 아치 미해당 시 **경고만 내고 빌드 스킵**("FlashKDA will not compile: CUDA >=12.0 and a supported architecture (SM90, SM10x, or SM12x) are required") — SM80 소스빌드 치명 아님.
- 블로그(https://vllm.ai/blog/2026-07-27-k3): prefill=FlashKDA(CUTLASS), decode=fused CUDA kernel + **"portable Triton fallback paths"** 명시 — 코드 확인과 일치.
- **함의**: Phase 2의 "KDA=순수 Triton" 우회는 K3 브랜치에 **이미 공식 폴백으로 내장**. FLA 외부 패키지 설치 불필요(vendored). 프로브에서 `--additional-config '{"kda_prefill_backend":"triton"}'`로 고정 가능(강제 명시 권장).

## (3) `mxfp4.py` 백엔드 게이트

**판정: [확인됨] `Mxfp4Config.get_min_capability() = 80` — SM80은 config 게이트 통과. 백엔드 선택 로직은 mxfp4.py에서 `fused_moe/oracle/mxfp4.py`로 이관. [추정] SM80 실선택 = MARLIN/BATCHED_MARLIN (우선순위 목록에서 SM80 생존 후보가 그것뿐).**

- `vllm/model_executor/layers/quantization/mxfp4.py` (main @ d223c900 raw fetch):
  - `get_min_capability() -> 80`, act dtype = bf16 전용.
  - 백엔드 선택은 `from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend, select_deepseek_v4_mxfp4_moe_backend` — 파일 내 하드 SM 게이트 없음.
  - 두 메서드 클래스: `GptOssMxfp4MoEMethod`(gpt-oss 계열) / `Mxfp4MoEMethod`(canonical, DeepSeek-V4 계열 셀렉터 사용).
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`:
  - canonical(DSv4·K3 계열) 우선순위 `_get_priority_backends()` L331-350: `[FLASHINFER_TRTLLM_MXFP4_MXFP8, DEEPGEMM_MXFP4, MARLIN, BATCHED_MARLIN]` — 주석 원문: "SM100+ prefers DeepGEMM FP4 / TRTLLM MXFP8; SM90 falls through to Triton_unfused or Marlin (the backend-level `is_supported_config` check filters by device capability)".
  - gpt-oss 우선순위 L307-328: `[FI_TRTLLM_BF16, FI_TRTLLM_MXFP8, AITER×3, TRITON(OAI triton_kernels), FI_CUTLASS×2, MARLIN, BATCHED_MARLIN, XPU, EMULATION]` (TRITON_UNFUSED는 MTP 버그로 주석 처리).
  - 후보별 `is_supported_config`가 capability로 필터 → SM80에서 FlashInfer(trtllm-gen)·DeepGEMM 탈락, **MARLIN 계열만 잔존** = #35922 crash 커널 가족. Marlin FP4 지원 게이트는 `marlin_utils_fp4.is_fp4_marlin_supported()` = `capability ≥ 75` [확인됨].
  - 강제 지정 매핑 존재: `{"triton", "triton_unfused", "marlin", ...}` → `kernel_config.moe_backend`(MoEBackend)로 오버라이드 (K3 코드가 같은 knob으로 `deep_gemm_mega_moe` 체크).
- **함의**: K3 MXFP4 체크포인트를 SM80에 그대로 올리면 MoE는 Marlin-MXFP4(bf16-act) 경로로 유도 → **#35922 리스크 직결**. Phase 2 우회는 재양자화(W4A16→moe_wna16 Triton, 항목 6) 또는 mxfp4 EMULATION(gpt-oss 목록에만 존재) 확인 필요.

## (4) K3 모델 코드의 hard SM90+/FlashInfer assert

**판정: [확인됨 @ `0498dc7`] K3 모델 코드는 main에 없음(파일 glob `kimi_k3*` 0건) — PR #50000(open)의 `vllm/models/kimi_k3/`에 존재. 코드 내 **hard SM90+ assert 없음**: capability 분기는 전부 soft fallback. FlashInfer는 코드 강제가 아니라 성능 경로+도커 의존성.**

- main @ d223c900: 트리 전체에 `k3` 경로 0건. Kimi 계열은 `kimi_linear.py`(KimiLinearForCausalLM), `kimi_k25.py`(K2.5 VL, DeepseekV2 랩), `kimi_vl/kimi_audio`뿐. → **구 조사문서가 가정한 `vllm/model_executor/models/kimi_k3*.py`는 존재하지 않는 경로**.
- K3 지원 현황: PR **#50000** "[New model] Kimi K3" open (선행 #49999 closed), 트래킹 이슈 #50001 open. 위성 PR: #50076(EPLB)·#50055(DCP)·#50080(MTP)·#50082(benchmark_moe)·#50054(ROCm KDA gate fuse)·#50063(MoRIIO KV transfer) — 전부 open.
- 구조: `vllm/models/kimi_k3/__init__.py`가 플랫폼 분기(nvidia/ ↔ amd/, "hardware-isolated entry point"), 등록 클래스 = `KimiK3ForConditionalGeneration` + `KimiK3MTP` + `KimiLinearForCausalLM`(kimi_linear.py는 이 PR에서 삭제·이전).
- capability 게이트 전수 grep 결과 (`k3_model.py`/`k3_kda.py`/`k3_mla.py`/`attn_res.py`):
  - KDA: 항목 (2)대로 auto→triton 폴백. hard assert는 `"flashkda"` **명시 강제 시에만** RuntimeError.
  - `attn_res.py` L197: cute_dsl 융합 경로가 `is_device_capability_family(100)` 조건 — 조건부 최적화(미해당 시 일반 경로).
  - MegaMoE: `kernel_config.moe_backend == "deep_gemm_mega_moe"`일 때만 활성(opt-in) — 활성 시 EP+SITU+latent-MoE+grouped-topk 요구 raise. SM80은 이 백엔드를 안 켜면 무관.
  - `k3_mla.py`: fp8 KV cache 사용 시에만 "backend that accepts quant query input" assert — `--kv-cache-dtype fp8`을 안 쓰면 무관. MLA 백엔드는 vLLM 표준 셀렉터 경유(TRITON_MLA 지정 가능 여부는 Phase 1 프로브 대상).
- 단, **공식 지원 하드웨어와의 괴리**: PR body는 "docker image `vllm/vllm-openai:kimi-k3` 사용" + 추가 의존성 flashinfer PR [#4178](https://github.com/flashinfer-ai/flashinfer/pull/4178)(MLA)·[#4180](https://github.com/flashinfer-ai/flashinfer/pull/4180)(SITU trtllmgen MoE). 블로그는 "At least one 8× B300 (or GB300 NVL72) node is required; 16× B200 is also supported" + 8×MI355X, "only Docker images are usable now"(pre-release FlashInfer 의존) — **A100/Ampere 언급 0**.
  출처: https://github.com/vllm-project/vllm/pull/50000 · https://vllm.ai/blog/2026-07-27-k3 · https://recipes.vllm.ai/moonshotai/Kimi-K3
- **함의**: SM80 시도는 **kimi-k3 브랜치 소스빌드**가 기준(모델 코드가 main에 없음). 코드 구조상 hard block은 없으므로 `probe_k3_modules`는 유효하되, "PR이 main에 머지되기 전까지 주 단위로 브랜치가 움직인다"를 전제로 커밋 고정 필수. FlashKDA·fused_kda_decode·cute_dsl은 SM80 빌드에서 자동 제외/미사용이라 빌드 자체는 성립 예상 [추정].

## (5) CMakeLists.txt — MARLIN_MOE 아치 목록

**판정: [확인됨] `MARLIN_MOE_ARCHS = "8.0+PTX;12.0f"`(CUDA≥12.8; 미만이면 `"8.0+PTX;12.0a;12.1a"`) → SM80 포함. **`MARLIN_MOE_FP4`라는 변수는 현재 main에 존재하지 않음** — FP4 인스턴스화는 CMake 변수가 아니라 `generate_kernels.py`의 타입 테이블이 결정하며, bf16-act FP4는 sm80 커널 파일에 포함된다.**

- `CMakeLists.txt` L1186-1290 (MoE 확장 `_moe_C_stable_libtorch`, 2026-06-11 #44565로 이전):
  - `MARLIN_MOE_ARCHS`: `8.0+PTX;12.0f` → `sm80_kernel_*.cu` 글롭 빌드.
  - `MARLIN_MOE_SM75_ARCHS`: `7.5` → `sm75_kernel_*.cu`.
  - `MARLIN_MOE_FP8_ARCHS`: `8.9;12.0f`(또는 `8.9;12.0a;12.1a`) → `sm89_kernel_*.cu`.
  - `MARLIN_MOE_OTHER_ARCHS`: `7.5;8.0+PTX` → `ops.cu`(런처/셀렉터).
- `csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py`:
  - QUANT_CONFIGS에 **NVFP4** (`b=kFE2M1f, s=kFE4M3fn, group_blocks=[1]`) 와 **MXFP4** (`a=kBFloat16, b=kFE2M1f, s=kFE8M0fnu, group_blocks=[2]`) 포함.
  - 파일명 분기 L291-296: `a_type == kFE4M3fn`(fp8-act)만 `sm89_kernel_*`, sm75 사전은 `sm75_*`, **나머지 전부 `sm80_kernel_*`** → bf16-act NVFP4/MXFP4 Marlin MoE 커널이 SM80 대상으로 빌드됨. (#35922에서 FP4 모드도 crash한 사실과 정합.)
- 참고: L963-1001의 `FP4_SM120_ARCHS`(12.0f)·`FP4_SM100_ARCHS`(10.0f;10.7f;11.0f)는 CUTLASS FP4 GEMM/MoE 블록("MXFP4 experts quantization is currently compiled only in this block") — Marlin과 별개, SM80 무관.
- K3 브랜치는 CMakeLists +48줄(FlashKDA 등) — SM80 빌드 시 위 목록 자체는 동일 [확인됨: PR files diff 목록], 세부 diff 내용은 미열람 [미확인].
- **함의**: SM80용 소스빌드에서 `TORCH_CUDA_ARCH_LIST=8.0` 시 Marlin MoE(INT4·FP4 bf16-act)는 컴파일된다 — 문제는 빌드가 아니라 런타임 crash(#35922). Phase 2 커널수정 경로의 대상 파일은 `csrc/libtorch_stable/moe/marlin_moe_wna16/{ops.cu, kernel.h, marlin_template.h}`.

## (6) Triton `moe_wna16` 경로의 SM80 지원

**판정: [확인됨] 지원. Triton MoE experts의 게이트는 `has_device_capability((7,5))` — SM80 통과. 단 자동 선택 우선순위에서 Marlin보다 뒤라서, compressed-tensors/GPTQ 체크포인트는 **명시 오버라이드**(`moe_backend="triton"`)가 필요.**

- `vllm/model_executor/layers/quantization/moe_wna16.py`: `get_min_capability() -> 70`. 백엔드는 oracle로 위임: `select_wna16_moe_backend(...)`.
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`:
  - `WNA16MoEBackend` = {MARLIN, BATCHED_MARLIN, HUMMING, CPU, FLASHINFER_TRTLLM, TRITON, XPU, EMULATION}.
  - CUDA 우선순위 L117-122: `FLASHINFER_TRTLLM → MARLIN → BATCHED_MARLIN → TRITON → HUMMING → EMULATION`.
  - `moe_wna16` 체크포인트 레이아웃은 MARLIN/BATCHED_MARLIN/EMULATION에서 명시 배제("the MoeWNA16 checkpoint layout is not supported") → 이 경우 자동으로도 TRITON 도달.
  - TRITON 제약: expert bias 미지원, AutoAWQ 레이아웃 미지원, GPTQ `desc_act`/`actorder="group"` 미지원.
  - 수동 지정 매핑: `"triton" → WNA16MoEBackend.TRITON` (runner `moe_backend` 문자열; K3 코드 기준 knob은 `vllm_config.kernel_config.moe_backend` [확인됨], CLI 표기는 `--kernel-config` 계열로 추정 [추정]).
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`: `TritonExperts.is_supported_config`에 `current_platform.has_device_capability((7, 5))` (L113) — **SM80 OK**; `TritonWNA16Experts(TritonExperts)` (L557)가 이를 상속.
- **함의**: Phase 2 우회 1안(MoE=Triton W4A16) 성립. 재양자화 시 llm-compressor 출력을 (a) `moe_wna16` 호환 포맷으로 만들거나 (b) compressed-tensors 유지 + `moe_backend="triton"` 강제, 두 방식 모두 코드 경로 존재. `desc_act=False`·bias 없음 조건 준수 필요.

---

## 종합 — 설계 스펙 대비 변경점 요약

| 스펙 가정 (2026-07-28 스냅샷) | 재확인 결과 |
|---|---|
| #35922 상태 불명 | **미수정 확정** (stale-close, 관련 PR 미병합, #36811 open) — probe_35922 유효 |
| KDA prefill Triton 노출 여부 불명 | main·K3 브랜치 모두 Triton 경로 내장, K3는 `kda_prefill_backend` knob 노출, SM80 자동 폴백 |
| KDA 수치검증 미완 리스크 | main에 naive-fp32 대조 테스트 존재, K3 브랜치는 전용 테스트 757줄 — 리스크 하향 |
| `kimi_k3*.py`가 main에 있다고 가정 | **없음** — PR #50000(open)의 `vllm/models/kimi_k3/`; SM80 작업은 `kimi-k3` 브랜치 `0498dc7` 기준으로 고정할 것 |
| hard SM90 assert 우려 | 없음 (soft fallback 구조). 단 공식 지원 HW는 B300/B200/MI355X, docker-only, pre-release FlashInfer 의존 |
| Marlin FP4 SM80 인스턴스화 필요 가능성 | 이미 인스턴스화됨(bf16-act) — 필요한 건 신규 인스턴스화가 아니라 **crash 수정** |
| moe_wna16 Triton 우회 | 성립 (capability 게이트 7.5), 단 자동선택이 Marlin 우선이므로 명시 오버라이드 필수 |

출처 일람: 본문 각 항목의 GitHub URL + raw fetch (전부 2026-07-28, main=`d223c900`, kimi-k3=`0498dc7e`).
