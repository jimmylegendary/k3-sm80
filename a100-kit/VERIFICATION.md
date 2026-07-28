# a100-kit 검증 리포트 (킷 검증자 패스)

검증일: 2026-07-28 · 대상: `a100-kit/` 5파일 (RUNBOOK.md, DECISION-TREE.md, probe_35922.sh,
probe_kda_fla.py, probe_k3_modules.py) · 방법: 정적 검사 + **pin된 원본 소스 raw 재fetch 대조**
(vLLM v0.26.0 태그 · main `d223c900` · kimi-k3 `0498dc7e` git trees API `truncated:false` 전수)
+ HF API + PyPI API + NVIDIA Compute Sanitizer 공식 문서. 검증 보조: codex(gpt-5.6-sol) 정적
리뷰 1회(교차 확인용) — 최종 판단·수정은 Claude.

## 1. 검사 결과 요약

| 검사 축 | 결과 |
|---|---|
| `bash -n probe_35922.sh` | PASS |
| `python3 -m py_compile` (probe 2종) | PASS (수정 후 재확인 포함) |
| 명령·플래그·env var·시그니처 실존 (pin 대조) | **결함 2건 발견·수정**: F1(존재하지 않는 pytest 경로), F11(k3 decode 커널 T==N 제약 위반 — T6 상시 CRASH). 나머지 호출·플래그·env는 전부 실존 확인 (§3) |
| HF 모델 ID 실존 | 전부 확인 (§3.4) — 킷의 "Kimi-K3-0.18B 부재(401)" 주장도 재확인 일치 |
| DECISION-TREE 분기 완결성 | verdict 전수 나열 대조 → 갭 발견·수정 (F4~F10, F18~F19) |
| exit-code 계약 (false-green) | 결함 4건 수정 (F2a/F2b/F12/F14) |
| GPU 없는 호스트 오실행 안전성 | 안전 (§4) — 수정 불요 |

## 2. 수정 내역 (총 20건 — 코드 9 + 문서 11)

### F1. RUNBOOK §3 — 존재하지 않는 pytest 경로 (중대)
- **결함**: `pytest tests/kernels/test_kda.py` 지시 — 이 파일은 main(`d223c900`)/v0.26.0에는
  있으나(raw 200) **kimi-k3 pin `0498dc7e` 트리에는 없다**(raw 404 + git trees API 전수 확인:
  `test_kda*` 매치는 `tests/models/kimi_k3/{test_kda.py,test_kda_metadata.py}` 2건뿐).
  Stage 2 checkout에서 pytest가 경로 오류(rc=4)로 죽어 러너를 막는다.
- **수정**: RUNBOOK §3 해당 줄 삭제 + 사유 주석(main vendored 사본 수치검증은 Stage 1
  `probe_kda_fla --impl main`이 동일 naive-fp32 대조로 커버) · §4 "3종→2종" ·
  트러블슈팅 행 보강 · DECISION-TREE 3곳("pytest 3종" 참조) 동기화.

### F2. probe_kda_fla.py — exit-code 계약 위반 2건 (false-green)
- **결함 a**: `--impl auto`(기본값)에서 세 구현 모두 import 실패(UNAVAILABLE)면 테스트 0개
  실행인데 **exit 0** — 깨진 venv에서 조용한 초록.
- **결함 b**: `--perf` 경로의 CRASH가 `run_guard`로 격리만 되고 exit code에 미반영
  (perf_smoke는 반환값이 없어 성공/크래시 모두 None — 기존 누적 방식으로 감지 불가).
  헤더 계약("3=크래시")과 SUMMARY의 CRASH 카운트에 모순.
- **수정**: `n_loaded==0`이면 `NO_IMPL_LOADED` 라인 emit 후 exit 3; 최종 exit를 RESULTS의
  FAIL/CRASH 카운트 기준으로 재산정(격리된 모든 CRASH 포함).
- 부가: env INFO 라인의 `__import__("triton")`을 try로 보호(부재 시 "?"로 기록, 미분류
  traceback 방지).

### F3. probe_k3_modules.py — DECISION-TREE 절 참조 오류
- **결함**: S1 import 실패 hint가 "§3"(probe_k3_modules 분기표)을 가리킴 — 커밋 재고정/import
  경로 수정 절차는 **§5(브랜치 드리프트)**.
- **수정**: hint를 §5로 교정.

### F4~F9. DECISION-TREE 분기 갭 (스크립트가 낼 수 있는 verdict 전수 대조)
`classify()`/emit 가능한 verdict 문자열을 전부 나열해 표와 대조한 결과, 다음 6개에 "다음
행동"이 없었다 — 각각 행 추가:
- **F4** §1.1: arm A `CRASH_NO_KERNEL_IMAGE` (wheel sm80 부재 의심 → 소스빌드 재실행 →
  recon (5) 재조사).
- **F5** §1.1: `FAIL_AFTER_INIT_rc=*` / `FAIL_OTHER_rc=*` (미분류 실패 — 로그 원문 우선,
  판정표에 우겨넣기 금지).
- **F6** §1.1: meta `PREFETCH_FAIL_*` (arm 자동 스킵 상태로 §4 종합판정 진입 금지).
- **F7** §1.2: arm B `SKIPPED_NO_SANITIZER` (우회 판정은 가능하나 커널수정 트랙 착수 금지 —
  툴킷 설치 후 `PROBE_ARMS="B"` 재실행).
- **F8** §2: `UNAVAILABLE`/`NO_IMPL_LOADED` (venv-스테이지 매핑 재확인 → §5).
- **F9** §3: S2 `FAIL_NOT_REGISTERED` + S3 `FAIL_MODEL_PREP` 행 추가.

### F10. DECISION-TREE §2 첫 행 — F1 연동 문구 정정
"pytest 3종 green" 전제 → "2종"으로 (F1과 동일 근거).

### F11. probe_kda_fla.py T6 — k3 decode 커널의 T==N 제약 위반 (중대; codex 발견 → 원본 재확인 확정)
- **결함**: 구 T6는 CU_MIX(T=556, N=4) 전체 시퀀스를 k3 `fused_recurrent_kda`에 넘겼는데,
  pin `0498dc7e`의 `fused_recurrent_kda_fwd`는 1-D `ssm_state_indices`에서 **`assert T == N`**
  (시퀀스당 1토큰 decode 전용 — main 사본의 범용 varlen recurrent와 다름; main 사본에는 해당
  assert 없음을 별도 확인 → T2/T3는 유효). 구 T6는 **항상 AssertionError→CRASH**로,
  DECISION-TREE §2의 "CRASH→최우선 에스컬레이션"을 매번 오발동시켰을 것.
- **수정**: T6를 서빙과 동형인 "chunk prefill(seq당 L-1) → state 핸드오프 →
  fused_recurrent_kda decode(seq당 1토큰, T==N 충족)" 구조로 재설계. decode 토큰 o를
  naive·chunk 전체-T의 해당 위치와, decode 후 state를 chunk 전체-T final state와 대조
  (기존 cross/naive/state 3지표 유지). 인덱스 산식은 CPU dry-run으로 검증
  (552 prefill + 4 decode = 556, 분할 완전성 assert 통과). perf_smoke의 k3 decode는
  원래 32seq×1tok(T==N=32)라 수정 불요 확인.

### F12. probe_kda_fla.py — top-level torch import 실패 시 exit 1 (계약은 3) [codex #11]
try로 감싸 `FAIL_TORCH_IMPORT` emit 후 exit 3.

### F13. probe_k3_modules.py S1 — WRONG_EXC·NO_RAISE 동시 emit 모순 [codex #12]
RuntimeError 외 예외 시 "예외는 났다"로 처리(forced_raises=True) — WRONG_EXC만 WARN으로
남고, 사실과 다른 NO_RAISE("조용히 통과")가 함께 찍히던 모순 제거.

### F14. probe_k3_modules.py — `--stages` 오타 시 아무것도 안 돌고 exit 0 [codex #14]
{S0,S1,S2,S3} 부분집합 검증 추가 — 위반 시 argparse error(exit 2).

### F15. probe_35922.sh — 알 수 없는 arm이 로그로만 무시됨 [codex #3]
`PROBE35922|arm=X|verdict=UNKNOWN_ARM` SUMMARY 라인 추가(기계판독 가시화).

### F16. RUNBOOK §3 빌드 블록 — `pip install -e . | tee`가 실패 exit code 은폐 [codex #7]
블록 서두에 `set -o pipefail` 추가.

### F17. RUNBOOK §2/§3/§4 — python 프로브 결과가 반출 tar에 누락 [codex #8]
python 프로브 2종은 stdout이 결과물인데 §4 반출은 `results/`만 수집 — 실행 명령을
`2>&1 | tee results/*.log`로 교체하고 §4 문구를 실제 동작대로 정정.

### F18. DECISION-TREE §1.1/§1.2 — verdict 문자열 매핑 불일치 [codex #5/#6]
§1.1 C열의 "FAIL(거부)/CRASH(런타임)"이 스크립트 verdict(`FAIL_OTHER_rc=*` 등)와 1:1이
아님 → 매핑 주석 추가(최종 분류는 로그 원문 기준). §1.2에 arm B `_WITH_SAN_ERRORS` 접미사
설명 추가(분기는 sanitizer 로그 내용 기준임을 명시).

### F19. DECISION-TREE §2/§3 — 환경 게이트·참고 라인 행 부재 [codex #17/#18]
§2에 FAIL_TORCH_IMPORT/FAIL_NO_CUDA/FAIL_NOT_SM80 행 + INFO(env/PERF)=비판정 규칙,
§3에 S0 환경/import 실패군 행 + S1 FAIL_IMPORT_KDA_FUNCS/FAIL_RESOLVE_AUTO/FAIL_AUTO_NOT_TRITON
확장 + `*_OK`=진행 라인·`WARN_*`=참고 라인 규칙 추가.

## 3. 실존 대조 결과 (수정 불요 — 킷 주장 그대로 확인됨)

전부 2026-07-28 raw fetch. 아래 항목은 프로브가 실제 호출하는 이름·키워드 인자 단위로 대조했다.

### 3.1 vLLM v0.26.0 (Stage 1 wheel 경로)
- `vllm/third_party/flash_linear_attention/ops/kda.py`: `chunk_kda`(L1458) 시그니처
  `q,k,v,g,beta,scale,initial_state,output_final_state,use_qk_l2norm_in_kernel,cu_seqlens,**kwargs` ·
  `fused_recurrent_kda`(L109) `...,inplace_final_state,use_qk_l2norm_in_kernel,cu_seqlens,ssm_state_indices(LongTensor),**kwargs`
  — probe_kda_fla T1/T2/T3 호출과 키워드 일치 (`torch.long` 인덱스 정합).
- `.../fused_moe/experts/marlin_moe.py`: `fused_marlin_moe`(L223) — RUNBOOK Stage-1 import 검증 유효.
- `vllm/envs.py`: `VLLM_ENABLE_V1_MULTIPROCESSING` 존재 (pin 0498dc7에도 동일).
- `vllm/config/kernel.py`: `MoEBackend` Literal에 `"triton"`/`"marlin"`/`"emulation"` 포함 —
  arm C/E의 `kernel_config={"moe_backend": ...}` 유효.
- `vllm/engine/arg_utils.py` L749-750: `kernel_config` dict→`KernelConfig(**dict)` 변환 존재.
- `CMakeLists.txt` L1181: `MARLIN_MOE_ARCHS "8.0+PTX;12.0f"` — RUNBOOK §2 주장 일치.
- `tests/kernels/test_kda.py` 존재(200) + naive 수식이 프로브 이식본과 **동일**
  (`S=S*exp(g); S+=βk⊗(v−k·S); o=q·S`, q 사전 scale, fp32) + **(V,K) 전치 관례 주석 원문
  확인**("h0 transposed to (V, K) layout for the kernel; naive uses (K, V)") + 임계 0.005 동일.

### 3.2 kimi-k3 pin `0498dc7e` (Stage 2 경로)
- `vllm/models/kimi_k3/__init__.py` 존재, nvidia/amd 분기 + `KimiK3ForConditionalGeneration`/
  `KimiK3MTP`/`KimiLinearForCausalLM` re-export — S0/S2 전제 일치.
- `vllm/models/kimi_k3/nvidia/kda.py`: `is_fused_kda_decode_supported(num_heads,head_dim,conv_width,num_spec,input_dtype,conv_state_dtype)`(L58) ·
  `is_flashkda_supported(head_dim,dtype,lower_bound)`(L85, `major in (9,10,12)`) ·
  `resolve_kda_prefill_backend(backend,head_dim,dtype,lower_bound)`(L148, "flashkda" 강제 시
  RuntimeError) — S1 호출 arity·기대값 전부 일치. `additional_config.get("kda_prefill_backend","auto")`(L350) 확인.
- `.../ops/third_party/kda/__init__.py`: `chunk_kda`/`chunk_kda_with_fused_gate`/
  `fused_recurrent_kda` re-export — probe IMPL_MODULES["k3"] 및 T4 유효.
- `chunk.py` L776 `chunk_kda_with_fused_gate(...raw_g,raw_beta,A_log,g_bias,...,lower_bound,use_qk_l2norm_in_kernel,cu_seqlens)` ·
  `fused_recurrent.py` L429 `fused_recurrent_kda(...,dt_bias,lower_bound,initial_state,cu_seqlens,ssm_state_indices,...,fuse_gate=None→True)`,
  wrapper가 `use_qk_l2norm_in_kernel=True` 상시 — T5/T6 호출·전제 일치.
- **게이트 수식 커널 원문 일치**: `_kda_gate_beta_fwd_kernel`·`kda_gate_chunk_cumsum_vector_kernel`
  모두 `USE_LOWER_BOUND: gate = lower_bound * sigmoid(exp(A_log) * (g + bias))`,
  `beta = sigmoid(raw_beta)`, bias는 flat `[H*D]` 인덱싱 — probe `ref_gate`/입력 shape 정합.
- `vllm/v1/attention/backends/registry.py`: `AttentionBackendEnum.TRITON_MLA`(L80)·`GDN_ATTN`(L180)
  존재; `arg_utils.py` L2321-2329 `attention_backend` 문자열→enum 변환("Reuse the validator...")
  — `--mla-backend TRITON_MLA` 유효.
- `requirements/cuda.txt`: `torch==2.13.0`, `flash-linear-attention==0.5.0`(# KDA),
  `flashinfer-python==0.6.15.post1`+`flashinfer-cubin==0.6.15.post1`+extra-index-url,
  "flashinfer-cubin is not on PyPI ... setup.py excludes" 주석 원문 — RUNBOOK §0/§1.3 일치.
- `requirements/build.txt` **부재** / `requirements/build/cuda.txt` 존재(cmake>=3.26.1,
  torch==2.13.0) — RUNBOOK §3 주석 정확.
- `docker/Dockerfile` `ARG CUDA_VERSION=13.0.3`(L25) · `pyproject.toml`
  `requires-python = ">=3.10,<3.15"` — RUNBOOK §0 일치.
- `vllm/third_party/flash_linear_attention/`(main vendored 사본) k3 브랜치에도 존속 —
  probe l2norm_ref의 1순위 경로가 Stage 2 venv에서도 성립.
- `vllm/model_executor/models/registry.py` 존재 — S2 import 유효.

### 3.3 도구 플래그
- Compute Sanitizer 공식 문서에서 `--tool`, `--launch-timeout`(초, 기본 10/60),
  `--error-exitcode`, `--log-file` 전부 확인 — probe_35922 arm B 커맨드라인 유효.
  출처: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- `timeout` rc 124 판정, `PIPESTATUS[0]` 사용 위치(직전 파이프라인) — bash 의미론상 정확.

### 3.4 HF / PyPI (2026-07-28 API)
| 대상 | 결과 |
|---|---|
| `RedHatAI/Qwen3-30B-A3B-quantized.w4a16` @ `1c842a50` | 200, gated:False |
| `openai/gpt-oss-20b` @ `6cee5e81` | 200, gated:False |
| `moonshotai/Kimi-K3` @ `9f62e4e9` | 200, gated:False; 소형파일 목록에 config.json·
  configuration/modeling/tokenization *.py·tiktoken.model·preprocessor_config.json 등 —
  allow_patterns `["*.py","*.json","*.model","LICENSE"]`로 필요 파일 전부 커버 (가중치 96 shard 제외 확인) |
| `inference-optimization/Kimi-K3-0.18B` | 401 — 킷의 "존재하지 않음" 주장 재확인 일치 |
| `facebook/opt-125m` / `nytopop/Qwen3-30B-A3B.w4a16`(§1.1 대체 ckpt) | 200 |
| PyPI `vllm==0.26.0` | cp38-abi3 manylinux_2_28 x86_64 wheel 존재 |
| PyPI `flashinfer-python==0.6.15.post1` | 존재 (extra-index 없이 설치 가능 — §1.3 주장 성립) |
| K3 tiny-config mutate 대상 필드 | 실물 config.json에 전부 존재 (hidden_size 7168, layers 93,
  attn_res_block_size 12, linear_attn_config.{num_heads 96, head_dim 128, gate_lower_bound -5.0,
  kda_layers, full_attn_layers}, vision_config.text_hidden_size 7168, quantization_config,
  architectures=["KimiK3ForConditionalGeneration"]) — 축소 규칙 유효 |

## 4. GPU 없는 호스트 오실행 안전성 (수정 불요 판정)

- `probe_35922.sh`: precheck가 `nvidia-smi` 부재 → 즉시 FATAL exit 1. 그 전 부수효과는
  cwd 아래 `results/probe_35922_*/` 빈 디렉터리+로그 생성뿐(다운로드·설치·시스템 변경 없음).
  non-SM80 GPU는 `PROBE_ALLOW_NON_SM80=1` 없이는 차단.
- `probe_kda_fla.py`: `torch.cuda.is_available()` False → `FAIL_NO_CUDA` exit 3. 부수효과 없음.
- `probe_k3_modules.py`: S0가 cuda→SM80→vllm→kimi_k3 순서로 게이트; `--stages`로 S0를 건너도
  최소 브랜치 import 가드가 실행되어 개발호스트(wheel/미설치)에서는 S3(HF 다운로드) 도달 전
  exit 10. HF 다운로드는 소형파일 한정 + S3 진입 후에만 발생.
- 세 스크립트 모두 sudo/시스템 경로 쓰기/파괴적 명령 없음.

## 5. 잔여 리스크 (수정하지 않음 — 사유 포함)

1. **T5(및 T6 초기 h0)의 state 레이아웃 하드코딩**: T1/T4는 h0 (V,K)/(K,V) 양방향
   자동판별하지만 T5·T6의 초기 h0 공급은 (V,K) 전치 고정. 업스트림 test_kda.py 주석으로
   (V,K) 관례를 원문 확인했으므로 정상 경로에선 문제없음 — 단 브랜치가 관례를 바꾸면
   T4 PASS·T5 FAIL 패턴이 "게이트 문제"로 오독될 수 있음(§2 T5/T6 행이 잡아주나 원인 분류가
   한 단계 늦음). T6의 state "비교"는 재설계 후 커널↔커널(chunk 산출 state를 그대로 decode에
   공급)이라 레이아웃 가정이 상쇄됨. 의도적 단순화로 유지.
2. **k3 `fused_recurrent_kda`의 `num_accepted_tokens`/`out` 미사용**: 프로브는 기본값(None)
   경로만 검증 — 서빙의 spec-decode 경로는 Phase 2 이후 소관.
3. **브랜치 드리프트**: PR #50000은 open/force-push 가능 — 본 검증은 전부 pin `0498dc7e` 기준.
   드리프트 시 §5 절차가 유일한 방어선 (킷 자체에 이미 명시).
4. **sanitizer arm 소요시간**: memcheck 10~100×, `TIMEOUT_SAN=14400`(4h)이 부족할 수 있음 —
   TIMEOUT 시 §1.1 TIMEOUT 행 절차 준용 (env로 연장 가능).
5. **probe_35922 verdict `FAIL_AFTER_INIT`**: 생성 단계 실패를 IMA/OOM 패턴 매칭으로만 분류 —
   미지 예외는 F5 행(미분류 실패 절차)으로 흡수되지만 자동 분류는 아님.
6. **codex 교차 리뷰 삼사 결과**: 18건 지적 중 9건 수용(F11~F19; 전부 Claude가 pin 원본
   재확인 후 반영 — 특히 #10/T6는 `fused_recurrent.py` L361 부근 assert 원문으로 확정),
   2건은 이미 자체 발견과 중복(F2a/F7), **7건 기각**(사유 명기):
   - #1 `$HOME` unset × `set -u`: 실패가 시끄럽고(unbound variable 즉시 종료) 실서버에서
     HOME 미설정은 비현실 — 코드 복잡도만 증가.
   - #2 탭 구분 `PROBE_ARMS`: 문서화된 사용법이 공백 구분(`PROBE_ARMS="A C"`)이고, 실패
     모드도 "prefetch 생략→arm 내부에서 vLLM이 직접 다운로드(타임아웃에 포함)"로 강등일 뿐
     오판정 없음.
   - #4 prefetch 실패 시 arm별 SKIPPED 라인: meta `PREFETCH_FAIL_*` 라인 + F6 행이 동일
     정보를 이미 제공(스킵된 arm = PROBE35922 라인 부재로 식별 가능).
   - #9 naive가 GVA(HV≠H) 미지원: 프로브는 H==HV 입력만 생성하고 K3도 96/96 — 도달 불가
     경로의 일반화는 범위 밖.
   - #13 `WARN_*`이 exit 0: 의도된 계약(WARN=참고) — §3에 "WARN=기록만" 규칙을 명시하는
     것으로 갈음(F19).
   - #15 vLLM logging handler 미정리: 원샷 프로세스로 종료 직전이라 실효 없음.
   - #16 `_WITH_SAN_ERRORS` 접미사 제거 요구: 접미사는 "memcheck가 보고 후 계속 진행" 정보를
     보존 — 제거 대신 §1.2에 해석 규칙 명시(F18).

## 6. 재검증 절차 (이 리포트의 재현)

```bash
cd a100-kit && bash -n probe_35922.sh && python3 -m py_compile probe_kda_fla.py probe_k3_modules.py
# pin 대조는 §3의 각 파일을 raw.githubusercontent.com/vllm-project/vllm/{v0.26.0|0498dc7e...}/...
# HF는 huggingface.co/api/models/<id>/revision/<rev> (200/401), PyPI는 pypi.org/pypi/<pkg>/<ver>/json
```
