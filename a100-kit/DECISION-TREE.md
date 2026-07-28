# Phase 1 프로브 결과별 의사결정 트리

작성: 2026-07-28 · 스펙 `docs/superpowers/specs/2026-07-28-k3-sm80-serving-design.md`의
Phase 2(우회→커널) / Phase 3(재양자화) 분기와 1:1로 연결. 근거는 전부 `recon/*.md`.

실행 순서(RUNBOOK §4): probe_35922 → probe_kda_fla(main) → [소스빌드] → pytest KDA 2종
(tests/models/kimi_k3/ — main 계열 test_kda.py는 pin 0498dc7 트리에 없음, RUNBOOK §3 주석)
→ probe_kda_fla(k3) → probe_k3_modules.
이후(S2): 같은 노드·venv-k3 에서 `kernel-kit/RUNNER.md` 절차(`run_all.sh`)로 커널 7종 단독
검증을 이어 돈다 — §4 매트릭스의 Phase 2 진입 형태와 무관하게 S2 4게이트 PASS 커널만
S4(vLLM 통합)에 들어간다 (docs/PLAN.md §S2).

전제 사실 (recon에서 확정, 프로브가 뒤집을 수 있는 것만 여기서 다룸):
- 원본 K3 ckpt(MXFP4)의 SM80 **auto** 백엔드 생존 후보는 **MARLIN/BATCHED_MARLIN뿐**
  (canonical 우선순위 [FI_TRTLLM, DEEPGEMM, MARLIN, BATCHED_MARLIN]에서 앞 둘은 SM100+/FP8 —
  recon/vllm-upstream.md (3)). 단 **명시 오버라이드**로는 `"emulation"`(dequant→bf16 GEMM,
  정확성 전용·성능 부적합)과 `"triton"`(OAI triton_kernels 계열, is_supported_config 통과 여부
  미검증)이 매핑에 존재 — `vllm/config/kernel.py` + `fused_moe/oracle/mxfp4.py` @ d223c900 확인.
- W4A16(INT4 compressed-tensors)은 `moe_backend="triton"` 명시 시 TritonExperts(cap≥7.5)로
  동작해야 함 (recon (6)) — 이것이 **Phase 2 우회 1안**이고 arm C가 그 실증.

---

## 1. probe_35922 (Marlin MoE crash) — arm 조합 분기

### 1.1 주 판정: arm A(W4A16 auto) × arm C(W4A16 triton)

verdict 문자열 매핑(스크립트 `classify()`는 아래 표보다 거친 분류를 낸다): 표의
"FAIL (ValueError/oracle 거부)"는 스크립트 verdict `FAIL_OTHER_rc=*`/`FAIL_AFTER_INIT_rc=*`
중 **로그에 거부 사유(ValueError/oracle)** 가 있는 경우, "CRASH (런타임)"는 `CRASH_*` 또는
트레이스가 CUDA/커널 예외인 `FAIL_AFTER_INIT` — 최종 분류는 `logs/arm_*.log` 원문 기준.

| A (재현) | C (우회) | 판정 | 다음 행동 |
|---|---|---|---|
| CRASH_IMA | SUCCESS | **예상 시나리오.** #35922 미수정 확정 + 우회 1안 성립 | ① Phase 2 우회 1안(MoE=Triton W4A16) 확정. ② **Phase 3 재양자화를 "조건부"→"필수 경로"로 격상** — 원본 MXFP4는 auto로 Marlin뿐이라 crash 미해결 시 서빙 불가(단 §1.3 E=SUCCESS면 완화). ③ arm B 결과로 커널 수정 트랙 병행 여부 결정(§1.2) |
| CRASH_IMA | FAIL (ValueError/oracle 거부) | Triton이 이 ckpt 레이아웃 거부 (bias/actorder=group/AWQ 제약 — recon (6)) | 로그의 거부 사유 확인 → 대체 W4A16 ckpt로 C 재실행(`PROBE_MODEL_W4A16=nytopop/Qwen3-30B-A3B.w4a16` 등). Phase 3 재양자화 시 `desc_act=False`·bias 없음·actorder≠group을 **양자화 레시피 제약으로 채택** |
| CRASH_IMA | CRASH (런타임) | Triton MoE 자체도 SM80에서 문제 — 우회 1안 관측 반증 | **중대.** 전체 로그 + env를 갖고 재조사: triton 버전, `VLLM_LOGGING_LEVEL=DEBUG` 재실행. 재현 시 Phase 2는 커널수정 트랙만 남음 → Jimmy 보고 후 진행 판단 |
| SUCCESS | (any) | #35922가 이 환경/버전에서 재현 안 됨 — recon (1)("버그픽스 커밋 0건")과 모순 | wheel(0.26.0)과 이슈(0.16.0)의 차이로 조용히 수정됐을 가능성. ① 소스빌드(Stage 2)에서 arm A 재실행해 wheel/빌드 차이 배제. ② 그래도 SUCCESS면: **원본 MXFP4 Marlin 직행 가능성 재평가**(arm E가 결정적) — Phase 3 필요성이 성능 사유로만 남음 |
| TIMEOUT | — | profile_run이 안 끝남 (이슈 본문은 7분 소요 사례) | `PROBE_TIMEOUT_ARM=7200` 재실행. 그래도면 hang으로 분류 — py-spy dump + 보고 |
| FAIL_OOM | — | 80GB에서 발생하면 설정 문제 | `PROBE_GPU_MEM_UTIL=0.8 PROBE_MAX_MODEL_LEN=8192`로 재실행 (판정 목적에는 충분) |
| CRASH_NO_KERNEL_IMAGE | — | 실행 바이너리에 sm80 Marlin INT4 커널 부재 — recon (5)("8.0+PTX" 릴리스 포함)와 모순 | wheel이면 Stage 2 소스빌드(TORCH_CUDA_ARCH_LIST=8.0)에서 arm A 재실행. 소스빌드도 같으면 recon (5) 재조사 + 판정 보류·보고 (§1.3 arm E와 동일 처리 계열) |
| FAIL_AFTER_INIT_rc=* / FAIL_OTHER_rc=* | — | 크래시·OOM·타임아웃 어느 패턴에도 안 걸린 실패 (HF 인증/디스크/의존성 등 환경 문제 가능성 우선) | `logs/arm_*.log` 전문에서 예외 원문 확인 → 환경 문제면 §1 사전조건 재점검 후 재실행, vLLM 예외면 트레이스 첨부해 보고 (판정표에 우겨넣지 말 것 — 미분류 실패는 재현 로그가 판정보다 우선) |
| (meta) PREFETCH_FAIL_W4A16 / PREFETCH_FAIL_MXFP4 | — | HF 다운로드 실패 — 해당 arm 자동 스킵됨 | §1.3/§1.4 사전조건(네트워크·디스크) 재점검 → `PROBE_SKIP_DOWNLOAD=1`(캐시 있을 때) 또는 재시도. 스킵된 arm이 있는 채로 종합 판정(§4)으로 넘어가지 말 것 |

### 1.2 arm B (compute-sanitizer memcheck) — 원인 특정

주: arm B의 SUMMARY verdict에는 `_WITH_SAN_ERRORS` 접미사가 붙을 수 있다(memcheck는 invalid
access를 보고만 하고 앱을 계속 진행시킬 수 있음 — 예: `SUCCESS_WITH_SAN_ERRORS`). 이 표는
접미사·base verdict와 무관하게 **sanitizer 로그 내용**(`logs/sanitizer_B.log`)으로 분기한다.

| sanitizer 출력 | 해석 | 다음 행동 |
|---|---|---|
| `Invalid __global__ write/read` in `marlin*` 커널, c_tmp/reduce 버퍼 관련 스택 | **#36889의 `c_tmp` reduction-buffer OOB 가설 확정** (recon (1)) | Phase 2 커널수정 트랙 개시: 대상 = `csrc/libtorch_stable/moe/marlin_moe_wna16/{ops.cu,kernel.h,marlin_template.h}`. 미병합 PR #36889 diff를 출발점으로 재구현(그대로 취하지 말고 sanitizer 좌표로 검증). 수정 후 arm A/B 재실행 = 회귀 게이트 |
| Invalid access이지만 다른 커널/위치 | 새 원인 | 스택·좌표 기록 → vLLM 이슈 갱신(새 이슈 or #36811에 코멘트) + 수정 난이도 재평가 |
| sanitizer clean인데 crash 재현 | 동기화/레이스 계열 의심 | `--tool racecheck`, `--tool synccheck` 재실행. CUDA graph 관련이면 eager 여부 재확인(프로브는 eager 고정) |
| sanitizer 하에서 crash 미재현 | 타이밍 민감 버그 | 이 자체가 레이스 증거에 가까움 — racecheck 우선 |
| SKIPPED_NO_SANITIZER | compute-sanitizer 미설치 (CUDA 툴킷 부재 노드) | §1.1 판정(A×C)만으로 Phase 2 우회 여부는 결정 가능. 단 **커널수정 트랙(§1.2 행 1)은 sanitizer 없이 착수 금지** — CUDA 툴킷 설치(RUNBOOK §6 nvcc 항목) 후 arm B만 재실행 (`PROBE_ARMS="B"`) |

### 1.3 arm D/E (gpt-oss-20b, MXFP4) — K3 원본 경로 대리 판정

주의: gpt-oss는 GptOss 전용 셀렉터(우선순위에 TRITON(OAI)/EMULATION 포함), K3는 canonical
셀렉터 — **auto 결과(D)는 참고, marlin 강제(E)가 본 판정**. 로더도 다름(gpt-oss 자체 포맷 vs
K3 compressed-tensors mxfp4-pack) — 커널 가족 판정으로만 사용.

| E (marlin 강제) | 판정 | 다음 행동 |
|---|---|---|
| SUCCESS | **MXFP4-Marlin(bf16-act) SM80 동작** — recon (5)의 "sm80 인스턴스화됨" 실증 | K3 **원본 ckpt 직행 lane 부활**: Phase 2에서 K3 실물 로드 시 auto(Marlin) 시도 가치 있음. Phase 3는 성능(§6 bytes/token 137GB→56GB) 사유로 여전히 유력하나 "유일 경로"는 아님 |
| CRASH_NO_KERNEL_IMAGE | 실행 바이너리에 sm80 FP4 Marlin 부재 | wheel이면 소스빌드(Stage 2, TORCH_CUDA_ARCH_LIST=8.0)에서 재실행. 소스빌드도 같으면 recon (5) 재조사(generate_kernels.py 산출 확인: `ls csrc/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_*fe2m1*`) |
| CRASH_IMA | INT4와 동족 crash — FP4도 같은 버그 계열 | §1.2 커널수정 트랙에 FP4 케이스 추가 (수정 1회로 INT4+FP4 동시 해결 겨냥). K3 원본 직행은 수정 전 불가 확정 |
| FAIL (oracle 거부) | gpt-oss 레이아웃이 marlin 강제와 비호환 | D(auto) 결과로 대체 판단, K3 실물 판정은 Phase 2로 이월 |

- D(auto)가 TRITON(OAI triton_kernels)로 성공하면: K3 canonical 셀렉터에는 그 경로가 자동
  후보에 없지만 **명시 `moe_backend="triton"` 매핑은 존재** — Phase 2에서 K3 실물에
  `kernel_config={"moe_backend":"triton"}` 시도를 백로그에 추가(성립 시 재양자화 없이 MXFP4
  Triton 서빙 가능성).
- `"emulation"`은 어느 결과에서든 **정확성 디버깅 전용**(dequant→bf16 GEMM, K3 급에서 성능
  불가) — Phase 2 트러블슈팅 도구로만 기록.

---

## 2. probe_kda_fla (KDA Triton 수치) — 분기

임계: rel-RMSE < 0.005 (T1~T4; vLLM 업스트림 test_kda.py와 동일), < 0.008 (T5/T6 게이트 경유).

| 결과 | 판정 | 다음 행동 |
|---|---|---|
| 전부 PASS (main+k3, bf16+fp16) + pytest 2종 green | **KDA SM80 리스크 해소** — recon/kda-fla.md §5-1의 "수치 검증 공백" 닫힘 | Phase 2 KDA=Triton 확정(`kda_prefill_backend="triton"` 명시). 스펙 리스크 항목("KDA prefill 수치 미검증") 종결. state 레이아웃 보고값(T2/T3의 `state_layout`)을 hat-kv-manager 산정 문서에 기록 |
| impl=..\|test=import\|verdict=UNAVAILABLE (또는 summary NO_IMPL_LOADED, exit 3) | 대상 구현 import 실패 — venv/브랜치 불일치 (§3 S0와 동일 원인 계열) | `--impl main`은 Stage 1 venv, `--impl k3`는 Stage 2 venv에서만 성립(RUNBOOK §2/§3). 올바른 venv인데도 실패면 브랜치 드리프트 절차(§5)로 import 경로 재확인 |
| FAIL_TORCH_IMPORT / FAIL_NO_CUDA / FAIL_NOT_SM80 (exit 3) | 환경 게이트 — GPU 노드가 아니거나 venv 미활성/비SM80 | RUNBOOK §1/§2 사전조건 재점검 후 올바른 노드·venv에서 재실행 (판정 아님). non-SM80 강행은 `PROBE_ALLOW_NON_SM80=1` — 단 SM80 판정과 비교 불가 |
| verdict=INFO (env, PERF_*) | 판정 라인 아님 — 환경 스냅샷·성능 참고치 | 행동 없음. PERF 수치는 Phase 0 hatir 판정치와 대조해 기록만 (게이트는 Phase 0 소관) |
| 일부 FAIL, rmse ≤ 4×임계 (회색지대) | SM80 수치 열화 의심 (H100 CI 기준과 괴리) | ① fp16과 bf16 중 어느 쪽인지 확인(bf16만 회색 = 정밀도 특성일 수 있음). ② 동일 시드로 3회 반복 — autotune 비결정성 분리. ③ H100 대비가 필요하면 결과를 갖고 upstream(vLLM tests/kernels/test_kda.py 값)과 비교 후 FLA/vLLM에 보고. Phase 2는 **보류 아님**(회색은 진행하되 e2e ppl 체크 추가) |
| FAIL, rmse > 4×임계 | 커널 버그 (SM80 특이) | FLA에 이슈 보고(재현 스크립트 = 이 프로브) + vLLM(벤더링 사본 소관)에도. 완화 실험: chunk 커널 autotune 후보 고정(BT=32/64/128 개별) 후 재측정 — 특정 config만 깨지면 해당 config 제외 패치로 Phase 2 진행 가능 |
| CRASH (triton 컴파일/런타임) | FLA #625류 (autotune×환경 상호작용) 또는 SM80 미지 이슈 | triton 버전 기록(env INFO 라인) → triton 업/다운그레이드 1회 시도(torch 동봉 버전 우선) → 재현 시 FLA 이슈 + vLLM 이슈. **KDA가 막히면 Phase 2 전체가 막히므로 최우선 에스컬레이션** |
| T5/T6만 FAIL (T1/T4 PASS) | 게이트/디코드 융합 경로 문제 (K3 브랜치 신규 코드) | vLLM PR #50000에 보고. 임시 우회: K3 `fused_recurrent_kda(fuse_gate=False)` 경로(비융합 게이트, 코드 존재 확인됨) 검증 후 Phase 2에서 해당 경로 강제 가능한지 확인 |
| main PASS · k3 FAIL | 브랜치 사본 드리프트/버그 | main↔k3 사본 diff (`diff vllm/third_party/.../kda.py vllm/models/kimi_k3/nvidia/ops/third_party/kda/`) → PR #50000 보고 |

---

## 3. probe_k3_modules (K3 모듈 SM80) — 분기

주: `MODEL_PREP_OK`/`ENGINE_INIT_OK`/`GENERATE_OK`는 진행 라인(판정 아님 — 다음 단계 계속).
`WARN_*`는 exit code에 반영되지 않는 참고 라인 — 드리프트 신호로 기록만 하되, 반복되면 recon 갱신.

| 결과 | 판정 | 다음 행동 |
|---|---|---|
| S0 FAIL_TORCH_IMPORT / FAIL_NO_CUDA / FAIL_NOT_SM80 / FAIL_VLLM_IMPORT | 환경 게이트 — GPU 노드/venv 문제 (K3 판정 아님) | RUNBOOK §1/§3 사전조건·venv 재점검 후 재실행 |
| S0 FAIL_NOT_K3_BRANCH | wheel/main 빌드에서 실행됨 | Stage 2 venv 확인. **PR #50000이 main에 머지된 뒤라면**: main 새 커밋으로 재고정하고 recon (4) 갱신 (모델 코드 경로가 `vllm/models/kimi_k3/` 그대로인지 확인) |
| S1 FAIL_IMPORT_KDA_FUNCS | 함수 이동/개명 의심 | 브랜치 드리프트 절차(§5) — import 경로 재확인 후 프로브 수정·재실행 |
| S1 UNEXPECTED_* / FAIL_RESOLVE_AUTO / FAIL_AUTO_NOT_TRITON | SM80 게이트 로직이 recon 스냅샷과 달라짐 | 해당 함수 소스 재열람 → recon/vllm-upstream.md (2) 갱신 → 프로브 기대값 수정 후 재실행 |
| S2 FAIL_NOT_REGISTERED | `KimiK3ForConditionalGeneration`이 레지스트리에 없음 — 브랜치 드리프트(등록 클래스 개명) 의심 | `ModelRegistry.get_supported_archs()` 출력에서 `Kimi*` 검색 → 개명 확인 시 §5 절차 + probe/recon의 클래스명 갱신. S3는 config `architectures` 매칭이 실질이므로 S3 결과와 교차 확인 |
| S3 FAIL_MODEL_PREP | HF 소형파일 다운로드/축소 config 생성 실패 (엔진 도달 전) | 네트워크·디스크 확인 → 재실행. `moonshotai/Kimi-K3` revision `9f62e4e9` 접근 불가면 recon/k3-config.md의 pin 재확인(리포 개편 가능성) 후 새 revision 기록. config 구조 드리프트(KeyError)면 recon/k3-config.md 갱신 + mutate 규칙 수정 |
| S3 FAIL_ENGINE_INIT — 트레이스가 quantization/MoE 계열 | K3 구현이 mxfp4 전제를 하드코딩했을 가능성 (recon [미확인] 항목) | tiny config에 실물 `quantization_config`(config.full.json에 보존됨)를 되살려 재시도 — 단 그러면 Marlin-MXFP4 경로를 타므로 #35922 crash와 얽힘(arm E 결과 참조). 하드코딩 확인 시 PR #50000에 보고 |
| S3 FAIL_ENGINE_INIT — processor/vision 계열 | 멀티모달 전처리 요구 파일 문제 | 다운로드 파일 목록 확인(kimi_k3_processor.py, preprocessor_config.json 포함 여부) → allow_patterns 확장 재시도 |
| S3 FAIL_ENGINE_INIT — OOM | dummy라도 활성/캐시 산정 초과 | `PROBE_GPU_MEM_UTIL=0.8` 또는 max_model_len 512로 재시도 |
| S3 FAIL_GENERATE | forward 크래시 — 모듈 경로 문제 (KDA/MLA/MoE 중 트레이스로 특정) | 트레이스의 모듈로 분류: KDA면 probe_kda_fla 결과와 교차 확인, MLA면 `--mla-backend TRITON_MLA` 강제 재시도, MoE면 tiny 파라미터(experts 32/top16) 재검토 |
| S3 PASS | **K3 모듈(KDA/AttnRes/LatentMoE/MLA) SM80 초기화·forward 성립** | Phase 2 진행 신호. S3_BACKEND_REPORT의 MLA/어텐션 백엔드 라인을 기록 → Phase 2 설정값으로 채택 |
| `--mla-backend TRITON_MLA` 강제 PASS | TRITON_MLA 지정 가능 확정 (recon (4)의 미해결 질문) | Phase 2 기본 MLA 백엔드 후보로 채택 (auto 선택과 성능 비교는 Phase 4) |
| 강제 FAIL, auto PASS | 백엔드 이름/지정 방식 드리프트 | auto가 고른 백엔드(S3_BACKEND_REPORT)를 그대로 쓰고, 강제 지정법은 소스에서 재확인 |

---

## 4. 종합 매트릭스 → Phase 2 진입 형태

축: ① MoE 우회 성립(=arm C SUCCESS) ② KDA 수치 OK(=probe_kda_fla PASS) ③ K3 모듈 OK(=S3 PASS)

| ① | ② | ③ | Phase 2 진입 형태 |
|---|---|---|---|
| ✓ | ✓ | ✓ | **우회 1안 풀셋**: K3 W4A16(Phase 3 산출물) + Triton MoE + Triton KDA + (TRITON_)MLA. 커널 신규작성 0으로 토큰 도달 시도 — 스펙 Phase 2-1 그대로 |
| ✓ | ✓ | ✗ | 커널은 되는데 모델 통합이 문제 — §3 분기로 원인 좁힌 뒤 PR #50000 이슈화. Phase 2는 브랜치 수정 대기 or 자체 패치 |
| ✓ | ✗ | — | KDA가 병목 — §2 에스컬레이션. MoE만으로는 서빙 불가(93층 중 69층이 KDA) |
| ✗ | ✓ | ✓ | MoE 우회 실패 — Phase 2-2(커널 수정: §1.2 c_tmp 트랙)로 직행. hatir 타일 사전선정은 이 트랙에서 무관(디버깅 문제, 성능 문제 아님) |
| ✗ | ✗ | — | 프로브 킷 전면 재검토 + Jimmy 보고 (조기 중단 후보 — 스펙 리스크 절 참조) |

추가 게이트: arm E=SUCCESS면 위 표와 독립적으로 "원본 MXFP4 직행" lane이 열림 —
Phase 3 착수 전에 K3 실물 1노드 부분로드(레이어 서브셋) 스모크로 확인할 것.

Phase 0(hatir 판정)과의 결합: Phase 0이 "모든 커널 완벽 가정 시 X tok/s"를 내면,
위 매트릭스의 어떤 성공 조합도 X를 넘지 못한다 — X가 실사용 불가면 프로브 결과와
무관하게 중단 보고가 우선이다(스펙 Phase 0 게이트).

---

## 5. 브랜치 드리프트 절차 (공통)

kimi-k3 브랜치가 pin(0498dc7)에서 움직였거나 main에 머지된 경우:
1. 새 커밋 해시 기록 (`K3_BRANCH_ACTUAL_COMMIT.txt` — RUNBOOK §3이 자동 생성).
2. 프로브 관련 표면적 diff 확인:
   `git diff 0498dc7..HEAD -- vllm/models/kimi_k3/ vllm/config/kernel.py csrc/libtorch_stable/moe/ | head -200`
3. 함수 이동/개명 시 이 킷의 import 경로 수정 (probe_k3_modules.py S1, probe_kda_fla.py IMPL_MODULES).
4. 결과 보고 시 반드시 "pin 대비 드리프트 있음 + 새 해시"를 명기 — recon 문서 갱신 트리거.

## 6. 결과 반출 체크리스트

- [ ] `results/probe_35922_*/SUMMARY.txt` + `logs/` (sanitizer 로그 포함)
- [ ] `probe_kda_fla` stdout 전문 (PROBE_KDA| 라인 + rmse 값들)
- [ ] `probe_k3_modules` stdout 전문 (PROBE_K3M| + S3_BACKEND_REPORT| 라인)
- [ ] pytest 2종 결과 (`-q` 요약이면 충분; tests/models/kimi_k3/ 2파일 — RUNBOOK §3 주석 참조)
- [ ] `build-k3.log` 마지막 200줄 + `K3_BRANCH_ACTUAL_COMMIT.txt` (있으면)
- [ ] env.txt (driver/nvcc/pip freeze)
