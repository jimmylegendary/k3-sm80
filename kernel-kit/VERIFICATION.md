# kernel-kit 검증-수정 라운드 기록 (S2)

- 일시: 2026-07-28 · 역할: 수정자(검증 결과 접수 → 실증 재확인 → major만 수정)
- 계약: 접수된 검증 결과 중 **실제 확인되는 critical/major만 수정**, 틀린 주장은 기각·사유 기록,
  minor는 판정만 기록(후속 백로그). 수정 후 호스트 게이트(py_compile + `run_all.sh --cpu-dry-run`) green 재확인.
- 접수: 2개 검증 리스트 합산 13건 → 중복 2건(스테일 집계·미추적 로그) 제외 **고유 11건**.
- 판정 합계: **major 2건 확인→수정** · minor 8건 확인(그중 2건은 본 라운드 재실행·커밋으로 치유,
  6건 기록·미수정) · **1건 기각(기제·규모 반증)**.

## 판정표

| # | 대상 | 주장 요지 | sev | 판정 | 처리 |
|---|---|---|---|---|---|
| 1 | common/report.py:198 | f-string 표현식 내 백슬래시 → py<3.12 SyntaxError, 킷 전체 실행 불능 | major | **확인** (호스트 실증: `python3.11 -m py_compile` → SyntaxError 재현, 타 파일 전부 통과) | **수정** — escape 를 f-string 밖으로 |
| 2 | k01 harness b32_T8k | 262,144 tok 단일 varlen 호출 = A100 80GB 확정 OOM → NUMERIC·SHAPES 영구 FAIL | major | **확인** (핀 0498dc7e 원문 재대조: chunk.py L796-798 l2norm 사본·L534/552 g fp32·L209-211 w/u/kg, chunk_delta_h.py **L352 `h = k.new_empty(B,NT,H,V,K)`**·L357 v_new — 동시 상주 ≈104GB ≫ 80GB) | **수정** — `b32_T2k`(32×2048) 교체 + 사전등록 주석 |
| 3 | kernel-report/summary.json | 커밋된 집계 스테일 (k01–k03 MISSING 표기 vs 리포트 존재) | minor | **확인** (counts INCOMPLETE 4/MISSING 3 vs 실리포트 7종) | 치유 — 본 라운드 dry-run 재실행으로 재집계(INCOMPLETE 7/MISSING 0)·커밋 |
| 4 | run_all.sh rc=2 / k01–k03 무리포트 종료 | GPU 모드 사전조건 실패(rc=2)가 FAIL 집계에 안 잡히고, k01–k03 은 리포트 없이 종료(k04 는 all-SKIP 기록 — 비대칭) | minor | **확인** (run_all.sh 판정 루프 rc==1/rc>=3 만 집계; k01 L328-334 write 이전 EXIT_PRECOND vs k04 L243-244 write 후 EXIT_PRECOND) | **수정됨 (후속 라운드 2026-07-28)** — run_all: rc=2 커널 목록 명시 + exit 2(성공 오독 방지); k01–k03: main except 에서 harness_crash evidence 포함 리포트 write 후 EXIT_ERR(k04 와 대칭) |
| 5 | k03 PERF upd_b32 | roofline ≈9.3µs < 30µs 인데 k04/k07 의 사전등록 제외 규칙 없이 eff_floor 0.50 판정 — 자매 하네스와 자기모순 | minor | **확인** (perf_bytes_model(b=32,update)=18.9MB→9.26µs; k04/k07 `PERF_JUDGE_MIN_ROOFLINE_US=30.0` 실재; k03 자신도 b1 launch-bound 를 docstring 에서 인지) | **수정됨 (후속 라운드 2026-07-28)** — k04 동일 30µs 사전등록 규칙 적용: upd roofline<30µs 시 판정 대표를 fn_b1_T8k(3파트 프로덕션 호출 벤치, roofline≈590µs)로 이관, upd_b32 는 excluded evidence 보존; fn 측정 실패 시 update 판정 유지+note_fallback |
| 6 | k01 bench 클로저 | `del inp…` 는 이름만 지울 뿐 셀 참조가 남아 inp ≈1.0GB 포함 ~1.2GB VRAM 상시 잠식 | minor | **기각(부분)** — CPython 3.2+ 에서 셀 변수 `del` 은 셀을 비워 객체를 **해제**함 (호스트 weakref 실증: del 직후 inp alive=False). 잔존은 미삭제 v_bench+h_bench ≈0.21GB 뿐 — 주장 기제·규모(5~6×) 오류 | 잔존 0.21GB 는 사실로 기록(영향 미미, b1_T128k 피크 ≈52GB 대비) — 미수정 |
| 7 | kernel-report 로그 세트 | k04/k07 로그 미추적 + k05/k06 로그 부재 — 반출 증거 세트 비대칭 | minor | **확인** (`git status` ?? 2건, ls-files 에 k01–k03 로그만) | 치유 — 전체 dry-run 재실행으로 7종 로그 재생성·전부 커밋(로그=추적 방침으로 통일) |
| 8 | report.py `--selftest --report-dir <미존재>` | makedirs 없이 open → FileNotFoundError 크래시 (write() 와 비대칭) | minor | **확인** (호스트 재현: L292 FileNotFoundError; run_all 은 L27 mkdir -p 선행이라 러너 플로우 무사) | **수정됨 (후속 라운드 2026-07-28)** — `_selftest` 초입 makedirs(exist_ok=True); 미존재 경로 재현 케이스 PASS 확인 |
| 9 | k02 보조 arm | 'spec-decode 경로' 주장이나 실제 spec 경로는 래퍼 fuse_gate=True(in-kernel gate) — 하네스는 외부 gate 선계산으로 False 분기만 검증 | minor | **확인** (핀 fused_recurrent.py: `fuse_gate None→True`→`use_gate_in_kernel=fuse_gate`; kda.py spec 경로가 raw_g/raw_beta 로 래퍼 기본값 호출; 하네스는 `kda_gate_ref`+sigmoid 선계산 후 fwd 직호출). 단 K3 는 `num_nextn_predict_layers=0`(MTP 없음) — SM80 주경로(packed, in-kernel gate)는 올바로 검증됨, 오판정 위험 없음 | 미수정(minor) — 백로그: arm 을 래퍼(fuse_gate=True) 경유로 추가/교체 + docstring 구성 차이 명기 |
| 10 | shapes.py perf 바이트 모델 | (a) k05 uniq=min(m·topk,E) 는 상한 — m=32 기대 유니크 ≈390 vs 512 계상(목표 ~1.3× 느슨) (b) k06 GEMV 스트림 케이스에 p·2 쓰기 계상(실이동량 과대) | minor | **확인** (기대 유니크 896·(1−(1−1/896)^512)≈390; k06 은 PERF_POLICY measure_only 라 게이트 무영향·evidence 예측치만 왜곡; k05 는 느슨한 방향이라 위양성 FAIL 없음) | 미수정(minor) — 백로그: k05 uniq 기대값 산식(또는 '상한 계상' 명기), k06 케이스별 bytes 분리 |
| 11 | k03 SHAPES 128k 부재 | PLAN §S2 'ctx 8k/128k 대표점' 대비 conv fn 은 8k·varlen 뿐 — 128k 축 int32류 감시 공백 | minor | **확인** (REQUIRED_SHAPES[k03] 에 128k 없음; PLAN §S2 완료판정 3 문구와 편차; 축소 근거 주석은 있으나 편차·감시 공백 미기록) | 미수정(minor) — 백로그: fn_b1_T128k 1점 추가 또는 'k01 이 128k 축 대리 + elem<2^31 산술 확인' 사전등록 명기 |

중복 접수 2건(리스트2 의 스테일 집계·미추적 로그)은 #3·#7 에 합산.

## 수정 상세

### #1 report.py — f-string 백슬래시 (major)

- 증상: `render_md()` 의 `f"{gd['reason'].replace('|', '\\|')}"` — Python 3.12 미만은
  f-string 표현식 내 백슬래시가 SyntaxError. report.py 는 모든 하네스·run_all 이 import 하는
  공통 모듈 → 3.10/3.11 venv 에서 킷 전체 실행 불능(run_all 1단계 py_compile exit 3).
  a100-kit/RUNBOOK.md §2 는 "3.10~3.14 허용" 을 선언하므로 선언 지원범위와 모순.
- 수정: escape 를 f-string 밖 `reason_md` 로 호이스팅(동작 동일). env_check/RUNBOOK 은
  손대지 않음 — 이 수정으로 선언 범위가 다시 참이 됨(fix_hint 의 안전한 쪽 채택).
- 재검증: `python3.11/-3.12/-3.14 -m py_compile common/*.py k0*/harness.py` 전부 rc=0.

### #2 k01 b32 프리필 케이스 (major)

- 증상: `b32_T8k = [8192]×32` — 262,144 tok 를 단일 varlen `chunk_kda_with_fused_gate` 호출로
  실행. 핀 0498dc7e 원문 기준 동시 상주(하네스 입력 q/k/v/raw_g 25.8GB + v clone 6.4 +
  l2norm q/k 사본 12.9 + g fp32 12.9 + w/u/kg 19.3 + h(1,4096,96,128,128) bf16 12.9 +
  v_new 6.4 등) ≈ **104GB ≫ 80GB** → 확정 OOM. OOM 은 케이스 단위로 잡혀 NUMERIC FAIL +
  SHAPES 미커버(FAIL) → 커널이 수치적으로 완벽해도 k01 영구 FAIL(S2 PASS 도달 불가).
  프로덕션 chunked-prefill 도 호출당 이 규모를 내보내지 않음(대표성 없음).
- 수정(사전등록, shapes.py 주석): `b32_T2k = dict(b=32, T=2048)` / `[2048]×32` (합 65,536 tok
  = b8_T8k_varlen 동일 총량, 상주 ≈26GB). PLAN §S2 '배치 32' 커버는 seq 수 32 로 성립,
  T축 8k/128k 대표점은 b1_T8k/b8_T8k_varlen/b1_T128k 담당(최대 상주 케이스 = b1_T128k ≈52GB).
- 파생 갱신: ① int32 감시 evidence 재작성 — 단일호출 최대 elem = b1_T128k 1.61e9 < 2^31,
  elem≥2^31 단일호출(≥174,763 tok)은 상주 산정상 80GB 초과라 S2 실행 불가·실경로 부재
  (감시를 상한 케이스 b1_T128k 로 이관, 사전등록) ② dry SHAPES evidence 의
  `planned_lengths` 하드코딩 `"[8192]×N"` → `"[{v[0]}]×N"` (교체 후 오표기 방지)
  ③ RUNNER.md §6 OOM 행에서 k01 b32/T8k 제거·경위 표기.

## 게이트 최종 상태 (수정 후 재실행)

| 게이트 | 결과 |
|---|---|
| py_compile (3.11 / 3.12 / 3.14) | **green** (rc=0 ×3 — #1 수정 전 3.11 은 SyntaxError) |
| `shapes.py --check` | PASS (상수 24건 대조, 7커널 케이스 30개) |
| `run_all.sh --cpu-dry-run` | **exit 0** · 프레임워크 selftest green · **7/7 INCOMPLETE**(dry 상한, PASS 사칭 없음) · MISSING 0 · 스키마 위반 0 |
| 집계(kernel-report/summary.json) | INCOMPLETE 7 / MISSING 0 (#3 치유), 로그 7종 전부 추적(#7 치유) |

후속 라운드(2026-07-28, 오케스트레이터 직접 수정): #4·#5·#8 수정 완료 — 호스트 게이트(py_compile 3.11/3.12 + run_all --cpu-dry-run exit 0) green 재확인. 잔여 백로그 = #6잔존·#9·#10·#11 (S2 판정 결과를 뒤집는 항목 아님)
(#5 만 GPU PERF 위양성 가능성이 있어 사내 실행 전 처리 권장).

---

# 라운드 2 — vendor화·도커화 (2026-07-29, PLAN §S2 전달 모델 변경)

배경: 사내 vllm pip 설치 실패 + "커널은 torch 로 단독 테스트해야" 지적 → S2 전달 모델을
"사내 pip 설치"에서 "**커널 vendor화 + 도커 이미지 반입**"으로 변경(PLAN §S2, Jimmy 지시).

## 산출물 (커밋 체인)

| 커밋 | 내용 |
|---|---|
| `a4b69ce` | **vendor화** — pin `0498dc7e` 원본에서 `kernel-kit/vendor/` 로 커널 소스 복사(출처·커밋·라이선스 헤더): `fla_kda/`(chunk·chunk_intra·chunk_intra_token_parallel·fused_recurrent), `fla_common/`(chunk_delta_h·cumsum·index·l2norm·op·utils), `mamba_ops/`(causal_conv1d), `attn_ops/`(triton_decode_attention), `situ.py`, `_vllm_shims/`. `vendor/__init__.py` = PEP 562 lazy `_EXPORTS` **13심볼**(k01·k02 KDA 9종 + k03 conv 2종 + k04 decode_attention_fwd + k07 situ_and_mul_native). `check_no_vllm.py` = vendor 트리 vllm import 0건 게이트 |
| `beb7859` | **도커 2티어 + 배포 2경로** — `docker/Dockerfile.triton`(torch 2.13.0+cu126, vLLM 없음; 빌드게이트 + vllm 존재 시 실패하는 역게이트), `docker/Dockerfile.full`(멀티스테이지 vLLM kimi-k3 `0498dc7e` SM80 소스빌드 + compute-sanitizer; GPU 불필요 크로스컴파일), `build.sh`/`push.sh`(ghcr + release 분할 tar 폴백)/`PULL.md`(사내 반입 절차) |
| `6d44e19` | **컨테이너 gate PASS** (`docker/GATE-RESULT.md`) — triton 이미지(`sha256:560f65bc…`) 안에서 ① `run_all.sh --cpu-dry-run` **exit 0**, 집계 PASS 0/FAIL 0/**INCOMPLETE 7**/MISSING 0(dry-run 상한 준수, PASS 사칭 없음) ② vendor import **13/13 callable** + `sys.modules` 에 vllm* 0건(vLLM 없이 torch+triton 만으로 전 체인 실로드) ③ `check_no_vllm.py` green(24파일) |

## 실행 경로 변경 (본 라운드 문서 동기화)

- 주 경로 = **도커** (`docker/PULL.md`): RUNBOOK §0.5·RUNNER §1/§2·DECISION-TREE 실행 전제를
  도커 기준으로 개정, bootstrap/venv 는 "대안(베어메탈)"으로 강등.
- 판정 체계 불변: 4게이트(NUMERIC/SANITIZER/SHAPES/PERF)·리포트 스키마·반출 tgz 루프 그대로.
  SANITIZER 게이트 실행 조건만 명시화 — compute-sanitizer 는 `:full` 이미지에만 동봉.

## 잔여 리스크 (이 라운드 종료 시점)

- **full 이미지 미빌드** — 컨테이너 gate 는 triton 티어만 실증됨. full 은 수 시간 소요
  (vLLM SM80 소스빌드)라 별도 실행 필요; k05(Marlin)·SANITIZER 게이트는 full 에서만 성립.
- full 의 compute-sanitizer PATH 인식(`env_check` 탐색 경로)은 빌드 후 게이트(L149)로 확인
  예정 — 실패 시 `/usr/local/cuda*/bin` 심링크 보정.
- GPU 실측은 사내 A100 에서만 가능 — dry-run 상한(INCOMPLETE)까지가 호스트 실증 한계.

---
# 라운드 3 — K3.1 P1: 핀 a3c14efb 재vendor + k11/k12 (2026-11-09~10, 호스트 검증만)

## 산출물
- vendor 재생성: fla_kda/fla_common/mamba_ops/attn_ops/math_utils 를 핀 a3c14efb 사본으로 교체(본문 무수정 — 예외는 conv1d CPU 블록 제거뿐),
  이전 사본의 인라인 SM80-FIX(kda-sm80-int64) 2건 퇴역. _vllm_shims 에 5개 신규 스텁(gpu_sync_debug·torch_utils·platform_utils·jit_warmup·jit_warmup_triton_helper) + platforms 갱신.
- vendor/dsa_ops = 핀 + `docker/patches-k31/triton-mla-sparse-sm80.patch`(sha256 74675e99…7ed8)의 신규 파일 본문(재생성 도구 `vendor/revendor_dsa_ops.py`,
  계약 테스트 `tests/kernel/test_dsa_ops_vendor.py` 25건). 원천 PR #47629 head 064801dd — **PR 원본이 아니라 배포 패치본**(통합 결함 B 수정, 아래).
- common/dsa_ref.py·dsa_gate.py(`kernel_provenance` evidence)·triton_interp.py, k11_dsa_indexer_sm80/·k12_sparse_mla_sm80/ 하네스.
- 등록: shapes(상수·케이스·perf 모델·_CROSSCHECK; `DSA_ROW_INT32_LIMIT` 3,728,270·`DSA_PACKED_BLOCK_STRIDE_ROWS` 1712) · timing.PERF_POLICY(measure_only, cv_max 0.15) ·
  tolerance.TOL · run_all KERNELS_DEFAULT · relay(`^k\d\d_` + `s2-kernel-report/v1` 만 집계 + ≤12줄 압축) · report.aggregate 타 게이트 스키마 제외.
- PINS 갱신(k01~k09), k04 sha 핀, k01 long-T int64 퇴역 probe.

## 호스트 게이트 (마스터 재실행 2026-11-10)
shapes --check(상수 31·커널 11·케이스 59) · timing --selftest · report --selftest(mixed_dir 7항목 true) · check_no_vllm 34파일 ·
`revendor_dsa_ops.py --check --src-tree <wt-lead>`(패치 본문 == 적용 트리 == vendored) · run_all --cpu-dry-run(exit 0, INC=11, RELAY 10줄, R! 없음) ·
`python3 -m unittest discover -s tests -t .` OK(skip 41) · venv(torch 2.15 cpu·triton 3.9) tests/kernel OK(`K31_VLLM_SRC` 포함) ·
인터프리터 k11 5/5(logits rmse_rel ≤ 9.2e-8, top-k 일치 1.0), k12 4/4(rmse_rel 1.55e-3 ~ 1.88e-3, max_abs ≤ 7.96e-3) PASS — **게이트 미반영 evidence**.

## k11/k12 — 판정 대상은 이미지에 실리는 패치본 (통합 결함 B 수정)
- 결함: `vendor/dsa_ops`가 PR #47629 원본이라 게이트가 배포 커널이 아닌 커널을 판정했다. 수정: 배포 패치의 신규 파일 본문으로 교체했고, 리포트 `evidence.kernel.files`에
  vendored sha256, 헤더가 주장하는 패치 sha256, `to_int64_markers`(≥4)가 남는다.
- k12 REQUIRED 11케이스. 신규 `dec_b1_ctx160k_packed`는 packed hybrid row를 흉내 낸다(블록 stride 1712행, ctx 163,840, 물리 row id가 3,728,270 너머). 풀 5.05GB,
  GPU 추정 5.66GB(≤ 8GiB). 선택 행 중 한계 너머가 0이면 `…/int32_zone` entry가 FAIL한다(공회전 방지). auto 변종은 `max_split_mid_out_numel`만큼 예약한 NaN 오염 workspace에서 스크래치를 받는다.
- 음성 대조(호스트 인터프리터, 서브프로세스): 배포 패치본 small PASS·big PASS / int64 승격 4곳을 뺀 변종 small PASS·**big SIGSEGV** / PR 원본 small PASS·**big SIGSEGV**.
  스크래치 `lse` 저장을 뺀 변종은 NaN 오염 workspace에서 출력 NaN/Inf 로 FAIL. workspace 예약 1원소 부족은 `ScratchReservationError`.
- REQUIRED packed 케이스를 CPU로 그대로 돌려 PASS(선택 2048행 중 308행이 한계 너머, 2.4초, RSS 803MB).

## 정직한 상한 / 미검증
- GPU 게이트(NUMERIC·SANITIZER·SHAPES·PERF 실행) 전부 미실행. CPU 인터프리터 대조는 PASS 근거가 아니다(bf16 `tl.dot`·RZ 캐스트·autotune 보정 3건 적용, triton 3.9.0 전용).
- PERF 는 측정 전용(속도 미보증). DeepGEMM bit-exact 비교 없음(오라클은 동일 양자화 입력 공유). A800 e2e 검증(PR 본문)은 PR 원본 기준이며 패치본은 미검증.
- k11 `dec_b8_ctx256k_pool27x`는 PR 원본 대비 변별력이 없다(paged 커널의 block_idx는 PR 원본도 int64). mqa의 `token_id`/`m`/`n_start` 승격은 logits ≥ 2^31 원소가 필요해 소스 마커로만 고정.
- `block_stride_rows=1712`는 패치 테스트(W1) 값을 인용했고 독립 재유도하지 않았다(recon 근거 추가 필요).
- k05/k06 docstring 의 옛 ops.cu 경로·줄 번호는 이력(PINS 만 갱신). vendor/K01-SM80-ANALYSIS.md 는 핀 0498dc7e 기준 이력 문서.
- 인터프리터에서 k01/k02/k03 커널은 실행 불가(`chunk_intra` `_semantic` ValueError, `causal_conv1d_update` mask dtype) → 정적 계약(AST 시그니처)으로 대체.

