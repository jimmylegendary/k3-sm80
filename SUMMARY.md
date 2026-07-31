# K3 → 32×A100(SM80) 종합 요약 — 원본 MXFP4 five-lane serving 준비

## 0. 2026-07-31 현재 구현 상태

- **checkpoint는 재양자화하지 않는다.** `moonshotai/Kimi-K3` 원본 MXFP4 weight를
  Marlin MoE가 BF16 activation/dequant compute로 실행한다. 따라서 W4A16 checkpoint를
  새로 만들 때 생길 수 있는 weight 정확도 손실은 이 경로에 없다.
- SM80용 MXFP4 Marlin MoE 수정, KDA int64 overflow 수정, KDA Triton 강제,
  software E4M3FN MLA KV cache opt-in이 pinned vLLM `0498dc7e` patchset으로 들어갔다.
- lane 05는 target MLA만 software FP8이고, RoPE DSpark draft layer 93–97은
  `--kv-cache-dtype-skip-layers`로 BF16 cache를 유지한다.
- 독립 gate는 `k08`(원본 MXFP4 MoE), `kda`(k01+k02), `c01`(4-node EP32),
  `k09`(SM80 software FP8 KV), `k10`(DSpark state/integration)로 분리했다.
- 5개 lane의 Compose/Makefile/단일 `.env`/Runbook은
  `deploy/01-*`부터 `deploy/05-*`에 있다. 모두 native vLLM MP이며 Ray를 쓰지 않는다.
- 개발 host의 정적·CPU contract PASS는 A100 kernel/service PASS가 아니다. 실제 결과는
  `docs/A100-5LANE-RESULTS.md`에서 현재 모두 `INCOMPLETE`로 시작한다.

권장 승격 순서: lane 01 TP8/PP4 256k → lane 02 TP8/DP4/EP32 256k →
lane 03 +DSpark → lane 04 1M+software FP8 KV → lane 05 두 기능 결합.

날짜: 2026-07-28 · 스펙: `docs/superpowers/specs/2026-07-28-k3-sm80-serving-design.md` (접근 A 승인본)

**버전 고정 (전 산출물 공통 핀)**

| 대상 | 핀 | 출처 |
|---|---|---|
| 모델 | `moonshotai/Kimi-K3` @ `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | https://huggingface.co/moonshotai/Kimi-K3 (recon/k3-config.md) |
| vLLM main | `d223c900d85224c02f2162ee2c757a769e99f519` (2026-07-28) | https://github.com/vllm-project/vllm (recon/vllm-upstream.md) |
| vLLM kimi-k3 브랜치 | `0498dc7ea9204f1863b46515816f5d8d458e01e7` (PR #50000 head) | https://github.com/vllm-project/vllm/pull/50000 |
| vLLM 릴리스 wheel | `vllm==0.26.0` (2026-07-27) | https://pypi.org/pypi/vllm/0.26.0/json |
| hatir / hat-schema | `3a60d38` / `98ea3f7` | phase0/verdict.json |

파일 지도: 조사 = `recon/{vllm-upstream,k3-config,kda-fla,a100-comms}.md` · Phase 0 = `phase0/{twin_a100_x32.json,k3_model.py,results.json,REPORT.md,verdict.json,VERIFICATION.md}` · 반입 킷 = `a100-kit/{RUNBOOK.md,DECISION-TREE.md,probe_35922.sh,probe_kda_fla.py,probe_k3_modules.py,VERIFICATION.md}`

아래 §1–§4의 Phase 0/1 내용은 2026-07-28 탐색 기록이다. 현재 실행 기준은
§0과 `deploy/01-*`…`deploy/05-*`의 lane별 `RUNBOOK.md`이며, 원본 MXFP4 경로를
폐기하고 W4A16으로 전환하던 과거 분기는 더 이상 실행 계획이 아니다.

---

## 1. Phase 0 게이트 판정: **PASS** — "쓸 만한 속도" 성립

**한 줄**: 현실(보수) 시나리오에서도 최선 전략(TP8PP4) 단일 스트림 decode **46.8 tok/s** ≥ 기준 5 tok/s (9.4× 여유). "모든 커널·통신이 완벽해도"의 낙관 상한 87.3 tok/s, 전략 무관 완전병렬 이론 상한 476 tok/s(2.10 ms/tok — recon 바이트 폐쇄 검증과 일치). 커널 작업에 들어갈 가치가 있다.

수치 전부 `phase0/k3_model.py` 실행 산출(손수치 0), 검증 렌즈 3개 수정 반영 후 값 (`phase0/VERIFICATION.md`).

### 전략별 tok/s (ctx 8k, 원본 ckpt = MXFP4 expert + bf16 잔여)

| 전략 | 단일 스트림 b1 (real → opt) | 처리량 b32 real, PP파이프 정상상태 | b32 지배 병목(real) |
|---|---|---|---|
| **TP8PP4** (노드=stage, 노드내 TP8) | **46.8 → 87.3** | **799 tok/s** (µb8, 최악 stage 10.0ms) | compute(dequant) + HBM |
| TP16PP2 | 42.4 → 83.8 | 697 | compute + comm-α |
| TP8EP8PP4 | 36.0 → 79.3 | 585 | compute + HBM |
| TP4EP8PP4 (과제 지정형) | 29.0 → 51.7 | 525 | compute + HBM |
| TP8EP4 (PP 없음) | 28.3 → 61.4 | 521 | comm-α + compute |
| EP32 (PP 없음) | 17.9 → 63.2 | 387 | comm-α (inter-node a2a) |

- **TP8PP4가 지연·처리량 양 축 모두 승자.** 초기 산출의 "단일 스트림=TP8PP4 / 처리량=EP32 레짐 이원화"는 검증에서 **폐기** — (i) PP 마이크로배치 파이프라이닝을 0으로 치던 지표 인공물, (ii) 노드 간 a2a α를 allreduce α로 낙관하던 오류(per-peer 구조로 정정 시 EP32 collective당 188µs)의 합작이었다.
- 과거 counterfactual로 전량 W4A16이면 TP8PP4 b1 real **65.7**로 모델링됐지만,
  정확도 손실 우려 때문에 선택하지 않았다. 현재 five-lane은 모두 원본 MXFP4다.
- 메모리: 전 전략 80GB 적재 OK(최악 61.3GB/GPU). seq당 KV+state @256k = 2.13GB(TP8PP4)~7.30GB(EP32) → @256k 동시 seq 상한 TP8PP4=11, EP32=4. ctx 민감도: TP8PP4 real b8 기준 8k 25.5 → 128k 19.0 tok/s (완만).

### 신뢰 구간과 가장 약한 가정

신뢰 구간은 통계 CI가 아니라 **real(보수)–opt(낙관) 파라미터 2안 밴드**다: 단일 스트림 46.8–87.3, 처리량 799–1341 tok/s. 절대치는 탐색등급(순위·규모 판단용, HATIR 트렌드-정합 원칙). 약한 가정 순위:

1. **`max_num_seqs`>1 실동작 [미확인]** — 배치>1 결과 전체(처리량 숫자 전부)의 전제. KDA state 다중 seq 배칭이 kimi-k3 브랜치/SM80에서 되는지 미검증. **배치 1 게이트 판정은 이 전제와 무관.**
2. **`dequant_ops_per_param`=4(real) [무근거 밴드]** — "b32에서 dequant ALU가 compute 70% 점유" 발견은 이 값의 함수(dq=2면 메모리바운드로 반전). 실측·1차 출처 없음.
3. **노드 간 a2a per-peer α 6µs [파생]** — NCCL Simple per-hop 기반 1차 근사. EP 전략 순위(387~521)에 직결.
4. PP 파이프라인은 **정상상태 상한**(램프업/드레인 버블·vLLM 스케줄러 실동작 미모델) — 799는 상한.
5. **게이트는 decode 한정** — TTFT/prefill 미모델. '대화형' 체감은 TTFT에도 걸림.
6. 무혼잡·무겹침 α-β 통신 + MoE 균등 라우팅 기대값(불균형 계수 1.5는 expert 커널에만).
7. minor 부채 6건 공시(REPORT §10): HBM 피크 100% 계상(46.8→~44 방향) 등 — 게이트·순위 불변 확인.
8. 사내 IB 구성 미상 → real=2×HDR(50GB/s)/opt=8×HDR(200GB/s) 2안 — TP8PP4 게이트에는 거의 무영향, EP·TP16 전략에만 유효.

1–3번이 Phase 1 마이크로벤치 대상(§3). #35922 수정/우회 성립은 전제가 아니라 **별도 게이트**(속도 모델은 커널이 "돌아간다"를 가정할 뿐).

---

## 2. 업스트림 델타 — 2026-07-28 조사문서(스펙 가정) 대비

| 스펙/구 조사 가정 | 재확인 결과 (recon/*, 전부 커밋 고정) | 계획 영향 |
|---|---|---|
| #35922 상태 불명 | **미수정 확정** — stale-bot `not_planned` close(2026-07-03), 원인 후보 PR #36889(`c_tmp` reduction-buffer OOB) **미병합** close, 동족 #36811 open | probe_35922 그대로 유효, 우회 필요성 불변 |
| K3 모델 코드가 main에 있다 | **없음** — PR #50000(open)의 `vllm/models/kimi_k3/`뿐. SM80 작업 기준 = kimi-k3 브랜치 `0498dc7e` 소스빌드 | Stage 2가 필수 경로로 확정. 브랜치 force-push 대비 드리프트 절차 내장 |
| KDA prefill Triton 노출 여부 불명 | **공식 폴백으로 내장** — `kda_prefill_backend ∈ {auto,triton,flashkda}`, SM80은 auto→triton 자동. FLA는 소스 벤더링(pip 의존 자동 해소) | Phase 2 "KDA=순수 Triton" 우회가 이미 업스트림 기본 경로 = 유리 |
| KDA 수치검증 미완 리스크 | main에 naive-fp32 대조 `tests/kernels/test_kda.py` 존재, K3 브랜치는 전용 테스트 757줄로 대체 — **리스크 하향** (단 A100 CI는 여전히 부재) | probe_kda_fla가 SM80 첫 공식 대조 |
| hard SM90 assert 우려 | **없음** — 전부 soft fallback (FlashKDA·fused_kda_decode·cute_dsl은 SM80에서 자동 스킵/미사용) | probe_k3_modules로 초기화 경로만 실증하면 됨 |
| Marlin FP4 SM80 신규 인스턴스화 필요 가능성 | **이미 인스턴스화됨**(bf16-act MXFP4/NVFP4 → `sm80_kernel_*.cu`) — 필요한 건 **crash 수정** | Phase 2-2 대상 파일 특정 완료(`csrc/libtorch_stable/moe/marlin_moe_wna16/`) |
| FlashInfer pre-release 수동 해소 필요 | **불필요** — #4178/#4180은 B300/B200 도커 전용. 표준 `flashinfer-python==0.6.15.post1`(PyPI에도 존재)로 충족 | 반입 절차 단순화, 설치 금지 명시 |
| 활성 52GB/token (104B×0.5B) | **2.6× 과소** — 원본 ckpt는 **136.96 GB/token** (활성 104.19B 중 53.3%=55.6B가 bf16: KDA+MLA+shared+latent). 바이트 단위 폐쇄 검증 완료 | 병목은 expert가 아니라 bf16 잔여부. W4A16 수치는 과거 counterfactual이며 현재 경로가 아님 |
| 현실 낙관치 단일 스트림 10~30 tok/s 추정 | **46.8(보수)–87.3(낙관)으로 상향 확정** — 93층 직렬화 세금(comm-α 5.6ms + launch 5.1ms/스텝)은 우려대로 실재하나 HBM 스트리밍(8.0ms)과 병렬 규모가 지배 | 게이트 PASS의 실체 |
| probe용 소형모델 `Kimi-K3-0.18B` | HF에 **존재하지 않음**(401) → 실물 K3 config/tokenizer/remote-code + 축소 config + `load_format="dummy"` 대안으로 킷 구현 | probe_k3_modules 설계 변경 완료 |
| hatir 그대로 적용 가능 | 기본 `_gemm_plan`이 이 트윈에서 디코드 compute **10–20× 과대계상**(SM당 smem reuse-tier 클램프) → einsum.tile 직접 저작 플랜으로 대체 | hatir 환류 백로그(부산물: 32×A100 트윈 + K3 케이스 스터디) |

---

## 3. Jimmy 사내 실행 계획 (Phase 1 — 러너: Jimmy)

전 절차 = `a100-kit/RUNBOOK.md`, 결과 판정 = `a100-kit/DECISION-TREE.md`. 반입은 git pull이면 끝(도커 불필요, HF/PyPI/GitHub 자유 전제 — RUNBOOK §1.3이 실검증).

### 실행 순서 + 예상 소요 (GPU 노드 실작업 1.5~2일)

| # | 작업 | 소요(예상) | 산출 |
|---|---|---|---|
| 0 | RUNBOOK §1 사전조건(드라이버/nvcc 13.0/네트워크/디스크 ~85GB/sanitizer) + **IB 구성 확인**(`ibstat`, 노드당 NIC 수) | 30분 | 환경 go/no-go + real/opt 통신 시나리오 중 실제안 확정 |
| 1 | Stage 1: `vllm==0.26.0` wheel venv + 검증 + 스모크 | 30–60분 | 빌드 없이 당일 프로브 가능 |
| 2 | `probe_35922.sh` arm A~E (다운로드 ~31GB 포함; arm B=compute-sanitizer가 지배) | A/C/D/E 각 10–30분, B 1–4h → 반나절 | `SUMMARY.txt` verdict 5종 |
| 3 | `probe_kda_fla.py --impl main` (+`--perf` 선택) | 10–20분 | KDA Triton vs naive-fp32 rel-RMSE |
| 4 | Stage 2: kimi-k3 `0498dc7e` 소스빌드 (`TORCH_CUDA_ARCH_LIST=8.0`, 별도 venv, torch 2.13.0) | 빌드 40–90분(64코어) + 검증 30분 | K3 모델 코드 실행 환경 |
| 5 | `pytest tests/models/kimi_k3/test_kda{,_metadata}.py` | 10–30분 | 업스트림 테스트의 SM80 첫 실행 |
| 6 | `probe_kda_fla.py --impl k3` (+`--impl fla` 교차 선택) | 10–20분 | K3 사본 T4/T5/T6 (fused gate·decode 핸드오프) |
| 7 | `probe_k3_modules.py` (+`PROBE_ATTN_BACKEND=TRITON_MLA` 재실행) | 20–40분 | S0→S3 (KDA 게이트/registry/dummy 엔진+생성) + S3_BACKEND_REPORT |
| 8 | 결과 반출 tar (RUNBOOK §4 체크리스트) | 10분 | 호스트 반출 → Phase 0 모델 재보정 입력 |

**킷 밖 추가 확인 4건** (Phase 0이 신규 등재한 약한 가정 — 스크립트 미포함, 수동 30분~2h씩):
- (a) `max_num_seqs>1` 스모크: probe_k3_modules의 tiny 모델로 batch 4~8 동시 생성 (§1 가정 1 해소 — 처리량 숫자의 성립 조건)
- (b) MXFP4 dequant 마이크로벤치 (가정 2: dq_ops 확정 — compute-bound 결론의 진위)
- (c) 노드 간 nccl-tests `alltoall_perf`/`allreduce_perf` 2노드 (가정 3+8: a2a α와 실제 IB 실효 busbw)
- (d) prefill 1회 실측(TTFT 봉투, ctx 8k) — decode 한정 게이트의 보완

### 각 결과가 계획을 바꾸는 방식 (DECISION-TREE 요지)

- **현재 분기:** patched 원본 MXFP4 Marlin MoE를 `k08`로 먼저 검증한다. PASS면 lane 01로
  진행하고, FAIL이면 해당 MXFP4 kernel을 계속 수정한다. W4A16 재양자화로 우회하지 않는다.
- **k08 sanitizer/수치/shape/perf**는 모두 같은 image digest와 commit에 묶인 stamp로
  남기며, CPU dry-run은 PASS로 승격하지 않는다.
- **probe_kda_fla 전부 PASS** → 스펙 리스크 "KDA prefill 수치 미검증" 종결, `kda_prefill_backend="triton"` 확정. **CRASH면 최우선 에스컬레이션** — 93층 중 69층이 KDA라 막히면 Phase 2 전체 불가.
- **probe_k3_modules S3 PASS** → K3 모듈(KDA/AttnRes/LatentMoE/MLA) SM80 성립 → Phase 2 진입, S3_BACKEND_REPORT 값을 설정으로 채택. FAIL은 원인별로 PR #50000 이슈화/자체 패치 분기.
- **종합**: ①MoE 우회 ②KDA 수치 ③K3 모듈의 8조합 매트릭스가 Phase 2 진입 형태를 결정(DECISION-TREE §4). 마이크로벤치 (a)–(d)는 Phase 0 모델 재보정 → 전략 최종 확정(TP8PP4 기본, TP16PP2 대안) → Phase 4 목표치 = Phase 0 판정치 대비 달성률 관리.

---

## 4. 확정된 현재 방침

1. checkpoint는 `moonshotai/Kimi-K3` 원본 MXFP4만 사용한다.
2. native vLLM MP로 4노드 × 8GPU를 구성하며 Ray를 쓰지 않는다.
3. lane 01→05를 순차 승격하고, 각 lane은 하나의 `.env`와 하나의 profile 기반
   Compose를 사용한다.
4. A100에서 새 image digest로 `k08`, KDA, `c01`, 필요 시 `k09`/`k10` 및 integration
   gate가 PASS하기 전에는 serving 가능을 확정하지 않는다.

---

*부산물(환류 대기): 32×A100 시스템 트윈 + K3 디코드 케이스 스터디 → hatir/hat-schema 자산, hatir `_gemm_plan` 디코드 과대계상 발견 → hatir 개선 백로그.*
