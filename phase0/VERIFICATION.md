# Phase 0 검증 렌즈 판정 및 수정 내역 (VERIFICATION)

날짜: 2026-07-28. 검증 렌즈 3개의 결함 주장 전수를 코드·recon·hatir 계약·외부 1차 출처로
재검증한 뒤, **확인된 critical/major만 수정 적용**하고 모델을 재실행해
`results.json`/`REPORT.md`/`verdict.json`을 갱신했다. minor는 수정하지 않고 여기와
REPORT §10에 공시했다(수정 범위 통제 — 발주 지시).

재검증 방법: 렌즈 주장별로 (i) 지목 라인 실재 확인, (ii) hatir 계약
(`hatir/impl/hatir/collectives.py`, `kernels.py` @ `3a60d38`) 대조, (iii) recon 4개 문서
대조, (iv) 외부 출처 재확인(DeepEP README, MARLIN arXiv — 2026-07-28 열람),
(v) 수치 반사실 재계산. 수정 후 재실행 2회로 결정론 확인, 그리고 **의도 밖 회귀 0 검증**:
변경된 결과 행 24/72개 = 정확히 real×a2a-보유 전략뿐(TP8PP4/TP16PP2 동기값·opt 전체 불변).

## 1. 확인 — critical/major (수정 적용 4건)

### A. PP 마이크로배치 파이프라이닝 미모델 (렌즈1 major / 렌즈2 major / 렌즈3 critical) — 수정
- **확인 근거**: `k3_model.py` `run()`이 `tok_s_system = G×1e3/step_ms`로 계산, `step_time()`은
  93층 전체+PP 경계의 직렬 합 — vLLM PP의 다중 in-flight 마이크로배치가 0으로 계상됨.
  반사실(모델 자체 수치): TP8PP4 real b8 스텝 39.2ms → 최악 stage ≈ 10ms → 8tok/10ms ≈
  800 tok/s로 종전 발표 366(동기)의 2.2배, EP32(576)를 역전. KV 적재도 성립(TP8PP4 max
  321 seq@8k ≥ 상주 32). 세 렌즈의 반사실 계산(790/815/816)과 정합.
- **수정**: `_accum_layers`/`_tail` 리팩터 + `pipelined_throughput()`(EXPLICIT P1) 추가 —
  상주 G를 p개 마이크로배치(G_mb=G/p)로 나눠 stage별 시간(가중치 재스트림·uniq expert를
  G_mb 기준으로 반영, PP 경계 송신, 마지막 stage tail, 마이크로배치당 CPU)을 계산,
  처리량 = G_mb / 최악 stage 시간. 전 행에 `tok_s_system_pipelined` 컬럼 추가, 처리량
  판정·verdict `best_throughput_real_b32`를 파이프라인 기준으로 교체(동기값은
  `tok_s_system_sync`로 병기). 한계(정상상태 상한, 버블 미모델)는 §3/§10에 공시.
- **결과**: real b32 orig 처리량 승자 EP32 576 → **TP8PP4 파이프라인 799 tok/s**
  (µb8×4, 최악 stage 10.0ms). 종전 핵심 발견 1·4(레짐 이원화, PP 세금)는 지표 정의
  인공물로 판명 — 발견 1·4 재작성.

### B. real 노드간 a2a α=44µs 오적용 + E6 dedup/MXFP8의 DeepEP 전제 (렌즈1 major) — 수정
- **확인 근거**: (i) 44µs 출처(nccl#949)는 2노드 16GPU **allreduce** 소메시지 α이며, 자체
  recon `a100-comms.md` §3이 "노드 간 a2a는 per-peer 소메시지로 쪼개져 α가 곱해지는 구조",
  §0이 "노드-로컬 EP 배치가 강제됨"이라 결론 — 32랭크 collective당 44µs 1회는 자기모순.
  (ii) `hierarchical_collective`는 `_fab(e,·,a,e−1)`로 α 총량 = a (랭크 수 무관) 소비 확인.
  (iii) E6 랭크-중복제거(12.75/16 sends)와 MXFP8 payload는 DeepEP류 커널 전제인데 DeepEP
  README Requirements = "Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA
  support" (https://github.com/deepseek-ai/DeepEP, 2026-07-28 열람) — SM80 불가. (iv)
  `recon/vllm-upstream.md`에 SM80 EP a2a 백엔드(deepep/pplx/all_to_all) 근거 0건(grep).
- **수정**: 트윈에 `a2a_alpha_inter_per_peer_us`(real 6.0 / opt 0.0)와
  `a2a_rank_dedup`(real false / opt true) 추가. 모델 E8: inter a2a α_eff = α_base +
  6µs×(노드 밖 peer 수) — per-peer 6µs는 NCCL Simple per-hop(arXiv:2507.04786 Table I,
  기존 트윈 provenance) 기반 [파생]; EP32 real = 44+6×24 = 188µs로 렌즈 제시 naive 밴드
  (100–300µs) 내. E6: real은 naive per-(token,expert) 복사(=topk 16), dedup+MXFP8은 opt
  전용(SM80 커스텀 커널 신규작성 전제의 상한으로 재정의, 트윈 `_comment`/provenance에
  DeepEP·PXN 출처 핀). Phase 1 a2a 마이크로벤치 항목을 §10에 명시.
- **결과**: EP32 real b32 576→387, b1 34.0→17.9 tok/s; TP8EP4 569→521. 렌즈 예상대로
  "보수 라벨의 EP32 우위"가 소멸(비-PP 중에서도 TP8EP4가 EP32 역전).

### C. `dequant_ops_per_param`(real 4.0/opt 2.0) 무근거 (렌즈3 major) — 수정(공시·조건화)
- **확인 근거**: recon 4개 문서 grep — dequant **ops** 유도 0건(read-bytes 논리만 존재).
  `mk_rq` 주석의 "Marlin 논문" 인용은 URL/버전 미고정(프로젝트 규칙 위반)이고 INT4 근거.
  수치 반사실 재계산 확인: TP8PP4 real b32 compute 61.3ms ≈ 전부 dequant 항
  (uniq(32)=392.7 × 33,030,144/8 × 4 / 9.74e12 × 92층 = 61.2ms) — 발견 2(compute 70%)가
  이 파라미터 하나의 함수이며, dq=2면 ALU:HBM 비 1.576→0.788로 메모리바운드 반전.
- **수정**: 트윈에 `_dequant_ops_note`([무근거 밴드] 명시 + 명령어 수준 유도 2–4 ops/param
  참고치) + provenance `marlin_paper` = https://arxiv.org/abs/2408.11743 (MARLIN, PPoPP
  2025 — FP16×INT4 실증, MXFP4 직접 근거 아님) 핀. `mk_rq` docstring 동일 정정. REPORT
  발견 2를 "dq_ops=4 가정 시" 조건부로 재작성 + §10에 Phase 1 dequant 마이크로벤치 명시.
  (파라미터 값 자체는 변경하지 않음 — 대체 근거가 없는 상태의 값 교체는 또 다른 무근거.)

### D. `max_num_seqs=1` 제약 미공시 (렌즈3 major) — 수정(공시)
- **확인 근거**: grep — `max_num_seqs`는 스펙(§물리 타당성 우려)·`recon/k3-config.md` §8
  ("state가 max_num_seqs를 먼저 제약할 수 있음")·`a100-kit/probe_k3_modules.py`에만 존재,
  phase0 산출물 0건. 배치 8/32 결과 전체(파이프라인 포함)가 KDA-state 다중 seq 배칭 +
  DP 어텐션 + EP dispatch의 실동작 전제 위에 있음에도 미공시.
- **수정**: REPORT §10 첫 항목으로 전제 명시(배치 1 게이트와 확실성 등급 분리), 발견 1에
  ⚠ 조건 병기, verdict `caveats`에 등재, Phase 1 프로브 요구(`max_num_seqs>1` 동작 확인)
  명시.

## 2. 확인 — minor (수정 보류, 공시 6건 + 문구 정정 2건)

REPORT §10 "알려진 minor 모델 부채" 항목으로 공시. 전부 게이트/순위 판정 불변 확인.

1. **HBM 피크 100% 계상** (렌즈1·렌즈2 각각 제기): `lin_us`/`ew_us`/`exp_kernel`이
   `HBM_BW=2039e9` 원값 사용, 트윈에 `bw_efficiency` 부재 — real 라벨과 비대칭(통신은
   실측 busbw derate). 영향: b1 real 46.8→~44 tok/s 방향, 게이트(≥5) 9× 여유로 불변.
2. **logits AG scope `"intra"` 하드코딩** (렌즈1): `_tail()`의 `ag_us(..., "intra")` —
   TP16PP2는 TP16이 2노드 스팬이라 inter여야(어텐션 AR는 `ar_scope="inter"`로 옳게 처리한
   것과 비일관). 과소분 스텝당 ~0.04ms(0.2%), 순위 불변.
3. **`ag_us`에 per-rank shard 바이트 전달** (렌즈2): hatir `_bw_factor("allgather")=(P−1)/P`
   링 공식은 msg=전체(gathered) 버퍼 관례 — bw 항이 ta배 과소(b32/ta8 real에서 ~5.7 vs
   ~46µs; 스텝의 <0.05%).
4. **`p2p_us` 25GB/s 고정** (렌즈2): real `ib_node_injection_bps=50e9`(2×HDR)를 p2p가
   미소비 — TP8 랭크쌍 8개 동시 송신 시 랭크당 ~6.25GB/s. 영향 b32 스텝당 ~0.17ms(0.2%).
5. **KDA conv state 4탭 계상** (렌즈3): `KDA_CONV_BYTES_PER_LAYER_SEQ = 3×12288×4×2`
   (294,912B) vs vLLM 공식 `(3·H·d_k/TP, K_conv−1=3)` = 221,184B/층/seq
   (`recon/kda-fla.md` §표) — 1.33× 과대, 게다가 conv 연산 코스트(L391)는 ×3으로 자체
   비일관. state 합계 대비 미미(conv 1.9 vs recurrent 51.8 MiB/GPU/seq), KV 상한 판정 불변.
6. **`MLA_Q_LORA=576` 미사용·오도 상수** (렌즈2): 참조 0건, 실제 q_lora_rank는
   `attn_mla()`에 1536 하드코딩(파라미터 폐쇄 MLA_REPL_P와 정합 확인됨). 유지보수 위험만.
7. **TTFT/prefill 부재 vs '대화형' 게이트 문구** (렌즈1): prefill/TTFT 산출 0건 확인 —
   **게이트 문구를 "decode 한정"으로 명시 정정**(REPORT §0·verdict criterion) + §10에
   Phase 1 prefill 봉투/실측 요구 공시. (모델 추가는 보류 — 스펙의 Phase 0 산출 정의는
   "전략별 tok/s 상한"이라 위반은 아니며, 문구 정확화로 대응.)
8. **방법 서술 부정확** (렌즈2, 부분 확인): `roofline_us`는 import만 되고 호출 0회(dead
   import), 실제 경로는 `einsum.tile` 플랜 + 자체 max(compute, 명시 bytes/HBM) 루프라인 —
   **docstring/REPORT §3 문구 정정 + dead import 제거**(수치 영향 0, 산출 문서의 허위
   방법 서술은 재생성 시 반복되므로 정정이 필요).

## 3. 기각 (1건 — 하위 주장)

- **렌즈2 "hatir에 kernels.build는 존재하지 않고"**: 거짓. `hatir/impl/hatir/kernels.py:289`
  `def build(kernel: str, hw, *shapes, **attrs)` 실재(@ `3a60d38`). 다만 같은 주장의 본체
  (k3_model이 그것을 실제로 쓰지 않으며 docstring이 오도)는 사실이라 §2-8로 부분 인정.
  그 외 세 렌즈의 결함 주장 중 전면 기각 대상은 없음 — 반사실 수치까지 전부 재현됨.

## 4. 수정 후 최종 판정 (재실행 산출, ctx 8k, orig 체크포인트)

- **게이트: PASS 유지** — real 최선(배치1) TP8PP4 46.8 tok/s (≥5, 9.4×), opt 87.3.
  게이트 수치는 수정 전후 불변(TP8PP4는 a2a 없음 — 회귀 diff로 확인).
- **처리량(real b32, P1 파이프라인)**: **TP8PP4 799 tok/s** (µb8×4 in-flight, 최악 stage
  10.0ms) > TP16PP2 697 > TP8EP8PP4 585 > TP4EP8PP4 525 > TP8EP4 521 > EP32 387.
  종전 "EP32 576 승자·레짐 이원화" 결론 폐기(지표 인공물 + a2a α 낙관의 합작).
- 전제·캐비앳: verdict.json `caveats` 4건 (max_num_seqs>1 [미확인], P1 정상상태 상한,
  dq_ops 무근거 밴드, per-peer α [파생]).
