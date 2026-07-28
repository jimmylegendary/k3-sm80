# Phase 0 — hatir 타당성 게이트: Kimi K3 디코드 @ 32×A100(SM80)

생성: `k3_model.py` 실행 산출물 (수치 전부 실행 출력; 손수치 없음). 날짜: 2026-07-28

## 0. 판정 (게이트)

**PASS** — 기준: 단일 스트림(배치 1) ≥ 5 tok/s가 대화형 최소선.

- 현실(보수) 시나리오 최선: **TP8PP4** = **46.8 tok/s** (스텝 21.4ms, ctx 8k)
- 낙관 시나리오 최선("모든 커널·통신이 완벽해도"): **TP8PP4** = **87.3 tok/s** (스텝 11.5ms)
- 완전병렬 이론 하한(전략 무관): 2.10 ms/tok = 476 tok/s 상한 — recon 앵커 2.1ms와 정합(바이트 폐쇄 True)

### 핵심 발견 (수치는 아래 표에서)

1. **레짐 이원화**: 단일 스트림(대화 지연)은 α-이벤트 최소인 **TP8PP4**가, 시스템 처리량(배치 32)은 expert 스트리밍을 32-way로 푸는 **EP32**(576 tok/s)가 승자 — 운영 모드에 따라 배치 전략이 갈린다.
2. **원본 MXFP4의 dequant ALU 세금**: SM80은 FP4 HW가 없어(#35922 커널 가족의 에뮬레이션 경로) 배치 32에서 TP-샤딩 전략의 compute 비중이 70%까지 상승 (TP8PP4/real). EP 분산(expert/rank↓) 또는 W4A16 재양자화가 이를 완화.
3. **Phase 3(W4A16) 가치 정량화**: bf16 잔여 111GB/tok이 지배 병목 — 전량 W4A16이면 현실 시나리오 단일 스트림 46.8 → 65.7 tok/s (1.40×). '조건부'가 아니라 사실상 필수 격상 근거(recon/k3-config.md §6과 합치).
4. **PP의 디코드 세금**: PP는 스테이지 직렬화로 가중치 스트리밍 병렬도를 32→8-way로 떨어뜨림 — 배치가 커질수록 EP(무-PP) 계열이 역전하는 구조적 이유.

## 1. 고정 핀 (재현성)

- hf_kimi_k3: `moonshotai/Kimi-K3 @ 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 (recon/k3-config.md)`
- vllm_main: `d223c900d85224c02f2162ee2c757a769e99f519 (2026-07-28, recon/vllm-upstream.md)`
- vllm_kimi_k3_branch: `0498dc7ea9204f1863b46515816f5d8d458e01e7 (PR #50000)`
- hatir: `3a60d38`
- hat_schema: `98ea3f7`
- twin: `phase0/twin_a100_x32.json`

## 2. 트윈 검증 (hat_schema.validate.validate_twin)

- 오류 0건 필수 → 통과. 전체 메시지 6건 (info/warn는 모델링 선택):
  - [info] ib: ib: no line_bytes — byte-granular movement assumed; set line_bytes = this tier's instruction/burst transfer size to model + validate over-fetch (the 'movement unit')
  - [info] ib: ib: no energy_pj_per_byte — energy for this tier is 0 (fine if you only want time)
  - [info] nvlink: nvlink: no line_bytes — byte-granular movement assumed; set line_bytes = this tier's instruction/burst transfer size to model + validate over-fetch (the 'movement unit')
  - [info] nvlink: nvlink: no energy_pj_per_byte — energy for this tier is 0 (fine if you only want time)
  - [info] l2: l2: no energy_pj_per_byte — energy for this tier is 0 (fine if you only want time)
  - [info] smem: smem: no energy_pj_per_byte — energy for this tier is 0 (fine if you only want time)

## 3. 방법

- 커널: hatir `einsum.tile`로 직접 저작한 플랜(+`roofline_us`) — 디코드 GEMV는 가중치 1회 스트림·n-블록 108 SM 팬·split-k 매핑, MLA attention/KDA recurrent/elementwise도 tile 플랜 (HBM→L2→SMEM→PE 서브스택, max(compute, HBM) 루프라인). hatir 기본 `_gemm_plan`은 이 트윈(SM당 smem 164KB reuse tier)에서 spatial fan이 클램프되어 디코드 compute를 10–20× 과대계상 → 직접 저작으로 대체(코드 주석).
- 통신: hatir `hierarchical_collective`(α-β, 무혼잡 이상치) — α/실효 busbw는 트윈 `meta.comm_scenarios`의 보수(real)/낙관(opt) 2안 (recon/a100-comms.md 근거).
- hatir가 표현 못 하는 항목은 명시 가산항(코드 EXPLICIT E1–E7): 런치/그래프 갭, dequant ALU, MoE 불균형, 혼합 dtype, 스텝 CPU, dispatch 랭크-중복제거, expert grouped-GEMV.
- 한계: 무혼잡·무겹침(통신-계산 오버랩 없음 = 보수), 균등 라우팅 기대값, 절대치는 탐색등급(순위/규모 판단용) — HATIR 검증 원칙(트렌드 정합)에 따름.

## 4. 전략 정의 (32 GPU = 4노드×8)

| 전략 | PP | 어텐션 | MoE | a2a 링크 | 비고 |
|---|---|---|---|---|---|
| TP8PP4 | 4 | TP8 | TP8 | — | 노드=PP stage, 노드 내 TP8 (expert도 텐서샤딩). a2a 없음 |
| TP4EP8PP4 | 4 | TP4×DP2 | EP8 | intra | 과제 지정 'TP4×EP8(노드내 EP)': stage=노드, 어텐션 TP4×DP2, MoE 노드로컬 EP8 |
| TP8EP8PP4 | 4 | TP8 | EP8 | intra | S2 변형: 어텐션 TP8 + 노드로컬 EP8 (bf16 스트리밍 8-way 유지) |
| EP32 | 1 | TP8×DP4 | EP32 | inter | PP 없음: 어텐션 TP8×DP4(노드=replica), MoE EP32 (a2a 노드 간) |
| TP8EP4 | 1 | TP8×DP4 | EP4×TP8 | inter | PP 없음: 어텐션 TP8×DP4, MoE EP4(노드=EP rank)×TP8 (a2a 노드 간) |
| TP16PP2 | 2 | TP16 | TP16 | — | 대조군: TP16이 노드를 넘어 AR이 IB를 타면 어떻게 되는가 |

## 5.1 결과 — 원본 체크포인트(MXFP4 expert + bf16 잔여 111GB/tok)

### 현실(보수) 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템 | 지배 병목 top-2 |
|---|---|---|---|---|---|
| TP8PP4 | 1 | 21.4 | 46.8 | 47 | HBM 8.0ms + comm-α 5.6ms |
| TP8PP4 | 8 | 39.2 | 25.5 | 204 | compute 18.8ms + HBM 9.4ms |
| TP8PP4 | 32 | 87.3 | 11.5 | 366 | compute 61.3ms + HBM 14.1ms |
| TP4EP8PP4 | 1 | 34.5 | 29.0 | 29 | HBM 14.7ms + comm-α 9.3ms |
| TP4EP8PP4 | 8 | 59.8 | 16.7 | 134 | compute 28.1ms + HBM 15.5ms |
| TP4EP8PP4 | 32 | 127.1 | 7.9 | 252 | compute 91.9ms + HBM 18.6ms |
| TP8EP8PP4 | 1 | 27.8 | 36.0 | 36 | comm-α 9.3ms + HBM 8.0ms |
| TP8EP8PP4 | 8 | 53.8 | 18.6 | 149 | compute 28.1ms + HBM 9.4ms |
| TP8EP8PP4 | 32 | 123.4 | 8.1 | 259 | compute 91.9ms + HBM 14.2ms |
| EP32 | 1 | 29.4 | 34.0 | 34 | comm-α 13.7ms + HBM 8.0ms |
| EP32 | 8 | 36.3 | 27.6 | 220 | comm-α 13.7ms + HBM 8.2ms |
| EP32 | 32 | 55.6 | 18.0 | 576 | compute 23.0ms + comm-α 13.7ms |
| TP8EP4 | 1 | 31.9 | 31.3 | 31 | comm-α 15.5ms + HBM 8.0ms |
| TP8EP4 | 8 | 38.4 | 26.1 | 208 | comm-α 15.5ms + HBM 8.2ms |
| TP8EP4 | 32 | 56.2 | 17.8 | 569 | compute 23.0ms + comm-α 15.5ms |
| TP16PP2 | 1 | 23.6 | 42.4 | 42 | comm-α 12.3ms + launch 5.1ms |
| TP16PP2 | 8 | 34.5 | 29.0 | 232 | comm-α 12.3ms + compute 9.4ms |
| TP16PP2 | 32 | 65.4 | 15.3 | 490 | compute 30.6ms + comm-α 12.3ms |

### 낙관 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템 | 지배 병목 top-2 |
|---|---|---|---|---|---|
| TP8PP4 | 1 | 11.5 | 87.3 | 87 | HBM 9.6ms + comm-α 1.0ms |
| TP8PP4 | 8 | 23.4 | 42.7 | 342 | HBM 21.3ms + comm-α 1.0ms |
| TP8PP4 | 32 | 56.0 | 17.9 | 572 | HBM 53.4ms + comm-α 1.0ms |
| TP4EP8PP4 | 1 | 19.3 | 51.7 | 52 | HBM 16.5ms + comm-α 1.9ms |
| TP4EP8PP4 | 8 | 32.1 | 31.2 | 249 | HBM 29.2ms + comm-α 1.9ms |
| TP4EP8PP4 | 32 | 66.6 | 15.0 | 481 | HBM 63.3ms + comm-α 1.9ms |
| TP8EP8PP4 | 1 | 12.6 | 79.3 | 79 | HBM 9.8ms + comm-α 1.9ms |
| TP8EP8PP4 | 8 | 26.1 | 38.3 | 307 | HBM 23.1ms + comm-α 1.9ms |
| TP8EP8PP4 | 32 | 62.6 | 16.0 | 511 | HBM 59.0ms + comm-α 1.9ms |
| EP32 | 1 | 15.8 | 63.2 | 63 | HBM 8.4ms + comm-α 6.5ms |
| EP32 | 8 | 19.1 | 52.3 | 418 | HBM 11.6ms + comm-α 6.5ms |
| EP32 | 32 | 28.6 | 35.0 | 1119 | HBM 20.6ms + comm-α 6.5ms |
| TP8EP4 | 1 | 16.3 | 61.4 | 61 | HBM 8.4ms + comm-α 6.9ms |
| TP8EP4 | 8 | 19.5 | 51.2 | 410 | HBM 11.6ms + comm-α 6.9ms |
| TP8EP4 | 32 | 28.8 | 34.7 | 1111 | HBM 20.7ms + comm-α 6.9ms |
| TP16PP2 | 1 | 11.9 | 83.8 | 84 | comm-α 5.6ms + HBM 5.4ms |
| TP16PP2 | 8 | 18.5 | 54.0 | 432 | HBM 11.8ms + comm-α 5.6ms |
| TP16PP2 | 32 | 37.0 | 27.0 | 865 | HBM 29.6ms + comm-α 5.6ms |

## 5.2 결과 — Phase 3 전량 W4A16 재양자화 가정(56GB/tok)

### 현실(보수) 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템 | 지배 병목 top-2 |
|---|---|---|---|---|---|
| TP8PP4 | 1 | 15.2 | 65.7 | 66 | comm-α 5.6ms + launch 5.1ms |
| TP8PP4 | 8 | 26.9 | 37.1 | 297 | HBM 15.8ms + comm-α 5.6ms |
| TP8PP4 | 32 | 59.2 | 16.9 | 541 | HBM 46.6ms + comm-α 5.6ms |
| TP4EP8PP4 | 1 | 23.0 | 43.5 | 43 | comm-α 9.3ms + HBM 7.0ms |
| TP4EP8PP4 | 8 | 39.0 | 25.7 | 205 | HBM 22.8ms + comm-α 9.3ms |
| TP4EP8PP4 | 32 | 81.9 | 12.2 | 391 | HBM 65.2ms + comm-α 9.3ms |
| TP8EP8PP4 | 1 | 21.1 | 47.3 | 47 | comm-α 9.3ms + launch 6.6ms |
| TP8EP8PP4 | 8 | 37.9 | 26.4 | 211 | HBM 21.5ms + comm-α 9.3ms |
| TP8EP8PP4 | 32 | 83.2 | 12.0 | 385 | HBM 65.3ms + comm-α 9.3ms |
| EP32 | 1 | 23.8 | 42.0 | 42 | comm-α 13.7ms + launch 6.6ms |
| EP32 | 8 | 28.4 | 35.2 | 282 | comm-α 13.7ms + HBM 7.3ms |
| EP32 | 32 | 41.6 | 24.1 | 770 | HBM 18.3ms + comm-α 13.7ms |
| TP8EP4 | 1 | 26.4 | 37.9 | 38 | comm-α 15.5ms + launch 7.3ms |
| TP8EP4 | 8 | 30.5 | 32.8 | 262 | comm-α 15.5ms + launch 7.3ms |
| TP8EP4 | 32 | 42.3 | 23.6 | 756 | HBM 18.5ms + comm-α 15.5ms |
| TP16PP2 | 1 | 20.3 | 49.2 | 49 | comm-α 12.3ms + launch 5.1ms |
| TP16PP2 | 8 | 28.2 | 35.4 | 283 | comm-α 12.3ms + HBM 8.9ms |
| TP16PP2 | 32 | 51.3 | 19.5 | 624 | HBM 26.0ms + comm-α 12.3ms |

### 낙관 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템 | 지배 병목 top-2 |
|---|---|---|---|---|---|
| TP8PP4 | 1 | 6.2 | 161.0 | 161 | HBM 4.3ms + comm-α 1.0ms |
| TP8PP4 | 8 | 17.9 | 56.0 | 448 | HBM 15.8ms + comm-α 1.0ms |
| TP8PP4 | 32 | 49.8 | 20.1 | 642 | HBM 46.6ms + comm-α 1.0ms |
| TP4EP8PP4 | 1 | 9.2 | 108.2 | 108 | HBM 6.4ms + comm-α 1.9ms |
| TP4EP8PP4 | 8 | 21.6 | 46.2 | 370 | HBM 18.7ms + comm-α 1.9ms |
| TP4EP8PP4 | 32 | 55.2 | 18.1 | 580 | HBM 52.0ms + comm-α 1.9ms |
| TP8EP8PP4 | 1 | 7.4 | 135.9 | 136 | HBM 4.5ms + comm-α 1.9ms |
| TP8EP8PP4 | 8 | 20.5 | 48.8 | 390 | HBM 17.5ms + comm-α 1.9ms |
| TP8EP8PP4 | 32 | 56.3 | 17.8 | 568 | HBM 52.0ms + comm-α 1.9ms |
| EP32 | 1 | 10.6 | 94.2 | 94 | comm-α 6.5ms + HBM 3.2ms |
| EP32 | 8 | 13.8 | 72.3 | 578 | comm-α 6.5ms + HBM 6.3ms |
| EP32 | 32 | 23.1 | 43.3 | 1387 | HBM 15.0ms + comm-α 6.5ms |
| TP8EP4 | 1 | 11.1 | 90.3 | 90 | comm-α 6.9ms + HBM 3.2ms |
| TP8EP4 | 8 | 14.2 | 70.2 | 562 | comm-α 6.9ms + HBM 6.3ms |
| TP8EP4 | 32 | 23.3 | 42.9 | 1374 | HBM 15.1ms + comm-α 6.9ms |
| TP16PP2 | 1 | 9.1 | 109.4 | 109 | comm-α 5.6ms + HBM 2.6ms |
| TP16PP2 | 8 | 15.6 | 64.1 | 513 | HBM 8.9ms + comm-α 5.6ms |
| TP16PP2 | 32 | 33.7 | 29.6 | 948 | HBM 26.0ms + comm-α 5.6ms |

## 6. 승자 분해 (배치 1 / 8 / 32, 현실 시나리오, 원본)

전략 **TP8PP4**:

| 배치 | HBM | compute | comm-α | comm-bw | launch | cpu | 합계 ms |
|---|---|---|---|---|---|---|---|
| 1 | 8.0 | 2.5 | 5.6 | 0.0 | 5.1 | 0.1 | 21.4 |
| 8 | 9.4 | 18.8 | 5.6 | 0.3 | 5.1 | 0.1 | 39.2 |
| 32 | 14.1 | 61.3 | 5.6 | 1.0 | 5.1 | 0.1 | 87.3 |

## 7. GPU당 메모리 (가중치 + KV/state)

KDA 69층 state + MLA 24층 latent KV (recon/k3-config.md §8: latent 27,648B/tok, state 6.29MB/층/seq fp32):

| 전략 | 가중치 GB(최악 stage) | seq당 @8k MB | seq당 @256k GB | state/seq MB | 최대 seq @8k | 최대 seq @256k | 적재 |
|---|---|---|---|---|---|---|---|
| TP8PP4 | 49.6 | 79 | 2.13 | 13.2 | 321 | 11 | OK |
| TP4EP8PP4 | 53.4 | 92 | 2.14 | 26.3 | 466 | 20 | OK |
| TP8EP8PP4 | 49.6 | 79 | 2.13 | 13.2 | 321 | 11 | OK |
| EP32 | 61.3 | 283 | 7.30 | 56.8 | 192 | 4 | OK |
| TP8EP4 | 61.3 | 283 | 7.30 | 56.8 | 192 | 4 | OK |
| TP16PP2 | 49.8 | 136 | 3.94 | 13.6 | 184 | 6 | OK |

## 8. 컨텍스트 민감도 (승자, 배치 8, 원본)

| 전략 | ctx | 시나리오 | 스텝 ms | tok/s/유저 |
|---|---|---|---|---|
| TP8PP4 | 2048 | opt | 22.7 | 44.0 |
| TP8PP4 | 2048 | real | 38.6 | 25.9 |
| TP8PP4 | 8192 | opt | 23.4 | 42.7 |
| TP8PP4 | 8192 | real | 39.2 | 25.5 |
| TP8PP4 | 32768 | opt | 26.1 | 38.4 |
| TP8PP4 | 32768 | real | 41.9 | 23.9 |
| TP8PP4 | 131072 | opt | 36.7 | 27.2 |
| TP8PP4 | 131072 | real | 52.6 | 19.0 |

## 9. 검증 앵커

- 모델 bytes/token(배치1, 원본) = 137.0 GB (bf16 111.1 + MXFP4 25.8) vs recon 136.96 GB → 폐쇄 True
- 완전병렬 하한 2.10 ms/tok vs recon 2.10 ms

## 10. 주요 가정/리스크

- 통신은 무혼잡 α-β(hatir collectives 계약) — EP32의 노드간 a2a는 실측에서 더 나쁠 수 있음(보수 α 44µs 반영으로 1차 방어).
- MoE 라우팅 균등 기대값 + 불균형 계수(real 1.5)로 근사 — 실측 라우팅 분포로 Phase 1에서 재보정.
- AttnRes 스트림의 PP 경계 폭([미확인], recon/k3-config.md §10) → real은 hidden 2배 전송 가정.
- latent 경계 collective(구현 [미확인]) → real만 +1 AR/MoE층.
- KDA state fp32(vLLM 하드코딩), FLA A100 수치검증은 Phase 1 프로브 (recon/kda-fla.md).
- Marlin/moe_wna16 커널이 #35922 수정 후 정상 동작한다는 전제(속도 게이트일 뿐 정합성 게이트 아님).
