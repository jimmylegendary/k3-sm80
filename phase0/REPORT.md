# Phase 0 — hatir 타당성 게이트: Kimi K3 디코드 @ 32×A100(SM80)

생성: `k3_model.py` 실행 산출물 (수치 전부 실행 출력; 손수치 없음). 날짜: 2026-07-28

## 0. 판정 (게이트)

**PASS** — 기준: 단일 스트림(배치 1) decode ≥ 5 tok/s (대화형 최소선의 decode 축; TTFT/prefill은 미모델 — §10).

- 현실(보수) 시나리오 최선: **TP8PP4** = **46.8 tok/s** (스텝 21.4ms, ctx 8k)
- 낙관 시나리오 최선("모든 커널·통신이 완벽해도"): **TP8PP4** = **87.3 tok/s** (스텝 11.5ms)
- 완전병렬 이론 하한(전략 무관): 2.10 ms/tok = 476 tok/s 상한 — recon 앵커 2.1ms와 정합(바이트 폐쇄 True)

### 핵심 발견 (수치는 아래 표에서)

1. **처리량 승자는 PP 파이프라인 정상상태 기준 TP8PP4**: 배치 32(상주), 현실 시나리오에서 PP 마이크로배치 파이프라이닝(P1) 반영 시 **TP8PP4 = 799 tok/s** (µb8×4 in-flight, 최악 stage 10.0ms). 동기 단일배치(비-파이프라인) 기준으로는 TP8EP4(521 tok/s)가 앞서 보이지만 이는 vLLM PP의 다중 in-flight 배치를 0으로 치는 지표 정의 인공물이다. 단일 스트림(대화 지연)은 α-이벤트 최소인 **TP8PP4**가 승자. ⚠ 배치>1 결과 전체가 `max_num_seqs`>1 성립을 전제([미확인] — §10).
2. **원본 MXFP4의 dequant ALU 세금 (dq_ops=4 가정 시)**: SM80은 FP4 HW가 없어(#35922 커널 가족의 에뮬레이션 경로) 배치 32에서 TP-샤딩 전략의 compute 비중이 70%까지 상승 (TP8PP4/real). 단 이 결론은 dequant_ops_per_param(real 4.0/opt 2.0)의 함수이며 이 값은 [무근거 밴드](실측·1차 출처 없음 — 트윈 _dequant_ops_note): dq=2면 같은 항이 메모리바운드로 반전된다. Phase 1 dequant 마이크로벤치로 확정 전까지 조건부. EP 분산(expert/rank↓) 또는 W4A16 재양자화가 완화 방향인 것은 dq 값과 무관하게 성립.
3. **Phase 3(W4A16) 가치 정량화**: bf16 잔여 111GB/tok이 지배 병목 — 전량 W4A16이면 현실 시나리오 단일 스트림 46.8 → 65.7 tok/s (1.40×). '조건부'가 아니라 사실상 필수 격상 근거(recon/k3-config.md §6과 합치).
4. **EP의 실제 세금은 노드 간 a2a α 구조**: naive 경로(real)에서 노드 간 a2a α는 per-peer 항이 지배(E8: EP32 = 44+6×24 = 188µs/collective, ×2회/MoE층) — EP32/TP8EP4의 처리량이 종전 산출 대비 하락하고, DeepEP류 dedup+MXFP8 payload는 SM90 전용 라이브러리라 SM80에서는 커스텀 커널 신규작성 없이는 못 쓴다(opt 시나리오 전용 가정). PP의 '가중치 스트리밍 8-way 직렬화' 세금은 동기 지표의 인공물이었고, 파이프라인 정상상태에서는 PP가 처리량도 이긴다(발견 1).

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

- 커널: hatir `einsum.tile`로 직접 저작한 플랜 + 명시 루프라인 max(플랜 compute_us, 명시 mem_bytes/HBM피크) — 디코드 GEMV는 가중치 1회 스트림·n-블록 108 SM 팬·split-k 매핑, MLA attention/KDA recurrent/elementwise도 tile 플랜 (HBM→L2→SMEM→PE 서브스택). hatir 기본 `_gemm_plan`/`roofline_us`는 이 트윈(SM당 smem 164KB reuse tier)에서 spatial fan이 클램프되어 디코드 compute를 10–20× 과대계상 → 미사용, 직접 저작으로 대체(코드 주석).
- 통신: hatir `hierarchical_collective`(α-β, 무혼잡 이상치) — α/실효 busbw는 트윈 `meta.comm_scenarios`의 보수(real)/낙관(opt) 2안 (recon/a100-comms.md 근거). 노드 간 a2a α는 per-peer 구조(E8): α_eff = α_base + 6µs×(노드 밖 peer 수) — EP32 real은 188µs/collective (naive NCCL a2a 밴드와 정합; 트윈 _a2a_alpha_note).
- EP 전송 복사본(E6): real = naive per-(token,expert) 복사(topk=16) — vLLM SM80에 전용 EP a2a 백엔드 근거 없음(recon/vllm-upstream.md); opt = DeepEP식 랭크-중복제거 + MXFP8 payload — 단 DeepEP는 SM90 전용이라 SM80 커스텀 커널 신규작성 전제의 상한.
- PP 처리량(P1): `tok/s/시스템`은 동기 단일배치(직렬 합 = 지연 근거) 기준이고, `tok/s/시스템(PP파이프)`는 상주 배치를 p개 마이크로배치로 나눠 in-flight 유지하는 vLLM PP 정상상태(최악 stage 시간 결정; 가중치 재스트림·uniq expert는 마이크로배치 기준으로 반영, 램프업/드레인 버블 미모델) — PP 전략의 처리량 판정은 후자 사용.
- hatir가 표현 못 하는 항목은 명시 가산항(코드 EXPLICIT E1–E8, P1): 런치/그래프 갭, dequant ALU([무근거 밴드] — 발견 2), MoE 불균형, 혼합 dtype, 스텝 CPU, dispatch 복사본 수, expert grouped-GEMV, per-peer a2a α, PP 파이프라인 정상상태.
- 한계: 무혼잡·무겹침(통신-계산 오버랩 없음 = 보수), 균등 라우팅 기대값, 절대치는 탐색등급(순위/규모 판단용) — HATIR 검증 원칙(트렌드 정합)에 따름. HBM은 datasheet 피크(2039GB/s) 무보정 — §10 minor 부채 참조.

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

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템(동기) | tok/s/시스템(PP파이프) | 지배 병목 top-2 |
|---|---|---|---|---|---|---|
| TP8PP4 | 1 | 21.4 | 46.8 | 47 | 47 | HBM 8.0ms + comm-α 5.6ms |
| TP8PP4 | 8 | 39.2 | 25.5 | 204 | 322 (µb2·stage 6.2ms) | compute 18.8ms + HBM 9.4ms |
| TP8PP4 | 32 | 87.3 | 11.5 | 366 | 799 (µb8·stage 10.0ms) | compute 61.3ms + HBM 14.1ms |
| TP4EP8PP4 | 1 | 34.5 | 29.0 | 29 | 29 | HBM 14.7ms + comm-α 9.3ms |
| TP4EP8PP4 | 8 | 59.8 | 16.7 | 134 | 204 (µb2·stage 9.8ms) | compute 28.1ms + HBM 15.5ms |
| TP4EP8PP4 | 32 | 127.3 | 7.9 | 251 | 525 (µb8·stage 15.2ms) | compute 91.9ms + HBM 18.6ms |
| TP8EP8PP4 | 1 | 27.8 | 36.0 | 36 | 36 | comm-α 9.3ms + HBM 8.0ms |
| TP8EP8PP4 | 8 | 53.9 | 18.6 | 148 | 246 (µb2·stage 8.1ms) | compute 28.1ms + HBM 9.4ms |
| TP8EP8PP4 | 32 | 123.6 | 8.1 | 259 | 585 (µb8·stage 13.7ms) | compute 91.9ms + HBM 14.2ms |
| EP32 | 1 | 55.9 | 17.9 | 18 | 18 | comm-α 40.2ms + HBM 8.0ms |
| EP32 | 8 | 63.0 | 15.9 | 127 | 127 | comm-α 40.2ms + HBM 8.2ms |
| EP32 | 32 | 82.7 | 12.1 | 387 | 387 | comm-α 40.2ms + compute 23.0ms |
| TP8EP4 | 1 | 35.3 | 28.3 | 28 | 28 | comm-α 18.8ms + HBM 8.0ms |
| TP8EP4 | 8 | 42.2 | 23.7 | 190 | 190 | comm-α 18.8ms + HBM 8.2ms |
| TP8EP4 | 32 | 61.4 | 16.3 | 521 | 521 | compute 23.0ms + comm-α 18.8ms |
| TP16PP2 | 1 | 23.6 | 42.4 | 42 | 42 | comm-α 12.3ms + launch 5.1ms |
| TP16PP2 | 8 | 34.5 | 29.0 | 232 | 280 (µb4·stage 14.3ms) | comm-α 12.3ms + compute 9.4ms |
| TP16PP2 | 32 | 65.4 | 15.3 | 490 | 697 (µb16·stage 23.0ms) | compute 30.6ms + comm-α 12.3ms |

### 낙관 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템(동기) | tok/s/시스템(PP파이프) | 지배 병목 top-2 |
|---|---|---|---|---|---|---|
| TP8PP4 | 1 | 11.5 | 87.3 | 87 | 87 | HBM 9.6ms + comm-α 1.0ms |
| TP8PP4 | 8 | 23.4 | 42.7 | 342 | 585 (µb2·stage 3.4ms) | HBM 21.3ms + comm-α 1.0ms |
| TP8PP4 | 32 | 56.0 | 17.9 | 572 | 1341 (µb8·stage 6.0ms) | HBM 53.4ms + comm-α 1.0ms |
| TP4EP8PP4 | 1 | 19.3 | 51.7 | 52 | 52 | HBM 16.5ms + comm-α 1.9ms |
| TP4EP8PP4 | 8 | 32.1 | 31.2 | 249 | 366 (µb2·stage 5.5ms) | HBM 29.2ms + comm-α 1.9ms |
| TP4EP8PP4 | 32 | 66.6 | 15.0 | 481 | 975 (µb8·stage 8.2ms) | HBM 63.3ms + comm-α 1.9ms |
| TP8EP8PP4 | 1 | 12.6 | 79.3 | 79 | 79 | HBM 9.8ms + comm-α 1.9ms |
| TP8EP8PP4 | 8 | 26.1 | 38.3 | 307 | 531 (µb2·stage 3.8ms) | HBM 23.1ms + comm-α 1.9ms |
| TP8EP8PP4 | 32 | 62.6 | 16.0 | 511 | 1206 (µb8·stage 6.6ms) | HBM 59.0ms + comm-α 1.9ms |
| EP32 | 1 | 15.8 | 63.2 | 63 | 63 | HBM 8.4ms + comm-α 6.5ms |
| EP32 | 8 | 19.1 | 52.3 | 418 | 418 | HBM 11.6ms + comm-α 6.5ms |
| EP32 | 32 | 28.6 | 35.0 | 1119 | 1119 | HBM 20.6ms + comm-α 6.5ms |
| TP8EP4 | 1 | 16.3 | 61.4 | 61 | 61 | HBM 8.4ms + comm-α 6.9ms |
| TP8EP4 | 8 | 19.5 | 51.2 | 410 | 410 | HBM 11.6ms + comm-α 6.9ms |
| TP8EP4 | 32 | 28.8 | 34.7 | 1111 | 1111 | HBM 20.7ms + comm-α 6.9ms |
| TP16PP2 | 1 | 11.9 | 83.8 | 84 | 84 | comm-α 5.6ms + HBM 5.4ms |
| TP16PP2 | 8 | 18.5 | 54.0 | 432 | 537 (µb4·stage 7.4ms) | HBM 11.8ms + comm-α 5.6ms |
| TP16PP2 | 32 | 37.0 | 27.0 | 865 | 1258 (µb16·stage 12.7ms) | HBM 29.6ms + comm-α 5.6ms |

## 5.2 결과 — Phase 3 전량 W4A16 재양자화 가정(56GB/tok)

### 현실(보수) 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템(동기) | tok/s/시스템(PP파이프) | 지배 병목 top-2 |
|---|---|---|---|---|---|---|
| TP8PP4 | 1 | 15.2 | 65.7 | 66 | 66 | comm-α 5.6ms + launch 5.1ms |
| TP8PP4 | 8 | 26.9 | 37.1 | 297 | 452 (µb2·stage 4.4ms) | HBM 15.8ms + comm-α 5.6ms |
| TP8PP4 | 32 | 59.2 | 16.9 | 541 | 1154 (µb8·stage 6.9ms) | HBM 46.6ms + comm-α 5.6ms |
| TP4EP8PP4 | 1 | 23.0 | 43.4 | 43 | 43 | comm-α 9.3ms + HBM 7.0ms |
| TP4EP8PP4 | 8 | 39.0 | 25.6 | 205 | 303 (µb2·stage 6.6ms) | HBM 22.8ms + comm-α 9.3ms |
| TP4EP8PP4 | 32 | 82.1 | 12.2 | 390 | 798 (µb8·stage 10.0ms) | HBM 65.2ms + comm-α 9.3ms |
| TP8EP8PP4 | 1 | 21.1 | 47.3 | 47 | 47 | comm-α 9.3ms + launch 6.6ms |
| TP8EP8PP4 | 8 | 37.9 | 26.4 | 211 | 328 (µb2·stage 6.1ms) | HBM 21.5ms + comm-α 9.3ms |
| TP8EP8PP4 | 32 | 83.4 | 12.0 | 384 | 826 (µb8·stage 9.7ms) | HBM 65.3ms + comm-α 9.3ms |
| EP32 | 1 | 50.3 | 19.9 | 20 | 20 | comm-α 40.2ms + launch 6.6ms |
| EP32 | 8 | 55.1 | 18.2 | 145 | 145 | comm-α 40.2ms + HBM 7.3ms |
| EP32 | 32 | 68.7 | 14.5 | 466 | 466 | comm-α 40.2ms + HBM 18.3ms |
| TP8EP4 | 1 | 29.7 | 33.6 | 34 | 34 | comm-α 18.8ms + launch 7.3ms |
| TP8EP4 | 8 | 34.3 | 29.1 | 233 | 233 | comm-α 18.8ms + launch 7.3ms |
| TP8EP4 | 32 | 47.5 | 21.0 | 673 | 673 | comm-α 18.8ms + HBM 18.5ms |
| TP16PP2 | 1 | 20.3 | 49.2 | 49 | 49 | comm-α 12.3ms + launch 5.1ms |
| TP16PP2 | 8 | 28.2 | 35.4 | 283 | 335 (µb4·stage 12.0ms) | comm-α 12.3ms + HBM 8.9ms |
| TP16PP2 | 32 | 51.3 | 19.5 | 624 | 873 (µb16·stage 18.3ms) | HBM 26.0ms + comm-α 12.3ms |

### 낙관 시나리오 — ctx 8k

| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템(동기) | tok/s/시스템(PP파이프) | 지배 병목 top-2 |
|---|---|---|---|---|---|---|
| TP8PP4 | 1 | 6.2 | 161.0 | 161 | 161 | HBM 4.3ms + comm-α 1.0ms |
| TP8PP4 | 8 | 17.9 | 56.0 | 448 | 944 (µb2·stage 2.1ms) | HBM 15.8ms + comm-α 1.0ms |
| TP8PP4 | 32 | 49.8 | 20.1 | 642 | 1735 (µb8·stage 4.6ms) | HBM 46.6ms + comm-α 1.0ms |
| TP4EP8PP4 | 1 | 9.2 | 108.2 | 108 | 108 | HBM 6.4ms + comm-α 1.9ms |
| TP4EP8PP4 | 8 | 21.6 | 46.2 | 370 | 675 (µb2·stage 3.0ms) | HBM 18.7ms + comm-α 1.9ms |
| TP4EP8PP4 | 32 | 55.2 | 18.1 | 580 | 1421 (µb8·stage 5.6ms) | HBM 52.0ms + comm-α 1.9ms |
| TP8EP8PP4 | 1 | 7.4 | 135.9 | 136 | 136 | HBM 4.5ms + comm-α 1.9ms |
| TP8EP8PP4 | 8 | 20.5 | 48.8 | 390 | 813 (µb2·stage 2.5ms) | HBM 17.5ms + comm-α 1.9ms |
| TP8EP8PP4 | 32 | 56.3 | 17.8 | 568 | 1518 (µb8·stage 5.3ms) | HBM 52.0ms + comm-α 1.9ms |
| EP32 | 1 | 10.6 | 94.2 | 94 | 94 | comm-α 6.5ms + HBM 3.2ms |
| EP32 | 8 | 13.8 | 72.3 | 578 | 578 | comm-α 6.5ms + HBM 6.3ms |
| EP32 | 32 | 23.1 | 43.3 | 1387 | 1387 | HBM 15.0ms + comm-α 6.5ms |
| TP8EP4 | 1 | 11.1 | 90.3 | 90 | 90 | comm-α 6.9ms + HBM 3.2ms |
| TP8EP4 | 8 | 14.2 | 70.2 | 562 | 562 | comm-α 6.9ms + HBM 6.3ms |
| TP8EP4 | 32 | 23.3 | 42.9 | 1374 | 1374 | HBM 15.1ms + comm-α 6.9ms |
| TP16PP2 | 1 | 9.1 | 109.4 | 109 | 109 | comm-α 5.6ms + HBM 2.6ms |
| TP16PP2 | 8 | 15.6 | 64.1 | 513 | 665 (µb4·stage 6.0ms) | HBM 8.9ms + comm-α 5.6ms |
| TP16PP2 | 32 | 33.7 | 29.6 | 948 | 1429 (µb16·stage 11.2ms) | HBM 26.0ms + comm-α 5.6ms |

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

- **`max_num_seqs`>1 성립 [미확인]**: 배치 8/32의 모든 결과(파이프라인 포함)는 hybrid KDA-state 다중 시퀀스 배칭 + DP 어텐션 + EP dispatch가 kimi-k3 브랜치/SM80에서 실동작한다는 전제 위에 있다. recon/k3-config.md §8은 KDA state가 `max_num_seqs`를 먼저 제약할 수 있다고 경고(스펙 '물리 타당성 우려'의 명시 항목). 배치 1 게이트 판정은 이 전제와 무관. → Phase 1 프로브에 `max_num_seqs>1` 동작 확인(스케줄러/state 경로) 추가.
- **PP 파이프라인 모델(P1)은 정상상태 상한**: 램프업/드레인 버블, vLLM PP 스케줄러 실동작(마이크로배치 수·스케줄링 정책) 미모델. 스테이지 간 패브릭은 본 스윕 전략에서 노드 단위로 분리라 무경합 가정 유지. KV 적재는 §7 기준 상주 32 seq 가능(TP8PP4 max 321@8k).
- **dequant_ops_per_param(real 4.0/opt 2.0)은 [무근거 밴드]** — 명령어 수준 유도만 있고 실측·1차 출처 없음(트윈 _dequant_ops_note; Marlin arXiv:2408.11743은 INT4 근거일 뿐). 발견 2(ALU-bound)는 dq=4 조건부. → Phase 1에 MXFP4 dequant 마이크로벤치 추가.
- 통신은 무혼잡 α-β(hatir collectives 계약). 노드 간 a2a α는 per-peer 구조(E8, [파생] — 실측 아님): per-peer 6µs는 NCCL Simple per-hop(arXiv:2507.04786 Table I) 기반 1차 근사 → Phase 1 a2a 마이크로벤치로 재고정. 노드 내 a2a α(20µs flat)는 per-peer 미적용(잔여 낙관).
- MoE 라우팅 균등 기대값 + 불균형 계수(real 1.5)로 근사 — expert 커널에만 적용, a2a hot-rank 수신 편중에는 미적용(같은 방향의 잔여 낙관). 실측 라우팅 분포로 Phase 1에서 재보정.
- AttnRes 스트림의 PP 경계 폭([미확인], recon/k3-config.md §10) → real은 hidden 2배 전송 가정.
- latent 경계 collective(구현 [미확인]) → real만 +1 AR/MoE층.
- KDA state fp32(vLLM 하드코딩), FLA A100 수치검증은 Phase 1 프로브 (recon/kda-fla.md).
- Marlin/moe_wna16 커널이 #35922 수정 후 정상 동작한다는 전제(속도 게이트일 뿐 정합성 게이트 아님).
- **게이트는 decode 한정**: TTFT/prefill(ctx 8k ≈ 1.7e15 FLOPs + KDA chunk 경로)은 미모델 — '대화형' 체감은 TTFT에도 걸리므로 Phase 1에서 prefill 봉투 계산/실측 필요.
- **알려진 minor 모델 부채(검증 렌즈 확인, 판정 불변 — phase0/VERIFICATION.md)**: ① HBM을 피크 2039GB/s 100%로 계상(실측 스트림 효율 ~85–90% 미반영, real 최선 46.8→~44 tok/s 방향), ② lm_head logits AG scope가 `intra` 하드코딩(TP16PP2는 inter여야 — 스텝당 ~0.04ms 과소), ③ logits AG에 per-rank shard 바이트 전달(hatir 관례는 full buffer — bw 항 ~40µs 과소), ④ PP p2p bw 25GB/s 고정(real 2×HDR 공유 시 ~6.25GB/s — 스텝당 ~0.17ms 과소), ⑤ KDA conv state를 kernel 4탭 전체로 계상(vLLM은 K_conv−1=3 저장, recon/kda-fla.md — state/seq ~1.33× 과대), ⑥ 미사용 상수 MLA_Q_LORA=576(실제 q_lora_rank=1536 하드코딩).
