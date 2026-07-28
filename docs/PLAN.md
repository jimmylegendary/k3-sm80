# k3-sm80 전체 계획 — 단계·목적·완료 판정 (v2, 2026-07-28)

원칙(Jimmy 지시): **커널은 먼저 단독으로 검증하고, 통과한 것만 vLLM에 넣는다.**
S2를 통과하지 않은 커널은 S4에 진입하지 않는다.

전제: 개발 호스트에 NVIDIA GPU 없음 → 저작·참조구현·하네스·CPU dry-run은 호스트,
컴파일·GPU 실행은 사내 A100(러너 = Jimmy, repo pull → 실행 → 로그 반출 → 호스트에서 수정).

## 커널 작업의 실제 형태 (recon 확정, "H100 코드 포팅" 아님)

| 커널 | 상태 | 작업 |
|---|---|---|
| KDA chunk (prefill) | FLA 순수 Triton, 아치 게이팅 0건 | A100 수치검증 + smem 164KB 타일 튜닝 |
| KDA recurrent (decode) + causal_conv1d | 동일 | 동일 |
| MLA decode | TRITON_MLA (SM80 지원) | 수치검증 |
| Marlin W4A16 MoE | SM80 커널 존재, crash(#35922, c_tmp OOB 후보 #36889) | **버그 수정** |
| Marlin FP4 MoE | sm80 인스턴스화 존재 | crash 수정 (조건부 lane, probe E 결과에 따름) |
| SiTU activation | — | 언퓨즈드 참조로 대체(성능 후순위) |

FlashKDA(CUTLASS sm90a) 포팅은 **하지 않는다** — Triton 폴백이 공식 경로.

## 단계

### S0 — hatir 투자 게이트 ✅ 완료
- 목적: 커널 작업 투자 가치 판정.
- 완료 판정: 3렌즈 적대검증 통과한 `phase0/verdict.json` = **PASS**
  (TP8PP4 real 46.8 / opt 87.3 tok/s, b32 파이프 799; W4A16 시 1.40×).

### S1 — 사내 프로브 (킷 ✅, 실행 대기) `a100-kit/`
- 목적: 어떤 커널 작업이 실제 필요한지 확정 + 약한 가정 실측
  (max_num_seqs>1, dequant_ops/param, inter-node a2a α, TTFT).
- 작업: probe_35922(A~E) → probe_kda_fla → 소스빌드(kimi-k3 `0498dc7e`) →
  probe_k3_modules + 킷 밖 4건 마이크로벤치.
- 완료 판정: DECISION-TREE.md 의 모든 프로브 결과가 기록되고, 마이크로벤치 값으로
  `phase0/k3_model.py` 재보정 커밋이 남을 것.

### S2 — 커널 단독 검증 `kernel-kit/` (저작 중)
- 목적: vLLM 없이 커널별 독립 PASS. 실패 격리 지점을 커널 내부로 좁힘.
- 작업: 커널별 하네스 = fp32 참조구현(순수 PyTorch) + K3 실제 shape + 기계판정 리포트.
- 완료 판정 (커널당 4게이트, `run_all.sh` → `kernel-report/` JSON):
  1. **NUMERIC**: fp32 참조 대비 오차 ≤ tol (bf16 연산 기준 tol 명시, seed 고정)
  2. **SANITIZER**: compute-sanitizer memcheck clean
  3. **SHAPES**: K3 실측 shape 커버 (KDA H=96 dk=dv=128; MoE 896/top16, inter 3072,
     shared fused 6144; MLA kv_lora 512; 배치 1/8/32; ctx 8k/128k 대표점)
  4. **PERF**: Phase 0 루프라인 목표 대비 임계(커널별 명시) 이상
- 대상: k01 KDA chunk / k02 KDA recurrent / k03 causal_conv1d / k04 MLA decode /
  k05 Marlin W4A16 MoE (crash 예상 → S3 입력) / k06 dequant 마이크로벤치(모델 재보정용) /
  k07 SiTU 언퓨즈드 참조.
- 호스트 게이트(사내 반입 전): py_compile + CPU dry-run(참조 경로만) green.

### S3 — 커널 수정 (조건부, S1·S2 결과로 개시)
- 목적: S2에서 FAIL/CRASH 난 커널의 수정 (#35922 최우선).
- 완료 판정: 최소 재현 케이스 PASS 전환 + 해당 커널 S2 4게이트 재통과 +
  기존 green 커널 회귀 전부 유지.

### S4 — vLLM 통합 (S2 통과 커널만)
- 목적: 검증된 커널을 소스빌드 vLLM에 배선.
- 완료 판정: ① 레이어 parity — vLLM 레이어 forward vs kernel-kit 참조 logits tol 내
  ② dummy-load K3(또는 소형 변형) E2E 생성 정상 ③ 커널 단독 perf 대비 절벽 없음(≥80%).

### S5 — 양자화 lane (⏸ Jimmy 결정 대기; 2026-07-28 정확도 논의로 재정의)
정확도 사실관계: MXFP4→bf16 dequant는 **exact**(e2m1 값·E8M0 스케일 모두 bf16 정확 표현)
→ 원본 MXFP4 직행이 무손실. MXFP4→INT4 재양자화는 실손실(비균등→균등 격자, QAT 적응 가중치).
**expert 재양자화는 속도 이득 0** (MXFP4=INT4=4bit, bytes 동일) — 1.40×의 원천은 전적으로
bf16 잔여 53.3%(KDA 30.6B + shared 12.2B + MLA·latent proj·lm_head)의 양자화.
- **3a (기본)**: mixed — expert 원본 MXFP4 유지(무손실, Marlin FP4 MoE lane 필요=probe E/S3),
  bf16 잔여만 W4A16. ~56GB/tok, 1.40×. 단 잔여 양자화도 신규 PTQ 손실(어텐션 민감) →
  품질 게이트 필수.
- **3b (보험)**: 전량 W4A16 — FP4 MoE 수정 불가 판명 시에만. expert 손실 감수.
- **3-none**: 재양자화 없이 46.8 tok/s로 충분하면 skip.
- W4A16 lane을 유지하는 이유: ① INT4/FP4 Marlin MoE는 같은 커널 계열 — c_tmp OOB 수정이
  양쪽에 적용될 가능성 ② INT4는 재현 체크포인트가 있어 crash 디버깅 진입로 ③ 3b 보험.
- 완료 판정: 선택 lane 체크포인트가 S4 스택에서 로드·생성 + 원본 대비 품질 대조 통과
  (방법은 lane 확정 시 정의).

### S6 — 32장 스케일업
- 목적: TP8PP4 배치 + 튜닝으로 "쓸 만한 속도" 실증.
- 완료 판정: 실측 tok/s ≥ Phase 0 real 예측의 50%에서 시작, 튜닝으로 격차 축소;
  최종 수용 기준은 Jimmy 재가('쓸 만한 속도' 조작화, SUMMARY §4-②).

## 반복 루프
사내 실행 로그(`kernel-report/` tar)를 repo로 반입 → 호스트에서 분석·수정 → push →
사내 pull 재실행. 모든 단계 전환은 커밋에 판정 근거 파일 포함.
