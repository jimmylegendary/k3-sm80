# Kimi K3 → 32×A100(SM80) 서빙 — 설계 스펙

날짜: 2026-07-28 · 승인: Jimmy (접근 A 승인, 구현 ultracode+effort max)

## 목표와 성공 기준

- **목표**: Kimi K3(2.78T MoE, MXFP4/MXFP8, KDA 69 + Gated MLA 24 하이브리드)를 사내
  4노드 × 8×A100-80GB(SM80) 클러스터에서 vLLM으로 서빙.
- **성공 기준**: "쓸 만한 속도" — 사내에서 대화형으로 실사용 가능한 수준.
  correctness-only가 아니라 커널 튜닝·병렬화 전략까지 범위에 포함.
- **환경 사실**(확인됨):
  - 사내 A100: **거의 상시 / 자유 반복** 가능. → 호스트 시뮬 검증은 대체재가 아님.
  - 반입: **docker registry만 차단, 나머지(HF·PyPI·GitHub) 자유.** → day-0 Docker
    이미지 불가는 어차피 SM80 소스 빌드가 필요하므로 치명적이지 않음. pre-release
    FlashInfer 의존성은 수동 해소 필요.
  - 개발 호스트: **NVIDIA GPU 0장**(AMD Phoenix APU). CUDA 컴파일·실행 불가.

## 승인된 접근: A) 실측 우선 + hatir 게이트 (C를 Phase 2에 흡수)

핵심 판단(승인됨):
- **HAT 스택은 검증 대체재가 아니라 투자 판단 게이트.** hatir는 tensor-free 해석
  모델이라 커널 수치정확도·인덱싱 버그(#35922)는 구조적으로 못 본다. 그건
  A100 + compute-sanitizer의 몫.
- **hat-vllm-mock 호스트 정찰은 강등.** A100 자유 접근이 이 카드를 죽였다.
- **B(HAT 정면 투입) 기각.**

### 물리 타당성 우려 (Phase 0가 답할 것)
- 토큰당 활성 가중치 ≈ 104B×0.5B ≈ 52GB → 32장 분산 시 GPU당 1.63GB → HBM
  2039GB/s 기준 이론 하한 ~0.8ms/token.
- 실제 병목은 대역폭이 아니라 **93레이어 직렬화 + 레이어당 통신 동기화**
  (100µs/layer만 잡아도 +9ms). 현실 낙관치 단일 스트림 10~30 tok/s 추정 —
  이 숫자를 커널 작업 **전에** hatir로 확정한다.
- `max_num_seqs=1`은 "쓸 만한 속도"와 긴장 관계: 배치를 못 키우면 MoE가 GEMV로
  떨어져 Marlin dequant가 ALU 바운드가 된다. 배치 축도 Phase 0 탐색 변수.

## 단계

### Phase 0 — hatir 타당성 게이트 (호스트, 지금)
- `hat-schema/examples/a100-80gb.hw` 기반 **32×A100 시스템 트윈** 작성
  (노드 내 NVLink, 노드 간 IB, `dgx-h100-x4.hw` 패턴 참조).
- K3 디코드 스텝 해석 모델: 레이어별 커널(KDA chunk/recurrent, MLA decode,
  MoE W4A16 dequant-GEMV/GEMM, dense) + 통신(TP all-reduce, PP p2p, EP all-to-all).
- 병렬화 전략 비교: TP8×PP4 / EP 변형 / 혼합, 배치 1·8·32.
- **산출**: 전략별 tok/s 상한(낙관/현실 오버헤드 2안) + 게이트 판정
  ("모든 커널이 완벽해도 X tok/s"). 판정이 실사용 불가 수준이면 여기서 중단하고
  Jimmy에게 보고.
- 부가: hat-kv-manager 관점 KV/state 메모리 산정(69 KDA state + 24 MLA latent @256k).

### Phase 1 — A100 정찰 프로브 (사내, 러너 = Jimmy)
- **업스트림 재확인 선행**(조사문서는 2026-07-28 스냅샷): #35922 상태, KDA prefill
  Triton 백엔드 노출 여부, mxfp4 게이트, K3 모델 코드의 hard SM90 assert 유무.
- 프로브 킷(`a100-kit/`): 오프라인 소스 빌드 절차(vLLM SM80, FlashInfer, FLA) +
  프로브 스크립트:
  - `probe_35922`: Qwen3-30B-A3B W4A16으로 INT4 Marlin MoE crash 재현 +
    compute-sanitizer 원인 특정.
  - `probe_k3_modules`: Kimi-K3-0.18B로 K3 모듈(KDA/AttnRes/LatentMoE) SM80 초기화
    경로 분리 검증.
  - 결과별 의사결정 트리 포함.

### Phase 2 — 우회 우선, 그다음 커널
1. **우회(구 접근 C)**: MoE=Triton `moe_wna16`(W4A16), KDA=FLA 순수 Triton
   `chunk_kda`(prefill 포함), MLA=`TRITON_MLA`. 커널 신규작성 0으로 토큰 도달 시도.
2. 우회 실패/성능 미달 시 커널 수정: #35922 원인 수정(인덱싱 버그, sanitizer 기반),
   필요 시 FP4 MoE SM80 인스턴스화. hatir는 **타일 사전 선정**(A100 smem 164KB vs
   H100 228KB)으로 탐색공간 축소에만 사용.

### Phase 3 — 재양자화 (조건부)
- llm-compressor W4A16 INT4 (K3 아키텍처 미지원 시 model_free_ptq/커스텀 정의).
- KDA 문제와 독립임을 전제.

### Phase 4 — 32장 스케일업 + 튜닝
- Phase 0 승자 전략으로 배치, 배치 크기·CUDA graph·통신 튜닝. 목표 수치는
  Phase 0 판정치 대비 달성률로 관리.

## HAT 활용 (승인된 역할 구분)
- **맞음**: ① Phase 0 사전 타당성 게이트(hatir), ② 커널 타일 사전 선정(hatir),
  ③ KV/state 산정(hat-kv-manager 관점).
- **안 붙임**: 커널 수치검증, #35922 디버깅, hat-vllm-mock 호스트 정찰.
- 부산물: 32×A100 시스템 트윈 + K3 케이스 스터디는 hatir/hat-schema 자산으로 환류.

## 리포 구조 (`~/repos/k3-sm80`, private)
```
docs/superpowers/specs/   이 문서
recon/                    업스트림·사실관계 재확인 결과 (출처 포함)
phase0/                   hatir 모델 코드 + 판정 리포트 (host에서 실행 가능)
a100-kit/                 사내 반입용 프로브 킷 + 러너 문서
```
실행 규약: hatir는 `PYTHONPATH=~/repos/hatir/impl:~/repos/hat-schema`로 사용.
모든 변경은 커밋+푸시(사내에서 pull).

## 리스크
- Phase 0 판정이 "물리적으로 불가"로 나올 수 있음 — 그 경우 조기 종료가 성공적
  결과임을 명시.
- vLLM 주 단위 변동 → recon 결과에 커밋/버전 고정 필수.
- KDA prefill Triton 경로의 수치 검증 미완(업스트림) → Phase 2에서 FLA 단위테스트
  대조를 프로브에 포함.
