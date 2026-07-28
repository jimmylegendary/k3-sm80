# kernel-kit RUNNER — S2 커널 단독 검증 사내 실행 절차

작성: 2026-07-28 · 러너: Jimmy (사내 A100 노드) · 계약: `docs/PLAN.md` §S2
핀: vLLM kimi-k3 `0498dc7e` / main `d223c900` / HF Kimi-K3 `9f62e4e9` (a100-kit RUNBOOK §0과 동일)

## 0. 무엇을 판정하는가

커널 7종을 **vLLM 없이 단독으로** 4게이트 판정한다. S2를 통과한 커널만 S4(vLLM 통합)에 들어간다.

| 게이트 | 판정 | 기준 원천 |
|---|---|---|
| NUMERIC | fp32 참조 대비 오차 ≤ tol (seed 고정) | `common/tolerance.py` TOL 표 (업스트림 바 + bf16 eps 유도, 근거 주석) |
| SANITIZER | compute-sanitizer memcheck clean | `common/report.py` sanitizer 파서 (ERROR SUMMARY: 0) |
| SHAPES | K3 실측 shape 요구 케이스 전부 커버 | `common/shapes.py` REQUIRED_SHAPES (배치 1/8/32 · ctx 8k/128k 대표점) |
| PERF | 측정 ≤ 루프라인/eff_floor | `common/timing.py` PERF_POLICY + `--phase0-verdict` 원천 |

게이트는 PASS/FAIL/SKIP(사유 필수) 3값, **커널 종합 = 4게이트 all-PASS**.
FAIL/CRASH도 버리는 결과가 아니다 — 특히 k05(#35922)는 crash 재현 로그가 **S3 입력물**이다.

## 1. 사전조건 — a100-kit RUNBOOK 환경 재사용

새 환경을 만들지 말 것. `a100-kit/RUNBOOK.md` §1(노드 점검) → §3(Stage 2 `venv-k3` 소스빌드)를
그대로 재사용한다. kernel-kit 는 **Stage 2 venv-k3 를 기본 환경**으로 한다
(K3 전용 KDA Triton ops `vllm.models.kimi_k3.nvidia.ops...` 가 이 브랜치에만 있음).

| 커널 | 필요한 환경 | 비고 |
|---|---|---|
| k01/k02/k03 (KDA·conv) | **venv-k3 (Stage 2)** | K3 vendored Triton ops. Stage 1로 돌릴 땐 main vendored 사본(`vllm.third_party.flash_linear_attention`)이라 게이트 수식(fused gate)이 다름 — k3 사본 기준이 정본 |
| k04 (MLA decode) | venv-k3 권장 (venv-probe 가능) | TRITON_MLA 커널은 양쪽에 존재 |
| k05 (Marlin MoE) | venv-k3 권장 (venv-probe 가능) | `_moe_C` sm80 빌드 필요 — Stage 2 소스빌드가 보장 |
| k06/k07 | 아무 쪽 | torch+triton만 있으면 됨 |

추가 확인(둘 다 RUNBOOK §1에 이미 포함):

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # A100, 8.0
which compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer   # 없으면 SANITIZER=SKIP으로 진행됨
```

## 2. 실행

```bash
cd <repo>/kernel-kit
source ~/k3-sm80-work/venv-k3/bin/activate

bash run_all.sh                                   # 전체 (k01..k07 순차 + 집계 + tar)
KERNELS="k05_marlin_moe_w4a16" bash run_all.sh    # 부분 재실행 (FAIL 커널만 재시도 등)
bash run_all.sh --cpu-dry-run                     # (호스트용) 참조·판정·리포트 경로만 — GPU 불필요
```

- PERF 목표 원천은 기본 `../phase0/verdict.json`. 다른 파일로 실험하려면
  `PHASE0_VERDICT=/path/to/verdict.json bash run_all.sh` (경로 하드코딩 없음).
- 러너가 시작 시 프레임워크 자체점검(py_compile + 4모듈 selftest)을 돌리고,
  실패하면 커널 실행 전에 exit 3으로 멈춘다.
- 하네스가 아직 저작되지 않은 커널은 **MISSING**으로 표시되고 넘어간다(오류 아님).

## 3. 커널별 예상 소요 (A100 1장, 첫 실행 기준)

| 커널 | 본 측정 | sanitizer arm | 비고 |
|---|---|---|---|
| k01 KDA chunk | 5–10분 | +5–15분 | Triton autotune 최초 1회가 지배(ctx 128k 케이스). 재실행은 캐시로 단축 |
| k02 KDA recurrent | 2–4분 | +3–8분 | state 6.29MB/seq 스트림 — 빠름 |
| k03 causal_conv1d | 1–2분 | +2–5분 | |
| k04 MLA decode | 3–6분 | +5–10분 | ctx 128k KV(b32 기준 ~4.8GB) 할당 포함 |
| k05 Marlin MoE | 3–8분 | +10–30분 | **crash 예상 경로** — sanitizer가 느리고 출력이 큼. hang ≠ 실패, 기다릴 것 |
| k06 dequant bench | 1–2분 | (해당 없음) | 측정 산출형 — CV≤10%면 PASS |
| k07 SiTU 참조 | 1분 | +1–2분 | PERF는 기본 SKIP(성능 후순위) |

전체 첫 실행 대략 30–60분(autotune 포함), 재실행 ~15분.
sanitizer arm은 각 하네스가 축소 shape로 자체 실행한다(전 shape sanitizer는 비용상 비표준).

## 4. 결과 해석

- `kernel-report/<id>.json` — 기계판정(스키마는 `common/report.py` docstring). `<id>.md` = 사람용 요약.
- `kernel-report/SUMMARY.md` + `summary.json` — 집계표. **S2 완료 = 7커널 전부 overall PASS.**
- overall 의미: `PASS`(4게이트 전부) / `FAIL`(어느 게이트든 FAIL — S3 후보) /
  `INCOMPLETE`(FAIL 없이 SKIP 잔존 — dry-run 은 원리상 여기까지; GPU 실행에서 INCOMPLETE 가 나오면
  SKIP 사유를 읽고 해소할 것: sanitizer 미설치, PERF 원천 미지정 등).
- exit code: `0` FAIL 없음 · `1` FAIL 존재 · `2` GPU 사전조건 미충족(skip) · `3` 프레임워크/하네스 오류.
- k05 가 FAIL/CRASH 인 것은 **예상 결과**다(#35922) — 리포트의 SANITIZER evidence(tail)와
  `k05_*.log` 가 S3 커널수정의 최소재현 입력이 된다. 나머지 6종의 green 은 S3 회귀 기준선.

## 5. 반출

```bash
# run_all.sh 가 자동 생성: kernel-kit/k3-sm80-kernel-report-<ts>.tgz  (kernel-report/ 전체)
# repo 반입(호스트에서):
tar xzf k3-sm80-kernel-report-<ts>.tgz -C kernel-kit/   # kernel-report/ 갱신
git add kernel-kit/kernel-report && git commit -m "kernel-kit: A100 S2 실행 결과 <ts>"
```

반복 루프는 PLAN 대로: 사내 실행 → tar 반입 → 호스트 분석·수정 → push → 사내 pull 재실행.

## 6. 트러블슈팅

| 증상 | 조치 |
|---|---|
| `env_check` exit 2 (GPU 미인식) | RUNBOOK §6.1 (드라이버/venv). kernel-kit 자체는 손대지 말 것 |
| capability ≠ (8,0) FAIL | 다른 GPU에 잡힘 — `CUDA_VISIBLE_DEVICES`로 A100 지정. 타 아치 결과는 판정 무효 |
| sanitizer 미발견 SKIP | CUDA 툴킷 `bin/` PATH 추가 (RUNBOOK §1.5). SKIP인 채로는 커널 종합 PASS 불가 |
| k01 autotune이 매우 느림 | 정상(최초 1회). `TRITON_CACHE_DIR`를 홈에 고정하면 재실행 단축 |
| k05 sanitizer 출력 폭주 | 정상 — 요약줄(ERROR SUMMARY)만 판정에 쓰인다. 로그는 그대로 반출 |
| 하네스 MISSING | 해당 커널 하네스가 아직 저작 전 — repo pull 최신화 확인 |
| OOM (k01 b32/T8k, k04 b32/128k) | 하네스가 케이스 단위로 실행하므로 실패 케이스만 FAIL 기록됨 — 리포트 evidence 확인 후 보고 |
