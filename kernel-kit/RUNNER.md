# kernel-kit RUNNER — S2 커널 단독 검증 사내 실행 절차

> **주 경로 = 도커 이미지** (2026-07-29 S2 전달 모델 변경, PLAN §S2): 사내에서는
> pull(또는 load) → `docker run ... bash /work/kernel-kit/run_all.sh` 만 한다 —
> 절차 전체는 **`docker/PULL.md`**. 이미지가 torch/triton(+full 은 vLLM 소스빌드·
> compute-sanitizer)을 동봉하므로 venv 구성이 필요 없다.
>
> **대안(베어메탈 venv)에서 환경이 안 잡힐 때(사전조건 미충족)**: `run_all.sh` 가 항목별
> FAIL 표와 해결 명령을 출력한다. 최다 원인 = venv 미활성(시스템 python 에 torch 없음).
> 자동 구성: `bash ../a100-kit/bootstrap.sh --stage2 --yes` (또는 `AUTO_FIX=1 bash run_all.sh`
> 로 stage1 자동 구성). run_all.sh 는 `~/k3-sm80-work` 의 venv-k3/venv-probe 를 자동 감지한다.

작성: 2026-07-28 (2026-07-29 도커 주 경로로 개정) · 러너: Jimmy (사내 A100 노드) · 계약: `docs/PLAN.md` §S2
핀: vLLM main `a3c14efb` (2026-11-09; 옛 kimi-k3 브랜치 `0498dc7e`·main `d223c900` 퇴역) / DSA 커널 PR #47629 head `064801dd`(패치본, `docker/patches-k31/triton-mla-sparse-sm80.patch`) / HF Kimi-K3 `9f62e4e9` · Kimi-K3.1-Pro `d9e99b07`

## 0. 무엇을 판정하는가

커널 11종(k01~k09 + K3.1 DSA k11·k12)을 **vLLM 없이 단독으로** 4게이트 판정한다. S2를 통과한 커널만 S4(vLLM 통합)에 들어간다.

| 게이트 | 판정 | 기준 원천 |
|---|---|---|
| NUMERIC | fp32 참조 대비 오차 ≤ tol (seed 고정) | `common/tolerance.py` TOL 표 (업스트림 바 + bf16 eps 유도, 근거 주석) |
| SANITIZER | compute-sanitizer memcheck clean | `common/report.py` sanitizer 파서 (ERROR SUMMARY: 0) |
| SHAPES | K3 실측 shape 요구 케이스 전부 커버 | `common/shapes.py` REQUIRED_SHAPES (배치 1/8/32 · ctx 8k/128k 대표점; k11·k12 는 ctx 8k/128k/256k × 배치 1/8) |
| PERF | 측정 ≤ 루프라인/eff_floor | `common/timing.py` PERF_POLICY + `--phase0-verdict` 원천 (k06·k09·k11·k12 는 측정 전용 measure_only: 반복 안정성 CV 만 판정, 속도 floor 는 A100 실측 후 사전등록) |

게이트는 PASS/FAIL/SKIP(사유 필수) 3값, **커널 종합 = 4게이트 all-PASS**.
FAIL/CRASH도 버리는 결과가 아니다 — 특히 k05(#35922)는 crash 재현 로그가 **S3 입력물**이다.

## 1. 사전조건 — 주 경로 = 도커 이미지 (`docker/PULL.md`)

**주 경로**: 새 환경을 만들지 말 것 — 이미지가 환경이다. `docker/PULL.md` §0(사전조건:
docker + nvidia-container-toolkit + repo clone) → §1/§2(pull 또는 release load)만 하면 된다.
k01~k04·k07 커널 소스는 `kernel-kit/vendor/` 로 vendor화되어 **vLLM 설치 없이** torch+triton
만으로 돈다(k05 만 vLLM C-확장 → full 이미지가 흡수).

| 커널 | 필요한 이미지 | 비고 |
|---|---|---|
| k01~k04·k06·k07 | `:triton` 이상 | vendored 커널(`vendor/fla_kda` 등, pin a3c14efb 사본) — vLLM 불필요 |
| k11·k12 (DSA indexer / sparse MLA) | `:full` 권장(`:triton` 도 실행되나 SANITIZER SKIP) | vendored Triton(`vendor/dsa_ops`, PR #47629 커널의 **패치본** 사본 = 이미지에 실리는 것과 동일) — vLLM 불필요 |
| k05 (Marlin MoE) | **`:full` 만** | `_moe_C` sm80 C-확장 = vLLM 소스빌드 필요 |
| SANITIZER 게이트 | **`:full` 만** | compute-sanitizer 동봉은 full 뿐 — triton 에서는 SKIP(사유 기록) |

재vendor 사본은 새 핀 이미지(torch 2.14, `k31-sm80-*`)를 전제한다. 옛 K3 이미지(torch 2.13)에서 k01~k04 가 import/launch 오류면 K3 회귀가 아니라 이미지 불일치다. K3.1 핀 이미지로 돌릴 때는 `bash run/21-k31-kernel-test.sh`(`docker/PULL-K31.md`)를 쓴다.

**대안(베어메탈 venv — 도커 불가 시)**: `a100-kit/RUNBOOK.md` §0.5 대안 절차(§1 노드 점검 →
§3 Stage 2 `venv-k3` 소스빌드)를 재사용한다. 이때 kernel-kit 는 **venv-k3 를 기본 환경**으로
한다(K3 전용 KDA Triton ops 가 이 브랜치에만 있음; venv-probe/Stage 1 은 main vendored 사본이라
게이트 수식(fused gate)이 다름 — k3 기준이 정본. k05 는 Stage 2 소스빌드 필수).
실행 시점: `a100-kit/DECISION-TREE.md` 의 S1 실행 순서 직후 같은 세션에서 이어 도는 것이 표준.

추가 확인(도커 경로는 PULL.md §0에 포함):

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # A100, 8.0
# (베어메탈일 때만) which compute-sanitizer — 없으면 SANITIZER=SKIP으로 진행됨
```

## 2. 실행

**주 경로 (도커 — 상세·부분 실행·비루트 옵션은 `docker/PULL.md` §3)**:

```bash
cd <repo>   # /work 로 마운트됨 — 결과 tgz 는 호스트 kernel-kit/ 에 남는다

docker run --rm --gpus all -v "$PWD":/work \
  ghcr.io/jimmylegendary/k3-sm80-kernelkit:full \
  bash /work/kernel-kit/run_all.sh                # 전체 (k01..k09·k11·k12 순차 + 집계 + tar)

docker run --rm --gpus all -v "$PWD":/work -e KERNELS="k05_marlin_moe_w4a16" \
  ghcr.io/jimmylegendary/k3-sm80-kernelkit:full \
  bash /work/kernel-kit/run_all.sh                # 부분 재실행 (FAIL 커널만 재시도 등)
```

**대안 (베어메탈 venv)**:

```bash
cd <repo>/kernel-kit
source ~/k3-sm80-work/venv-k3/bin/activate

bash run_all.sh                                   # 전체 (k01..k09·k11·k12 순차 + 집계 + tar)
KERNELS="k05_marlin_moe_w4a16" bash run_all.sh    # 부분 재실행 (FAIL 커널만 재시도 등)
bash run_all.sh --cpu-dry-run                     # (호스트용) 참조·판정·리포트 경로만 — GPU 불필요
```

**저작 호스트(GPU 없음) — k11/k12 CPU 인터프리터 대조(게이트 판정 아님; torch+triton venv, 예: `~/.venvs/k31-cpu`)**:

```bash
TRITON_INTERPRET=1 python kernel-kit/k11_dsa_indexer_sm80/harness.py --interpret --report-dir /tmp/k11-interp
TRITON_INTERPRET=1 python kernel-kit/k12_sparse_mla_sm80/harness.py --interpret --report-dir /tmp/k12-interp
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
| k04 MLA decode | 3–6분 | +5–10분 | ctx 128k KV(b32 기준 ~4.8GB) 할당 포함. bf16 KV 단일 arm(SM80은 fp8 KV 불가 — 리포트 evidence.fp8_kv_arm). PERF는 roofline≥30µs 4케이스 판정(b1_ctx8k는 launch 지배 → evidence만) |
| k05 Marlin MoE | 20–50분 | +30–120분 | **crash 예상 경로**. 케이스=독립 프로세스(사다리 E 8→64→256→896, K 512→3584→7168) — 한 crash가 전체를 안 죽임. 시간은 E=896 가중치 생성·양자화가 지배(케이스당 수 분). sanitizer는 가중치 빌드까지 계측되어 매우 느림 — hang ≠ 실패, 기다릴 것 |
| k06 dequant bench | 3–6분 | +1–3분 (소형 arm) | 측정 산출형 — CV≤10%면 PASS. `calibration-k06-dequant.json`(dequant_ops/param)이 phase0 재보정 입력 |
| k11 DSA indexer | 15–30분 | +5–15분 | child 11개(케이스당 데이터 생성·오라클·autotune); `dec_b8_ctx256k_pool27x` 는 ≈7.5GB 풀 (미실측 추정) |
| k12 sparse MLA | 10–25분 | +5–15분 | child 11개; kv 풀 b8·256k=2.4GB; 신규 `dec_b1_ctx160k_packed` 풀 약 5.05GB(추정 피크 5.66GB, 실측 `case_results.*.extra.gpu_mem`); split 변종(auto/1/4, prefill auto/2) 전부 대조 (미실측 추정) |
| k07 SiTU 참조 | 1분 | +1–2분 | arm=native(언퓨즈드)+fused(`_C.situ_and_mul`, vLLM 빌드 존재 시 = `:full` 이미지/venv-k3). PERF 기본 판정=fused arm·m8192 케이스만(floor 0.30); `--no-perf` 시 정책상 SKIP(성능 후순위) |

전체 첫 실행 대략 60–110분(추정, k11/k12 가산; autotune 포함), 재실행 ~15–25분.
sanitizer arm은 각 하네스가 축소 shape로 자체 실행한다(전 shape sanitizer는 비용상 비표준).

## 4. 결과 해석

- `kernel-report/<id>.json` — 기계판정(스키마는 `common/report.py` docstring). `<id>.md` = 사람용 요약.
- `kernel-report/SUMMARY.md` + `summary.json` — 집계표. **S2 완료 = 11커널 전부 overall PASS.**
- overall 의미: `PASS`(4게이트 전부) / `FAIL`(어느 게이트든 FAIL — S3 후보) /
  `INCOMPLETE`(FAIL 없이 SKIP 잔존 — dry-run 은 원리상 여기까지; GPU 실행에서 INCOMPLETE 가 나오면
  SKIP 사유를 읽고 해소할 것: sanitizer 미설치, PERF 원천 미지정 등).
- exit code: `0` FAIL 없음 · `1` FAIL 존재 · `2` GPU 사전조건 미충족(skip) · `3` 프레임워크/하네스 오류.
- RELAY 는 ≤12줄·줄당 ≤76자: PASS 커널은 `R1 PASS k01 …` 한 줄, 비-PASS 는 `Rn kid 게이트8자 overall | 꼬리`, 줄이 모자라면 뒤쪽 커널부터 꼬리 없이 4개씩 한 줄(`common/relay.py` docstring).
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
| `env_check` exit 2 (GPU 미인식) | 컨테이너면 대부분 `--gpus all` 누락/드라이버 부족 — `docker/PULL.md` §4. 베어메탈이면 RUNBOOK §6.1 (드라이버/venv). kernel-kit 자체는 손대지 말 것 |
| capability ≠ (8,0) FAIL | 다른 GPU에 잡힘 — `CUDA_VISIBLE_DEVICES`로 A100 지정. 타 아치 결과는 판정 무효 |
| sanitizer 미발견 SKIP | CUDA 툴킷 `bin/` PATH 추가 (RUNBOOK §1.5). SKIP인 채로는 커널 종합 PASS 불가 |
| k01 autotune이 매우 느림 | 정상(최초 1회). `TRITON_CACHE_DIR`를 홈에 고정하면 재실행 단축 |
| k05 sanitizer 출력 폭주 | 정상 — 요약줄(ERROR SUMMARY)만 판정에 쓰인다. 로그는 그대로 반출 |
| 하네스 MISSING | 해당 커널 하네스가 아직 저작 전 — repo pull 최신화 확인 |
| `children/*.json classification=MEMORY_VIOLATION` | child 가 CUDA illegal access — evidence.cases 의 log_tail 이 S3 입력 |
| k11 NUMERIC `[케이스/topk]` 위반 | logits(rmse_rel)도 함께 위반이면 디코드/스케일/주소 버그, logits 정상인데 top-k 만이면 TOPK_* 정의(`common/dsa_ref.py`) 재검토 |
| `dec_b8_ctx256k_pool27x` OOM | 풀 ≈7.5GB — 다른 프로세스의 GPU 점유 확인; 케이스 단위 FAIL 이라 나머지는 유효 |
| `dec_b1_ctx160k_packed` OOM | 풀 약 5.05GB — GPU 점유 확인; 케이스 단위 FAIL 이라 나머지는 유효. `int32_zone` entry FAIL 은 선택 행이 한계 너머에 하나도 없다는 뜻(공회전 방지) |
| OOM (k04 b32/128k 등) | 하네스가 케이스 단위로 실행하므로 실패 케이스만 FAIL 기록됨 — 리포트 evidence 확인 후 보고. (k01 b32 프리필은 상주 산정상 OOM 확정이라 b32_T2k 로 사전 축소 — `VERIFICATION.md` #2) |
