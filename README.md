# k3-sm80 — Kimi K3 를 32×A100(SM80) 에서 서빙하기

사내 A100 노드에서는 **`run/` 스크립트 4개만 순서대로 실행**하면 된다. ghcr(도커 레지스트리)
차단 환경 전제 — 이미지도 GitHub Release 로 받는다. 상세 배경은 `docs/PLAN.md`.

## 사내 러너 빠른 시작 (스크립트 → 결과)

```bash
git clone git@github.com:jimmylegendary/k3-sm80.git && cd k3-sm80
```

| 순서 | 스크립트 | 하는 일 | 얻는 결과 |
|---|---|---|---|
| 1 | `bash run/00-check-node.sh` | 노드 사전점검 (docker/GPU/드라이버/zstd — 설치 없음) | PASS/FAIL 표. exit 0 이면 진행 가능 |
| 2 | `bash run/10-load-image.sh triton` | GitHub Release 에서 이미지 tar 다운로드→검증→`docker load` (멱등) | `k3-sm80-kernelkit:triton` 사용 가능 (~3.7GB 전송) |
| 3 | `bash run/20-kernel-test.sh triton` | **커널 단독검증** — k01~k04·k06·k07 을 각각 fp32 참조와 대조 (vLLM 불필요) | 커널별 4게이트 판정 + 실행 끝의 **RELAY 블록** |
| 4 | **화면의 RELAY 블록(≤12줄) + exit 코드를 그대로 전달** | 파일 반출 불가 전제의 회신 채널 — 타이핑/전사 가능하게 압축돼 있음 | 개발 호스트가 이것만으로 커널 수정(S3) 진행 |

회신 규약(중요): 실행 마지막에 `==== RELAY ====` 블록이 항상 출력된다. **그 블록 전체 +
run_all 최종 exit 코드**만 옮겨주면 된다. FAIL/CRASH 가 있으면 해당 커널 로그
(`kernel-kit/kernel-report/<커널id>.log`)의 **Error/Traceback 이 있는 줄 1~3개**를 추가로
옮겨주면 진단이 훨씬 빨라진다. (파일 반출이 가능해지면 `run/30-send-results.sh` 로 tgz
커밋·푸시가 더 완전한 채널 — 선택사항.)

full 티어(k05 Marlin crash 재현 + compute-sanitizer 포함)가 Release 에 올라오면 2~4를
`full` 로 반복: `bash run/10-load-image.sh full && bash run/20-kernel-test.sh full`.

## 처음부터 다시 (재클론했을 때 — S3 진단 라운드 기준)

git conflict 등으로 지우고 다시 받았어도 **도커 이미지는 노드에 남아 있다**(`docker load` 는
repo 와 무관) — `run/10` 은 멱등이라 이미 있으면 즉시 skip 된다. 실행 결과(`kernel-report/`)는
git 미추적(로컬 전용)이므로 재클론하면 사라진다 → 스윕을 다시 돌려 재생성하면 된다.

```bash
git clone git@github.com:jimmylegendary/k3-sm80.git && cd k3-sm80
bash run/00-check-node.sh                 # 사전점검 (exit 0 확인)
bash run/10-load-image.sh triton          # 이미 load 돼 있으면 "skip" 한 줄 뜨고 끝
bash run/10-load-image.sh full
bash run/20-kernel-test.sh full           # ① 전체 스윕 (k05 sanitizer arm 포함, 수십 분)
bash run/27-perf-calib.sh full            # ② PERF 교정 실측 (~2분)
bash run/28-k05-sanitizer.sh              # ③ k05 memcheck 상세 (GPU 불필요, ① 이후)
bash run/26-k01-ab.sh full                # ④ (선택) k01 int32 수정 확정 A/B (~2분)
```

**회신할 것 (전부 화면 텍스트, 순서대로):** ① 의 RELAY 블록 + exit 코드 → ② 의 C0~C9 →
③ 의 S0~S9 → ④ 의 AB-VERDICT 한 줄. 이 네 묶음이 S3(커널 수정)의 전체 입력이다.
`R!` 로 시작하는 줄이 보이면 이전 실행 결과가 섞였다는 경고 — 그 줄도 같이 전달.

## 커널 테스트 결과를 읽는 법

- 커널별 판정 = 4게이트: **NUMERIC**(fp32 참조 대비 오차) / **SANITIZER**(memcheck) /
  **SHAPES**(K3 실측 shape 커버) / **PERF**(루프라인 대비). 전부 PASS = 그 커널은 S4(vLLM 통합) 진입 가능.
- **기대 결과**: k01~k04·k07 = PASS 기대(순수 Triton, 포팅 불필요) · k06 = 측정 전용(dequant
  비용 실측) · **k05 = CRASH 가 예상이자 목적**(#35922 재현 — crash 지점·sanitizer 로그가 수정 입력).
- `run/20` 의 exit: 0=FAIL 없음 · 1=FAIL/CRASH 존재(그것도 유효한 결과) · **2=실행 안 됨(성공 아님)**.
- 요약은 `kernel-kit/kernel-report/SUMMARY.md`, 커널별 상세는 같은 폴더의 `<커널id>.md/.json`.

## Kimi K3.1-Pro 지원 (P1, 2026-11-09 — 호스트 검증까지, A100 미실행)

K3 이미지·lane(위 절차)은 그대로 두고, **최신 vLLM main 핀 `a3c14efb` 위에 K3.1(DSA) 경로를 얹은 별도 이미지·lane**을 추가했다
(설계·근거: `docs/K31-SUPPORT-REVIEW.md`, 핀 사실: `recon/vllm-upstream-k31.md`, 체크포인트 diff: `recon/k31-config.md`).

| 구성 | 위치 | 비고 |
|---|---|---|
| vLLM 패치 세트(K3.1 전용) | `docker/patches-k31/` (`README.md`에 적용 순서·근거) | marlin SM80 rebase · `TRITON_MLA_SPARSE`+SM80 indexer(PR #47629 rebase, int64 수정) · K3.1 DSA glue |
| 이미지 레시피 | `docker/Dockerfile.k31-full` → `docker/Dockerfile.k31-openai`, `docker/build-k31.sh`, `docker/release-k31.sh`, 검증기 `docker/verify_k31_image.py` | CUDA 13.0.3 / torch 2.14 / Ubuntu 24.04(핀 CMake가 g++≥13 요구). 반입 절차 `docker/PULL-K31.md` |
| 사내 러너 스크립트 | `bash run/11-load-k31-image.sh` → `bash run/21-k31-kernel-test.sh` | K3용 `run/10`·`run/20`은 변경 없음 |
| kernel-kit 게이트 | `k11_dsa_indexer_sm80`, `k12_sparse_mla_sm80` (+ k01~k09 핀 재vendor) | NUMERIC/SANITIZER/SHAPES 판정, PERF 측정 전용. CPU 인터프리터 대조는 PASS 근거 아님 |
| 서빙 lane | `deploy/11-k31-tp8-pp4-256k-safe/` (`make -C … gates up gate-load gate-context smoke`) | TP8/PP4 · 256k · `--kv-cache-dtype bfloat16` 명시 · `--attention-backend TRITON_MLA_SPARSE` · DSpark 없음 · enforce_eager |
| 증거 표 | `docs/A100-K31-RESULTS.md` | 전부 INCOMPLETE에서 시작 — 사내 1라운드 결과로만 PASS/FAIL 기입 |

**정직한 상한**: 호스트에서는 핀 worktree 패치 적용·py_compile·AST·ruff·unittest·CPU dry-run·Triton 인터프리터 수치 대조·sm_80 PTX 컴파일까지만 확인했다.
GPU 실행(커널 4게이트, 서빙 smoke), 속도, 1M, DSpark, FP8 KV는 P1 범위 밖이거나 사내 A100 라운드에서 판정한다.

## 리포 지도

| 경로 | 내용 |
|---|---|
| `run/` | **사내 러너용 스크립트** (위 표 — 이것만 쓰면 됨) |
| `docker/patches-k31/`, `docker/Dockerfile.k31-*`, `docker/PULL-K31.md` | **K3.1 전용** 패치 세트·이미지 레시피·반입 절차 (K3 것과 분리) |
| `deploy/11-k31-*/`, `docs/A100-K31-RESULTS.md` | **K3.1 lane 11**(TP8/PP4 256k bf16)과 그 증거 표 |
| `kernel-kit/` | S2 커널 단독검증 하네스 (k01~k07) + `vendor/` (vLLM 에서 vendor 한 Triton 커널 — torch+triton 만으로 실행) |
| `docker/` | 이미지 2티어 빌드·배포 (`PULL.md` = 수동 절차 상세, `RELEASE-TAG.txt` = 최신 릴리스 태그) |
| `a100-kit/` | S1 프로브 킷 (vLLM 서빙 경로 프로브 — probe_35922 등; full 이미지 이후 단계) |
| `phase0/` | hatir 해석 모델 — 타당성 게이트 **PASS** (TP8PP4 real 46.8 tok/s, `verdict.json`) |
| `recon/` | 업스트림·사실관계 조사 (버전 pin 근거) |
| `docs/PLAN.md` | 전체 계획 S0~S6 + 단계별 완료 판정 |
| `docs/K31-SUPPORT-REVIEW.md` | **Kimi K3.1-Pro 지원 검토(2026-11-09)** — 체크포인트 diff·vLLM main DSA 경로의 SM80 블록 2건·리포 변경 지도·phase 계획·결정 요청 |
| `results/incoming/` | 사내 실행 결과 회신함 (`run/30` 이 커밋) |

## 지금 어디까지 왔나 (2026-07-29)

S0 게이트 PASS → S1/S2 킷·이미지 준비 완료(triton 티어 배포됨, full 빌드 중) →
**다음 = 사내에서 `run/` 4단계 실행** → 결과로 S3(커널 수정: #35922 등) 착수.
