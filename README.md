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
| 3 | `bash run/20-kernel-test.sh triton` | **커널 단독검증** — k01~k04·k06·k07 을 각각 fp32 참조와 대조 (vLLM 불필요) | 커널별 4게이트 판정 + 반출용 `kernel-report-*.tgz` |
| 4 | `bash run/30-send-results.sh` | 결과 tgz 를 repo 에 커밋·푸시 (회신) | 개발 호스트가 분석 → 커널 수정(S3) 진행 |

full 티어(k05 Marlin crash 재현 + compute-sanitizer 포함)가 Release 에 올라오면 2~4를
`full` 로 반복: `bash run/10-load-image.sh full && bash run/20-kernel-test.sh full`.

## 커널 테스트 결과를 읽는 법

- 커널별 판정 = 4게이트: **NUMERIC**(fp32 참조 대비 오차) / **SANITIZER**(memcheck) /
  **SHAPES**(K3 실측 shape 커버) / **PERF**(루프라인 대비). 전부 PASS = 그 커널은 S4(vLLM 통합) 진입 가능.
- **기대 결과**: k01~k04·k07 = PASS 기대(순수 Triton, 포팅 불필요) · k06 = 측정 전용(dequant
  비용 실측) · **k05 = CRASH 가 예상이자 목적**(#35922 재현 — crash 지점·sanitizer 로그가 수정 입력).
- `run/20` 의 exit: 0=FAIL 없음 · 1=FAIL/CRASH 존재(그것도 유효한 결과) · **2=실행 안 됨(성공 아님)**.
- 요약은 `kernel-kit/kernel-report/SUMMARY.md`, 커널별 상세는 같은 폴더의 `<커널id>.md/.json`.

## 리포 지도

| 경로 | 내용 |
|---|---|
| `run/` | **사내 러너용 스크립트** (위 표 — 이것만 쓰면 됨) |
| `kernel-kit/` | S2 커널 단독검증 하네스 (k01~k07) + `vendor/` (vLLM 에서 vendor 한 Triton 커널 — torch+triton 만으로 실행) |
| `docker/` | 이미지 2티어 빌드·배포 (`PULL.md` = 수동 절차 상세, `RELEASE-TAG.txt` = 최신 릴리스 태그) |
| `a100-kit/` | S1 프로브 킷 (vLLM 서빙 경로 프로브 — probe_35922 등; full 이미지 이후 단계) |
| `phase0/` | hatir 해석 모델 — 타당성 게이트 **PASS** (TP8PP4 real 46.8 tok/s, `verdict.json`) |
| `recon/` | 업스트림·사실관계 조사 (버전 pin 근거) |
| `docs/PLAN.md` | 전체 계획 S0~S6 + 단계별 완료 판정 |
| `results/incoming/` | 사내 실행 결과 회신함 (`run/30` 이 커밋) |

## 지금 어디까지 왔나 (2026-07-29)

S0 게이트 PASS → S1/S2 킷·이미지 준비 완료(triton 티어 배포됨, full 빌드 중) →
**다음 = 사내에서 `run/` 4단계 실행** → 결과로 S3(커널 수정: #35922 등) 착수.
