# Kimi-K3.1-Pro 32×A100 lane 11 evidence

개발 host에서 파일·CPU contract를 통과한 상태와 사내 A100 실행 결과를 분리한다.
아래 행은 실제 명령을 실행한 뒤에만 `PASS` 또는 `FAIL`로 바꾼다. 앞 gate에서 막혀
실행하지 않은 행은 `INCOMPLETE`를 유지한다. K3 five-lane 증거
[A100-5LANE-RESULTS.md](/home/jimmy/repos/k3-sm80/docs/A100-5LANE-RESULTS.md)와는 이미지·lane이
달라 행 ID(`k08`)가 같아도 별개의 증거이며, 한쪽의 PASS가 다른 쪽을 대신하지 않는다.

| id | command | image digest | env hash | GPU/driver | result | artifact |
|---|---|---|---|---|---|---|
| k08 | `make -C deploy/11-k31-tp8-pp4-256k-safe gate-k08` | not-run | not-run | not-run | INCOMPLETE | not-run |
| kda | `make -C deploy/11-k31-tp8-pp4-256k-safe gate-kda` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k11 | `make -C deploy/11-k31-tp8-pp4-256k-safe gate-k11` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k12 | `make -C deploy/11-k31-tp8-pp4-256k-safe gate-k12` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane11 | `make -C deploy/11-k31-tp8-pp4-256k-safe gates up gate-load gate-context smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |

검증:

```bash
python3 -m deploy.tooling.validate_lanes --evidence-k31 docs/A100-K31-RESULTS.md
```

각 실행 뒤 `image digest`(`sha256:` 시작), `.env` SHA-256(`sha256:` 시작), GPU/driver, report JSON
절대경로를 해당 행에 기입한다. 결과 파일 전체를 회신할 필요는 없고, 실패 시 stable error code와
마지막 20줄만 공유한다.

P1 상한(설계 [K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md) §5):
`k11`, `k12`는 NUMERIC, SANITIZER, SHAPES 판정이고 PERF는 측정 전용이다. `lane11`은 256k,
BF16 KV, `max_num_seqs=1`, `enforce_eager`, DSpark 없음이며 속도와 1M은 보증하지 않는다.
