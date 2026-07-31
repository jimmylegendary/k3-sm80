# Kimi-K3 32×A100 five-lane evidence

개발 host에서 파일·CPU contract를 통과한 상태와 사내 A100 실행 결과를 분리한다.
아래 행은 실제 명령을 실행한 뒤에만 `PASS` 또는 `FAIL`로 바꾼다. 앞 gate에서 막혀
실행하지 않은 행은 `INCOMPLETE`를 유지한다.

| id | command | image digest | env hash | GPU/driver | result | artifact |
|---|---|---|---|---|---|---|
| k08 | `make -C deploy/01-tp8-pp4-256k-safe gate-k08` | not-run | not-run | not-run | INCOMPLETE | not-run |
| c01 | `make -C deploy/02-tp8-dp4-ep32-256k-safe gate-c01` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k10 | `make -C deploy/03-tp8-dp4-ep32-256k-dspark gate-dspark` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k09 | `make -C deploy/04-tp8-dp4-ep32-1m-fp8kv gate-k09` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane01 | `make -C deploy/01-tp8-pp4-256k-safe gates up gate-load gate-context smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane02 | `make -C deploy/02-tp8-dp4-ep32-256k-safe gates up gate-load gate-context smoke baseline` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane03 | `make -C deploy/03-tp8-dp4-ep32-256k-dspark gates up gate-load gate-context gate-dspark smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane04 | `make -C deploy/04-tp8-dp4-ep32-1m-fp8kv gates up gate-load gate-context calibrate smoke baseline` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane05 | `make -C deploy/05-tp8-dp4-ep32-1m-fp8kv-dspark gates up gate-load gate-context gate-dspark calibrate smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |

검증:

```bash
python3 -m deploy.tooling.validate_lanes --evidence docs/A100-5LANE-RESULTS.md
```

각 실행 뒤 `image digest`, `.env` SHA-256, GPU/driver, report JSON 절대경로를 해당 행에
기입한다. 결과 파일 전체를 회신할 필요는 없고, 실패 시 stable error code와 마지막
20줄만 공유한다.
