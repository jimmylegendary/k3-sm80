# 01-tp8-pp4-256k-safe

## 목적

- topology: TP8 × PP4 × DP1, EP=0
- maximum context: 262144
- DSpark=0, FP8 KV=0

이 lane은 GPU gate가 연결되기 전까지 `INCOMPLETE`다.

## Preflight

```bash
make preflight
```

## 실행

```bash
make gates
make up
make smoke
make status
```

## 오류 코드

각 명령은 `K3-*` stable error code와 최대 20줄의 요약을 출력한다.

## 정리

```bash
make down
```

## 승격

필수 gate stamp와 OpenAI smoke가 모두 PASS한 뒤 다음 lane으로 이동한다.
