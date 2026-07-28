# k02_kda_recurrent — KDA recurrent decode (fused_recurrent_kda, Triton)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-28T09:01:18+00:00 · 소요 0.0s · repo de8dd04

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | selftest — 실비교 없음 |
| SANITIZER | PASS (통과) | compute-sanitizer memcheck clean (0 errors) |
| SHAPES | PASS (통과) | 요구 5케이스 전부 커버 |
| PERF | SKIP (생략) | selftest — 측정 없음 |

환경: GPU 없음 · torch None · triton None · vllm None
핀: {"selftest": true}
