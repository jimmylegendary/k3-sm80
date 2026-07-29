# k02_kda_recurrent — KDA recurrent decode (fused_recurrent_kda, Triton)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-29T07:50:36+00:00 · 소요 0.0s · repo d00c7d5

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | selftest — 실비교 없음 |
| SANITIZER | PASS (통과) | compute-sanitizer memcheck clean (0 errors) |
| SHAPES | PASS (통과) | 요구 5케이스 전부 커버 |
| PERF | SKIP (생략) | selftest — 측정 없음 |

환경: GPU 없음 · torch 2.13.0+cu130 · triton 3.7.1 · vllm 0.26.1rc1.dev25+g0498dc7ea
핀: {"selftest": true}
