# k05_marlin_moe_w4a16 — Marlin W4A16 MoE (marlin_moe_wna16, #35922 대상)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-28T10:03:14+00:00 · 소요 0.041s · repo db3c9de

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run — 순수 python 참조 자기검증 green, 실커널 비교 없음 |
| SANITIZER | SKIP (생략) | cpu-dry-run — GPU 미실행 |
| SHAPES | SKIP (생략) | cpu-dry-run — 케이스 정의 9건 검증만(필수 4 포함), 실행 커버 아님 |
| PERF | SKIP (생략) | cpu-dry-run — 측정 없음 |

환경: GPU 없음 · torch None · triton None · vllm None
핀: {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e", "hf": "moonshotai/Kimi-K3@9f62e4e9"}
