# k05_marlin_moe_w4a16 — Marlin W4A16 MoE (marlin_moe_wna16, #35922 대상)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-29T05:07:49+00:00 · 소요 0.866s · repo a4b69ce

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run — 순수 python 참조 자기검증 green, 실커널 비교 없음 |
| SANITIZER | SKIP (생략) | cpu-dry-run — GPU 미실행 |
| SHAPES | SKIP (생략) | cpu-dry-run — 케이스 정의 9건 검증만(필수 4 포함), 실행 커버 아님 |
| PERF | SKIP (생략) | cpu-dry-run — 측정 없음 |

환경: GPU 없음 · torch 2.13.0+cu126 · triton 3.7.1 · vllm None
핀: {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e", "hf": "moonshotai/Kimi-K3@9f62e4e9"}
