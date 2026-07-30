# k06_dequant_bench — W4A16 dequant 마이크로벤치 (dequant_ops/param 재보정)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-30T05:44:40+00:00 · 소요 1.084s · repo 1e24e34

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run — 순수 python dequant 판정경로 green, 실커널 비교 없음 |
| SANITIZER | SKIP (생략) | cpu-dry-run — GPU 미실행 |
| SHAPES | SKIP (생략) | cpu-dry-run — 케이스 정의 3건 검증만, 실행 커버 아님 |
| PERF | SKIP (생략) | cpu-dry-run — 측정 없음 (measure_only 커널) |

환경: GPU 없음 · torch 2.13.0+cu126 · triton 3.7.1 · vllm None
핀: {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e", "hf": "moonshotai/Kimi-K3@9f62e4e9"}
