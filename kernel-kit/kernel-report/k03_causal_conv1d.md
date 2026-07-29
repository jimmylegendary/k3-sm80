# k03_causal_conv1d — causal_conv1d fn/update (depthwise conv4 + SiLU, Triton)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-29T07:50:45+00:00 · 소요 0.002s · repo d00c7d5

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run: 실커널 미실행 — 순수 python 참조 자기검증 14건(fn↔update 토큰 동치 포함) + 판정 배선 양방향 green |
| SANITIZER | SKIP (생략) | cpu-dry-run: GPU 미실행 |
| SHAPES | SKIP (생략) | cpu-dry-run: 커널 미실행 — 요구 케이스 정의만 확인 |
| PERF | SKIP (생략) | 측정값 없음 (GPU 미실행) |

환경: GPU 없음 · torch 2.13.0+cu130 · triton 3.7.1 · vllm 0.26.1rc1.dev25+g0498dc7ea
핀: {"vllm_kimi_k3": "0498dc7e", "vllm_main": "d223c900", "hf": "moonshotai/Kimi-K3@9f62e4e9"}
