# k01_kda_chunk — KDA chunk prefill (chunk_kda_with_fused_gate, Triton)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-30T05:44:35+00:00 · 소요 0.002s · repo 1e24e34

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run: 실커널 미실행 — 순수 python 참조 자기검증 14건 + 판정 배선 양방향 green |
| SANITIZER | SKIP (생략) | cpu-dry-run: GPU 미실행 |
| SHAPES | SKIP (생략) | cpu-dry-run: 커널 미실행 — 요구 케이스 정의만 확인 |
| PERF | SKIP (생략) | 측정값 없음 (GPU 미실행) |

환경: GPU 없음 · torch 2.13.0+cu126 · triton 3.7.1 · vllm None
핀: {"vllm_kimi_k3": "0498dc7e", "vllm_main": "d223c900", "hf": "moonshotai/Kimi-K3@9f62e4e9"}
