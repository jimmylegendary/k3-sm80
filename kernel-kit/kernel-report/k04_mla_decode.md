# k04_mla_decode — MLA decode (TRITON_MLA, latent 576 absorbed)

- 종합: **INCOMPLETE** (cpu-dry-run — 실커널 미실행)
- 생성: 2026-07-30T05:44:38+00:00 · 소요 0.029s · repo 1e24e34

| 게이트 | 판정 | 사유 |
|---|---|---|
| NUMERIC | SKIP (생략) | cpu-dry-run: 실커널 미실행 — 참조·판정 파이프 자체점검만 수행(evidence.dry_run) |
| SANITIZER | SKIP (생략) | cpu-dry-run: GPU 없음 |
| SHAPES | SKIP (생략) | cpu-dry-run: 축소 shape 실행 — K3 실측 shape 판정은 GPU 모드 전용 |
| PERF | SKIP (생략) | cpu-dry-run: 측정 없음 |

환경: GPU 없음 · torch 2.13.0+cu126 · triton 3.7.1 · vllm None
핀: {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e", "hf_kimi_k3": "9f62e4e9"}
