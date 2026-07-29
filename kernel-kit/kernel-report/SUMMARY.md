# S2 kernel-kit 집계

- 생성: 2026-07-29T05:07:52+00:00 · 대상 7커널
- 종합: PASS 0 / FAIL 0 / INCOMPLETE 7 / MISSING 0 → S2 완료(전 커널 PASS): 아니오

| 커널 | 종합 | NUMERIC | SANITIZER | SHAPES | PERF |
|---|---|---|---|---|---|
| k01_kda_chunk (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k02_kda_recurrent (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k03_causal_conv1d (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k04_mla_decode (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k05_marlin_moe_w4a16 (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k06_dequant_bench (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k07_situ_act (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |

해석: MISSING=하네스 미저작/미실행 · INCOMPLETE=FAIL 없이 SKIP 잔존(dry-run 상한) · S4 진입은 해당 커널 overall=PASS 가 조건 (docs/PLAN.md).
