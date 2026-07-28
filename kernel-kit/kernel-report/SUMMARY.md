# S2 kernel-kit 집계

- 생성: 2026-07-28T09:01:26+00:00 · 대상 7커널
- 종합: PASS 0 / FAIL 0 / INCOMPLETE 4 / MISSING 3 → S2 완료(전 커널 PASS): 아니오

| 커널 | 종합 | NUMERIC | SANITIZER | SHAPES | PERF |
|---|---|---|---|---|---|
| k01_kda_chunk | **MISSING** | - | - | - | - |
| k02_kda_recurrent | **MISSING** | - | - | - | - |
| k03_causal_conv1d | **MISSING** | - | - | - | - |
| k04_mla_decode (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k05_marlin_moe_w4a16 (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k06_dequant_bench (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |
| k07_situ_act (dry) | **INCOMPLETE** | SKIP | SKIP | SKIP | SKIP |

해석: MISSING=하네스 미저작/미실행 · INCOMPLETE=FAIL 없이 SKIP 잔존(dry-run 상한) · S4 진입은 해당 커널 overall=PASS 가 조건 (docs/PLAN.md).
