# K01 (KDA chunk prefill) SM80 IMA — 원인 후보 순위표 (종합·교차검증)

작성: 2026-07-29, 렌즈1(타일/스테이징 예산)·렌즈2(인덱스 산술) 발견의 상호 반박·수치 재계산 종합.
증상: 사내 A100(SM80), 시작 112초 후 `CUDA error: an illegal memory access`, rc=3.
동일 이미지에서 k02(`fused_recurrent`) 4게이트 PASS → 스택 정상, chunk 계열 고유 문제.
스택: Triton 3.7.1 / torch 2.13.0+cu126 / A100 smem 166,912B(`fla_common/utils.py:183 Backend.AMPERE`).

---

## 0. 실행 경로 (소스 재확인 확정)

harness(`k01_kda_chunk/harness.py`)는 전 케이스 `cu_seqlens` 전달 + `lower_bound=-5.0` + `use_qk_l2norm_in_kernel=True`
→ 전 커널 `IS_VARLEN=True`, `safe_gate=True`(chunk.py:740). 케이스 순서(harness.py:163):
`b1_T8k → b1_T128k → b8_T8k_varlen → b32_T2k → state_carry_2x256`.

한 호출의 커널 순서(chunk.py:698-741 → 608-652):

| # | 커널 | 파일:라인 | autotune key (실측) |
|---|---|---|---|
| K1 | `l2norm_fwd_kernel2` | fla_common/l2norm.py:81 | 없음 (USE_DEFAULT_FLA_NORM=0 기본) |
| K2 | `kda_gate_chunk_cumsum_vector_kernel` | fla_kda/chunk.py:414 | `["H","S","BT","IS_VARLEN"]` |
| K3 | `chunk_kda_fwd_kernel_intra_sub_chunk` | fla_kda/chunk_intra.py:366 | `['BK','NC','BT','HV']` |
| K4 | `chunk_kda_fwd_kernel_inter_solve_fused` | fla_kda/chunk_intra.py:40 | BK∈{32,64} |
| K5 | `recompute_w_u_fwd_kernel` | fla_kda/chunk.py:49 | `["H","K","V","BT","BK","BV","IS_VARLEN"]` |
| K6 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (h **생산**) | fla_common/chunk_delta_h.py:47 | `["H","K","V","BT"]` |
| K7 | `chunk_gla_fwd_kernel_o` (h **소비**, `o=v` in-place, **마지막**) | fla_kda/chunk.py:251 | `["BT"]` (chunk.py:248) |

autotune key 값(H=96,K=V=128,BT=64,IS_VARLEN=True)이 5케이스 전부 동일 → **컴파일·튜닝은 케이스1(b1_T8k)에서 1회**, 이후 캐시 재사용(재튜닝 없음). `safe_gate=True`라 `chunk_kda_fwd_kernel_intra_token_parallel`은 미도달(chunk_intra.py:519-531 else 분기).

**int64 캐스트 전수 grep 결과 (결정적):** `fla_kda/chunk.py`·`chunk_intra.py`·`chunk_intra_token_parallel.py`에 `tl.int64` **0건**. `fla_common/chunk_delta_h.py`에만 11건(L99-104, 137-167, 210-212). k02가 쓰는 `fla_kda/fused_recurrent.py`는 `T/N: tl.int64` 선언 + bos/eos int64 로드(L157-158, 186-187) — **k02만 통과한 것과 정확히 정합**. 같은 vendor의 `attn_ops/triton_decode_attention.py:138`은 "page_number * page stride overflows int32" 주석과 함께 캐스트 — 이 버그 클래스는 업스트림이 이미 아는 종류이며 chunk 소비자 경로만 누락.

---

## 1. 교차검증 결과 (렌즈 간 일치·반박·보정)

| # | 항목 | 판정 |
|---|---|---|
| C1 | **주범 일치**: 렌즈1 F3 ≡ 렌즈2 발견1 = chunk.py:310 int32 오버플로. 독립 산출 수치 전 항목 일치(본 종합에서 3자 재계산 재확인, §5) | 채택 (순위 1) |
| C2 | 렌즈2 F3 목록 중 `chunk_delta_h.py:235/246/257/268`(gk last-idx): L210 `last_idx = min((i_t.to(tl.int64)+1)*BT, T)-1`이 **int64** → `(bos+last_idx)*H*K`는 i64 승격 연산 → **어떤 T에서도 안전. 패밀리에서 제외** | 렌즈2 보정 |
| C3 | 렌즈1의 "K7 autotune key = H,K,V,BT,BK,BV,IS_VARLEN" 서술은 부정확 — 실제 `key=["BT"]`(chunk.py:248). 단 결론(케이스 간 재튜닝 없음)은 동일 성립 | 렌즈1 보정 (결론 불변) |
| C4 | 렌즈2의 발화 임계 "총 토큰 ≥ 87,382"는 연속 근사 — 정확 임계는 **글로벌 청크 인덱스 1365 존재(=NT_total ≥ 1366) ∧ i_h ≥ 32**, 단일 seq로는 **T ≥ 87,361** (재계산: NT=1365(T=87,360)까지 max off 2,146,942,976 < 2^31 안전) | 렌즈2 보정 |
| C5 | 렌즈1 F1(smem 초과 config): K7 BK64/BV128/ns4 ≈ 188,416B > 166,912B는 실재하나 Triton 3.x autotuner가 컴파일단 `OutOfResources/PTXASError`를 잡아 inf 처리·배제(런칭 자체 불가) → **IMA로 전이 불가**. vendor에 `prune_configs_by`/smem 게이팅 전무한 구조 지적은 유효(성능·이식성 이슈) | 크래시 후보 기각, 참고 강등 |
| C6 | 렌즈1 F2(H=96 배수) 기각 = 렌즈2 배제와 정합. 도달 커널 전부 head는 스칼라 인덱스(`i_bh % H` 등), head축 타일 로드 없음 | 기각 확정 |
| C7 | 렌즈2 배제1(varlen len=2): inter_solve는 `if i_tc1/2/3 < T` 가드 + 전면 OOB 블록은 `boundary_check` 전량 마스크(L127-245), sub_chunk는 `if i_ti >= T: return`(chunk_intra.py 확인), 전진대입 `range(2, min(BC, T-i_tc0))`는 빈 루프 — 소스 재확인으로 유지 | 기각 유지 |
| C8 | **신규 (본 종합 발견)**: harness.py:213 `add_evidence("cases", case_meta)`가 **루프 종료 후에만** 실행 — sticky IMA면 `except RuntimeError`(L205) 후 `finally: torch.cuda.empty_cache()`(L211)가 오염 컨텍스트에서 재발화 → 루프 탈출 → main except(L367) `EXIT_ERR=3`. 이때 cases evidence 미기록 → **25-diag의 D7 covered가 사망 케이스와 무관하게 ∅(covered(0))로 나올 수 있음**. rc=3 관측과 정합하는 유일 경로가 바로 이 crash-out | §2 판독표에 반영 + 패치 P4 |

---

## 2. 원인 후보 순위표

| 순위 | 후보 | 근거 수치 | 죽을 케이스 예측 | 사내 1줄 검증법 | 수정 방향 |
|---|---|---|---|---|---|
| **1** | **`chunk.py:310` `chunk_gla_fwd_kernel_o`의 h 베이스 오프셋 int32 오버플로** — `h + (i_tg * H + i_h) * K * V`, i_tg=program_id(1) 유래 i32, 캐스트 전무. 생산자 K6는 int64 방어(chunk_delta_h.py:99,137) → **생산 성공·소비만 wrap** 비대칭 | h = `k.new_empty(B,NT,H,V,K)` bf16(chunk_delta_h.py:354); b1_T128k: NT=2048 → numel **3,221,225,472 > 2^31**(6.00GiB, 유일 초과 텐서). 오프셋 (i_tg·96+i_h)·16384: **i_tg=1365,i_h=32에서 정확히 2^31** → wrap −2,147,483,648 elem = **h−4.00GiB**; 최대(2047,95) 3,221,209,088 → −2.00GiB. 위반 쌍 **65,536/196,608 = 정확히 1/3**, 전부 [h−4GiB, h−2GiB) 접근. block_ptr shape (V,K)=(128,128) 기준 전 lane in-bounds 판정 → 마스크 없이 풀 로드 발행 = IMA | **b1_T128k 단독**(2번째 케이스), 그 안에서도 **7커널 중 마지막 K7**. 타 케이스 max off: b1_T8k 0.094× / b8 0.751~0.752× / b32 0.750× / state_carry 0.006× — 전부 안전. 시그니처: faulting kernel=`chunk_gla_fwd_kernel_o`, fault addr = h베이스 −2~−4GiB, 최초 위반 program_id(1)=1365. 타이밍: 케이스1(튜닝+참조+PERF 60회)+케이스2 입력생성(1.61e9 elem×5 CPU randn)+선행 6커널 ≈ 112s 정합 | 단일 seq **A/B 1줄**: T=87,360(NT=1365)→통과 vs **T=87,424(NT=1366)→IMA** (§4 스니펫). 또는 b1_T128k에 `CUDA_LAUNCH_BLOCKING=1 compute-sanitizer --tool memcheck` → 커널명 확인 | 패치 P1 (1줄, 곱하기 **전** `.to(tl.int64)`) |
| **2** | **`chunk_delta_h.py:99-104` 캐스트 순서 오류(무효 방어)** — `h += ((boh*H + i_h) * V * K).to(tl.int64)`: i32 곱이 wrap된 **뒤** 캐스트. L100-102,104(bos 기반)도 동일 패턴 | h 라인 발화 조건: boh ≥ 1365(i_h≥32) = 선행 누적 청크 1365개(full-64 기준 선행 ~87,360 tok). 현 케이스 max boh: b32=992 / b8=771(로컬)~815(과제 문안) / b1은 0 — **전부 미발화**. bos 라인들은 1.61e9 스케일(후보 3과 동일 임계) | **현 5케이스 어디서도 안 죽음** (이번 크래시 원인 아님 확정적). varlen 누적 ≥ 1366청크 케이스 추가 시 후보1과 동일 임계에서 생산자·소비자 동시 발화 | (P1 적용 후) `cu_seqlens=[0, 87360, 87424]` 2-seq 1회 → 죽으면 본 후보 잔존 확인 | 패치 P2 (5줄 캐스트 순서 교정) |
| **3** | **1.61e9 i32 오프셋 패밀리 (잠복 감시)** — per-token 스케일 raw i32 산술: `chunk.py:174`(`gk + ((bos+last_idx)*H + i_h)*K + o_k`, last_idx i32), K2/K3/K4/K5/K7의 전 block-ptr 베이스 `(bos*H+i_h)*K` 류 + 행 오프셋 `(i_t·BT)·H·K` 류, `chunk_intra.py:93-98/141/159/183/261-279`, `l2norm.py:90`(`X + rindex + N*row_idx`, b1_T128k 행수 12,582,912) | b1_T128k 최대 **1,610,612,608 = 0.750×2^31** (여유 1.33×). 일제 발화 임계: **총 토큰 ≥ 174,763** (T·H·K ≥ 2^31; l2norm M·N도 동일 지점). harness의 int32_overflow_watch 주석(L216-221)은 이 1.61e9만 감시하고 **h(3.22e9, 실제 최초 임계 87,361 tok)를 누락** — 감시 상한 자체가 오설정 | 현 5케이스 없음. 하네스 주석 근거상 ≥174,763 tok 단일호출은 A100 80GB OOM이라 실행 불가 — 실기 발화 경로는 사실상 없음(장래 H100 96GB+ 또는 케이스 확장 시 위험) | (실행 불요) 정적 확인만 — grep로 `.to(tl.int64)` 0건 재확인 | 패치 P3 (베이스 i64 선-advance 위생; 별도 PR 권고) + harness 주석 정정 |
| 4 (기각·참고) | K7/K6 smem 한계 autotune config — K7 BK64/BV128/ns4 스테이징 ≈ 188,416B > 166,912B, K6 BV64/ns4 ≈ 165,888B(여유 1,024B) | Triton 3.x autotuner `_bench`가 `(OutOfResources, CompileTimeAssertionFailure, PTXASError)` 포착 → 해당 config inf 배제, **런칭 자체 불발 → IMA 전이 불가**. 차선 BK64/BV64/ns4=147,456B 등 통과 config 존재 | **죽지 않음**. 이 계열이 원인이려면 b1_T8k **튜닝 창(시작 후 첫 수십 초)** 사망이어야 — 112s와 불합치 | `TRITON_PRINT_AUTOTUNING=1`로 b1_T8k 재실행 → 선택 config 확인 | (선택) SM80용 config pruning — 성능 최적화 차원 |
| 5 (기각) | H=96 비배수 / varlen len=2<BC 경계 / raw_beta 비연속 stride / grid 한계 | head 전부 스칼라 인덱스·타일 로드 무 / `if i_tc<T` 가드+전량 마스크+빈 루프 / stride 3종 명시 전달(chunk.py:579-581Δ→커널 L428-430,460-468) / 최대 grid y=2048·z=3072 < 65,535 | 해당 없음 (전 케이스 안전) | — | 불요 |

확신도: 후보1 = 버그 실재 **확실**(산술 전수 재검산·소스 3자 확인), 이번 크래시의 원인 **0.85+** — 잔여 불확실성은 (a) 112s가 정말 b1_T128k인가(diag 회신 대기), (b) wrap 주소 [h−4GiB, h−2GiB)가 우연히 전부 매핑돼 침묵 부패로 빠질 가능성(2GiB 연속 창 전부가 매핑돼야 하므로 희박; 관측이 IMA이므로 정합) 뿐.

---

## 3. diag 회신 D7(covered) 값별 판독표

D7 = `run/25-diag.sh`가 리포트 `evidence.cases`에서 error 없는 케이스를 나열한 것.
**전제 주의(C8):** sticky IMA crash-out 경로에서는 cases evidence 자체가 미기록 → **D7이 `covered(0)`이어도 "b1_T8k에서 죽었다"의 증거가 아님**. 이때는 D5(CRASH 문자열)·D9(로그 에러줄)·케이스 duration으로 보조 판별. 패치 P4 적용 후 재실행하면 D7이 진짜 판별력을 가진다.

| D7(covered) 회신 값 | 판독 | 액션 |
|---|---|---|
| `covered(0)` **+ D5 CRASH 있음** | **판별 불능(C8) — 예측과 모순 아님.** crash-out으로 기록 유실. 112s(≫ 케이스1 튜닝 창)는 b1_T128k 사망을 지지 | §4 A/B 1줄(87,360 vs 87,424)로 즉시 확정. P4 패치 후 재실행하면 D7 복원 |
| `covered(0)` + D5 없음 + 리포트 자체 없음(D0) | b1_T8k 이전(입력 생성/import/튜닝) 사망 — 후보1 기각(0.094×). 렌즈1 F1 계열 재조사 대상이나 OOR-배제 논리와 모순 → 양 렌즈 무설명 | compute-sanitizer + `TRITON_PRINT_AUTOTUNING=1` 재실행 회수 |
| `covered(1): b1_T8k` | **후보 1 확정** (예측 정중앙 적중) | 패치 P1 적용 → harness 재실행 → 4게이트 재판정 |
| `covered(2): b1_T8k,b1_T128k` | 후보1의 **크래시-원인** 지위 반증… 이려면 b1_T128k가 '실행 완료'여야 하는데, wrap 산술은 확실하므로 이 경우 = **침묵 부패로 완주**(b1_T128k/o NUMERIC 오차 폭발 동반해야 함) + 사망은 b8 고유 원인. 단 112s 안에 b1_T128k의 131k-step fp32 참조 계산까지 끝나는 건 시간상 거의 불가 → 이 회신 자체가 타이밍 모순 | b1_T128k/o의 rmse 값 회수(침묵부패 검증) + 사내 실행본 `CASE_LENGTHS` 회수(과제 문안 15,**2**,…,23000,13547 vs 로컬 15,8192,…,16261 불일치 확인됨 — 어느 쪽이든 오버플로 산술 결론 불변) + sanitizer | 
| `covered(3): …,b8_T8k_varlen` (b32 사망) | 양 렌즈 무설명 (b32 오프셋 최대 0.750×, grid (1024,3072) 한계 내, bos 최대 63,488 안전) | sanitizer 필수 — 새 원인 탐색 |
| `covered(4): …,b32_T2k` (state_carry 사망) | 양 렌즈 무설명 (T=512, h0/ht 오프셋 `i_nh*V*K` ≤ 1.57e6) | sanitizer 필수 |
| `covered(5)` 전부 | 크래시가 케이스 루프 밖(SANITIZER/PERF/리포트 단계) | trace_tail 회수 |

**사내 1줄 회신 요청문(그대로 전달 가능):** "`bash run/25-diag.sh k01_kda_chunk` 출력 전체 + k01 로그에서 케이스별 시작~에러 타임스탬프(가능하면), 그리고 사내 실행본 harness.py의 CASE_LENGTHS['b8_T8k_varlen'] 값."

---

## 4. 최소 수정 패치 스케치

### P1 — 주범 (chunk.py:310, 1줄)
```diff
--- a/kernel-kit/vendor/fla_kda/chunk.py
+++ b/kernel-kit/vendor/fla_kda/chunk.py
@@ -307,9 +307,9 @@ def chunk_gla_fwd_kernel_o(
         p_h = tl.make_block_ptr(
-            h + (i_tg * H + i_h) * K * V,
+            h + (i_tg * H + i_h).to(tl.int64) * K * V,
             (V, K),
             (K, 1),
```
근거: `i_tg*H + i_h ≤ 196,607`(i32 안전)까지 계산 후 **곱하기 전** i64 승격 — 생산자 chunk_delta_h.py:137 `i_t.to(tl.int64) * stride_h` 패턴과 동일. 상계 3,221,209,088 < 2^63.

### P2 — 잠복 동계열 (chunk_delta_h.py:99-104, 캐스트 순서 교정)
```diff
--- a/kernel-kit/vendor/fla_common/chunk_delta_h.py
+++ b/kernel-kit/vendor/fla_common/chunk_delta_h.py
@@ -96,10 +96,10 @@
     # calculate offset
-    h += ((boh * H + i_h) * V * K).to(tl.int64)
-    v += ((bos * H + i_h) * V).to(tl.int64)
-    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
-    w += ((bos * H + i_h) * K).to(tl.int64)
+    h += (boh * H + i_h).to(tl.int64) * V * K
+    v += (bos * H + i_h).to(tl.int64) * V
+    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
+    w += (bos * H + i_h).to(tl.int64) * K
     if SAVE_NEW_VALUE:
-        v_new += ((bos * H + i_h) * V).to(tl.int64)
+        v_new += (bos * H + i_h).to(tl.int64) * V
```
내부 합(`boh*H+i_h` ≤ ~1.3e7)은 i32 안전 — 스케일 곱만 i64로. 현 케이스 미발화지만 P1과 같은 임계(누적 1366청크)에서 터지는 시한폭탄.

### P3 — 1.61e9 패밀리 위생 (방어적, 별도 PR 권고)
발화 임계 총 T ≥ 174,763 (A100 80GB에선 OOM으로 실행 불가 — 즉시 위험 아님). 대표 diff 2곳 + 원칙:
```diff
# chunk.py:174 (recompute_w_u STORE_KG)
-            b_gn = tl.load(
-                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
-            )
+            b_gn = tl.load(
+                gk + ((bos + last_idx) * H + i_h).to(tl.int64) * K + o_k,
+                mask=m_k, other=0.0,
+            )
# l2norm.py:90,94 (kernel2)
-    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
+    xs = tl.load(X + (rindex + N * row_idx.to(tl.int64)), mask, other=0.0).to(tl.float32)
     ...
-    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)
+    tl.store(Y + (rindex + N * row_idx.to(tl.int64)), xs * rsqrt, mask)
```
원칙: block-ptr 경로는 베이스 raw 산술만 고쳐선 부족 — 행 오프셋 `(i_t·BT)·(H·K)`가 block_ptr 내부 i32로 계산되므로, 총 T>174k 지원 시 **베이스를 `(bos*H+i_h).to(tl.int64)*K + (i_t*BT).to(tl.int64)*H*K`로 선-advance하고 block_ptr는 청크-상대 offset**으로 재작성해야 한다(업스트림 fla 방식). 동시에 harness L216-221 int32_overflow_watch 주석을 "최초 임계 = chunk-state h(NT·H·V·K) 3.22e9, 87,361 tok"로 정정.

### P4 — 진단성 (harness.py, diag 재발 방지; C8 대응)
```diff
--- a/kernel-kit/k01_kda_chunk/harness.py
+++ b/kernel-kit/k01_kda_chunk/harness.py
@@ -163,6 +163,7 @@ def gpu_run(r: report.KernelReport, a) -> None:
     order = [PERF_CASE] + [c for c in CASE_LENGTHS if c != PERF_CASE] + list(EXTRA_CASES)
     for name in order:
+        print(f"[{KID}] case {name} start", flush=True)   # 사망 케이스 로그 특정용
         seed = shapes.stable_seed(KID, name)
         t0 = time.time()
         try:
@@ -196,6 +197,7 @@ def gpu_run(r: report.KernelReport, a) -> None:
             results.extend(entries)
             covered.append(name)   # 실행 완료 = 커버 (오차 판정은 NUMERIC 게이트 몫)
             case_meta[name] = {"seed": seed, "duration_s": round(time.time() - t0, 1),
                                "entries": [e.metrics.get("case") for e in entries]}
+            r.add_evidence("cases", dict(case_meta))   # 케이스마다 갱신 — crash-out에도 보존
```
(+ 선택: `finally: empty_cache()`를 try/except로 감싸 원 예외 보존, run_kernel 직후 `torch.cuda.synchronize()`로 IMA 표면화 지점 고정.)

### 검증 스니펫 — 후보1 A/B 1줄 (사내 즉시 실행용)
```bash
cd <repo>/kernel-kit && python3 - <<'EOF'
# A: T=87,360 (NT=1365, max off 2,146,942,976 < 2^31) → 통과해야 함
# B: T=87,424 (NT=1366, i_tg=1365·i_h>=32에서 wrap) → IMA면 후보1 확정
import torch
from vendor import fla_kda as m
for T in (87360, 87424):
    d = "cuda"
    f = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=torch.float32).to(dt).to(d)
    o, ht = m.chunk_kda_with_fused_gate(
        q=f(1,T,96,128), k=f(1,T,96,128), v=f(1,T,96,128), raw_g=f(1,T,96,128),
        raw_beta=f(1,T,96), A_log=f(96, dt=torch.float32)*0.5,
        g_bias=f(96*128, dt=torch.float32)*0.1, lower_bound=-5.0,
        initial_state=torch.zeros(1,96,128,128, device=d), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0,T], dtype=torch.int32, device=d))
    torch.cuda.synchronize(); print(f"T={T}: OK")
EOF
# 기대: "T=87360: OK" 출력 후 T=87424에서 illegal memory access → 후보1 확정
```

---

## 5. 수치 부록 (본 종합의 독립 재계산 — 렌즈 수치 대조 완료)

h(chunk-state) 텐서와 K7 h-오프셋 (H=96, K=V=128, BT=64, K·V=16,384; 2^31=2,147,483,648):

| 케이스 | ΣT | NT_total | h.numel | h-오프셋 최대 | ×2^31 | 판정 |
|---|---|---|---|---|---|---|
| b1_T8k | 8,192 | 128 | 201,326,592 | 201,310,208 | 0.094 | 안전 |
| **b1_T128k** | 131,072 | 2,048 | **3,221,225,472 (>2^31)** | **3,221,209,088 → i32 −1,073,758,208** | **1.500** | **WRAP** |
| b8 (로컬 harness.py:74) | 65,536 | 1,026 | 1,613,758,464 | 1,613,742,080 | 0.751 | 안전 |
| b8 (과제 문안 길이) | 65,536 | 1,027 | 1,615,331,328 | 1,615,314,944 | 0.752 | 안전 |
| b32_T2k | 65,536 | 1,024 | 1,610,612,736 | 1,610,596,352 | 0.750 | 안전 |
| state_carry(512) | 512 | 8 | 12,582,912 | 12,566,528 | 0.006 | 안전 |

- 최초 위반: `(1365·96+32)·16384 = 2,147,483,648 = 2^31` 정확 일치 → i32 −2^31 elem = h−4.00GiB(bf16). 위반 구간 elem [−2,147,483,648, −1,073,758,208] = byte **[h−4.00GiB, h−2.00GiB)**, 위반 (i_tg,i_h) 쌍 65,536개(정확히 1/3).
- 단일 seq 발화 임계 **T ≥ 87,361**; varlen 일반형 **NT_total ≥ 1,366**.
- b1_T128k에서 2^31을 넘는 산술은 도달 커널 전체를 통틀어 **chunk.py:310이 유일** (per-token 최대 1,610,612,608 = 0.750×; K6의 h는 int64 방어, boh=0).
- 1.61e9 패밀리 일제 발화 임계: 총 T ≥ **174,763** (T·H·K ≥ 2^31; l2norm 행수 M=T·96 ≥ 16,777,216 동일 지점).
- 참고 dtype: q/k/v/w/u/kg/h/Aqk/Akk = bf16, g(cumsum 후)/Akkd/h0/ht = fp32.

미해소 잔여(사내 회신 필요): ① 112s 사망 케이스 확정(D7 or 로그) ② 사내 실행본 b8 길이 목록(문안·로컬 불일치 — 결론 불변이나 실행본 특정용) ③ (선택) faulting kernel명/주소 — 후보1 시그니처 대조.
