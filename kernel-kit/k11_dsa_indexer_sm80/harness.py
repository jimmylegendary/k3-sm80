#!/usr/bin/env python3
"""k11_dsa_indexer_sm80 — K3.1 DSA indexer MQA logits (SM80 Triton) 단독 검증.

대상 (vendor/dsa_ops = **이미지에 실리는 패치본**: vLLM a3c14efb + docker/patches-k31/triton-mla-sparse-sm80.patch
      (sha256 74675e99…7ed8), 원천 PR #47629 head 064801dd · 원저자 haosdent/thomaslwang — PR 원본이 아니다):
  mqa_logits_triton.fp8_paged_mqa_logits_triton  decode 형 paged logits (K 를 uint8 LUT 로 in-kernel 디코드)
  mqa_logits_triton.fp8_mqa_logits_triton        prefill 형 dense chunk logits ([ks, ke) 구간 마스크)
  vLLM 호출 형태는 PR 의 sparse_attn_indexer.py 분기를 그대로 재현한다: clean_logits=False, decode 는
  max_model_len=배치 활성 최대 길이, prefill 은 질의 청크 × 전체 K. top-k 는 stock CUDA op(`top_k_per_row_*`,
  `persistent`)라 이 게이트의 대상이 아니다 — 이 하네스는 logits 의 top-k 소비 가능성(집합 일치)을 본다.

입력 생성 = K3.1 indexer 와 같은 경로 (common/dsa_ref.py 원천 주석 참조):
  index_k → fp32 LayerNorm(weight, bias)→bf16 → ue8m0 scale + E4M3 코드 → 페이지 캐시 [bs×128 fp8 | bs×4B scale]
  index_q/weights → head 별 ue8m0 E4M3 + 가중치 접기(w·scale·D^-0.5·H^-0.5).
  커널과 오라클은 **동일한 양자화 입력**을 공유한다 → 비교에는 커널 오차만 들어간다(양자화 오차는 evidence 로만).

판정 (docs/PLAN.md §S2 4게이트):
  NUMERIC   ① logits: 유효 구간을 rms(ref)로 정규화해 tolerance.TOL[k11] (rmse_rel ≤ 1e-4, max_abs ≤ 1e-3) +
              -inf 마스크 위치 완전 일치(NaN/inf 즉시 FAIL)
            ② **top-k 집합 일치율**: 행마다 K=min(2048, 유효 수), τ=ref 의 K번째 값, 경계 동률 밴드
              δ=TOPK_TIE_BAND_REL(1e-3)·rms(ref). 커널 상위 K 가 τ−δ 미만을 뽑거나(fp) τ+δ 초과를 놓친(fn)
              확정 불일치 = max(|fp|,|fn|). 일치율 = 1 − 불일치/K ≥ TOPK_AGREEMENT_MIN(1.0) 이어야 PASS.
              근거: 커널·오라클이 같은 양자화 입력이라 편차는 fp32 합산 순서(~1e-6)뿐이고 δ 는 그 ~1000배 —
              밴드 밖 불일치가 하나라도 있으면 디코드/스케일/주소 버그다. 밴드 안 교환은 동률로 허용.
            ③ clean_logits=True 와 False 가 유효 구간에서 비트 동일(PR 회귀 테스트와 동형)
  SHAPES    REQUIRED_SHAPES 11케이스 (decode b1/8 × ctx 8k/128k/256k + strided-pool 1 + prefill m256 × n 8k/128k/256k
            + varlen) 전부 실행 완료. decode 길이는 ctx−1−(i·37 mod 61) 지터로 마지막 블록을 항상 부분 블록으로 만든다.
            dec_b8_ctx256k_pool27x 는 블록 stride 를 27배로 벌려(≈7.5 GB 풀) packed hybrid KV row 안의 index-K strided
            view 를 모사 — 블록 오프셋이 2^31 B 를 넘는 구간의 int64 주소 계산(`block_idx.to(tl.int64)`)을 본다.
            정직한 상한: 패치본이 PR 원본에 더한 logits 행/타일 오프셋 int64 승격(`token_id`·`m`·`n_start`)은 logits 버퍼가
            2^31 원소(fp32 ≈ 8.6 GB)를 넘어야 발현하는데 REQUIRED 케이스(최대 prefill 256×256k)는 그에 한참 못 미친다 —
            이 승격은 소스 마커 수(tests/kernel/test_dsa_ops_vendor.py)로만 고정되고 수치로는 측정하지 않는다.
  SANITIZER compute-sanitizer memcheck — 소형 child 2종(decode b2·ctx1000·next_n2 / prefill varlen m70·n300)
  PERF      **측정 전용**(PERF_POLICY k11 measure_only, cv_max 0.15): 대표 케이스 dec_b1_ctx256k 의 반복 안정성(CV)
            만 판정, 전 케이스 median·루프라인 효율은 evidence. 속도 floor 는 A100 실측 후 사전등록.

모드:
  (기본)          GPU parent/child — run_all.sh 가 부른다.
  --cpu-dry-run   커널 미실행. 순수 python 참조·판정 자체점검 + 게이트 전부 SKIP(INCOMPLETE 상한).
  --interpret     TRITON_INTERPRET=1 CPU 인터프리터로 vendored 커널을 소형 shape 에서 실제 실행해 같은 판정 코드로
                  fp32 오라클과 대조 → evidence.interpret_cpu 에만 기록, 게이트는 SKIP 유지(**CPU 대조는 PASS 근거 아님**).

exit: 0 FAIL 없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import dsa_gate, dsa_ref, env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k11_dsa_indexer_sm80"
PINS = {
    "vllm_pin": "a3c14efb",
    "dsa_ops": "a3c14efb+triton-mla-sparse-sm80.patch(sha256 74675e99…7ed8; origin PR#47629@064801dd haosdent/thomaslwang)",
    "hf_k31": "moonshotai/Kimi-K3.1-Pro@d9e99b07",
}
KERNEL_MODULE = ("vendor/dsa_ops = 배포 패치본(vLLM a3c14efb + docker/patches-k31/triton-mla-sparse-sm80.patch "
                 "sha256 74675e99…7ed8; 원천 PR #47629 head 064801dd)")
KERNEL_FILES = ("mqa_logits_triton.py",)
CHILD_SCHEMA = "k11-child/v1"
CASE_ORDER = tuple(shapes.DSA_INDEXER_REQUIRED_CASES)
PERF_CASE = "dec_b1_ctx256k"

HEADS, HEAD_DIM, TOPK = shapes.DSA_INDEX_HEADS, shapes.DSA_INDEX_HEAD_DIM, shapes.DSA_TOPK
BLOCK = shapes.DSA_BLOCK_SIZE
WEIGHT_SCALE = HEAD_DIM ** -0.5 * HEADS ** -0.5          # index_softmax_scale · indexer_weight_scale (dsa.py L135·L189·L373)
K_NORM_EPS = 1e-6      # 예시값 — 실제는 text_config.rms_norm_eps (recon/k31-config 미기재, 커널 비교엔 영향 없음)

# 소형 sanitizer 케이스 (REQUIRED 아님 — SHAPES 에 안 들어간다)
SANITIZER_CASES = {
    "san_decode_b2_ctx1000_n2": dict(mode="decode", b=2, ctx=1000, next_n=2, topk=64),
    "san_prefill_m70_n300": dict(mode="prefill", m=70, n=300, varlen=True, topk=64),
}
# --interpret (CPU 인터프리터) 소형 케이스: 커널 차원(H=32, D=128, block 64)은 실물 그대로, ctx/배치/topk 만 축소
INTERPRET_CASES = {
    "int_decode_b2_ctx512_n1": dict(mode="decode", b=2, ctx=512, next_n=1, topk=64),
    "int_decode_b2_ctx300_pool3x": dict(mode="decode", b=2, ctx=300, next_n=1, topk=64, pool_stride_x=3),
    "int_decode_b1_ctx130_n2": dict(mode="decode", b=1, ctx=130, next_n=2, topk=32),
    "int_prefill_causal_m16_n192": dict(mode="prefill", m=16, n=192, topk=64),
    "int_prefill_varlen_m16_n256": dict(mode="prefill", m=16, n=256, varlen=True, topk=64),
}
QUANT_EFFECT_CASES = ("dec_b1_ctx8k", "pre_m256_n8k")      # 양자화 영향(정보용) 측정 대상


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ===========================================================================
# 케이스 실행 (GPU child 와 --interpret 가 같은 코드를 쓴다)
# ===========================================================================
def decode_lengths(cfg: dict) -> list[int]:
    """시퀀스별 컨텍스트 길이 — 마지막 블록이 항상 부분 블록이 되도록 ctx−1−(i·37 mod 61)."""
    return [max(cfg.get("next_n", 1) + 1, cfg["ctx"] - 1 - ((i * 37) % 61)) for i in range(cfg["b"])]


def _make_k_params(torch, gen, device):
    w = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32, device=device, generator=gen))
    b = 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32, device=device, generator=gen)
    return w, b


def _compare_logits(torch, ref, out, limit, case: str) -> tolerance.NumericResult:
    """logits 비교: 유효(ref 유한) 구간을 rms(ref)로 정규화해 TOL 적용 + -inf 마스크 위치 일치."""
    r, n = ref.shape
    if limit is not None:
        region = torch.arange(n, device=ref.device)[None, :] < limit.to(ref.device)[:, None]
    else:
        region = torch.ones_like(ref, dtype=torch.bool)
    ref_ninf = region & torch.isinf(ref) & (ref < 0)
    out_ninf = region & torch.isinf(out) & (out < 0)
    mask_diff = int((ref_ninf != out_ninf).sum())
    fin = region & torch.isfinite(ref)
    rv, ov = ref[fin], out[fin]
    rms = float(rv.pow(2).mean().sqrt()) if rv.numel() else 1.0
    rms = rms if rms > 0 else 1.0
    res = tolerance.compare(KID, rv / rms, ov / rms, case=case)
    res.metrics.update(ref_rms=rms, n_valid=int(fin.sum()), mask_diff=mask_diff)
    if mask_diff:
        res.violations.append(f"-inf 마스크 위치 불일치 {mask_diff}칸")
        res.passed = False
    return res


def _topk_entry(torch, ref, out, limit, topk: int, case: str) -> dict:
    tk = dsa_ref.topk_agreement(ref, out, topk, limit=limit)
    viol = []
    if tk["min_agreement"] < dsa_ref.TOPK_AGREEMENT_MIN:
        viol.append(f"top-k 일치율 {tk['min_agreement']:.6f} < {dsa_ref.TOPK_AGREEMENT_MIN} "
                    f"(밴드 밖 확정 불일치 {tk['mismatch_total']}건 / {tk['rows_with_mismatch']}행)")
    return {"case": case, "metrics": tk, "violations": viol, "passed": not viol}


def _entry(res: tolerance.NumericResult) -> dict:
    return {"case": res.metrics.get("case"), "metrics": res.metrics, "violations": res.violations,
            "passed": res.passed}


def _true_logits(torch, q_raw, w_raw, y):
    """양자화 없는 fp32 logits (정보용): Σ_h (w·scale_w)·ReLU(q·y) — [rows,H,D]·[N,D]."""
    s = torch.einsum("rhd,nd->rhn", q_raw.float(), y.float()).relu_()
    return (s * (w_raw.float() * WEIGHT_SCALE)[:, :, None]).sum(1)


def run_case(cfg: dict, case: str, device: str, do_timing: bool, warmup: int, reps: int, sanity: bool = True,
             kernel_only: bool = False) -> dict:
    """케이스 하나: 입력 생성 → 커널 실행 → (오라클 대조·top-k 일치·타이밍). kernel_only=True 는 sanitizer child 용 —
    memcheck 하에서 불필요한 torch 오라클 연산을 돌리지 않고 커널(clean_logits 두 모드)만 실행한다."""
    import torch
    from vendor.dsa_ops import fp8_mqa_logits_triton, fp8_paged_mqa_logits_triton
    seed = shapes.stable_seed(KID, case)
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    topk = cfg.get("topk", TOPK)
    on_cuda = device.startswith("cuda")
    t_build = time.monotonic()
    nw, nb = _make_k_params(torch, gen, device)
    entries: list[dict] = []
    extra: dict = {"seed": seed, "topk": topk}
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False

    if cfg["mode"] == "decode":
        b, nn = cfg["b"], cfg.get("next_n", 1)
        lens = decode_lengths(cfg)
        nblk = [_cdiv(length, BLOCK) for length in lens]
        total = sum(nblk) + 2                                   # 미사용 블록 2개(독극물 채움)
        perm = torch.randperm(total, generator=cpu_gen)
        bt = torch.zeros((b, max(nblk)), dtype=torch.int32)
        o = 0
        for i, n_i in enumerate(nblk):
            bt[i, :n_i] = perm[o:o + n_i].to(torch.int32)
            o += n_i
        bt = bt.to(device)
        seqs, ys = [], []
        for length in lens:
            raw = dsa_gate.gpu_randn_bf16((length, HEAD_DIM), gen, device)
            y = dsa_ref.k_layer_norm(raw, nw, nb, K_NORM_EPS)
            seqs.append(dsa_ref.quantize_index_k(y))
            ys.append(y)
        row_bytes = BLOCK * (HEAD_DIM + 4)
        stride_bytes = row_bytes * cfg["pool_stride_x"] if cfg.get("pool_stride_x") else None
        cache = dsa_ref.pack_index_k_cache(seqs, bt, total, BLOCK, device=device, stride_bytes=stride_bytes)
        if stride_bytes:
            extra.update(pool_stride_bytes=stride_bytes, max_block_offset_bytes=(total - 1) * stride_bytes,
                         int32_overflow_region=(total - 1) * stride_bytes > 2**31)
        q_raw = dsa_gate.gpu_randn_bf16((b * nn, HEADS, HEAD_DIM), gen, device)
        w_raw = dsa_gate.gpu_randn_bf16((b * nn, HEADS), gen, device)
        q_codes, w_fold = dsa_ref.quantize_index_q(q_raw, w_raw, WEIGHT_SCALE)
        q8 = q_codes.view(b, nn, HEADS, HEAD_DIM).view(torch.float8_e4m3fn)
        ctx_lens = torch.tensor(lens, dtype=torch.int32, device=device)
        mml = max(lens)

        def call(clean: bool):
            return fp8_paged_mqa_logits_triton(q8, cache, w_fold, ctx_lens, bt, mml, clean_logits=clean)

        extra.update(lens_head=lens[:4], max_model_len=mml, blocks=total)
        build_s = time.monotonic() - t_build
        out = call(False)
        if kernel_only:
            call(True)
            if on_cuda:
                torch.cuda.synchronize()
            return {"schema_version": CHILD_SCHEMA, "case": case, "shape": dict(cfg), "status": "PASS",
                    "entries": [], "timing": None, "extra": {**extra, "kernel_only": True}}
        if on_cuda:
            torch.cuda.synchronize()
        ref = dsa_ref.paged_mqa_logits(q_codes.view(b, nn, HEADS, HEAD_DIM), w_fold, cache, ctx_lens, bt, mml)
        limit = ctx_lens.repeat_interleave(nn)
        if sanity:
            out_clean = call(True)
            region = torch.arange(mml, device=device)[None, :] < limit[:, None]
            same = bool(torch.equal(torch.where(region, out, torch.zeros_like(out)),
                                    torch.where(region, out_clean, torch.zeros_like(out_clean))))
            extra["clean_logits_equal_on_valid"] = same
            del out_clean
        if case in QUANT_EFFECT_CASES and b == 1 and nn == 1:
            tl = _true_logits(torch, q_raw, w_raw, ys[0])
            tq = torch.full_like(tl, float("-inf"))
            tq[:, :lens[0]] = ref[:, :lens[0]]
            extra["quantization_effect"] = dsa_ref.topk_agreement(tl, tq, topk)
            extra["quantization_effect"]["note"] = "정보용: FP8 양자화(q·k) vs 무양자화 fp32 의 top-k — 게이트 아님"
    else:
        m, n = cfg["m"], cfg["n"]
        raw = dsa_gate.gpu_randn_bf16((n, HEAD_DIM), gen, device)
        y = dsa_ref.k_layer_norm(raw, nw, nb, K_NORM_EPS)
        k_codes, k_scale = dsa_ref.quantize_index_k(y)
        q_raw = dsa_gate.gpu_randn_bf16((m, HEADS, HEAD_DIM), gen, device)
        w_raw = dsa_gate.gpu_randn_bf16((m, HEADS), gen, device)
        q_codes, w_fold = dsa_ref.quantize_index_q(q_raw, w_raw, WEIGHT_SCALE)
        idx = torch.arange(m, device=device, dtype=torch.int64)
        if cfg.get("varlen"):
            ks = ((idx * 977) % max(1, n // 4)).to(torch.int32)
            ke = torch.minimum(ks.long() + 1 + (idx * 1237) % max(1, n // 2),
                               torch.full_like(idx, n)).to(torch.int32)
        else:
            ks = torch.zeros(m, dtype=torch.int32, device=device)
            ke = (n - m + 1 + idx).to(torch.int32)             # 질의 = 컨텍스트의 마지막 m 토큰 (인과)
        q8 = q_codes.view(torch.float8_e4m3fn)
        k8 = k_codes.view(torch.float8_e4m3fn)

        def call(clean: bool):
            return fp8_mqa_logits_triton(q8, (k8, k_scale), w_fold, ks, ke, clean_logits=clean)

        extra.update(ks_head=ks[:4].tolist(), ke_head=ke[:4].tolist())
        build_s = time.monotonic() - t_build
        out = call(False)
        if kernel_only:
            call(True)
            if on_cuda:
                torch.cuda.synchronize()
            return {"schema_version": CHILD_SCHEMA, "case": case, "shape": dict(cfg), "status": "PASS",
                    "entries": [], "timing": None, "extra": {**extra, "kernel_only": True}}
        if on_cuda:
            torch.cuda.synchronize()
        ref = dsa_ref.mqa_logits(q_codes, w_fold, k_codes, k_scale, ks, ke)
        limit = None
        if sanity:
            out_clean = call(True)
            extra["clean_logits_equal_everywhere"] = bool(torch.equal(out, out_clean))
            del out_clean
        if case in QUANT_EFFECT_CASES:
            tl = _true_logits(torch, q_raw, w_raw, y)
            pos = torch.arange(n, device=device)[None, :]
            tl = tl.masked_fill(~((pos >= ks.long()[:, None]) & (pos < ke.long()[:, None])), float("-inf"))
            extra["quantization_effect"] = dsa_ref.topk_agreement(tl, ref, topk)
            extra["quantization_effect"]["note"] = "정보용: FP8 양자화(q·k) vs 무양자화 fp32 의 top-k — 게이트 아님"

    # --- 판정 entry 3종 ---
    entries.append(_entry(_compare_logits(torch, ref, out, limit, f"{case}/logits")))
    entries.append(_topk_entry(torch, ref, out, limit, topk, f"{case}/topk"))
    flag = extra.get("clean_logits_equal_on_valid", extra.get("clean_logits_equal_everywhere", True))
    entries.append({"case": f"{case}/clean_logits_equiv", "metrics": {"equal": flag},
                    "violations": [] if flag else ["clean_logits=True 와 False 결과가 유효 구간에서 다름"],
                    "passed": bool(flag)})
    stats = None
    if do_timing and on_cuda:
        stats = timing.measure(lambda: call(False), warmup=warmup, reps=reps)
    del ref, out
    if on_cuda:
        torch.cuda.empty_cache()
    extra["build_s"] = round(build_s, 1)
    return {"schema_version": CHILD_SCHEMA, "case": case, "shape": dict(cfg),
            "status": "PASS" if all(e["passed"] for e in entries) else "FAIL",
            "entries": entries, "timing": stats, "extra": extra}


def _run_child(case: str, child_out: Path, warmup: int, reps: int) -> int:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        raise RuntimeError(f"k11 requires SM80, got {capability}")
    cfg = SANITIZER_CASES.get(case) or shapes.DSA_INDEXER_REQUIRED_CASES.get(case)
    if cfg is None:
        raise ValueError(f"unknown case: {case}")
    print(f"K11-CHILD start case={case} cfg={cfg}", flush=True)
    san = case.startswith("san_")           # sanitizer child: 커널만(오라클·타이밍 생략)
    payload = run_case(cfg, case, "cuda", do_timing=not san, warmup=warmup, reps=reps, kernel_only=san)
    child_out.parent.mkdir(parents=True, exist_ok=True)
    child_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"K11-CHILD done case={case} status={payload['status']}", flush=True)
    return 0 if payload["status"] == "PASS" else 1


# ===========================================================================
# --cpu-dry-run : 순수 python 자체점검 (torch/GPU 불필요)
# ===========================================================================
def dry_selfcheck() -> dict:
    """참조·판정 파이프 자체점검: 양자화 경계 → 한 쌍 logit → TOL 양방향 → top-k 일치 양방향."""
    import random
    rnd = random.Random(shapes.stable_seed(KID, "dry"))
    h, d, n = 4, 16, 40
    checks: dict[str, bool] = {}
    keys = []
    for _ in range(n):
        row = [rnd.gauss(0, 1) for _ in range(d)]
        keys.append(dsa_ref.quantize_k_row(row))
    qs = [[dsa_ref.quantize_q_head([rnd.gauss(0, 1) for _ in range(d)]) for _ in range(h)] for _ in range(1)]
    w = [rnd.gauss(0, 1) * (d ** -0.5 * h ** -0.5) * qs[0][i][1] for i in range(h)]
    q_codes = [c for c, _ in qs[0]]
    ref = [dsa_ref.mqa_logit_row(q_codes, w, kc, ks) for kc, ks in keys]
    checks["scales_are_pow2"] = all(math.frexp(s)[0] == 0.5 for _, s in keys)
    # 독립 재계산(디코드 값으로 직접)과 일치
    manual = []
    for kc, ks in keys:
        kd = [dsa_ref.fp8_ref.decode_e4m3fn(c) for c in kc]
        tot = 0.0
        for qh, wh in zip(q_codes, w):
            s = sum(dsa_ref.fp8_ref.decode_e4m3fn(a) * b for a, b in zip(qh, kd)) * ks
            tot += wh * max(s, 0.0)
        manual.append(tot)
    checks["logit_formula"] = all(abs(a - b) < 1e-12 for a, b in zip(ref, manual))
    # TOL 판정 양방향 (순수 python 경로)
    scale = math.sqrt(sum(v * v for v in ref) / len(ref)) or 1.0
    rn = [v / scale for v in ref]
    checks["tol_pass_case"] = tolerance.compare(KID, rn, [v * (1 + 1e-6) for v in rn], "dry_ok").passed
    checks["tol_fail_case"] = not tolerance.compare(KID, rn, [v * (1 + 5e-3) for v in rn], "dry_bad").passed
    # top-k 일치 양방향
    same = dsa_ref.topk_agreement_py(ref, list(ref), 10)
    swapped = list(ref)
    top = sorted(range(n), key=lambda i: -ref[i])
    swapped[top[0]], swapped[top[-1]] = swapped[top[-1]], swapped[top[0]]
    checks["topk_identical_passes"] = same["agreement"] >= dsa_ref.TOPK_AGREEMENT_MIN
    checks["topk_definite_swap_fails"] = dsa_ref.topk_agreement_py(ref, swapped, 10)["agreement"] < dsa_ref.TOPK_AGREEMENT_MIN
    checks["required_shapes_cover_modes"] = {c["mode"] for c in shapes.DSA_INDEXER_REQUIRED_CASES.values()} == {"decode", "prefill"}
    return {"mode": "cpu-dry-run(순수 python, torch/GPU 불필요)", "checks": checks,
            "selfcheck": "PASS" if all(checks.values()) else "FAIL"}


def kernel_evidence() -> dict:
    """리포트 evidence['kernel'] — 검증 대상이 PR 원본이 아니라 배포 패치본임을 파일 해시·int64 마커로 남긴다."""
    return {"module": KERNEL_MODULE,
            "entry_points": ["fp8_paged_mqa_logits_triton", "fp8_mqa_logits_triton"],
            "files": dsa_gate.kernel_provenance(KERNEL_FILES)}


def build_dry_report() -> dict:
    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    sc = dry_selfcheck()
    result.add_evidence("dry_run", sc)
    result.add_evidence("kernel", kernel_evidence())
    result.add_evidence("constants", {"heads": HEADS, "head_dim": HEAD_DIM, "topk": TOPK, "block": BLOCK,
                                      "index_k_bytes": shapes.DSA_INDEX_K_BYTES, "weight_scale": WEIGHT_SCALE})
    result.add_evidence("topk_policy", {"tie_band_rel": dsa_ref.TOPK_TIE_BAND_REL,
                                        "agreement_min": dsa_ref.TOPK_AGREEMENT_MIN})
    result.add_evidence("required_shapes", shapes.REQUIRED_SHAPES[KID])
    result.set_gate("NUMERIC", "SKIP", f"cpu-dry-run: 실커널 미실행 — 순수 python 참조·판정 자체점검 "
                    f"{len(sc['checks'])}건 {sc['selfcheck']}")
    result.set_gate("SANITIZER", "SKIP", "cpu-dry-run: compute-sanitizer 는 A100 실행 필요")
    result.set_gate("SHAPES", "SKIP", f"cpu-dry-run: {len(CASE_ORDER)}케이스 등록만 확인, 미실행")
    result.set_gate("PERF", "SKIP", "cpu-dry-run: CUDA 타이밍 미측정 (측정 전용 정책)")
    return result.as_dict()


# ===========================================================================
# --interpret : CPU Triton 인터프리터 대조 (게이트 판정 아님)
# ===========================================================================
def run_interpret(report_dir: Path) -> int:
    os.environ["TRITON_INTERPRET"] = "1"        # triton import 이전
    from common import triton_interp
    info = triton_interp.install()
    results, ok = [], True
    for name, cfg in INTERPRET_CASES.items():
        t0 = time.monotonic()
        try:
            payload = run_case(cfg, name, "cpu", do_timing=False, warmup=0, reps=0)
            payload["duration_s"] = round(time.monotonic() - t0, 1)
        except Exception as exc:  # 인터프리터 한계/오류도 증거로 남긴다
            traceback.print_exc()
            payload = {"case": name, "shape": dict(cfg), "status": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"[:500], "entries": []}
        ok &= payload["status"] == "PASS"
        results.append(payload)
        print(f"K11-INTERPRET case={name} status={payload['status']}", flush=True)
    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    result.add_evidence("interpret_cpu", {
        "note": "CPU 인터프리터 대조 — PASS 근거 아님(정직한 상한). GPU 게이트는 전부 SKIP 유지.",
        "overall": "PASS" if ok else "FAIL", "interpreter": info,
        "cases": [{k: p.get(k) for k in ("case", "shape", "status", "error", "duration_s", "extra")}
                  | {"entries": [{"case": e["case"], "passed": e["passed"], "violations": e["violations"],
                                  "metrics": e["metrics"]} for e in p["entries"]]} for p in results]})
    for gate in report.GATES:
        result.set_gate(gate, "SKIP", f"--interpret: CPU 인터프리터 대조 {('PASS' if ok else 'FAIL')} — "
                        "게이트 판정 아님(GPU 실행 필요)")
    write_dir = report_dir
    dsa_gate.write_payload(KID, result.as_dict(), write_dir)
    print(f"K11-INTERPRET-OVERALL {'PASS' if ok else 'FAIL'} (게이트 미반영) report={write_dir / (KID + '.json')}")
    return 0 if ok else 1


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--interpret", action="store_true",
                    help="TRITON_INTERPRET=1 CPU 대조(게이트 판정 아님; 기존 리포트와 같은 이름으로 기록)")
    ap.add_argument("--report-dir", type=Path, default=KIT / "kernel-report")
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--twin", default=None)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cases", nargs="*", default=None, help="REQUIRED 케이스 부분 실행(SHAPES 는 FAIL 이 된다)")
    ap.add_argument("--child-case")
    ap.add_argument("--child-out", type=Path)
    args = ap.parse_args()

    if args.child_case:
        if args.child_out is None:
            ap.error("--child-case requires --child-out")
        return _run_child(args.child_case, args.child_out, args.warmup, args.reps)
    if args.interpret:
        return run_interpret(args.report_dir)
    if args.cpu_dry_run:
        payload = build_dry_report()
        dsa_gate.write_payload(KID, payload, args.report_dir)
        sc = payload["evidence"]["dry_run"]["selfcheck"]
        print(f"K11-OVERALL INCOMPLETE mode=cpu-dry-run selfcheck={sc} report={args.report_dir / (KID + '.json')}")
        return report.EXIT_OK if sc == "PASS" else report.EXIT_ERR
    extra = {"kernel": kernel_evidence(),
             "call_contract": {"clean_logits": False, "decode_max_model_len": "active batch max",
                               "q": "E4M3 per-(token,head) ue8m0 + folded weights", "k_block": BLOCK},
             "topk_policy": {"tie_band_rel": dsa_ref.TOPK_TIE_BAND_REL, "agreement_min": dsa_ref.TOPK_AGREEMENT_MIN},
             "honest_limit": "oracle 은 DeepGEMM 과 bit-exact 가 아니다; 양자화 영향은 case_results.*.extra.quantization_effect"}
    return dsa_gate.run_parent(KID, Path(__file__), PINS, CASE_ORDER, PERF_CASE, list(SANITIZER_CASES), args, extra)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
