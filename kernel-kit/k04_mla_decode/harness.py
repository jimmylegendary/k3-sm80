"""k04_mla_decode — vLLM TRITON_MLA 디코드 커널 단독 검증 하네스 (S2).

대상 커널 (핀 확정, 2026-07-28 raw fetch로 원문 대조):
  vllm.v1.attention.ops.triton_decode_attention.decode_attention_fwd
  - vLLM main `d223c900` 과 kimi-k3 브랜치 `0498dc7e` 에서 **바이트 동일**
    (md5 de7ec6c04696f595bd79e83ce0943206, sha256 아래 PINNED_KERNEL_SHA256).
  - 호출부(vllm/v1/attention/backends/mla/triton_mla.py)는 브랜치에 DSpark(비인과
    draft) 분기만 추가 — causal decode 경로(우리 대상)는 main 과 동일.
  - import 는 **vendor 우선**(kernel-kit/vendor/attn_ops/triton_decode_attention.py —
    동일 핀 vendored 사본, vLLM 설치 불필요; sha 는 VENDORED_KERNEL_SHA256 로 별도 핀)
    → 실패 시 vllm 폴백. 출처는 evidence.kernel.source.

vLLM 결합 방식 (triton_mla.py `TritonMLAImpl.forward_mqa` L217-286 @ d223c900 1:1 재현):
  q            (B, 96, 576)  bf16   = absorbed q: [q_nope@W_UK (512) ‖ q_pe (64)]
  k_buffer     (P, page, 1, 576) bf16 = kv_c_and_k_pe_cache.unsqueeze(2)  [latent 512+rope 64]
  v_buffer     = k_buffer[..., :512] (view; is_mla → v = trans(k) 공유)
  o            (B, 96, 512)  bf16  / lse (B, 96) bf16
  block_table  (B, n_pages) int32 / seq_lens (B,) int32
  attn_logits  (B, 96, num_kv_splits, 513) fp32   [+1 = split별 LSE]
  num_kv_splits = min(next_pow2(max(1, max_seq_len//512)), sm_count*2)  [L41-47 재현]
  sm_scale     = qk_head_dim^-0.5 = 192^-0.5
    근거: kimi-k3 `0498dc7e` vllm/models/kimi_k3/nvidia/mla.py L127 (qk_head_dim=128+64),
    L134 (`self.scale = self.qk_head_dim**-0.5`; mscale 은 rope_parameters 있을 때만 —
    K3는 NoPE(mla_use_nope, recon/k3-config.md §3)라 미적용), L287/L314 로 impl 전달.
  is_mla=True → grouped 커널: Lk=576 → BLOCK_DMODEL=512 + BLOCK_DPE=64 경로
    (triton_decode_attention.py L491-497, kv_group_num = 96/1 = 96).

KV dtype arm 판정 — bf16 단일 arm:
  triton_mla.py L187-198: SM80에서 fp8 KV 는 __init__ 즉시 ValueError
  ("native FP8 (fp8e4nv) requires SM89+") → A100 에는 fp8 arm 이 존재하지 않는다.
  "fp8 저장 경로가 기본이면 두 arm" 조건 자체가 SM80 에서 거짓 (bf16 이 유일 경로).
  tolerance.py k04 정책("S2는 bf16 KV만 다룬다")과 일치. evidence 에 전사.

참조 구현 (NUMERIC 기준):
  latent 캐시를 block_table 로 전개(un-page)한 뒤 fp32 MQA attention:
    o_ref = softmax(q·kᵀ·192^-0.5)·v,  k=(L,576), v=k[:, :512]  — 전부 fp32.
  torch sdpa(MATH 백엔드) 와의 교차검증을 최소 케이스에서 1회 수행해 참조 자체의
  전사 오류를 차단한다(두 공식 max|Δ| < 1e-4 요구). lse 는 evidence 로 대조.

케이스: shapes.REQUIRED_SHAPES["k04_mla_decode"] (decode 5점, b∈{1,8,32}, ctx∈{8k,128k}).
  seq_lens 는 ctx - (i*37)%61 로 지터를 줘 페이지 비정렬 경계(cdiv split 마스크)를 검증.

PERF: timing.perf_gate(eff_floor 0.50) 를 케이스별 적용하되, **roofline < 30µs 케이스는
  판정 제외**(측정치는 evidence 기록). 근거: 커널 2회 launch+동기화 고정비 O(10µs)가
  루프라인의 수 배가 되는 구간은 '스트림 효율' 판정이 성립하지 않음 — PERF_POLICY k04
  basis("latent KV 스트림 지배, 특히 ctx 128k")의 전제도 스트림 지배 케이스다.
  제외 규칙은 사전등록(코드 상수) — 측정값을 보고 정하지 않는다. b1_ctx8k(≈4.6µs)만 제외.

CPU dry-run (--cpu-dry-run): GPU/torch 없이 순수 python 으로
  (1) naive fp64 softmax 참조 vs (2) 커널의 split-KV 2-stage 온라인 소프트맥스 알고리즘
  재현(stage1 분할 누적 + stage2 LSE 병합, triton_decode_attention.py 와 동일 수식)을
  bf16 반올림 입력·출력으로 대조해 판정 파이프(tolerance/report)를 실검증한다.
  음성 대조(의도적 오차 주입 → FAIL)까지 확인. 실커널 미실행이므로 4게이트는 전부
  SKIP(사유) — overall INCOMPLETE 상한 유지. 자체점검 실패 시 exit 3.

exit: 0 FAIL 없음 / 1 FAIL / 2 사전조건(GPU·vllm) 미충족 / 3 하네스 오류.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import random
import struct
import subprocess
import sys
import traceback

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k04_mla_decode"

# ---- K3 실측 상수 (common/shapes.py = recon 1:1 대조본) ----
H = shapes.MLA_HEADS            # 96
D_CACHE = shapes.MLA_CACHE_D    # 576 = kv_lora 512 + rope 64 (NoPE, 캐시엔 존재)
D_V = shapes.MLA_KV_LORA        # 512
SCALE = (shapes.MLA_QK_NOPE + shapes.MLA_QK_ROPE) ** -0.5   # 192^-0.5 (k3 mla.py L134)
PAGE_SIZE = 64                  # TRITON_MLA 요구 16 배수 (triton_mla.py supports_block_size)

PINS = {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e", "hf_kimi_k3": "9f62e4e9"}
# 핀 커밋의 vllm/v1/attention/ops/triton_decode_attention.py (main=branch 동일 파일)
PINNED_KERNEL_SHA256 = "b9823c65eba22d76abf64dad7d7d16a950f67471c86d6cb5b3e983aaf14381ac"
# vendor 사본 sha (= 핀 원본 + 출처 헤더 2줄 + import 2행 상대화 — vendor/STUBS.md).
# 원본 계보는 위 PINNED_ 로, vendor 사본 드리프트는 이 값으로 각각 감시한다.
VENDORED_KERNEL_SHA256 = "2ab9887c2d42a5674da1724c07d18ab2c0a6c97aeb80c4af5cb30e4cf30ef1de"

# PERF 판정 제외 임계 (사전등록): roofline < 30µs 면 launch 고정비 지배 → evidence 만
PERF_JUDGE_MIN_ROOFLINE_US = 30.0

FP8_ARM_NOTE = (
    "fp8 KV arm 미실행 — SM80에서 TRITON_MLA는 fp8 KV를 __init__에서 즉시 거부 "
    "(vllm@d223c900 triton_mla.py L187-198: 'native FP8 (fp8e4nv) requires SM89+'). "
    "A100 경로는 bf16 KV 단일 arm이 유일하며 K3 기본 kv-cache-dtype=auto(bf16). "
    "fp8 대조 바(upstream test atol=5e-1)는 SM89+ 전용 — S2 범위 밖."
)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    """triton_mla.py L37-47 재현: 512 tok/split, 2^n, SM×2 상한."""
    ideal = 1 << max(0, (max(1, max_seq_len // 512) - 1)).bit_length()
    return min(ideal, sm_count * 2)


def _seq_lens_for(b: int, ctx: int) -> list[int]:
    """페이지 비정렬 경계 검증용 지터 (결정적)."""
    return [ctx - ((i * 37) % 61) for i in range(b)]


# ===========================================================================
# GPU 경로
# ===========================================================================
def _import_kernel():
    """vendor(vLLM 불필요) 우선 → vllm 폴백. 출처·sha 핀 대조를 meta 로 반환."""
    import importlib
    try:
        mod = importlib.import_module("vendor.attn_ops.triton_decode_attention")
        source, pin_sha = "vendor", VENDORED_KERNEL_SHA256
        vendor_err = None
    except Exception as e_v:
        mod = importlib.import_module("vllm.v1.attention.ops.triton_decode_attention")
        source, pin_sha = "vllm", PINNED_KERNEL_SHA256
        vendor_err = f"{type(e_v).__name__}: {e_v}"[:300]
    with open(mod.__file__, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    meta = {"module": mod.__name__, "file": mod.__file__, "source": source,
            "sha256": sha, "matches_pin": sha == pin_sha, "pin_sha256": pin_sha}
    if vendor_err:
        meta["vendor_import_error"] = vendor_err
    return mod.decode_attention_fwd, meta


def _build_case_gpu(torch, case_name: str, b: int, ctx: int, dev):
    seed = shapes.stable_seed(KID, case_name)
    g = torch.Generator()          # CPU 생성기 — 아치 무관 결정성
    g.manual_seed(seed)
    seq_lens = _seq_lens_for(b, ctx)
    pages_per_seq = [_cdiv(L, PAGE_SIZE) for L in seq_lens]
    num_pages = sum(pages_per_seq)

    perm = torch.randperm(num_pages, generator=g)
    block_table = torch.zeros((b, max(pages_per_seq)), dtype=torch.int32)
    ofs = 0
    for i, n in enumerate(pages_per_seq):
        block_table[i, :n] = perm[ofs:ofs + n].to(torch.int32)
        ofs += n

    cache = torch.empty((num_pages, PAGE_SIZE, D_CACHE), dtype=torch.bfloat16, device=dev)
    CHUNK = 8192   # pages/chunk ≈ 1.2GB fp32 CPU 상한
    for p0 in range(0, num_pages, CHUNK):
        p1 = min(num_pages, p0 + CHUNK)
        chunk = torch.randn((p1 - p0, PAGE_SIZE, D_CACHE), generator=g, dtype=torch.float32)
        cache[p0:p1].copy_(chunk.to(torch.bfloat16))
    q = torch.randn((b, H, D_CACHE), generator=g, dtype=torch.float32) \
             .to(torch.bfloat16).to(dev)
    return {
        "name": case_name, "seed": seed, "b": b, "ctx": ctx,
        "seq_lens": seq_lens, "num_pages": num_pages,
        "cache": cache, "q": q,
        "block_table": block_table.to(dev),
        "seq_lens_t": torch.tensor(seq_lens, dtype=torch.int32, device=dev),
    }


def _kernel_buffers(torch, case, num_kv_splits, dev):
    b = case["b"]
    kv = case["cache"].unsqueeze(2)          # (P, page, 1, 576) — forward_mqa L263
    kv_c = kv[..., :D_V]                     # (P, page, 1, 512) view — L264
    o = torch.zeros(b, H, D_V, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(b, H, dtype=torch.bfloat16, device=dev)
    logits = torch.empty((b, H, num_kv_splits, D_V + 1), dtype=torch.float32, device=dev)
    return kv, kv_c, o, lse, logits


def _run_kernel(decode_attention_fwd, case, bufs, num_kv_splits):
    kv, kv_c, o, lse, logits = bufs
    decode_attention_fwd(
        case["q"], kv, kv_c, o, lse,
        case["block_table"], case["seq_lens_t"], logits,
        num_kv_splits, SCALE, PAGE_SIZE, is_mla=True,
    )
    return o, lse


def _reference_fp32(torch, case, dev):
    """latent 캐시 전개(un-page) 후 fp32 MQA attention (수식은 sdpa 동일)."""
    flat = case["cache"].reshape(-1, D_CACHE)
    b = case["b"]
    o_ref = torch.empty(b, H, D_V, dtype=torch.float32, device=dev)
    lse_ref = torch.empty(b, H, dtype=torch.float32, device=dev)
    ar_page = torch.arange(PAGE_SIZE, device=dev, dtype=torch.int64)
    for i in range(b):
        L = case["seq_lens"][i]
        nb = _cdiv(L, PAGE_SIZE)
        tok = (case["block_table"][i, :nb].to(torch.int64) * PAGE_SIZE).unsqueeze(1) + ar_page
        k = flat.index_select(0, tok.reshape(-1)[:L]).float()      # (L, 576)
        v = k[:, :D_V]
        qi = case["q"][i].float()                                  # (H, 576)
        logits = (qi @ k.T) * SCALE                                # (H, L) fp32
        lse_ref[i] = torch.logsumexp(logits, dim=-1)
        o_ref[i] = torch.softmax(logits, dim=-1) @ v
        del k, v, logits
    return o_ref, lse_ref


def _sdpa_crosscheck(torch, case, o_ref, dev) -> dict:
    """참조 전사 오류 차단: torch sdpa(MATH, fp32) 로 동일 o 재계산 — 1회/최소 케이스."""
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        cm = sdpa_kernel(SDPBackend.MATH)
        backend = "torch.nn.attention.sdpa_kernel(MATH)"
    except Exception:
        cm = contextlib.nullcontext()
        backend = "default(sdpa 컨텍스트 API 없음)"
    flat = case["cache"].reshape(-1, D_CACHE)
    i = 0
    L = case["seq_lens"][i]
    nb = _cdiv(L, PAGE_SIZE)
    ar = torch.arange(PAGE_SIZE, device=dev, dtype=torch.int64)
    tok = ((case["block_table"][i, :nb].to(torch.int64) * PAGE_SIZE).unsqueeze(1) + ar)
    k = flat.index_select(0, tok.reshape(-1)[:L]).float()
    q4 = case["q"][i].float().unsqueeze(0).unsqueeze(2)            # (1,H,1,576)
    k4 = k.unsqueeze(0).unsqueeze(0).expand(1, H, L, D_CACHE)
    v4 = k[:, :D_V].unsqueeze(0).unsqueeze(0).expand(1, H, L, D_V)
    with cm:
        o = F.scaled_dot_product_attention(q4, k4, v4, scale=SCALE)
    diff = (o.squeeze(2).squeeze(0) - o_ref[i]).abs().max().item()
    return {"backend": backend, "max_abs_vs_manual": diff, "ok": diff < 1e-4}


def _perf_case(torch, decode_attention_fwd, case, bufs, num_kv_splits,
               warmup: int, reps: int) -> dict:
    def fn():
        _run_kernel(decode_attention_fwd, case, bufs, num_kv_splits)
    return timing.measure(fn, warmup=warmup, reps=reps)


def run_gpu(args, rep: report.KernelReport) -> int:
    import torch
    dev = "cuda"
    try:
        decode_attention_fwd, kmeta = _import_kernel()
    except Exception as e:
        for gname in report.GATES:
            rep.set_gate(gname, "SKIP",
                         f"vLLM 커널 import 실패 — venv-k3(RUNBOOK §3) 확인: {type(e).__name__}: {e}")
        rep.add_evidence("import_error", traceback.format_exc()[-1500:])
        rep.write(args.report_dir)
        return report.EXIT_PRECOND
    rep.add_evidence("kernel", kmeta)
    rep.add_evidence("fp8_kv_arm", FP8_ARM_NOTE)
    rep.add_evidence("call_contract", {
        "scale": SCALE, "scale_basis": "qk_head_dim(192)^-0.5 — k3 mla.py@0498dc7e L127/L134 (NoPE→mscale 없음)",
        "page_size": PAGE_SIZE, "is_mla": True, "kv_dtype": "bfloat16",
        "num_kv_splits_rule": "min(next_pow2(max(1, max_seq_len//512)), sm_count*2) — triton_mla.py L37-47",
    })
    if not kmeta["matches_pin"]:
        rep.add_evidence("kernel_pin_warning",
                         f"[source={kmeta['source']}] 커널 파일 sha256 이 핀과 다름 "
                         "(vendor→VENDORED_KERNEL_SHA256, vllm→PINNED_KERNEL_SHA256 "
                         "d223c900=0498dc7e 동일본) — 판정은 유효하되 드리프트를 기록")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    # PERF 목표 원천 (경로 하드코딩 금지 — 인자만)
    vp = args.phase0_verdict if args.phase0_verdict and os.path.isfile(args.phase0_verdict) else None
    peaks = None
    if vp:
        tp = timing.resolve_twin_path(vp, args.twin)
        if tp:
            peaks = timing.load_twin_peaks(tp)

    wanted = list(shapes.REQUIRED_SHAPES[KID].items())
    if args.cases:
        wanted = [(n, c) for n, c in wanted if n in args.cases]

    numeric_results: list[tolerance.NumericResult] = []
    covered: list[str] = []
    case_ev: dict = {}
    perf_judged: dict = {}
    perf_excluded: dict = {}
    case_errors: dict = {}

    for name, cfg in wanted:
        b, ctx = cfg["b"], cfg["ctx"]
        try:
            case = _build_case_gpu(torch, name, b, ctx, dev)
            nks = _num_kv_splits(max(case["seq_lens"]), sm_count)
            bufs = _kernel_buffers(torch, case, nks, dev)
            o_k, lse_k = _run_kernel(decode_attention_fwd, case, bufs, nks)
            torch.cuda.synchronize()
            o_ref, lse_ref = _reference_fp32(torch, case, dev)

            r = tolerance.compare(KID, o_ref, o_k, case=name)
            numeric_results.append(r)
            lse_rel = ((lse_ref - lse_k.float()).pow(2).mean().sqrt()
                       / (lse_ref.pow(2).mean().sqrt() + 1e-8)).item()
            ev = {"seed": case["seed"], "b": b, "ctx": ctx,
                  "seq_lens_head": case["seq_lens"][:4], "num_pages": case["num_pages"],
                  "num_kv_splits": nks, "lse_rmse_rel": lse_rel,
                  "numeric_passed": r.passed}
            if name == "b1_ctx8k":
                ev["sdpa_crosscheck"] = _sdpa_crosscheck(torch, case, o_ref, dev)
                if not ev["sdpa_crosscheck"]["ok"]:
                    raise RuntimeError(f"참조 자체 교차검증 실패: {ev['sdpa_crosscheck']}")
            del o_ref, lse_ref

            model = shapes.perf_bytes_model(KID, b=b, ctx=ctx)
            stats = _perf_case(torch, decode_attention_fwd, case, bufs, nks,
                               args.warmup, args.reps)
            ev["measured"] = {k: stats[k] for k in
                              ("median_us", "mean_us", "min_us", "p90_us", "cv", "reps")}
            if peaks:
                rl = timing.roofline_us(model["bytes"], model.get("macs", 0.0),
                                        model.get("alu_ops", 0.0), peaks)
                ev["roofline_us"] = rl["roofline_us"]
                if rl["roofline_us"] >= PERF_JUDGE_MIN_ROOFLINE_US:
                    perf_judged[name] = timing.perf_gate(
                        KID, stats["median_us"], model, vp, args.twin, stats=stats)
                else:
                    perf_excluded[name] = {
                        "roofline_us": rl["roofline_us"], "median_us": stats["median_us"],
                        "reason": f"roofline {rl['roofline_us']:.1f}µs < "
                                  f"{PERF_JUDGE_MIN_ROOFLINE_US}µs — launch 고정비 지배 구간, 판정 제외(사전등록 규칙)"}
            covered.append(name)
            case_ev[name] = ev
            del case, bufs, o_k, lse_k
            torch.cuda.empty_cache()
        except Exception as e:
            case_errors[name] = {"error": f"{type(e).__name__}: {e}",
                                 "trace_tail": traceback.format_exc()[-1200:]}
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    rep.add_evidence("cases", case_ev)
    if case_errors:
        rep.add_evidence("case_errors", case_errors)

    # --- NUMERIC / SHAPES ---
    rep.set_gate("NUMERIC", *tolerance.numeric_gate(KID, numeric_results))
    rep.set_gate("SHAPES", *shapes.shapes_gate(KID, covered))

    # --- SANITIZER (축소 shape 자체 실행 — RUNNER §3) ---
    rep.set_gate("SANITIZER", *_sanitizer_gate(args))

    # --- PERF 집계 ---
    if vp is None:
        rep.set_gate("PERF", "SKIP",
                     f"PERF 목표 원천(verdict.json) 없음: {args.phase0_verdict} — 임의 목표 판정 금지",
                     {"measured_cases": {n: e.get("measured") for n, e in case_ev.items()}})
    elif peaks is None:
        rep.set_gate("PERF", "SKIP", "트윈(HW 피크 원천) 미해석 — verdict files/--twin 확인",
                     {"measured_cases": {n: e.get("measured") for n, e in case_ev.items()}})
    else:
        pev = {"judge_rule": f"케이스별 eff_floor 판정, roofline < {PERF_JUDGE_MIN_ROOFLINE_US}µs 는 "
                             "제외(launch 고정비 지배 — 사전등록)",
               "judged": {n: {"status": g[0], "reason": g[1], **g[2]} for n, g in perf_judged.items()},
               "excluded": perf_excluded}
        fails = {n: g for n, g in perf_judged.items() if g[0] == "FAIL"}
        if fails:
            n0, g0 = next(iter(fails.items()))
            rep.set_gate("PERF", "FAIL",
                         f"{len(fails)}/{len(perf_judged)} 판정 케이스 미달 — 예: [{n0}] {g0[1]}", pev)
        elif perf_judged:
            rep.set_gate("PERF", "PASS",
                         f"판정 {len(perf_judged)}케이스 전부 floor 충족 "
                         f"(제외 {len(perf_excluded)}건은 evidence)", pev)
        else:
            rep.set_gate("PERF", "SKIP", "판정 가능 케이스 없음(전부 roofline<임계 또는 실행 실패)", pev)

    rep.write(args.report_dir)
    return rep.exit_code()


# ---------------------------------------------------------------------------
# SANITIZER: compute-sanitizer memcheck 하 축소 케이스 (b=2, ctx=4096, 전체 차원 유지)
# ---------------------------------------------------------------------------
def _sanitizer_gate(args) -> tuple:
    san = env_check.find_compute_sanitizer()
    if not san:
        return ("SKIP", "compute-sanitizer 미발견 — RUNBOOK §1.5 (CUDA 툴킷 bin/)", {})
    cmd = [san, "--tool", "memcheck", sys.executable,
           os.path.abspath(__file__), "--sanitizer-inner"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.sanitizer_timeout)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return report.sanitizer_gate_from_output(out, " ".join(cmd), p.returncode)
    except subprocess.TimeoutExpired:
        return ("SKIP", f"sanitizer {args.sanitizer_timeout}s 타임아웃 — 판독 불가(수동 재실행 필요)",
                {"cmd": " ".join(cmd)})


def _sanitizer_inner() -> int:
    """sanitizer 하에서 1회 실행할 축소 케이스 (D 차원은 K3 실측 유지 — OOB 검출 목적)."""
    import torch
    decode_attention_fwd, _ = _import_kernel()
    dev = "cuda"
    case = _build_case_gpu(torch, "sanitizer_b2_ctx4096", 2, 4096, dev)
    nks = _num_kv_splits(max(case["seq_lens"]),
                         torch.cuda.get_device_properties(0).multi_processor_count)
    bufs = _kernel_buffers(torch, case, nks, dev)
    _run_kernel(decode_attention_fwd, case, bufs, nks)
    torch.cuda.synchronize()
    print("[k04 sanitizer-inner] kernel 1회 실행 완료 (b=2, ctx=4096, splits=%d)" % nks)
    return 0


# ===========================================================================
# CPU dry-run — 순수 python: naive fp64 참조 vs split-KV 2-stage 알고리즘 재현
# ===========================================================================
def _bf16(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    lo, hi = b & 0xFFFF, b >> 16
    if lo > 0x8000 or (lo == 0x8000 and (hi & 1)):
        hi += 1
    return struct.unpack("<f", struct.pack("<I", (hi << 16) & 0xFFFFFFFF))[0]


def _dry_naive(q, ks, vs, scale):
    """naive fp64 softmax attention — 1 (seq, head)."""
    logits = [sum(qd * kd for qd, kd in zip(q, k)) * scale for k in ks]
    m = max(logits)
    es = [math.exp(x - m) for x in logits]
    s = sum(es)
    dv = len(vs[0])
    o = [sum(es[j] * vs[j][d] for j in range(len(vs))) / s for d in range(dv)]
    return o, m + math.log(s)


def _dry_splitkv(q, ks, vs, scale, splits):
    """triton_decode_attention.py stage1(L278-467)+stage2(L575-639) 수식 재현 (fp64)."""
    L = len(ks)
    dv = len(vs[0])
    per = _cdiv(L, splits)
    parts = []
    for sp in range(splits):                       # stage1: split별 온라인 소프트맥스
        st, en = per * sp, min(per * (sp + 1), L)
        if en <= st:
            parts.append(None)
            continue
        e_max, e_sum = -math.inf, 0.0
        acc = [0.0] * dv
        for j in range(st, en):
            qk = sum(qd * kd for qd, kd in zip(q, ks[j])) * scale
            n_max = max(e_max, qk)
            re = math.exp(e_max - n_max) if e_max != -math.inf else 0.0
            p = math.exp(qk - n_max)
            for d in range(dv):
                acc[d] = acc[d] * re + p * vs[j][d]
            e_sum = e_sum * re + p
            e_max = n_max
        parts.append(([a / e_sum for a in acc], e_max + math.log(e_sum)))
    e_max, e_sum = -math.inf, 0.0                  # stage2: LSE 병합
    acc = [0.0] * dv
    for pt in parts:
        if pt is None:
            continue
        tv, tlog = pt
        n_max = max(tlog, e_max)
        old = math.exp(e_max - n_max) if e_max != -math.inf else 0.0
        ex = math.exp(tlog - n_max)
        for d in range(dv):
            acc[d] = acc[d] * old + ex * tv[d]
        e_sum = e_sum * old + ex
        e_max = n_max
    return [a / e_sum for a in acc], e_max + math.log(e_sum)


def run_dry(args, rep: report.KernelReport) -> int:
    DQ, DV_, HS, SPLITS, PAGE = 24, 16, 4, 3, 8
    scale = DQ ** -0.5
    results = []
    lse_max_abs = 0.0
    scaled_cases = []
    for name, cfg in shapes.REQUIRED_SHAPES[KID].items():
        b = min(cfg["b"], 2)
        ctx = 96 if cfg["ctx"] == 8192 else 160    # 축소 대표점 (page 비정렬 지터 유지)
        seed = shapes.stable_seed(KID, name)
        rnd = random.Random(seed)
        ref_flat, out_flat = [], []
        for i in range(b):
            L = ctx - ((i * 37) % 61) % 29         # 축소판 지터
            ks = [[_bf16(rnd.gauss(0, 1)) for _ in range(DQ)] for _ in range(L)]
            vs = [k[:DV_] for k in ks]             # is_mla: v = k[:, :dv] 공유 재현
            for h in range(HS):
                q = [_bf16(rnd.gauss(0, 1)) for _ in range(DQ)]
                o_ref, lse_ref = _dry_naive(q, ks, vs, scale)
                o_sim, lse_sim = _dry_splitkv(q, ks, vs, scale, SPLITS)
                ref_flat += o_ref
                out_flat += [_bf16(v) for v in o_sim]   # 커널 bf16 출력 반올림 모사
                lse_max_abs = max(lse_max_abs, abs(lse_ref - lse_sim))
        r = tolerance.compare(KID, ref_flat, out_flat, case=f"{name}(scaled)")
        results.append(r)
        scaled_cases.append({"case": name, "b": b, "ctx_scaled": ctx, "seed": seed,
                             "passed": r.passed, "rmse_rel": r.metrics["rmse_rel"]})
    # 음성 대조: 오차 주입 시 FAIL 해야 판정기가 살아있는 것
    bad = tolerance.compare(KID, ref_flat, [v * 1.02 for v in out_flat], case="negative_control")
    selfcheck_ok = all(r.passed for r in results) and (not bad.passed) and lse_max_abs < 1e-9

    ev = {"mode": "cpu-dry-run(순수 python, torch/GPU 불필요)",
          "algorithm_check": "naive fp64 softmax vs split-KV 2-stage(LSE 병합) 재현 — "
                             "triton_decode_attention.py stage1/stage2 수식 대조",
          "scaled_cases": scaled_cases,
          "lse_merge_max_abs_err": lse_max_abs,
          "negative_control_failed_as_expected": not bad.passed,
          "selfcheck": "PASS" if selfcheck_ok else "FAIL"}
    rep.add_evidence("dry_run", ev)
    rep.add_evidence("fp8_kv_arm", FP8_ARM_NOTE)
    rep.set_gate("NUMERIC", "SKIP",
                 "cpu-dry-run: 실커널 미실행 — 참조·판정 파이프 자체점검만 수행(evidence.dry_run)")
    rep.set_gate("SHAPES", "SKIP",
                 "cpu-dry-run: 축소 shape 실행 — K3 실측 shape 판정은 GPU 모드 전용")
    rep.set_gate("SANITIZER", "SKIP", "cpu-dry-run: GPU 없음")
    rep.set_gate("PERF", "SKIP", "cpu-dry-run: 측정 없음")
    rep.write(args.report_dir)
    print(f"[k04 dry-run] selfcheck={'PASS' if selfcheck_ok else 'FAIL'} "
          f"(scaled {len(results)}케이스, lse Δ={lse_max_abs:.2e}, 음성대조 FAIL 확인={not bad.passed})")
    return report.EXIT_OK if selfcheck_ok else report.EXIT_ERR


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=f"{KID} S2 하네스")
    ap.add_argument("--phase0-verdict", default=None, help="PERF 목표 원천 verdict.json")
    ap.add_argument("--twin", default=None, help="트윈 json 오버라이드")
    ap.add_argument("--report-dir",
                    default=os.path.join(_KIT, "kernel-report"))
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None, help="케이스 부분 실행")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--sanitizer-inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sanitizer-timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.sanitizer_inner:
        return _sanitizer_inner()

    rep = report.KernelReport(KID, dry_run=args.cpu_dry_run, pins=PINS)
    if args.cpu_dry_run:
        return run_dry(args, rep)

    env = env_check.collect_env()
    ok, checks = env_check.check_preconditions(env, cpu_dry_run=False)
    if not ok:
        print(f"[{KID}] GPU 사전조건 미충족 — env_check 참조 "
              f"(--cpu-dry-run 으로 참조 경로만 검증 가능)", file=sys.stderr)
        for c in checks:
            if c["status"] == "FAIL":
                print(f"  FAIL {c['check']}: {c['detail']}", file=sys.stderr)
        return report.EXIT_PRECOND
    return run_gpu(args, rep)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
