#!/usr/bin/env python3
# =============================================================================
# probe_kda_fla.py — KDA Triton 커널(SM80) 단독 실행 + 소형 수치 대조
#
# k3-sm80 Phase 1 · 2026-07-28 · 러너: 사내 A100 노드
#
# 배경 (recon/kda-fla.md): FLA 공식 CI는 H100 전용 — A100(SM80) 수치는 업스트림이
# 보증하지 않는다. 이 프로브가 SM80 첫 공식 대조가 된다. 커널은 전부 순수 Triton
# (아치 게이팅 0건)이라 "돌아는 가야" 하고, 여기서 보는 것은 수치 정합 + autotune 안정성.
#
# 검증 대상 구현 (--impl):
#   main  vllm.third_party.flash_linear_attention.ops.kda
#         (vLLM main d223c900 / 릴리스 0.26.0 wheel에 존재 — Stage 1에서 실행 가능)
#   k3    vllm.models.kimi_k3.nvidia.ops.third_party.kda
#         (kimi-k3 브랜치 0498dc7 소스빌드 전용 — K3 서빙이 실제로 타는 사본)
#   fla   fla.ops.kda (pip flash-linear-attention==0.5.0 — 교차대조용, 선택)
#   auto  가능한 것 전부
#
# 테스트 (참조 = naive fp32 recurrence, FLA fla/ops/kda/naive.py @ 9c8e42e 및
#         vLLM tests/kernels/test_kda.py @ d223c900 에서 이식·인용):
#   T1  chunk_kda vs naive              (varlen, l2norm in-kernel)      [main/k3/fla]
#   T2  fused_recurrent_kda vs naive    (paged state pool + indices)    [main]
#   T3  chunk→recurrent state 핸드오프  (prefill 후 decode 1스텝 연속성) [main]
#   T4  = T1 을 k3 사본에               ...                             [k3]
#   T5  chunk_kda_with_fused_gate vs 독립 naive 게이트 수식             [k3]
#       gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias)), lower_bound=-5.0
#       beta = sigmoid(raw_beta)   ← K3 브랜치 커널( _kda_gate_beta_fwd_kernel )에서 추출한 수식
#   T6  chunk prefill(seq당 L-1) → fused_recurrent_kda decode(seq당 1토큰) 핸드오프
#       vs naive/chunk 전체-T — k3 decode 커널은 T==N(시퀀스당 1토큰) 전용이라
#       서빙과 동일한 prefill→decode 구조로 검증 [k3]
#
# 판정: RMSE 비율(rel_err = rmse(ref-out)/rmse(ref)) — vLLM 업스트림 테스트와 동일 척도.
#   기본 임계 0.005 (T5/T6 게이트 경유는 0.008). PROBE_KDA_THR_* env로 조절.
#   회색지대(임계~4x임계)는 DECISION-TREE §2 참조.
#
# 종료코드: 0=PASS · 2=수치 FAIL · 3=크래시/임포트 실패
# 출력: PROBE_KDA|impl=..|test=..|verdict=..|o_rmse=..|... (기계판독)
# =============================================================================

import argparse
import importlib
import os
import sys
import time
import traceback

try:
    import torch
    import torch.nn.functional as F
except Exception as _e:  # 계약: 3=크래시/임포트 실패 (기본 traceback exit 1 방지)
    print(f"PROBE_KDA|impl=-|test=env|verdict=FAIL_TORCH_IMPORT|error={_e!r}")
    sys.exit(3)

LOWER_BOUND = -5.0  # K3 config linear_attn_config.gate_lower_bound (recon/k3-config.md)
D_K3 = 128          # K3 head_dim

THR_BASE = float(os.environ.get("PROBE_KDA_THR", "0.005"))
THR_GATE = float(os.environ.get("PROBE_KDA_THR_GATE", "0.008"))

RESULTS: list[dict] = []


def emit(impl: str, test: str, verdict: str, **kv):
    extra = "".join(f"|{k}={v}" for k, v in kv.items())
    print(f"PROBE_KDA|impl={impl}|test={test}|verdict={verdict}{extra}", flush=True)
    RESULTS.append({"impl": impl, "test": test, "verdict": verdict, **kv})


# ----------------------------------------------------------------------------
# 참조 구현 (fp32) — FLA naive.py (MIT, Songlin Yang, Yu Zhang) 이식.
# 원본: https://github.com/fla-org/flash-linear-attention/blob/9c8e42e/fla/ops/kda/naive.py
# vLLM 포팅본: tests/kernels/test_kda.py @ d223c900 (동일 수식)
# ----------------------------------------------------------------------------
def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False):
    dtype = v.dtype
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = (x.to(torch.float) for x in [q, k, v, g, beta])
    q = q * scale
    S = k.new_zeros(B, H, K, v.shape[-1])
    if initial_state is not None:
        S = S + initial_state.to(torch.float)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv", b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    return o.to(dtype), (S if output_final_state else None)


def ref_gate(raw_g, dt_bias, A_log, lower_bound=LOWER_BOUND):
    """K3 게이트 수식 (kimi-k3 브랜치 _kda_gate_beta_fwd_kernel / kda_gate_fwd_kernel
    USE_LOWER_BOUND 분기에서 추출): gate = lower_bound * sigmoid(exp(A_log)*(g+bias)) [자연로그 공간]"""
    x = raw_g.to(torch.float)
    if dt_bias is not None:
        H, D = raw_g.shape[-2], raw_g.shape[-1]
        x = x + dt_bias.to(torch.float).view(H, D)
    a = torch.exp(A_log.to(torch.float))  # [H]
    return lower_bound * torch.sigmoid(a[None, None, :, None] * x)


def l2norm_ref(x):
    """참조용 L2 정규화. 가능하면 vendored l2norm_fwd(커널)를 그대로 사용해
    eps 정의 차이를 제거한다(vLLM 업스트림 테스트와 동일 수법)."""
    try:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        return l2norm_fwd(x.contiguous())
    except Exception:
        return F.normalize(x, p=2, dim=-1)


def rel_rmse(ref, out):
    ref = ref.detach().float().flatten()
    out = out.detach().float().flatten()
    rmse = (ref - out).square().mean().sqrt().item()
    base = ref.square().mean().sqrt().item()
    return rmse / (base + 1e-8)


def state_match(ref_kv, kernel_state):
    """naive state는 [N,H,K,V], 커널 저장은 (V,K) 전치 관례(vLLM test_kda.py 주석).
    양방향 비교해 (더 잘 맞는 rmse, 방향) 반환 — K=V=128 정방이라 shape로는 구분 불가."""
    r_t = rel_rmse(ref_kv, kernel_state.transpose(-1, -2))
    r_d = rel_rmse(ref_kv, kernel_state)
    return (r_t, "VK(transposed)") if r_t <= r_d else (r_d, "KV(direct)")


# ----------------------------------------------------------------------------
# 입력 생성 (vLLM/FLA 테스트 패턴 준수: q/k/v=rand, g=logsigmoid(randn), beta=sigmoid)
# ----------------------------------------------------------------------------
def make_inputs(cu, H, D, dtype, seed=42, device="cuda"):
    torch.manual_seed(seed)
    T = cu[-1]
    N = len(cu) - 1
    q = torch.rand(1, T, H, D, dtype=dtype, device=device)
    k = torch.rand(1, T, H, D, dtype=dtype, device=device)
    v = torch.rand(1, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device)).to(dtype)
    beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()
    raw_g = (torch.randn(1, T, H, D, dtype=torch.float32, device=device) * 0.5).to(dtype)
    raw_beta = torch.randn(1, T, H, dtype=dtype, device=device)
    A_log = (torch.randn(H, dtype=torch.float32, device=device) * 0.5).contiguous()
    dt_bias = (torch.randn(H * D, dtype=torch.float32, device=device) * 0.1).contiguous()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    return dict(q=q, k=k, v=v, g=g, beta=beta, raw_g=raw_g, raw_beta=raw_beta,
                A_log=A_log, dt_bias=dt_bias, h0=h0, cu=cu, cu_t=cu_t, N=N, T=T)


def naive_over_varlen(inp, g_override=None, beta_override=None, l2norm=True):
    """세그먼트별 naive 실행 → (o 연결, per-seq final state [N,H,K,V])"""
    q, k, v = inp["q"], inp["k"], inp["v"]
    g = inp["g"] if g_override is None else g_override
    beta = inp["beta"] if beta_override is None else beta_override
    outs, states = [], []
    for i in range(inp["N"]):
        s, e = inp["cu"][i], inp["cu"][i + 1]
        q_i = l2norm_ref(q[:, s:e]) if l2norm else q[:, s:e]
        k_i = l2norm_ref(k[:, s:e]) if l2norm else k[:, s:e]
        o_i, ht_i = naive_recurrent_kda(
            q_i, k_i, v[:, s:e], g[:, s:e], beta[:, s:e],
            initial_state=inp["h0"][i], output_final_state=True)
        outs.append(o_i)
        states.append(ht_i)
    return torch.cat(outs, dim=1), torch.cat(states, dim=0)


# ----------------------------------------------------------------------------
# 구현 로더
# ----------------------------------------------------------------------------
IMPL_MODULES = {
    "main": "vllm.third_party.flash_linear_attention.ops.kda",
    "k3": "vllm.models.kimi_k3.nvidia.ops.third_party.kda",
    "fla": "fla.ops.kda",
}


def load_impl(name):
    try:
        return importlib.import_module(IMPL_MODULES[name])
    except Exception as e:
        emit(name, "import", "UNAVAILABLE", error=repr(e).replace("|", "/")[:200])
        return None


# ----------------------------------------------------------------------------
# 테스트들
# ----------------------------------------------------------------------------
def t_chunk_vs_naive(impl_name, mod, cu, H, dtype, test_id):
    """initial_state 레이아웃은 구현별로 다를 수 있다:
    vLLM vendored(main/k3)는 (V,K) 전치 관례(tests/kernels/test_kda.py 주석),
    FLA 업스트림 문서는 [N,H,K,V]. 두 방향 모두 실행해 o가 맞는 쪽을 채택·보고한다
    (h0가 랜덤 비대칭이라 잘못된 방향은 o부터 크게 어긋남 — 자기판별 가능)."""
    inp = make_inputs(cu, H, D_K3, dtype)
    ref_o, ref_ht = naive_over_varlen(inp)
    best = None
    for layout, h0_in in (("VK", inp["h0"].transpose(-1, -2).contiguous()),
                          ("KV", inp["h0"].contiguous())):
        tri_o, tri_ht = mod.chunk_kda(
            q=inp["q"].clone(), k=inp["k"].clone(), v=inp["v"].clone(),
            g=inp["g"].clone(), beta=inp["beta"].clone(),
            initial_state=h0_in.clone(),
            output_final_state=True, cu_seqlens=inp["cu_t"],
            use_qk_l2norm_in_kernel=True)
        assert not torch.isnan(tri_o).any(), "o NaN"
        assert not torch.isnan(tri_ht).any(), "ht NaN"
        ro = rel_rmse(ref_o, tri_o)
        rh, _ = state_match(ref_ht, tri_ht)
        if best is None or ro < best[1]:
            best = (layout, ro, rh)
    layout, ro, rh = best
    v = "PASS" if (ro < THR_BASE and rh < THR_BASE) else "FAIL"
    emit(impl_name, test_id, v, o_rmse=f"{ro:.6f}", ht_rmse=f"{rh:.6f}",
         h0_layout=layout, thr=THR_BASE, dtype=str(dtype).split(".")[-1],
         cu=str(cu).replace(" ", ""))
    return v == "PASS"


def t2_recurrent_vs_naive(mod, cu, H, dtype):
    """main 사본 fused_recurrent_kda(g,beta 사전계산 시그니처) vs naive."""
    inp = make_inputs(cu, H, D_K3, dtype, seed=43)
    ref_o, ref_ht = naive_over_varlen(inp)
    pool = inp["h0"].transpose(-1, -2).contiguous().clone()  # 커널 레이아웃 pool
    idx = torch.arange(inp["N"], dtype=torch.long, device="cuda")
    tri_o, final = mod.fused_recurrent_kda(
        q=inp["q"].clone(), k=inp["k"].clone(), v=inp["v"].clone(),
        g=inp["g"].clone().float(), beta=inp["beta"].clone().float(),
        initial_state=pool, inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_t"], ssm_state_indices=idx)
    ro = rel_rmse(ref_o, tri_o)
    rh, orient = state_match(ref_ht, pool)
    v = "PASS" if (ro < THR_BASE and rh < THR_BASE) else "FAIL"
    emit("main", "T2_recurrent_vs_naive", v, o_rmse=f"{ro:.6f}",
         state_rmse=f"{rh:.6f}", state_layout=orient, thr=THR_BASE,
         dtype=str(dtype).split(".")[-1])
    return v == "PASS"


def t3_handoff(mod, H, dtype):
    """main 사본: T-1 토큰 chunk prefill → final state → 1토큰 recurrent decode.
    naive 전체 T와 비교 — 서빙의 prefill→decode state 연속성 검증."""
    T = 200
    inp = make_inputs([0, T], H, D_K3, dtype, seed=44)
    ref_o, ref_ht = naive_over_varlen(inp)

    cu_prefill = torch.tensor([0, T - 1], dtype=torch.int32, device="cuda")
    o_pre, ht_pre = mod.chunk_kda(
        q=inp["q"][:, :T - 1].clone(), k=inp["k"][:, :T - 1].clone(),
        v=inp["v"][:, :T - 1].clone(), g=inp["g"][:, :T - 1].clone(),
        beta=inp["beta"][:, :T - 1].clone(),
        initial_state=inp["h0"].transpose(-1, -2).contiguous(),
        output_final_state=True, cu_seqlens=cu_prefill,
        use_qk_l2norm_in_kernel=True)

    pool = ht_pre.contiguous().clone()
    idx = torch.arange(1, dtype=torch.long, device="cuda")
    cu_dec = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    o_dec, _ = mod.fused_recurrent_kda(
        q=inp["q"][:, T - 1:].clone(), k=inp["k"][:, T - 1:].clone(),
        v=inp["v"][:, T - 1:].clone(), g=inp["g"][:, T - 1:].clone().float(),
        beta=inp["beta"][:, T - 1:].clone().float(),
        initial_state=pool, inplace_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_dec, ssm_state_indices=idx)

    ro_pre = rel_rmse(ref_o[:, :T - 1], o_pre)
    ro_dec = rel_rmse(ref_o[:, T - 1:], o_dec)
    rh, orient = state_match(ref_ht, pool)
    v = "PASS" if (ro_pre < THR_BASE and ro_dec < THR_BASE and rh < THR_BASE) else "FAIL"
    emit("main", "T3_handoff", v, prefill_o_rmse=f"{ro_pre:.6f}",
         decode_o_rmse=f"{ro_dec:.6f}", final_state_rmse=f"{rh:.6f}",
         state_layout=orient, thr=THR_BASE, dtype=str(dtype).split(".")[-1])
    return v == "PASS"


def t5_fused_gate_vs_naive(mod, cu, H, dtype):
    """k3 사본 chunk_kda_with_fused_gate(raw_g, raw_beta, A_log, dt_bias, lower_bound)
    vs 독립 naive(게이트 수식 포함) — K3 prefill 실경로의 수치 e2e."""
    inp = make_inputs(cu, H, D_K3, dtype, seed=45)
    g_ref = ref_gate(inp["raw_g"], inp["dt_bias"], inp["A_log"])
    beta_ref = inp["raw_beta"].float().sigmoid()
    ref_o, ref_ht = naive_over_varlen(inp, g_override=g_ref, beta_override=beta_ref)
    tri_o, tri_ht = mod.chunk_kda_with_fused_gate(
        q=inp["q"].clone(), k=inp["k"].clone(), v=inp["v"].clone(),
        raw_g=inp["raw_g"].clone(), raw_beta=inp["raw_beta"].clone(),
        A_log=inp["A_log"], g_bias=inp["dt_bias"], lower_bound=LOWER_BOUND,
        initial_state=inp["h0"].transpose(-1, -2).contiguous(),
        output_final_state=True, cu_seqlens=inp["cu_t"],
        use_qk_l2norm_in_kernel=True)
    ro = rel_rmse(ref_o, tri_o)
    rh = rel_rmse(ref_ht, tri_ht.transpose(-1, -2))
    v = "PASS" if (ro < THR_GATE and rh < THR_GATE) else "FAIL"
    emit("k3", "T5_fused_gate_vs_naive", v, o_rmse=f"{ro:.6f}", ht_rmse=f"{rh:.6f}",
         thr=THR_GATE, dtype=str(dtype).split(".")[-1], cu=str(cu).replace(" ", ""))
    return v == "PASS"


def t6_recurrent_vs_chunk_k3(mod, cu, H, dtype):
    """k3 사본 decode 커널의 교차일관성 + state 핸드오프.

    주의: k3 사본 fused_recurrent_kda_fwd는 1-D ssm_state_indices에서 `assert T == N`
    (pin 0498dc7 fused_recurrent.py L361 부근) — **시퀀스당 1토큰 decode 전용**이다
    (main 사본의 범용 varlen recurrent와 다름; 킷 검증 2026-07-28에서 확인).
    따라서 서빙과 동일한 구조로 검증한다:
      chunk_kda_with_fused_gate(각 seq 앞 L-1 토큰) → state 핸드오프
      → fused_recurrent_kda(각 seq 마지막 1토큰, T==N) 를
      naive 전체-T(마지막 토큰 위치) 및 chunk 전체-T(o 마지막 토큰·final state)와 대조."""
    inp = make_inputs(cu, H, D_K3, dtype, seed=46)
    N = inp["N"]
    lens = [inp["cu"][i + 1] - inp["cu"][i] for i in range(N)]
    assert all(l >= 2 for l in lens), f"T6는 seq당 >=2 토큰 필요: {lens}"
    g_ref = ref_gate(inp["raw_g"], inp["dt_bias"], inp["A_log"])
    beta_ref = inp["raw_beta"].float().sigmoid()
    ref_o, _ = naive_over_varlen(inp, g_override=g_ref, beta_override=beta_ref)

    # (a) 전체 T chunk — o 비교 기준 + 전체-T final state 기준
    chk_o, chk_ht = mod.chunk_kda_with_fused_gate(
        q=inp["q"].clone(), k=inp["k"].clone(), v=inp["v"].clone(),
        raw_g=inp["raw_g"].clone(), raw_beta=inp["raw_beta"].clone(),
        A_log=inp["A_log"], g_bias=inp["dt_bias"], lower_bound=LOWER_BOUND,
        initial_state=inp["h0"].transpose(-1, -2).contiguous(),
        output_final_state=True, cu_seqlens=inp["cu_t"],
        use_qk_l2norm_in_kernel=True)

    # (b) prefill: 각 seq 마지막 1토큰 제외한 L-1 토큰
    pre_pos, cu_pre = [], [0]
    for i in range(N):
        pre_pos.extend(range(inp["cu"][i], inp["cu"][i + 1] - 1))
        cu_pre.append(cu_pre[-1] + lens[i] - 1)
    dec_pos = [inp["cu"][i + 1] - 1 for i in range(N)]
    idx_pre = torch.tensor(pre_pos, dtype=torch.long, device="cuda")
    idx_dec = torch.tensor(dec_pos, dtype=torch.long, device="cuda")
    cu_pre_t = torch.tensor(cu_pre, dtype=torch.int32, device="cuda")

    def sel(x, idx):
        return x.index_select(1, idx).contiguous()

    _, ht_pre = mod.chunk_kda_with_fused_gate(
        q=sel(inp["q"], idx_pre), k=sel(inp["k"], idx_pre), v=sel(inp["v"], idx_pre),
        raw_g=sel(inp["raw_g"], idx_pre), raw_beta=sel(inp["raw_beta"], idx_pre),
        A_log=inp["A_log"], g_bias=inp["dt_bias"], lower_bound=LOWER_BOUND,
        initial_state=inp["h0"].transpose(-1, -2).contiguous(),
        output_final_state=True, cu_seqlens=cu_pre_t,
        use_qk_l2norm_in_kernel=True)

    # (c) decode: seq당 1토큰 (T==N 충족), state pool은 prefill 산출 그대로 (핸드오프)
    pool = ht_pre.contiguous().clone()
    cu_dec = torch.arange(N + 1, dtype=torch.int32, device="cuda")
    sidx = torch.arange(N, dtype=torch.long, device="cuda")
    rec_o, _ = mod.fused_recurrent_kda(
        q=sel(inp["q"], idx_dec), k=sel(inp["k"], idx_dec), v=sel(inp["v"], idx_dec),
        raw_g=sel(inp["raw_g"], idx_dec), raw_beta=sel(inp["raw_beta"], idx_dec),
        A_log=inp["A_log"], dt_bias=inp["dt_bias"], lower_bound=LOWER_BOUND,
        initial_state=pool, cu_seqlens=cu_dec, ssm_state_indices=sidx)

    ref_dec = ref_o.index_select(1, idx_dec)
    chk_dec = chk_o.index_select(1, idx_dec)
    ro_x = rel_rmse(chk_dec, rec_o)        # 커널 간 (decode 토큰)
    ro_n = rel_rmse(ref_dec, rec_o)        # naive 대비 (decode 토큰)
    rh_x = rel_rmse(chk_ht, pool)          # 전체-T state 등가 (동일 (V,K) 레이아웃)
    v = "PASS" if (ro_x < THR_GATE and ro_n < THR_GATE and rh_x < THR_GATE) else "FAIL"
    emit("k3", "T6_recurrent_vs_chunk", v, cross_o_rmse=f"{ro_x:.6f}",
         naive_o_rmse=f"{ro_n:.6f}", cross_state_rmse=f"{rh_x:.6f}",
         thr=THR_GATE, dtype=str(dtype).split(".")[-1],
         note="prefill(L-1)+decode(1tok/seq,T==N)")
    return v == "PASS"


def perf_smoke(mod_name, mod):
    """성능 스모크(판정 없음): K3 TP8 환산 per-GPU 헤드수 H=12, D=128."""
    H, D = 12, D_K3
    dev = "cuda"
    torch.manual_seed(7)

    def timeit(fn, iters=10, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # prefill: 1 seq × 8192 tok
    inp = make_inputs([0, 8192], H, D, torch.bfloat16, seed=7)

    def run_chunk():
        mod.chunk_kda(q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta=inp["beta"],
                      initial_state=inp["h0"].transpose(-1, -2).contiguous(),
                      output_final_state=True, cu_seqlens=inp["cu_t"],
                      use_qk_l2norm_in_kernel=True)

    ms = timeit(run_chunk)
    emit(mod_name, "PERF_chunk_prefill", "INFO", shape=f"T8192_H{H}_D{D}",
         ms=f"{ms:.3f}", tok_per_s=f"{8192 / ms * 1e3:.0f}")

    # decode: 32 seq × 1 tok (main 사본 시그니처 기준 — k3 사본은 raw 입력)
    cu32 = [0] + list(range(1, 33))
    inp2 = make_inputs(cu32, H, D, torch.bfloat16, seed=8)
    pool = inp2["h0"].transpose(-1, -2).contiguous()
    idx = torch.arange(32, dtype=torch.long, device=dev)
    if mod_name == "k3":
        def run_dec():
            mod.fused_recurrent_kda(
                q=inp2["q"], k=inp2["k"], v=inp2["v"],
                raw_g=inp2["raw_g"], raw_beta=inp2["raw_beta"],
                A_log=inp2["A_log"], dt_bias=inp2["dt_bias"], lower_bound=LOWER_BOUND,
                initial_state=pool, cu_seqlens=inp2["cu_t"], ssm_state_indices=idx)
    else:
        gf = inp2["g"].float()
        bf = inp2["beta"].float()

        def run_dec():
            mod.fused_recurrent_kda(
                q=inp2["q"], k=inp2["k"], v=inp2["v"], g=gf, beta=bf,
                initial_state=pool, inplace_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=inp2["cu_t"],
                ssm_state_indices=idx)

    ms = timeit(run_dec)
    emit(mod_name, "PERF_recurrent_decode", "INFO", shape=f"B32_T1_H{H}_D{D}",
         ms=f"{ms:.3f}", note="69층 환산≈" + f"{ms * 69:.2f}ms/step(KDA만,통신제외)")


# ----------------------------------------------------------------------------
def run_guard(fn, impl, test_id, *a, **kw):
    """크래시를 verdict=CRASH로 격리(다음 테스트 계속)."""
    try:
        return fn(*a, **kw)
    except Exception:
        tb = traceback.format_exc().splitlines()
        emit(impl, test_id, "CRASH", error=tb[-1][:200].replace("|", "/"))
        print("\n".join(tb[-12:]), file=sys.stderr)
        return None  # None = crash (False=수치FAIL과 구분)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default="auto", choices=["auto", "main", "k3", "fla"])
    p.add_argument("--dtype", default="both", choices=["bf16", "fp16", "both"])
    p.add_argument("--perf", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        emit("-", "env", "FAIL_NO_CUDA")
        sys.exit(3)
    cap = torch.cuda.get_device_capability()
    try:
        triton_ver = __import__("triton").__version__
    except Exception:
        triton_ver = "?"  # triton 부재 시에도 env 라인은 남긴다 (import 실패는 각 테스트에서 격리)
    emit("-", "env", "INFO", gpu=torch.cuda.get_device_name(0).replace(" ", "_"),
         capability=str(cap).replace(" ", ""), torch=torch.__version__,
         triton=triton_ver)
    if cap != (8, 0) and os.environ.get("PROBE_ALLOW_NON_SM80") != "1":
        emit("-", "env", "FAIL_NOT_SM80", capability=str(cap).replace(" ", ""))
        sys.exit(3)

    torch.manual_seed(args.seed)
    dtypes = {"bf16": [torch.bfloat16], "fp16": [torch.float16],
              "both": [torch.float16, torch.bfloat16]}[args.dtype]
    impls = ["main", "k3", "fla"] if args.impl == "auto" else [args.impl]

    CU_MIX = [0, 15, 100, 300, 556]   # 청크(64) 비정렬 + 장단 혼합 varlen
    CU_ONE = [0, 1024]

    any_crash = False
    any_fail = False
    n_loaded = 0
    for name in impls:
        mod = load_impl(name)
        if mod is None:
            if args.impl != "auto":
                sys.exit(3)
            continue
        n_loaded += 1
        H = 4
        for dt in dtypes:
            for cu in (CU_MIX, CU_ONE):
                tid = "T1_chunk_vs_naive" if name != "k3" else "T4_chunk_vs_naive"
                r = run_guard(t_chunk_vs_naive, name, tid, name, mod, cu, H, dt, tid)
                any_crash |= r is None
                any_fail |= r is False
            if name == "main":
                r = run_guard(t2_recurrent_vs_naive, name, "T2_recurrent_vs_naive",
                              mod, CU_MIX, H, dt)
                any_crash |= r is None
                any_fail |= r is False
                r = run_guard(t3_handoff, name, "T3_handoff", mod, H, dt)
                any_crash |= r is None
                any_fail |= r is False
            if name == "k3":
                r = run_guard(t5_fused_gate_vs_naive, name, "T5_fused_gate_vs_naive",
                              mod, CU_MIX, H, dt)
                any_crash |= r is None
                any_fail |= r is False
                r = run_guard(t6_recurrent_vs_chunk_k3, name, "T6_recurrent_vs_chunk",
                              mod, CU_MIX, H, dt)
                any_crash |= r is None
                any_fail |= r is False
        if args.perf and name in ("main", "k3"):
            run_guard(perf_smoke, name, "PERF", name, mod)

    print("\n==== PROBE_KDA SUMMARY ====")
    npass = sum(1 for r in RESULTS if r["verdict"] == "PASS")
    nfail = sum(1 for r in RESULTS if r["verdict"] == "FAIL")
    ncrash = sum(1 for r in RESULTS if r["verdict"] == "CRASH")
    print(f"PASS={npass} FAIL={nfail} CRASH={ncrash} "
          f"(thr base={THR_BASE} gate={THR_GATE})")
    if n_loaded == 0:
        # auto 모드에서 모든 구현 import 실패 — false-green 방지 (exit 계약: 3=임포트 실패)
        print("PROBE_KDA|impl=-|test=summary|verdict=NO_IMPL_LOADED")
        sys.exit(3)
    # RESULTS 기반으로 재산정: run_guard가 격리한 CRASH(perf 포함)를 exit code에 반영
    any_crash = any_crash or ncrash > 0
    any_fail = any_fail or nfail > 0
    sys.exit(3 if any_crash else (2 if any_fail else 0))


if __name__ == "__main__":
    main()
