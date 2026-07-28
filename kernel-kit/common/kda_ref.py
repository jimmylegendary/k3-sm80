"""KDA fp32 참조구현 (k01/k02/k03 공용) — torch 경로(GPU/CPU) + 순수 python 경로(dry-run).

원천·재배포 고지 (전부 핀 커밋에서 원문 확인, 2026-07-28):
  - naive recurrent KDA: vLLM kimi-k3 브랜치 @ 0498dc7ea9204f1863b46515816f5d8d458e01e7
    tests/models/kimi_k3/test_kda.py `naive_recurrent_kda` (L54-89) 이식.
    SPDX-License-Identifier: Apache-2.0, Copyright contributors to the vLLM project.
    해당 함수는 FLA(flash-linear-attention) naive.py 포팅본
    (MIT License, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li).
  - causal_conv1d 참조: 동 커밋 tests/kernels/mamba/test_causal_conv1d.py
    `causal_conv1d_ref`(L30-69) / `causal_conv1d_update_ref`(L72-122) 이식 (Apache-2.0).
  - gate 수식: 동 커밋 vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py
    `kda_gate_fwd_kernel`(L827-889)에서 확정 —
      lower_bound 有: g = lower_bound · sigmoid(exp(A_log) · (raw_g + g_bias))
      lower_bound 無: g = −exp(A_log) · softplus(raw_g + g_bias; beta=1, threshold=20)
    fused 경로(chunk fused-gate / fused_recurrent / packed_decode)는 beta = sigmoid(raw_beta)
    를 커널 내부에서 적용 (fused_recurrent.py `_kda_gate_beta_fwd_kernel`,
    `fused_recurrent_kda_packed_decode_kernel` L575-577 확인).
  - l2norm: 동 커밋 vllm/third_party/flash_linear_attention/ops/l2norm.py —
    y = x / sqrt(Σx² + eps), eps=1e-6. packed_decode 커널 내부도 동일 eps(L552-554).
  - state 레이아웃: 커널 = (…, H, V, K). naive 참조는 (…, H, K, V) — 전치 관계
    (test_kda.py L177-192 "h0 transposed to (V, K) layout for the kernel").

수학 (KDA 점화식, delta-rule gated — Kimi Linear arXiv:2510.26692):
  S ← S ⊙ exp(g_t)[k축, v로 broadcast]
  S ← S + (β_t k_t) ⊗ (v_t − k_tᵀ S)
  o_t = q_tᵀ S,  q_t·scale(=K^-1/2), q/k는 l2norm.
  chunk(prefill)와 recurrent(decode)는 동일 점화식의 다른 평가 순서 —
  본 참조 recurrent 하나로 k01(chunk)·k02(recurrent) 모두 판정한다(둘의 일치 = 수학 검증).

dry-run 계약: 이 모듈의 selftest()는 torch/numpy 없이 stdlib만으로
참조구현의 자기일관성(전체 실행 vs state-이월 분할 실행 동치, gate/l2norm/conv 성질,
결정성)을 검증한다. 실커널 판정이 아니다.
"""

from __future__ import annotations

import json
import math
import random
import sys

try:  # 저작 호스트에는 torch 없음 — torch 경로는 GPU 하네스에서만 사용
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

L2NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0


# ===========================================================================
# 1) torch 경로 (fp32 참조 — GPU 모드에서 사용, CPU에서도 동작)
# ===========================================================================
def l2norm_ref(x, eps: float = L2NORM_EPS):
    """fp32 l2 정규화(마지막 축). vLLM l2norm.py 와 동일 산식: x/sqrt(Σx²+eps)."""
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate_ref(raw_g, A_log, g_bias, lower_bound: float | None):
    """raw_g (..., H, D) → 게이트(log-공간, fp32). kda_gate_fwd_kernel 수식 그대로."""
    H = A_log.numel()
    D = raw_g.shape[-1]
    a = A_log.float().exp().view(H, 1)          # (H,1) — head별 exp(A_log)
    x = raw_g.float()
    if g_bias is not None:
        x = x + g_bias.float().view(H, D)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * x)
    return -a * F.softplus(x, beta=1.0, threshold=SOFTPLUS_THRESHOLD)


def naive_recurrent_kda_raw(q, k, v, raw_g, raw_beta, A_log, dt_bias,
                            lower_bound: float | None, scale: float | None = None,
                            initial_state=None, l2norm_qk: bool = True,
                            progress: int | None = None):
    """raw 입력(bf16 가능)에서 fp32 참조 산출 — 토큰 단위 스트리밍(fp32 전체 사본 없음).

    q/k/v/raw_g: (B, T, H, D), raw_beta: (B, T, H).
    initial_state: (B, H, K, V) fp32 naive 레이아웃 (None=0).
    progress: N 토큰마다 stderr 진행 로그(대형 T 에서 hang 오인 방지; None=무출력).
    반환: (o fp32 (B,T,H,V), S fp32 (B,H,K,V)).
    test_kda.py naive_recurrent_kda 의 점화식과 동일(입력만 raw → gate/beta 내부 계산).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        if progress and t and t % progress == 0:
            print(f"    [kda_ref] 참조 recurrent {t}/{T} tok", file=sys.stderr, flush=True)
        q_t = q[:, t].float()
        k_t = k[:, t].float()
        if l2norm_qk:
            q_t = l2norm_ref(q_t)
            k_t = l2norm_ref(k_t)
        q_t = q_t * scale
        v_t = v[:, t].float()
        g_t = kda_gate_ref(raw_g[:, t], A_log, dt_bias, lower_bound)   # (B,H,K)... (B,H,D)
        b_t = torch.sigmoid(raw_beta[:, t].float())                     # (B,H)
        S = S * g_t.exp()[..., None]
        S = S + torch.einsum("bhk,bhv->bhkv",
                             b_t[..., None] * k_t,
                             v_t - (k_t[..., None] * S).sum(-2))
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q_t, S)
    return o, S


def kda_reference_varlen(q, k, v, raw_g, raw_beta, A_log, dt_bias,
                         lower_bound: float | None, initial_state_vk, cu_seqlens,
                         progress: int | None = None):
    """커널 호출과 동일 인자(varlen, B=1 레이아웃)의 fp32 참조.

    q/k/v/raw_g: (1, T_total, H, D), raw_beta: (1, T_total, H).
    initial_state_vk: (N, H, V, K) fp32 — **커널 레이아웃**. 내부에서 (K,V)로 전치.
    반환: (o fp32 (1,T_total,H,V), final_state_vk fp32 (N,H,V,K) — 커널 레이아웃).
    균일 길이면 배치 일괄 처리(정확히 동일 수학, T-루프만 단축).
    """
    cu = [int(x) for x in cu_seqlens]
    lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    N = len(lens)
    h0_kv = initial_state_vk.float().transpose(-1, -2)   # (N,H,K,V) naive 레이아웃
    if len(set(lens)) == 1:  # 균일 길이 → (N, L, H, D) 배치 경로
        L = lens[0]
        qb = q.reshape(N, L, *q.shape[2:])
        kb = k.reshape(N, L, *k.shape[2:])
        vb = v.reshape(N, L, *v.shape[2:])
        gb = raw_g.reshape(N, L, *raw_g.shape[2:])
        bb = raw_beta.reshape(N, L, raw_beta.shape[2])
        o, S = naive_recurrent_kda_raw(qb, kb, vb, gb, bb, A_log, dt_bias,
                                       lower_bound, initial_state=h0_kv,
                                       progress=progress)
        return (o.reshape(1, N * L, *o.shape[2:]),
                S.transpose(-1, -2).contiguous())
    outs, finals = [], []
    for i in range(N):
        s, e = cu[i], cu[i + 1]
        o_i, S_i = naive_recurrent_kda_raw(
            q[:, s:e], k[:, s:e], v[:, s:e], raw_g[:, s:e], raw_beta[:, s:e],
            A_log, dt_bias, lower_bound, initial_state=h0_kv[i:i + 1],
            progress=progress)
        outs.append(o_i)
        finals.append(S_i)
    return (torch.cat(outs, dim=1),
            torch.cat(finals, dim=0).transpose(-1, -2).contiguous())


# --- causal_conv1d 참조 (k03) ----------------------------------------------
def causal_conv1d_ref(x, weight, initial_state=None, silu: bool = True):
    """prefill(fn) 참조 — test_causal_conv1d.py causal_conv1d_ref 이식(fp32).

    x: (C, L), weight: (C, W), initial_state: (C, W-1)|None. bias 없음(K3 conv bias=False).
    반환: (out fp32 (C, L), final_state fp32 (C, W-1)).
    """
    x = x.float()
    weight = weight.float()
    C, W = weight.shape
    L = x.shape[-1]
    if initial_state is None:
        xin = x
        out = F.conv1d(x.unsqueeze(0), weight.unsqueeze(1), None,
                       padding=W - 1, groups=C).squeeze(0)
    else:
        xin = torch.cat([initial_state.float(), x], dim=-1)
        out = F.conv1d(xin.unsqueeze(0), weight.unsqueeze(1), None,
                       padding=0, groups=C).squeeze(0)
    out = out[..., :L]
    # 최종 state = (init‖x)의 마지막 W-1 열 (짧으면 좌측 0 패드) — 업스트림 F.pad 크롭 방식
    final = F.pad(xin, (W - 1 - xin.shape[-1], 0))
    if silu:
        out = F.silu(out)
    return out, final


def causal_conv1d_update_ref(x, conv_state, weight, silu: bool = True):
    """decode(update) 참조 — causal_conv1d_update_ref 이식(fp32, cache_seqlens 없음).

    x: (B, C) fp32, conv_state: (B, C, W-1) fp32 (in-place 갱신), weight: (C, W).
    반환: out fp32 (B, C).
    """
    x = x.float()
    weight = weight.float()
    W = weight.shape[-1]
    x_new = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)   # (B, C, W)
    conv_state.copy_(x_new[..., -(W - 1):])
    out = (x_new[..., -W:] * weight).sum(-1)                    # depthwise 마지막 위치
    if silu:
        out = F.silu(out)
    return out


# ===========================================================================
# 2) 순수 python 경로 (dry-run 전용 — 소형 shape, stdlib만)
# ===========================================================================
def py_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def py_softplus(x: float, threshold: float = SOFTPLUS_THRESHOLD) -> float:
    return x if x > threshold else math.log1p(math.exp(x))


def py_l2norm(vec: list, eps: float = L2NORM_EPS) -> list:
    n = math.sqrt(sum(v * v for v in vec) + eps)
    return [v / n for v in vec]


def py_gate(raw_g_hd: list, a_log_h: list, bias_hd: list | None,
            lower_bound: float | None) -> list:
    """raw_g[H][D] → gate[H][D] (log-공간)."""
    out = []
    for h, row in enumerate(raw_g_hd):
        a = math.exp(a_log_h[h])
        r = []
        for d, gv in enumerate(row):
            x = gv + (bias_hd[h][d] if bias_hd is not None else 0.0)
            if lower_bound is not None:
                r.append(lower_bound * py_sigmoid(a * x))
            else:
                r.append(-a * py_softplus(x))
        out.append(r)
    return out


def py_recurrent_kda(q, k, v, raw_g, raw_beta, a_log, dt_bias, lower_bound,
                     S0=None, l2norm_qk: bool = True):
    """q/k/v/raw_g: [T][H][D], raw_beta: [T][H], S0: [H][K][V]|None.

    반환: (o [T][H][V], S [H][K][V]). torch 경로와 동일 점화식·동일 순서.
    """
    T = len(q)
    H = len(q[0])
    K = len(q[0][0])
    V = len(v[0][0])
    scale = K ** -0.5
    S = ([[row[:] for row in h] for h in S0] if S0 is not None
         else [[[0.0] * V for _ in range(K)] for _ in range(H)])
    o = []
    for t in range(T):
        gate = py_gate(raw_g[t], a_log, dt_bias, lower_bound)
        o_t = []
        for h in range(H):
            q_t = py_l2norm(q[t][h]) if l2norm_qk else q[t][h][:]
            k_t = py_l2norm(k[t][h]) if l2norm_qk else k[t][h][:]
            q_t = [x * scale for x in q_t]
            beta = py_sigmoid(raw_beta[t][h])
            Sh = S[h]
            for kk in range(K):
                dec = math.exp(gate[h][kk])
                for vv in range(V):
                    Sh[kk][vv] *= dec
            for vv in range(V):
                kts = sum(k_t[kk] * Sh[kk][vv] for kk in range(K))
                delta = v[t][h][vv] - kts
                for kk in range(K):
                    Sh[kk][vv] += beta * k_t[kk] * delta
            o_t.append([sum(q_t[kk] * Sh[kk][vv] for kk in range(K))
                        for vv in range(V)])
        o.append(o_t)
    return o, S


def py_conv1d(x, w, state0=None, silu: bool = True):
    """x: [C][L], w: [C][W], state0: [C][W-1]|None → (out [C][L], final [C][W-1])."""
    C = len(x)
    W = len(w[0])
    out = []
    final = []
    for c in range(C):
        hist = (state0[c][:] if state0 is not None else [0.0] * (W - 1)) + x[c]
        row = []
        for i in range(len(x[c])):
            acc = sum(hist[i + j] * w[c][j] for j in range(W))
            row.append(acc / (1.0 + math.exp(-acc)) if silu else acc)
        out.append(row)
        final.append(hist[-(W - 1):])
    return out, final


def py_conv1d_update(x, state, w, silu: bool = True):
    """x: [C], state: [C][W-1] (in-place shift), w: [C][W] → out [C]."""
    out = []
    for c in range(len(x)):
        hist = state[c] + [x[c]]
        acc = sum(hist[j] * w[c][j] for j in range(len(w[c])))
        state[c] = hist[1:]
        out.append(acc / (1.0 + math.exp(-acc)) if silu else acc)
    return out


def py_flat(nested):
    if isinstance(nested, (int, float)):
        return [float(nested)]
    out = []
    for e in nested:
        out.extend(py_flat(e))
    return out


def _py_rand_case(seed: int, T: int, H: int, D: int):
    rnd = random.Random(seed)

    def g3(t, h, d):
        return [[[rnd.gauss(0, 1) for _ in range(d)] for _ in range(h)]
                for _ in range(t)]

    q, k, v, raw_g = g3(T, H, D), g3(T, H, D), g3(T, H, D), g3(T, H, D)
    raw_beta = [[rnd.gauss(0, 1) for _ in range(H)] for _ in range(T)]
    a_log = [0.5 * rnd.gauss(0, 1) for _ in range(H)]
    dt_bias = [[0.1 * rnd.gauss(0, 1) for _ in range(D)] for _ in range(H)]
    S0 = [[[rnd.gauss(0, 1) for _ in range(D)] for _ in range(D)] for _ in range(H)]
    return q, k, v, raw_g, raw_beta, a_log, dt_bias, S0


def _max_abs_diff(a, b) -> float:
    fa, fb = py_flat(a), py_flat(b)
    assert len(fa) == len(fb)
    return max(abs(x - y) for x, y in zip(fa, fb))


def selftest(verbose: bool = True) -> tuple[bool, dict]:
    """순수 python 참조 자기검증 (결정적, stdlib만). (ok, details) 반환.

    핵심: '전체 한 번' vs 'state 이월 분할 실행' 동치 — chunk/recurrent 동치와
    prefill→decode 연속성의 수학이 이 검증 하나로 담보된다(평가 순서만 다른 동일 점화식).
    """
    checks: list[tuple[str, bool, str]] = []
    T, H, D = 11, 2, 4
    for lb in (-5.0, None):
        tag = f"lb={lb}"
        q, k, v, rg, rb, al, db, S0 = _py_rand_case(20260728, T, H, D)
        o_full, S_full = py_recurrent_kda(q, k, v, rg, rb, al, db, lb, S0)
        # (a) state 이월 분할(5+6) 동치
        o1, S1 = py_recurrent_kda(q[:5], k[:5], v[:5], rg[:5], rb[:5], al, db, lb, S0)
        o2, S2 = py_recurrent_kda(q[5:], k[5:], v[5:], rg[5:], rb[5:], al, db, lb, S1)
        d_o = _max_abs_diff(o_full, o1 + o2)
        d_s = _max_abs_diff(S_full, S2)
        checks.append((f"kda_split_carry_o/{tag}", d_o < 1e-12, f"max|Δ|={d_o:.2e}"))
        checks.append((f"kda_split_carry_S/{tag}", d_s < 1e-12, f"max|Δ|={d_s:.2e}"))
        # (b) 결정성 (동일 seed 재구축 → 동일 출력)
        q2, k2, v2, rg2, rb2, al2, db2, S02 = _py_rand_case(20260728, T, H, D)
        o_re, _ = py_recurrent_kda(q2, k2, v2, rg2, rb2, al2, db2, lb, S02)
        d_det = _max_abs_diff(o_full, o_re)
        checks.append((f"kda_determinism/{tag}", d_det == 0.0, f"max|Δ|={d_det:.2e}"))
        # (c) gate 범위: lb 有 → (lb, 0) / lb 無 → (-inf, 0)
        gate = py_gate(rg[0], al, db, lb)
        flat_g = py_flat(gate)
        ok_rng = (all(lb < x < 0 for x in flat_g) if lb is not None
                  else all(x < 0 for x in flat_g))
        checks.append((f"kda_gate_range/{tag}", ok_rng,
                       f"min={min(flat_g):.3f} max={max(flat_g):.3f}"))
    # (d) softplus threshold 분기(>20 → 선형)
    g_lin = py_gate([[25.0]], [0.0], None, None)[0][0]
    checks.append(("gate_softplus_linear", abs(g_lin - (-25.0)) < 1e-9, f"{g_lin:.6f}"))
    # (e) l2norm 단위노름
    nv = py_l2norm([3.0, 4.0])
    nrm = math.sqrt(sum(x * x for x in nv))
    checks.append(("l2norm_unit", abs(nrm - 1.0) < 1e-6, f"|y|={nrm:.8f}"))
    # (f) conv: 전체 vs state 이월 분할(4+5) 동치 + fn↔update 토큰 단위 동치
    rnd = random.Random(20260729)
    C, L, W = 3, 9, 4
    x = [[rnd.gauss(0, 1) for _ in range(L)] for _ in range(C)]
    w = [[rnd.gauss(0, 1) for _ in range(W)] for _ in range(C)]
    s0 = [[rnd.gauss(0, 1) for _ in range(W - 1)] for _ in range(C)]
    of, sf = py_conv1d(x, w, s0)
    o1, s1 = py_conv1d([r[:4] for r in x], w, s0)
    o2, s2 = py_conv1d([r[4:] for r in x], w, s1)
    d_o = _max_abs_diff(of, [a + b for a, b in zip(o1, o2)])
    d_s = _max_abs_diff(sf, s2)
    checks.append(("conv_split_carry_o", d_o < 1e-12, f"max|Δ|={d_o:.2e}"))
    checks.append(("conv_split_carry_S", d_s < 1e-12, f"max|Δ|={d_s:.2e}"))
    st_u = [row[:] for row in s0]
    upd_o = [[] for _ in range(C)]
    for t in range(L):
        step = py_conv1d_update([x[c][t] for c in range(C)], st_u, w)
        for c in range(C):
            upd_o[c].append(step[c])
    d_u = _max_abs_diff(of, upd_o)
    d_us = _max_abs_diff(sf, st_u)
    checks.append(("conv_fn_vs_update_o", d_u < 1e-12, f"max|Δ|={d_u:.2e}"))
    checks.append(("conv_fn_vs_update_S", d_us < 1e-12, f"max|Δ|={d_us:.2e}"))

    ok = all(c[1] for c in checks)
    details = {"checks": len(checks),
               "failed": [{"name": n, "detail": d} for n, p, d in checks if not p],
               "names": [c[0] for c in checks]}
    if verbose:
        for n, p, d in checks:
            print(f"  {'PASS' if p else 'FAIL'} {n:32s} {d}")
        print(json.dumps({"kda_ref_selftest": "PASS" if ok else "FAIL",
                          "checks": len(checks)}, ensure_ascii=False))
    return ok, details


if __name__ == "__main__":
    ok, _ = selftest(verbose=True)
    sys.exit(0 if ok else 1)
