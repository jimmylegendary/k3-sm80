"""K3.1 DSA(DeepSeek Sparse Attention) fp32 참조구현 — k11(indexer logits)·k12(sparse MLA) 공용.

두 갈래: (A) **순수 python**(stdlib 만, 작은 shape — 호스트 dry-run 자기검증·테스트의 독립 오라클) /
(B) **torch 벡터화**(GPU·CPU 모두 — 하네스 본 대조). 두 갈래는 test_dsa_ref.py 에서 서로·upstream 참조와 대조한다.

원천 (전부 vLLM main @ a3c14efbc77c018e8bf579935095a87721349e05 원문 확인, 2026-11-09):
  - K 경로: vllm/models/kimi_k3/nvidia/dsa.py `KimiK3DSAIndexerKNorm`(L84-106: fp32 LayerNorm weight+bias →
    입력 dtype) + ops/dsa_indexer.py `_index_k_insert`(L36-80): y=LN(x)를 bf16 으로 둥글린 뒤 fp32,
    scale = ue8m0(max|y|, 바닥 1e-4, FP8_MAX 448), 코드 = e4m3(div_rn(y, scale)). CUDA
    `indexer_k_quant_and_cache_kernel`(csrc/cache_kernels.cu:705-773)과 같은 산식.
  - Q 경로: ops/dsa_indexer.py `_indexer_q_quant_fold_weights_kernel`(L178-206): head 별 ue8m0 scale(바닥 1e-10)
    → clamp ±448 → e4m3, 가중치 w' = w·scale·weight_scale. weight_scale = index_softmax_scale·
    indexer_weight_scale = D^-0.5 · H^-0.5 (dsa.py L135·L189·L373).
  - logits: DeepGEMM fp8_mqa_logits / fp8_paged_mqa_logits 의미 — logits[t,n] = Σ_h w'[t,h]·ReLU(q[t,h]·k[n]·ks[n])
    (vllm/v1/attention/ops/rocm_aiter_mla_sparse.py `fp8_mqa_logits_torch` L902-951 ·
    `fp8_paged_mqa_logits_torch` L471-569 와 동치; PR #47629 mqa_logits_triton 커널이 같은 식).
  - 캐시 레이아웃: 블록당 [block_size×128 fp8 | block_size×4B fp32 scale] (cache_kernels.cu:755-764) =
    132 B/token. 논리 shape [num_blocks, block_size, 1, 132] 는 이 바이트열의 stride trick 이다.
  - sparse MLA: FlashMLA tests/ref.py `reference_mla_sparse_prefill`(vLLM tests/kernels/attention/
    test_xpu_mla_sparse.py L24-77) — 선택 행 gather → absorbed MQA(qk 576, v = 앞 512 lane) → softmax.

정직한 상한: 이 모듈은 **오라클**이지 DeepGEMM 과 bit-exact 하지 않다. E4M3 양자화 오차(실제 서빙의 FP8 indexer 가
들이는 오차)는 비교 대상이 아니다 — 하네스는 커널과 오라클이 **같은 양자화 입력**을 공유하게 해 커널 오차만 본다.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

try:  # 저작 호스트에는 torch 없음 — torch 갈래는 GPU/CPU torch 환경에서만 사용
    import torch
except Exception:  # pragma: no cover
    torch = None

from . import fp8_ref

E4M3_MAX = 448.0
INDEX_K_AMAX_FLOOR = 1e-4     # _index_k_insert: _ue8m0_scale(amax, 1e-4, FP8_MAX)
INDEX_Q_AMAX_FLOOR = 1e-10    # _indexer_q_quant_fold_weights_kernel: per_token_group_quant_fp8(eps=1e-10)
TOPK_TIE_BAND_REL = 1e-3      # top-k 경계 동률 밴드: δ = TOPK_TIE_BAND_REL · rms(유효 ref logits)
TOPK_AGREEMENT_MIN = 1.0      # 밴드 밖 확정 불일치 0건 (docstring: 하네스 k11 판정 근거)
PAD_CODE = 0x7E               # 캐시 tail 독극물: 448 (마스킹 실수는 큰 logit 으로 드러난다)
PAD_SCALE = 1.0e30


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ===========================================================================
# (A) 순수 python 갈래 — stdlib 만 (dry-run 자기검증 · 독립 오라클)
# ===========================================================================
def ue8m0_scale(amax: float, floor: float, fp8_max: float = E4M3_MAX) -> float:
    """`_ue8m0_scale`: 2^ceil(log2(max(amax, floor)/fp8_max)) — fp32 산술을 모사(정확한 2의 거듭제곱 경계 보존)."""
    s = _f32(_f32(max(_f32(amax), _f32(floor))) / fp8_max)
    m, e = math.frexp(s)           # s = m·2^e, m ∈ [0.5, 1)
    return math.ldexp(1.0, e - 1 if m == 0.5 else e)


def quantize_k_row(row: Sequence[float]) -> tuple[list[int], float]:
    """index-K 한 행(LayerNorm→bf16 반올림된 값) → (E4M3 코드 D개, ue8m0 scale)."""
    scale = ue8m0_scale(max(abs(x) for x in row), INDEX_K_AMAX_FLOOR)
    return [fp8_ref.encode_e4m3fn(_f32(x / scale)) for x in row], scale


def quantize_q_head(row: Sequence[float]) -> tuple[list[int], float]:
    """index-Q 한 (토큰, head) 행 → (E4M3 코드 D개, ue8m0 scale). 코드는 ±448 로 clamp 후 인코딩."""
    scale = ue8m0_scale(max(abs(x) for x in row), INDEX_Q_AMAX_FLOOR)
    return [fp8_ref.encode_e4m3fn(max(-E4M3_MAX, min(E4M3_MAX, _f32(x / scale)))) for x in row], scale


def decode_codes(codes: Sequence[int]) -> list[float]:
    return [fp8_ref.decode_e4m3fn(c) for c in codes]


def mqa_logit_row(q_codes: Sequence[Sequence[int]], w: Sequence[float],
                  k_codes: Sequence[int], k_scale: float) -> float:
    """한 (질의 행, 키) 쌍: Σ_h w_h · ReLU( Σ_d q_hd·k_d · k_scale ). q_codes = [H][D] 코드."""
    kd = decode_codes(k_codes)
    total = 0.0
    for qh, wh in zip(q_codes, w):
        s = sum(a * b for a, b in zip(decode_codes(qh), kd)) * k_scale
        total += wh * (s if s > 0.0 else 0.0)
    return total


def topk_agreement_py(ref_row: Sequence[float], out_row: Sequence[float], k: int,
                      band_rel: float = TOPK_TIE_BAND_REL) -> dict:
    """한 행의 top-k 집합 일치 (경계 동률 허용) — 정의는 topk_agreement(torch)와 동일.

    유효 위치 = ref 가 유한한 위치. K = min(k, 유효 수). τ = ref 의 K번째로 큰 값, δ = band_rel·rms(유효 ref).
    S_ref/S_out = 각 행의 상위 K 위치 집합. 확정 불일치:
      fp = S_out \\ S_ref 중 ref < τ−δ (커널이 기준 문턱보다 확실히 낮은 것을 뽑음)
      fn = S_ref \\ S_out 중 ref > τ+δ (커널이 확실히 높은 것을 놓침)
    mismatch = max(|fp|, |fn|). agreement = 1 − mismatch/K. (밴드 안의 교환 = 동률로 간주해 불일치 아님)
    """
    valid = [i for i, v in enumerate(ref_row) if math.isfinite(v)]
    kk = min(k, len(valid))
    if kk == 0:
        return {"k": 0, "mismatch": 0, "agreement": 1.0, "overlap": 1.0}
    rms = math.sqrt(sum(ref_row[i] ** 2 for i in valid) / len(valid))
    delta = band_rel * max(rms, 1e-30)

    def top(vals):
        def key(i):
            v = vals[i]
            return -math.inf if (v != v) else v
        return sorted(valid, key=lambda i: (-key(i), i))[:kk]

    s_ref, s_out = set(top(ref_row)), set(top(out_row))
    tau = min(ref_row[i] for i in s_ref)
    fp = sum(1 for i in s_out - s_ref if ref_row[i] < tau - delta)
    fn = sum(1 for i in s_ref - s_out if ref_row[i] > tau + delta)
    mism = max(fp, fn)
    return {"k": kk, "mismatch": mism, "agreement": 1.0 - mism / kk,
            "overlap": len(s_ref & s_out) / kk}


def sparse_mla_py(q: Sequence[Sequence[Sequence[float]]], kv: Sequence[Sequence[float]],
                  indices: Sequence[Sequence[int]], sm_scale: float, d_v: int) -> list[list[list[float]]]:
    """순수 python sparse MLA 참조. q[T][H][Dq], kv[S][Dq], indices[T][topk](-1/범위 밖 = 무효) → out[T][H][d_v]."""
    out: list[list[list[float]]] = []
    s_kv = len(kv)
    for t, qt in enumerate(q):
        sel = [i for i in indices[t] if 0 <= i < s_kv]
        rows = []
        for qh in qt:
            if not sel:
                rows.append([0.0] * d_v)
                continue
            logits = [sm_scale * sum(a * b for a, b in zip(qh, kv[i])) for i in sel]
            m = max(logits)
            es = [math.exp(x - m) for x in logits]
            z = sum(es)
            rows.append([sum(e * kv[i][d] for e, i in zip(es, sel)) / z for d in range(d_v)])
        out.append(rows)
    return out


# ===========================================================================
# (B) torch 벡터화 갈래
# ===========================================================================
def _require_torch():
    if torch is None:
        raise RuntimeError("torch 가 필요한 함수 — torch venv 에서 실행할 것")
    return torch


_POS_VALUES = tuple(fp8_ref.decode_e4m3fn(c) for c in range(0x7F))   # 코드 0x00..0x7E (오름차순, 0 … 448)
_TABLE_CACHE: dict = {}


def _tables(device):
    key = str(device)
    t = _TABLE_CACHE.get(key)
    if t is None:
        pos = torch.tensor(_POS_VALUES, dtype=torch.float32, device=device)
        dec = torch.tensor([fp8_ref.decode_e4m3fn(c) for c in range(256)], dtype=torch.float32, device=device)
        t = _TABLE_CACHE[key] = (pos, dec)
    return t


def e4m3_decode(codes):
    """uint8 코드 → fp32 (fp8_ref 표). 0x7F/0xFF 는 NaN."""
    _require_torch()
    return _tables(codes.device)[1][codes.long()]


def _encode_flat(x):
    pos, _ = _tables(x.device)
    a = x.abs()
    hi = torch.bucketize(a, pos).clamp_(max=pos.numel() - 1)
    lo = (hi - 1).clamp_(min=0)
    d_lo = a - pos[lo]
    d_hi = pos[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))      # 동률 → 짝수 코드 (RNE)
    code = torch.where(pick_hi, hi, lo)
    code = torch.where(a >= E4M3_MAX, torch.full_like(code, 0x7E), code)   # 포화 (inf 포함)
    code = torch.where(torch.isnan(x), torch.full_like(code, fp8_ref.CANONICAL_NAN), code)
    sign = (torch.signbit(x) & ~torch.isnan(x)).to(code.dtype) * 0x80
    return (code + sign).to(torch.uint8)


def e4m3_encode(x):
    """fp32 → uint8 E4M3FN 코드: RNE·±448 포화·NaN→0x7F·부호 있는 0 보존 (fp8_ref.encode_e4m3fn 의 벡터판)."""
    _require_torch()
    x = x.float().contiguous()
    flat = x.view(-1)
    chunk = 1 << 24
    if flat.numel() <= chunk:
        return _encode_flat(flat).view(x.shape)
    out = torch.empty(flat.numel(), dtype=torch.uint8, device=x.device)
    for i in range(0, flat.numel(), chunk):
        out[i:i + chunk] = _encode_flat(flat[i:i + chunk])
    return out.view(x.shape)


def ue8m0_scale_t(amax, floor: float, fp8_max: float = E4M3_MAX):
    """ue8m0_scale 의 텐서판 — log2 구현 차이 없이 frexp 로 정확한 2의 거듭제곱 경계를 보존."""
    s = torch.clamp(amax.float(), min=floor) / fp8_max
    m, e = torch.frexp(s)
    exp = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(s), exp)


def k_layer_norm(x, weight, bias, eps: float):
    """KimiK3DSAIndexerKNorm: fp32 LayerNorm(weight, bias) → 입력 dtype (bf16)."""
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).type_as(x)


def quantize_index_k(y):
    """LayerNorm 출력 y [N, D] (bf16) → (코드 uint8 [N, D], scale fp32 [N]) — `_index_k_insert` 산식."""
    yf = y.float()
    scale = ue8m0_scale_t(yf.abs().amax(dim=-1), INDEX_K_AMAX_FLOOR)
    return e4m3_encode(yf / scale[:, None]), scale


def quantize_index_q(q, w, weight_scale: float):
    """index_q [T, H, D], index_weights [T, H] → (코드 uint8 [T, H, D], 접힌 가중치 fp32 [T, H])."""
    qf = q.float()
    scale = ue8m0_scale_t(qf.abs().amax(dim=-1), INDEX_Q_AMAX_FLOOR)           # [T, H]
    codes = e4m3_encode(torch.clamp(qf / scale[..., None], -E4M3_MAX, E4M3_MAX))
    return codes, w.float() * scale * weight_scale


class _Fp32Exact:
    """matmul 을 TF32 없이 fp32 로 — 오라클 정밀도 보장(컨텍스트 매니저)."""

    def __enter__(self):
        self._old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self._old


def _row_logits(qd, w, kd, k_scale):
    """qd [R,H,D], w [R,H], kd [N,D], k_scale [N] (모두 fp32) → [R,N] — 커널과 같은 연산 순서(scale → ReLU → w → Σ_h)."""
    s = torch.einsum("rhd,nd->rhn", qd, kd)
    s.mul_(k_scale[None, None, :])
    s.relu_()
    s.mul_(w[:, :, None])
    return s.sum(dim=1)


def mqa_logits(q_codes, w, k_codes, k_scale, ks=None, ke=None, temp_elems: int = 1 << 27):
    """prefill 형 dense logits 오라클. q_codes [M,H,D] u8, w [M,H] fp32, k_codes [N,D] u8, k_scale [N] fp32,
    ks/ke [M] int (행별 유효 키 구간 [ks, ke)) → fp32 [M,N] (구간 밖 -inf)."""
    _require_torch()
    m, h, _ = q_codes.shape
    n = k_codes.shape[0]
    with _Fp32Exact():
        qd, kd = e4m3_decode(q_codes), e4m3_decode(k_codes)
        out = torch.empty((m, n), dtype=torch.float32, device=q_codes.device)
        rc = max(1, temp_elems // max(1, h * n))
        for r0 in range(0, m, rc):
            r1 = min(m, r0 + rc)
            out[r0:r1] = _row_logits(qd[r0:r1], w[r0:r1].float(), kd, k_scale.float())
    if ks is not None and ke is not None:
        idx = torch.arange(n, device=out.device)[None, :]
        valid = (idx >= ks.long()[:, None]) & (idx < ke.long()[:, None])
        out.masked_fill_(~valid, float("-inf"))
    return out


def unpack_index_k_cache(cache):
    """[NB, bs, 1, D+4] u8 (또는 [NB, bs*(D+4)]) → (코드 u8 [NB, bs, D], scale fp32 [NB, bs])."""
    nb, bs = cache.shape[0], cache.shape[1]
    flat = cache.reshape(nb, -1)
    d = flat.shape[1] // bs - 4
    return (flat[:, : bs * d].reshape(nb, bs, d),
            flat[:, bs * d:].contiguous().view(torch.float32).reshape(nb, bs))


def pack_index_k_cache(seqs, block_tables, num_blocks: int, block_size: int = 64, device=None,
                       stride_bytes: int | None = None):
    """시퀀스별 (코드 [L,D], scale [L]) → 페이지 캐시 u8 [num_blocks, block_size, 1, D+4].

    block_tables [B, maxblk] int: 논리 블록 j → 물리 블록. 쓰이지 않은 블록과 마지막 블록의 tail 은
    독극물(코드 0x7E=448, scale 1e30)로 채워 마스킹 실수가 큰 logit 으로 드러나게 한다.
    stride_bytes: 블록 간 바이트 stride(기본 = 블록 크기). K3.1 은 index-K 페이지가 packed hybrid KV row 안의
    strided view 라서 `block_idx*stride` 가 int32 를 넘길 수 있다 — 큰 stride 를 주면 그 풀 모양을 만들고, 블록 사이
    틈은 0xA5 로 채워 커널이 stride 를 무시하고 읽으면 값이 어긋나게 한다."""
    _require_torch()
    d = seqs[0][0].shape[-1]
    dev = device or seqs[0][0].device
    row = block_size * (d + 4)
    if stride_bytes is None or stride_bytes == row:
        cache = torch.empty((num_blocks, row), dtype=torch.uint8, device=dev)
    else:
        assert stride_bytes > row and stride_bytes % 4 == 0, (stride_bytes, row)
        backing = torch.empty(num_blocks * stride_bytes, dtype=torch.uint8, device=dev)
        backing.fill_(0xA5)
        cache = backing.as_strided((num_blocks, row), (stride_bytes, 1))
    codes_region = cache[:, : block_size * d].view(num_blocks, block_size, d)
    scale_region = cache[:, block_size * d:].view(torch.float32)             # [NB, bs]
    codes_region.fill_(PAD_CODE)
    scale_region.fill_(PAD_SCALE)
    for i, (codes, scale) in enumerate(seqs):
        length = codes.shape[0]
        nblk = -(-length // block_size)
        blocks = block_tables[i, :nblk].long()
        pad = nblk * block_size - length
        if pad:
            codes = torch.cat([codes, torch.full((pad, d), PAD_CODE, dtype=torch.uint8, device=codes.device)])
            scale = torch.cat([scale, torch.full((pad,), PAD_SCALE, dtype=torch.float32, device=scale.device)])
        codes_region[blocks] = codes.reshape(nblk, block_size, d)
        scale_region[blocks] = scale.reshape(nblk, block_size)
    if stride_bytes is None or stride_bytes == row:
        return cache.view(num_blocks, block_size, 1, d + 4)
    return cache.as_strided((num_blocks, block_size, 1, d + 4), (stride_bytes, d + 4, d + 4, 1))


def paged_mqa_logits(q_codes, w, cache, context_lens, block_tables, max_model_len: int):
    """decode 형 paged logits 오라클 (DeepGEMM fp8_paged_mqa_logits 의미).

    q_codes [B,next_n,H,D] u8, w [B*next_n,H] fp32, cache [NB,bs,1,D+4] u8, context_lens [B],
    block_tables [B,maxblk] → fp32 [B*next_n, max_model_len]. 행 (b,n) 의 인과 한계 q_off = ctx_b−next_n+n:
    키 위치 k 는 k < ctx_b 이고 k ≤ q_off 일 때만 유효, 나머지는 -inf."""
    _require_torch()
    b, nn, h, _ = q_codes.shape
    nb, bs = cache.shape[0], cache.shape[1]
    out = torch.full((b * nn, max_model_len), float("-inf"), dtype=torch.float32, device=q_codes.device)
    kc, ksc = unpack_index_k_cache(cache)
    with _Fp32Exact():
        qd_all = e4m3_decode(q_codes)
        for i in range(b):
            length = int(context_lens[i])
            nblk = -(-length // bs)
            blocks = block_tables[i, :nblk].long()
            kd = e4m3_decode(kc[blocks].reshape(nblk * bs, -1)[:length])
            sc = ksc[blocks].reshape(-1)[:length]
            lg = _row_logits(qd_all[i], w[i * nn:(i + 1) * nn].float(), kd, sc)       # [nn, length]
            pos = torch.arange(length, device=lg.device)[None, :]
            q_off = (length - nn + torch.arange(nn, device=lg.device))[:, None]
            lg.masked_fill_(pos > q_off, float("-inf"))
            out[i * nn:(i + 1) * nn, :length] = lg
    return out


def topk_agreement(ref, out, k: int, limit=None, band_rel: float = TOPK_TIE_BAND_REL) -> dict:
    """행별 top-k 집합 일치 (경계 동률 허용) — 정의는 topk_agreement_py 와 동일.

    ref/out [R,N] fp32, limit [R] int: 행별로 앞 limit 위치만 본다(decode logits tail 은 미정의 메모리)."""
    _require_torch()
    rows = ref.shape[0]
    tot_mism = rows_bad = 0
    min_agree = 1.0
    overlaps = []
    ks_seen = set()
    for r in range(rows):
        n = int(limit[r]) if limit is not None else ref.shape[1]
        rr, oo = ref[r, :n], out[r, :n]
        fin = torch.isfinite(rr)
        nv = int(fin.sum())
        kk = min(k, nv)
        if kk == 0:
            continue
        ks_seen.add(kk)
        rms = float(rr[fin].pow(2).mean().sqrt())
        delta = band_rel * max(rms, 1e-30)
        oo = torch.where(torch.isnan(oo), torch.full_like(oo, float("-inf")), oo)
        idx_ref = torch.topk(torch.where(fin, rr, torch.full_like(rr, float("-inf"))), kk).indices
        idx_out = torch.topk(oo, kk).indices
        m_ref = torch.zeros(n, dtype=torch.bool, device=rr.device)
        m_out = torch.zeros(n, dtype=torch.bool, device=rr.device)
        m_ref[idx_ref] = True
        m_out[idx_out] = True
        tau = rr[idx_ref].min()
        fp = int((m_out & ~m_ref & fin & (rr < tau - delta)).sum())
        fn = int((m_ref & ~m_out & (rr > tau + delta)).sum())
        mism = max(fp, fn)
        tot_mism += mism
        rows_bad += int(mism > 0)
        min_agree = min(min_agree, 1.0 - mism / kk)
        overlaps.append(float((m_ref & m_out).sum()) / kk)
    return {"rows": rows, "k_values": sorted(ks_seen), "mismatch_total": tot_mism, "rows_with_mismatch": rows_bad,
            "min_agreement": min_agree,
            "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else 1.0,
            "min_overlap": min(overlaps) if overlaps else 1.0,
            "band_rel": band_rel}


def sparse_mla(q, kv, indices, sm_scale: float, d_v: int = 512, temp_elems: int = 1 << 27):
    """sparse MLA 오라클 (fp32). q [T,H,Dq], kv [S,Dq] 또는 [S,1,Dq], indices [T,topk] 또는 [T,1,topk]
    (물리 행, <0 또는 ≥S 는 무효) → (out fp32 [T,H,d_v], lse fp32 [T,H]).
    V = kv 의 앞 d_v lane (absorbed latent 공유). 유효 행이 하나도 없는 질의는 out=0, lse=-inf."""
    _require_torch()
    if kv.dim() == 3:
        kv = kv[:, 0, :]
    if indices.dim() == 3:
        indices = indices[:, 0, :]
    t, h, dq = q.shape
    s_kv, topk = kv.shape[0], indices.shape[1]
    out = torch.zeros((t, h, d_v), dtype=torch.float32, device=q.device)
    lse = torch.full((t, h), float("-inf"), dtype=torch.float32, device=q.device)
    tc = max(1, temp_elems // max(1, topk * dq))
    with _Fp32Exact():
        for t0 in range(0, t, tc):
            t1 = min(t, t0 + tc)
            idx = indices[t0:t1].long()
            valid = (idx >= 0) & (idx < s_kv)
            g = kv[torch.where(valid, idx, torch.zeros_like(idx)).reshape(-1)].reshape(t1 - t0, topk, dq).float()
            p = torch.einsum("thd,tkd->thk", q[t0:t1].float(), g) * sm_scale
            p.masked_fill_(~valid[:, None, :], float("-inf"))
            l = torch.logsumexp(p, dim=-1)                                   # [tc,H]
            none = ~valid.any(dim=-1)                                        # 유효 행 0개
            l_safe = torch.where(torch.isinf(l), torch.zeros_like(l), l)
            wgt = torch.exp(p - l_safe[..., None])
            wgt = torch.where(none[:, None, None], torch.zeros_like(wgt), wgt)
            out[t0:t1] = torch.einsum("thk,tkd->thd", wgt, g[..., :d_v])
            lse[t0:t1] = l
    return out, lse
