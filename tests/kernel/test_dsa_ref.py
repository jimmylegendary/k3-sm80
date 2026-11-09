"""common/dsa_ref.py — K3.1 DSA fp32 참조구현의 계약 테스트.

순수 python 갈래는 호스트 python 만으로, torch 갈래는 torch 가 있을 때만 실행한다(없으면 skip):
    /home/jimmy/.venvs/k31-cpu/bin/python -m unittest tests.kernel.test_dsa_ref

torch 갈래는 upstream 참조 함수 3개를 **그대로 이식한 오라클**과 작은 shape 에서 수치 대조한다:
  - fp8_mqa_logits_torch / fp8_paged_mqa_logits_torch : vLLM vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
    @ a3c14efbc77c018e8bf579935095a87721349e05 L902-951 / L471-569 (DeepGEMM tests/test_attention.py 에서 가져온 것)
  - reference_mla_sparse_prefill : vLLM tests/kernels/attention/test_xpu_mla_sparse.py L24-77
    (FlashMLA tests/ref.py 에서 가져온 것; attn_sink 없는 경로만 이식)
  - _fp8_paged_mqa_logits_ref : vLLM PR #47629 head 064801dd tests/kernels/attention/test_mqa_logits_triton.py
    (블록 분리 레이아웃 [bs×128 fp8 | bs×4B scale] 을 올바로 푸는 next_n>1 오라클). 주의: aiter 판
    fp8_paged_mqa_logits_torch 의 next_n>1 분기는 `kv_cache[..., :dim]` 로 **토큰 인터리브** 레이아웃을
    가정해(ROCm 캐시용) CUDA 블록 분리 캐시에서 쓰레기가 나온다 — next_n=1 분기만 대조에 쓴다.
"""

from __future__ import annotations

import math
import random
import unittest

from kernel_kit_import import import_common

dsa_ref = import_common("dsa_ref")
fp8_ref = import_common("fp8_ref")
shapes = import_common("shapes")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

needs_torch = unittest.skipUnless(torch is not None and hasattr(torch, "float8_e4m3fn"), "torch(+fp8 dtype) 필요")


# ---------------------------------------------------------------------------
# 순수 python 갈래 (torch 불필요)
# ---------------------------------------------------------------------------
class PurePythonReferenceTest(unittest.TestCase):
    def test_ue8m0_scale_keeps_exact_power_of_two_boundaries(self) -> None:
        # amax = 448·2^k 이면 scale 이 정확히 2^k (올림 없음), 그보다 한 ulp 이상 크면 2^(k+1)
        self.assertEqual(dsa_ref.ue8m0_scale(448.0, 1e-4), 1.0)
        self.assertEqual(dsa_ref.ue8m0_scale(448.0 * 8, 1e-4), 8.0)
        self.assertEqual(dsa_ref.ue8m0_scale(448.0 * 8 * (1 + 2.0**-20), 1e-4), 16.0)
        self.assertEqual(dsa_ref.ue8m0_scale(3.7, 1e-4), 2.0**-6)       # 3.7/448 = 0.00826 → 2^-6
        # 바닥: amax 가 바닥보다 작으면 바닥/448 의 올림 2의 거듭제곱
        self.assertEqual(dsa_ref.ue8m0_scale(0.0, 1e-4), dsa_ref.ue8m0_scale(1e-9, 1e-4))

    def test_k_row_quantization_error_is_bounded_and_scale_is_pow2(self) -> None:
        rnd = random.Random(7)
        row = [rnd.gauss(0, 1) for _ in range(128)]
        codes, scale = dsa_ref.quantize_k_row(row)
        self.assertEqual(math.frexp(scale)[0], 0.5)                      # 2의 거듭제곱
        dec = [fp8_ref.decode_e4m3fn(c) * scale for c in codes]
        amax = max(abs(x) for x in row)
        for x, y in zip(row, dec):
            # E4M3 정규 구간 상대오차 ≤ 2^-4, 서브노멀/0 근처는 절대 한 스텝 (scale·2^-10) 이내
            self.assertLessEqual(abs(x - y), max(abs(x) * 2.0**-4, scale * 2.0**-9), (x, y))
        self.assertLessEqual(max(abs(fp8_ref.decode_e4m3fn(c)) for c in codes), 448.0)
        self.assertGreaterEqual(amax / scale, 224.0)                     # ue8m0 올림 → 최대값이 [224, 448]

    def test_q_head_quantization_clamps_to_448(self) -> None:
        codes, scale = dsa_ref.quantize_q_head([1e6, -1e6, 0.0, 1.0])
        self.assertTrue(all(math.isfinite(fp8_ref.decode_e4m3fn(c)) for c in codes))
        self.assertLessEqual(max(abs(fp8_ref.decode_e4m3fn(c)) for c in codes), 448.0)

    def test_mqa_logit_row_matches_manual_formula(self) -> None:
        q = [[fp8_ref.encode_e4m3fn(v) for v in (1.0, -2.0, 0.5, 0.0)],
             [fp8_ref.encode_e4m3fn(v) for v in (-1.0, 1.0, 1.0, 1.0)]]
        k = [fp8_ref.encode_e4m3fn(v) for v in (2.0, 1.0, 4.0, -8.0)]
        w = [0.5, -2.0]
        s0 = (1.0 * 2.0 + -2.0 * 1.0 + 0.5 * 4.0 + 0.0) * 0.25          # = 0.5·... k_scale 0.25
        s1 = (-1.0 * 2.0 + 1.0 + 4.0 - 8.0) * 0.25
        expect = 0.5 * max(s0, 0.0) + -2.0 * max(s1, 0.0)
        self.assertAlmostEqual(dsa_ref.mqa_logit_row(q, w, k, 0.25), expect, places=12)

    def test_topk_agreement_ties_and_definite_mismatch(self) -> None:
        ref = [float(i) for i in range(100)] + [-math.inf] * 5
        same = dsa_ref.topk_agreement_py(ref, list(ref), 10)
        self.assertEqual((same["mismatch"], same["agreement"], same["overlap"]), (0, 1.0, 1.0))
        # 경계 근방 near-tie: ref[89] 를 τ=90 바로 아래(1e-6)로 두고 커널이 둘의 순서를 뒤집어 89 를 뽑는다
        # → 밴드(δ=1e-3·rms) 안의 교환이므로 불일치 0 이지만 집합은 달라 overlap<1
        ref_nt = list(ref)
        ref_nt[89] = 90.0 - 1e-6
        near = list(ref_nt)
        near[89], near[90] = 90.0, 90.0 - 1e-6
        res = dsa_ref.topk_agreement_py(ref_nt, near, 10)
        self.assertEqual(res["mismatch"], 0)
        self.assertLess(res["overlap"], 1.0)
        # 확정 불일치: 최상위 값 하나를 -inf 로 떨어뜨리면 놓친 것이 τ+δ 위
        bad = list(ref)
        bad[99] = -math.inf
        res = dsa_ref.topk_agreement_py(ref, bad, 10)
        self.assertEqual(res["mismatch"], 1)
        self.assertAlmostEqual(res["agreement"], 0.9)
        # 유효 위치가 k 보다 적으면 K=유효 수
        few = [1.0, 2.0, -math.inf, -math.inf]
        self.assertEqual(dsa_ref.topk_agreement_py(few, few, 10)["k"], 2)

    def test_sparse_mla_py_masks_invalid_rows_and_zeroes_empty_queries(self) -> None:
        kv = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [9.0, 9.0, 9.0]]
        q = [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]]
        out = dsa_ref.sparse_mla_py(q, kv, [[0, 1, -1, 7], [-1, -1, -1, -1]], 1.0, 2)
        e0, e1 = math.exp(1.0), 1.0
        self.assertAlmostEqual(out[0][0][0], e0 / (e0 + e1), places=12)     # 행 0 의 V[0]=1, 행 1 의 V[0]=0
        self.assertEqual(out[1][0], [0.0, 0.0])                              # 유효 행 없음 → 0

    def test_constants_agree_with_shapes(self) -> None:
        self.assertEqual(shapes.DSA_INDEX_K_BYTES, shapes.DSA_INDEX_HEAD_DIM + 4)
        self.assertEqual(shapes.DSA_TOPK % 16, 0)


# ---------------------------------------------------------------------------
# upstream 이식 오라클 (원문 로직 유지; 이식 외 변경 없음)
# ---------------------------------------------------------------------------
def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def upstream_fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """rocm_aiter_mla_sparse.py fp8_mqa_logits_torch (L902-951) 이식."""
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device
    mask_lo = torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale.reshape(-1)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def upstream_fp8_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables, max_model_len):
    """rocm_aiter_mla_sparse.py fp8_paged_mqa_logits_torch (L471-569) 이식 (FP8_DTYPE = float8_e4m3fn)."""
    import torch.nn.functional as F
    fp8 = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full([batch_size, max_model_len], float("-inf"), device=q.device, dtype=torch.float32)
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = _cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            scale_offset = block_size * dim
            cache_value = cache[..., :scale_offset].view(dtype=fp8).to(torch.float32)
            cache_scale = cache[..., scale_offset:].view(dtype=torch.float32).contiguous()
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.view(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float("-inf"), device=q.device, dtype=torch.float32)
    for i in range(batch_size):
        context_len = context_lens[i]
        context_len_i = int(context_len.item())
        q_offsets = torch.arange(context_len_i - next_n, context_len_i, device=q.device)
        context_limit = torch.full((next_n,), context_len_i, dtype=torch.int32, device=q.device)
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        max_context_len = int(context_limit.max().item())
        for block_rk in range(_cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device=q.device)
            mask = (k_offsets[None, :] < context_limit[:, None]) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size:(block_rk + 1) * block_size] = torch.where(
                k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def pr_fp8_paged_mqa_logits_ref(q, kv_cache, weights, context_lens, block_tables, max_model_len):
    """PR #47629 tests/kernels/attention/test_mqa_logits_triton.py `_fp8_paged_mqa_logits_ref` 이식."""
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * dim
    kv_data = flat[:, :k_end].reshape(num_blocks, block_size, 1, dim).view(fp8_dtype).float()
    scale = flat[:, k_end:].view(torch.float32).reshape(num_blocks, block_size, 1, 1)
    q = q.float()
    kv_data = kv_data * scale
    logits = torch.full([batch_size * next_n, max_model_len], float("-inf"), device=q.device, dtype=torch.float32)
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range(_cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_data[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device=q.device)
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :],
                            (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
                            float("-inf"))
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            block_start = block_rk * block_size
            block_end = min((block_rk + 1) * block_size, max_model_len)
            block_width = block_end - block_start
            logits[i * next_n:(i + 1) * next_n, block_start:block_end] = torch.where(
                k_offsets[None, :block_width] <= q_offsets[:, None], s[:, :block_width], float("-inf"))
    return logits


def upstream_reference_mla_sparse_prefill(q, kv, indices, sm_scale, d_v):
    """test_xpu_mla_sparse.py reference_mla_sparse_prefill (L24-77) 이식 — attn_sink/topk_length 없는 경로."""
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape
    indices = indices.clone().squeeze(1)
    invalid_mask = (indices < 0) | (indices >= s_kv)
    indices[invalid_mask] = 0
    q = q.float()
    gathered_kv = kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    P = q @ gathered_kv.transpose(1, 2)
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")
    orig_lse = torch.logsumexp(P, dim=-1)
    lse_for_o = orig_lse.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]
    orig_lse[orig_lse == float("-inf")] = float("+inf")
    return out, orig_lse


def _rmse_rel(a, b):
    return float(((a - b).pow(2).mean().sqrt()) / (a.pow(2).mean().sqrt() + 1e-30))


def _make_keys(n, d, gen):
    raw = torch.randn(n, d, generator=gen).to(torch.bfloat16)
    return dsa_ref.quantize_index_k(raw)


def _make_queries(rows, h, d, gen):
    q = torch.randn(rows, h, d, generator=gen).to(torch.bfloat16)
    w = torch.randn(rows, h, generator=gen).to(torch.bfloat16)
    return dsa_ref.quantize_index_q(q, w, (d ** -0.5) * (h ** -0.5))


@needs_torch
class TorchReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = torch.Generator().manual_seed(1234)

    # --- E4M3 / ue8m0 -------------------------------------------------------
    def test_e4m3_encode_matches_scalar_reference_including_ties(self) -> None:
        rnd = random.Random(0)
        vals = [0.0, -0.0, 1.0, 448.0, 449.0, 1e9, -1e9, math.inf, -math.inf, 2.0**-10, 3 * 2.0**-10,
                1.0625, 1.1875, 2.0**-9, 464.0, 447.9]
        vals += [rnd.uniform(-500, 500) for _ in range(2000)]
        vals += [rnd.uniform(-1, 1) * 10 ** rnd.uniform(-6, 2) for _ in range(2000)]
        x = torch.tensor(vals, dtype=torch.float32)
        got = dsa_ref.e4m3_encode(x).tolist()
        want = [fp8_ref.encode_e4m3fn(float(v)) for v in x.tolist()]
        self.assertEqual(got, want)
        self.assertEqual(dsa_ref.e4m3_encode(torch.tensor([float("nan")])).tolist(), [0x7F])

    def test_e4m3_encode_matches_torch_native_cast_in_range(self) -> None:
        x = (torch.randn(50000, generator=self.gen) * 60).clamp(-448, 448)
        native = x.to(torch.float8_e4m3fn).view(torch.uint8)
        self.assertTrue(torch.equal(native, dsa_ref.e4m3_encode(x)))
        codes = torch.arange(256, dtype=torch.uint8)
        dec = dsa_ref.e4m3_decode(codes)
        nat = codes.view(torch.float8_e4m3fn).float()
        finite = torch.isfinite(nat)
        self.assertTrue(torch.equal(dec[finite], nat[finite]))

    def test_ue8m0_tensor_matches_python_scalar(self) -> None:
        amax = torch.tensor([0.0, 1e-9, 3.7, 448.0, 3584.0, 3584.0004, 12345.678], dtype=torch.float32)
        got = dsa_ref.ue8m0_scale_t(amax, 1e-4).tolist()
        want = [dsa_ref.ue8m0_scale(float(a), 1e-4) for a in amax.tolist()]
        self.assertEqual(got, want)

    def test_quantizers_match_python_rows(self) -> None:
        raw = torch.randn(5, 128, generator=self.gen).to(torch.bfloat16)
        codes, scale = dsa_ref.quantize_index_k(raw)
        for r in range(5):
            c, s = dsa_ref.quantize_k_row([float(v) for v in raw[r]])
            self.assertEqual(codes[r].tolist(), c)
            self.assertEqual(float(scale[r]), s)
        q = (torch.randn(3, 4, 128, generator=self.gen) * 5).to(torch.bfloat16)
        w = torch.randn(3, 4, generator=self.gen).to(torch.bfloat16)
        ws = 128 ** -0.5 * 32 ** -0.5
        qc, wf = dsa_ref.quantize_index_q(q, w, ws)
        for t in range(3):
            for h in range(4):
                c, s = dsa_ref.quantize_q_head([float(v) for v in q[t, h]])
                self.assertEqual(qc[t, h].tolist(), c)
                self.assertAlmostEqual(float(wf[t, h]), float(w[t, h]) * s * ws, places=9)

    # --- logits 오라클 vs upstream ------------------------------------------
    def test_mqa_logits_matches_upstream_fp8_mqa_logits_torch(self) -> None:
        m, n, h, d = 12, 200, 32, 128
        kc, ksc = _make_keys(n, d, self.gen)
        qc, wf = _make_queries(m, h, d, self.gen)
        ks = torch.tensor([0, 5, 40, 64, 0, 100, 129, 7, 0, 150, 3, 60], dtype=torch.int32)
        ke = torch.tensor([200, 70, 41, 200, 3, 200, 130, 150, 200, 151, 9, 199], dtype=torch.int32)
        mine = dsa_ref.mqa_logits(qc, wf, kc, ksc, ks, ke)
        theirs = upstream_fp8_mqa_logits_torch(
            qc.view(torch.float8_e4m3fn), (kc.view(torch.float8_e4m3fn), ksc), wf, ks, ke)
        self.assertTrue(torch.equal(torch.isinf(mine), torch.isinf(theirs)))
        fin = torch.isfinite(theirs)
        # upstream 은 bf16 einsum 출력(반올림)을 쓴다 → 상대 RMS 5e-3 이내, 우리 쪽은 순수 fp32
        self.assertLess(_rmse_rel(theirs[fin], mine[fin]), 5e-3)

    def test_paged_mqa_logits_matches_upstream_for_next_n_1_and_4(self) -> None:
        h, d, bs = 32, 128, 64
        for next_n, ctxs in ((1, [150, 193]), (4, [130, 257])):
            b = len(ctxs)
            nblk = [_cdiv(c, bs) for c in ctxs]
            total = sum(nblk) + 3
            perm = torch.randperm(total, generator=self.gen)
            bt = torch.zeros(b, max(nblk), dtype=torch.int32)
            o = 0
            seqs = []
            for i, nb_i in enumerate(nblk):
                bt[i, :nb_i] = perm[o:o + nb_i].to(torch.int32)
                o += nb_i
                seqs.append(_make_keys(ctxs[i], d, self.gen))
            cache = dsa_ref.pack_index_k_cache(seqs, bt, total, bs)
            qc, wf = _make_queries(b * next_n, h, d, self.gen)
            qc4 = qc.reshape(b, next_n, h, d)
            cl = torch.tensor(ctxs, dtype=torch.int32)
            # upstream 참조는 max_model_len 이 블록 배수가 아니면 마지막 블록 슬라이스 대입에서 깨진다 → 배수로 맞춘다
            mml = _cdiv(max(ctxs), bs) * bs
            mine = dsa_ref.paged_mqa_logits(qc4, wf, cache, cl, bt, mml)
            refs = {"pr": pr_fp8_paged_mqa_logits_ref(qc4.view(torch.float8_e4m3fn), cache, wf, cl, bt, mml)}
            if next_n == 1:     # aiter 판 next_n>1 분기는 레이아웃 가정이 달라 제외 (모듈 docstring)
                refs["aiter_next_n_1"] = upstream_fp8_paged_mqa_logits_torch(
                    qc4.view(torch.float8_e4m3fn), cache, wf, cl, bt, mml)
            for name, theirs in refs.items():
                self.assertTrue(torch.equal(torch.isinf(mine), torch.isinf(theirs)), f"mask {name} next_n={next_n}")
                fin = torch.isfinite(theirs)
                self.assertLess(_rmse_rel(theirs[fin], mine[fin]), 1e-5, f"{name} next_n={next_n}")

    def test_cache_layout_scale_region_follows_fp8_region(self) -> None:
        d, bs = 128, 64
        codes, scale = _make_keys(70, d, self.gen)
        bt = torch.tensor([[2, 0]], dtype=torch.int32)
        cache = dsa_ref.pack_index_k_cache([(codes, scale)], bt, 3, bs)
        self.assertEqual(tuple(cache.shape), (3, bs, 1, d + 4))
        self.assertEqual(shapes.DSA_INDEX_K_BYTES, cache.shape[-1])
        flat = cache.view(3, -1)
        # 논리 토큰 5 → 물리 블록 2 슬롯 5: fp8 바이트는 [5·128, 6·128), scale 은 bs·128 + 5·4 부터 4B
        self.assertTrue(torch.equal(flat[2, 5 * d:6 * d], codes[5]))
        self.assertEqual(float(flat[2, bs * d + 5 * 4:bs * d + 5 * 4 + 4].contiguous().view(torch.float32)),
                         float(scale[5]))
        # 논리 토큰 66 → 논리 블록 1 = 물리 블록 0 슬롯 2; 블록 0 의 tail(슬롯 6~)과 미사용 블록 1 은 독극물
        self.assertTrue(torch.equal(flat[0, 2 * d:3 * d], codes[66]))
        self.assertTrue(bool((flat[0, 6 * d:7 * d] == dsa_ref.PAD_CODE).all()))
        self.assertTrue(bool((flat[1, :bs * d] == dsa_ref.PAD_CODE).all()))
        c2, s2 = dsa_ref.unpack_index_k_cache(cache)
        self.assertTrue(torch.equal(c2[2, :bs], codes[:bs]))
        self.assertTrue(torch.equal(s2[2], scale[:bs]))

    def test_strided_pool_keeps_layout_and_poisons_gaps(self) -> None:
        d, bs = 128, 64
        row = bs * (d + 4)
        codes, scale = _make_keys(100, d, self.gen)
        bt = torch.tensor([[3, 1]], dtype=torch.int32)
        cache = dsa_ref.pack_index_k_cache([(codes, scale)], bt, 4, bs, stride_bytes=row * 3)
        self.assertEqual(tuple(cache.shape), (4, bs, 1, d + 4))
        self.assertEqual(cache.stride(0), row * 3)
        flat = cache.view(4, -1)                                   # 커널 wrapper 와 같은 view(num_blocks, -1)
        self.assertEqual(flat.stride(0), row * 3)
        c2, s2 = dsa_ref.unpack_index_k_cache(cache)
        self.assertTrue(torch.equal(c2[3], codes[:bs]))
        self.assertTrue(torch.equal(s2[3], scale[:bs]))
        # 블록 사이 틈(0xA5)은 건드리지 않았다 — stride 를 무시하고 읽는 커널은 이 값을 읽게 된다
        backing = cache.as_strided((4 * row * 3,), (1,))
        gap = backing[row:row * 3]
        self.assertTrue(bool((gap == 0xA5).all()))
        # strided 풀에서도 오라클 logits 이 contiguous 풀과 같다
        q, w = _make_queries(1, 32, d, self.gen)
        plain = dsa_ref.pack_index_k_cache([(codes, scale)], bt, 4, bs)
        cl = torch.tensor([100], dtype=torch.int32)
        a = dsa_ref.paged_mqa_logits(q.reshape(1, 1, 32, d), w, plain, cl, bt, 100)
        b = dsa_ref.paged_mqa_logits(q.reshape(1, 1, 32, d), w, cache, cl, bt, 100)
        self.assertTrue(torch.equal(a, b))

    # --- top-k 일치 ----------------------------------------------------------
    def test_topk_agreement_torch_matches_python_including_ties(self) -> None:
        rnd = random.Random(3)
        for trial in range(30):
            n = rnd.choice([20, 64, 200])
            ref = [round(rnd.gauss(0, 1), 1 if trial % 3 == 0 else 6) for _ in range(n)]   # 1/3 은 동률 많음
            for i in rnd.sample(range(n), n // 5):
                ref[i] = -math.inf
            out = [v + (rnd.gauss(0, 1e-5) if trial % 2 else rnd.gauss(0, 0.3)) if math.isfinite(v) else v
                   for v in ref]
            k = rnd.choice([5, 17, n])
            py = dsa_ref.topk_agreement_py(ref, out, k)
            tt = dsa_ref.topk_agreement(torch.tensor([ref]), torch.tensor([out]), k)
            # mismatch(밴드 기반)는 동률 처리와 무관한 정의 — overlap 은 동률 선택에 의존하므로 동률 없는 trial 만 비교
            self.assertEqual(py["mismatch"], tt["mismatch_total"], (trial, py, tt))
            if trial % 3 != 0:
                self.assertAlmostEqual(py["overlap"], tt["mean_overlap"], places=9)

    def test_topk_agreement_limit_ignores_undefined_tail(self) -> None:
        ref = torch.tensor([[3.0, 2.0, 1.0, -float("inf"), -float("inf")]])
        out = torch.tensor([[3.0, 2.0, 1.0, 1e30, 1e30]])        # tail 은 미정의 메모리
        self.assertEqual(dsa_ref.topk_agreement(ref, out, 2, limit=torch.tensor([3]))["mismatch_total"], 0)
        self.assertGreater(dsa_ref.topk_agreement(ref, out, 2)["mismatch_total"], -1)

    # --- sparse MLA ----------------------------------------------------------
    def test_sparse_mla_matches_upstream_reference(self) -> None:
        t, h, dq, s_kv, topk, dv = 5, 12, 576, 300, 64, 512
        q = torch.randn(t, h, dq, generator=self.gen).to(torch.bfloat16)
        kv = torch.randn(s_kv, 1, dq, generator=self.gen).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(s_kv, generator=self.gen)[:topk] for _ in range(t)]).to(torch.int32)
        idx[1, 20:] = -1                  # 부분 무효
        idx[2, :] = -1                    # 전부 무효
        idx[3, 0] = s_kv + 5              # 범위 밖 = 무효
        scale = 192 ** -0.5
        out, lse = dsa_ref.sparse_mla(q, kv, idx, scale, dv)
        up_out, up_lse = upstream_reference_mla_sparse_prefill(q, kv, idx.unsqueeze(1), scale, dv)
        self.assertLess(float((out - up_out).abs().max()), 1e-5)
        self.assertTrue(torch.equal(out[2], torch.zeros_like(out[2])))
        both = torch.isfinite(lse) & torch.isfinite(up_lse)
        self.assertLess(float((lse[both] - up_lse[both]).abs().max()), 1e-5)
        self.assertTrue(bool(torch.isinf(lse[2]).all()))              # 유효 행 없음 → lse=-inf

    def test_sparse_mla_python_and_torch_agree(self) -> None:
        t, h, dq, s_kv, topk, dv = 2, 2, 6, 9, 5, 4
        q = torch.randn(t, h, dq, generator=self.gen).to(torch.bfloat16)
        kv = torch.randn(s_kv, dq, generator=self.gen).to(torch.bfloat16)
        idx = torch.tensor([[0, 3, 8, -1, 4], [1, 1, 2, 7, -1]], dtype=torch.int32)
        want = dsa_ref.sparse_mla_py(q.float().tolist(), kv.float().tolist(), idx.tolist(), 0.3, dv)
        got, _ = dsa_ref.sparse_mla(q, kv, idx, 0.3, dv)
        self.assertLess(float((got - torch.tensor(want)).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
