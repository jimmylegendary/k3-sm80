"""Portable Triton byte-storage kernels for an E4M3FN MLA KV cache.

These kernels are intentionally software conversion kernels.  SM80 stores one
E4M3FN code per byte and converts to/from BF16 at the cache boundary; no native
FP8 tensor-core operation or FP8 tensor dtype is used.
"""

from __future__ import annotations

import math
from typing import Any


try:  # Host-side contract tests must import without a CUDA Python stack.
    import torch
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on the GPU image
    torch = None
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _round_positive_rne(value):
        lower = tl.floor(value)
        fraction = value - lower
        lower_i = lower.to(tl.int32)
        round_up = (fraction > 0.5) | (
            (fraction == 0.5) & ((lower_i & 1) != 0)
        )
        return lower_i + round_up.to(tl.int32)


    @triton.jit
    def e4m3_bits_to_fp32(bits):
        """Decode uint8 E4M3FN codes without a hardware FP8 conversion."""
        bits_i = bits.to(tl.int32)
        sign = (bits_i >> 7) & 1
        exponent = (bits_i >> 3) & 0xF
        mantissa = bits_i & 0x7
        subnormal = mantissa.to(tl.float32) * (2.0**-9)
        normal = (1.0 + mantissa.to(tl.float32) / 8.0) * tl.exp2(
            exponent.to(tl.float32) - 7.0
        )
        value = tl.where(exponent == 0, subnormal, normal)
        value = tl.where(sign == 0, value, -value)
        is_nan = (exponent == 0xF) & (mantissa == 0x7)
        return tl.where(is_nan, float("nan"), value)


    @triton.jit
    def _fp32_to_e4m3_bits(value):
        """Saturating RNE encoder matching common.fp8_ref."""
        is_nan = value != value
        # The reciprocal preserves the sign of zero on CUDA.
        is_negative = (value < 0.0) | (
            (value == 0.0) & ((1.0 / value) < 0.0)
        )
        magnitude = tl.abs(value)

        sub_q = _round_positive_rne(magnitude * 512.0)
        safe_magnitude = tl.maximum(magnitude, 2.0**-6)
        exponent = tl.floor(tl.log2(safe_magnitude)).to(tl.int32)
        exponent = tl.maximum(-6, tl.minimum(8, exponent))
        step = tl.exp2(exponent.to(tl.float32) - 3.0)
        normal_q = _round_positive_rne(magnitude / step)
        carry = normal_q >= 16
        exponent = exponent + carry.to(tl.int32)
        normal_q = tl.where(carry, 8, normal_q)
        normal_code = ((exponent + 7) << 3) + (normal_q - 8)

        positive = tl.where(magnitude < 2.0**-6, sub_q, normal_code)
        positive = tl.minimum(positive, 0x7E)
        positive = tl.where(magnitude >= 448.0, 0x7E, positive)
        signed = positive | (is_negative.to(tl.int32) << 7)
        return tl.where(is_nan, 0x7F, signed)


    @triton.jit
    def _pack_bf16_to_e4m3_kernel(
        src,
        dst,
        block_table,
        scale,
        n_elements: tl.constexpr,
        cache_dim: tl.constexpr,
        page_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        token = offsets // cache_dim
        channel = offsets - token * cache_dim
        logical_page = token // page_size
        page_offset = token - logical_page * page_size
        physical_page = tl.load(block_table + logical_page, mask=mask, other=0)
        physical_offset = (
            (physical_page * page_size + page_offset) * cache_dim + channel
        )
        values = tl.load(src + offsets, mask=mask, other=0.0).to(tl.float32)
        bits = _fp32_to_e4m3_bits(values / scale)
        tl.store(dst + physical_offset, bits.to(tl.uint8), mask=mask)


    @triton.jit
    def _load_e4m3_to_bf16_kernel(
        src,
        block_table,
        scale,
        dst,
        n_elements: tl.constexpr,
        cache_dim: tl.constexpr,
        page_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        token = offsets // cache_dim
        channel = offsets - token * cache_dim
        logical_page = token // page_size
        page_offset = token - logical_page * page_size
        physical_page = tl.load(block_table + logical_page, mask=mask, other=0)
        physical_offset = (
            (physical_page * page_size + page_offset) * cache_dim + channel
        )
        bits = tl.load(src + physical_offset, mask=mask, other=0).to(tl.uint8)
        values = e4m3_bits_to_fp32(bits) * scale
        tl.store(dst + offsets, values.to(tl.bfloat16), mask=mask)

else:

    def e4m3_bits_to_fp32(bits: Any) -> Any:
        raise RuntimeError("Triton is required for e4m3_bits_to_fp32")


def _require_stack() -> tuple[Any, Any]:
    if torch is None or triton is None:
        raise RuntimeError("torch with CUDA and Triton are required")
    return torch, triton


def _validate_common(
    tensor: Any,
    block_table: Any,
    scale: float,
    n_tokens: int,
    page_size: int,
    cache_dim: int,
) -> None:
    torch_module, _ = _require_stack()
    if not tensor.is_cuda or not block_table.is_cuda:
        raise ValueError("tensor and block_table must be CUDA tensors")
    if block_table.dtype != torch_module.int32 or block_table.ndim != 1:
        raise ValueError("block_table must be contiguous 1-D torch.int32")
    if not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    if n_tokens <= 0 or page_size <= 0 or cache_dim <= 0:
        raise ValueError("n_tokens, page_size and cache_dim must be positive")
    required_pages = (n_tokens + page_size - 1) // page_size
    if block_table.numel() < required_pages:
        raise ValueError(
            f"block_table has {block_table.numel()} entries, "
            f"requires {required_pages}"
        )


def pack_bf16_to_e4m3(
    src: Any,
    dst: Any,
    scale: float,
    *,
    block_table: Any,
    n_tokens: int,
    page_size: int = 64,
    cache_dim: int = 576,
) -> None:
    """Pack logical ``[token, cache_dim]`` BF16 values into physical pages."""
    torch_module, triton_module = _require_stack()
    _validate_common(
        src, block_table, scale, n_tokens, page_size, cache_dim
    )
    if src.dtype != torch_module.bfloat16 or not src.is_contiguous():
        raise ValueError("src must be contiguous CUDA torch.bfloat16")
    if dst.dtype != torch_module.uint8 or not dst.is_contiguous():
        raise ValueError("dst must be contiguous CUDA torch.uint8")
    n_elements = n_tokens * cache_dim
    if src.numel() < n_elements:
        raise ValueError("src is smaller than n_tokens * cache_dim")
    physical_pages = int(block_table[: (n_tokens + page_size - 1) // page_size].max()) + 1
    if dst.numel() < physical_pages * page_size * cache_dim:
        raise ValueError("dst is too small for the physical block table")
    block = 256
    _pack_bf16_to_e4m3_kernel[(triton_module.cdiv(n_elements, block),)](
        src,
        dst,
        block_table,
        float(scale),
        n_elements=n_elements,
        cache_dim=cache_dim,
        page_size=page_size,
        block_size=block,
        num_warps=4,
    )


def load_e4m3_to_bf16(
    src: Any,
    scale: float,
    dst: Any,
    *,
    block_table: Any,
    n_tokens: int,
    page_size: int = 64,
    cache_dim: int = 576,
) -> None:
    """Load physical E4M3FN pages into logical BF16 token order."""
    torch_module, triton_module = _require_stack()
    _validate_common(
        src, block_table, scale, n_tokens, page_size, cache_dim
    )
    if src.dtype != torch_module.uint8 or not src.is_contiguous():
        raise ValueError("src must be contiguous CUDA torch.uint8")
    if dst.dtype != torch_module.bfloat16 or not dst.is_contiguous():
        raise ValueError("dst must be contiguous CUDA torch.bfloat16")
    n_elements = n_tokens * cache_dim
    if dst.numel() < n_elements:
        raise ValueError("dst is smaller than n_tokens * cache_dim")
    physical_pages = int(block_table[: (n_tokens + page_size - 1) // page_size].max()) + 1
    if src.numel() < physical_pages * page_size * cache_dim:
        raise ValueError("src is too small for the physical block table")
    block = 256
    _load_e4m3_to_bf16_kernel[(triton_module.cdiv(n_elements, block),)](
        src,
        block_table,
        float(scale),
        dst,
        n_elements=n_elements,
        cache_dim=cache_dim,
        page_size=page_size,
        block_size=block,
        num_warps=4,
    )
