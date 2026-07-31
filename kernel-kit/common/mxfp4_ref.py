"""Small, auditable MXFP4/E8M0 CPU reference for the k08 kernel gate.

Logical checkpoint layout:
  packed_w: [experts, output_features, input_features / 2] uint8
  scales:   [experts, output_features, input_features / 32] uint8 E8M0FNU
  routing:  (expert_ids [tokens, topk], weights [tokens, topk])

Each packed byte stores the earlier input feature in its low nibble. This module
does not know Marlin's internal reordered layout; k08 performs that conversion
separately and compares the kernel result with this checkpoint-layout reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


E2M1_VALUES: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
MXFP_BLOCK_SIZE = 32


def decode_e2m1(packed: bytes) -> list[float]:
    """Decode low nibble then high nibble for every packed byte."""

    values: list[float] = []
    for byte in packed:
        values.append(E2M1_VALUES[byte & 0x0F])
        values.append(E2M1_VALUES[(byte >> 4) & 0x0F])
    return values


def decode_e8m0(scales: bytes) -> list[float]:
    """Decode OCP E8M0FNU scales; byte 255 is the sole NaN encoding."""

    return [
        float("nan") if code == 0xFF else math.ldexp(1.0, code - 127)
        for code in scales
    ]


def _routing_tensors(routing, torch):
    if isinstance(routing, dict):
        try:
            expert_ids = routing["expert_ids"]
            weights = routing["weights"]
        except KeyError as exc:
            raise ValueError("routing requires expert_ids and weights") from exc
    elif isinstance(routing, Sequence) and len(routing) == 2:
        expert_ids, weights = routing
    else:
        raise ValueError("routing must be (expert_ids, weights) or a mapping")
    expert_ids = torch.as_tensor(expert_ids, dtype=torch.int64, device="cpu")
    weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
    if expert_ids.ndim != 2 or weights.shape != expert_ids.shape:
        raise ValueError("routing ids/weights must have equal [tokens, topk] shape")
    if not torch.isfinite(weights).all():
        raise ValueError("routing weights must be finite")
    return expert_ids, weights


def _decode_expert(packed, scales, expert: int, torch):
    packed_e = packed[expert].contiguous()
    scales_e = scales[expert].contiguous()
    low = packed_e & 0x0F
    high = (packed_e >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(packed_e.shape[0], -1)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    values = lut[codes.to(torch.int64)]

    if torch.any(scales_e == 0xFF):
        raise ValueError("E8M0 scale byte 255 is NaN")
    exponent = scales_e.to(torch.int32) - 127
    decoded_scales = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    expanded_scales = decoded_scales.repeat_interleave(MXFP_BLOCK_SIZE, dim=-1)
    if expanded_scales.shape != values.shape:
        raise ValueError(
            f"scale layout does not cover weights: {tuple(scales_e.shape)} vs "
            f"{tuple(values.shape)}"
        )
    return values * expanded_scales


def mxfp4_moe_reference(x, packed_w, scales, routing):
    """Compute a routed weight-only MXFP4 expert linear layer in FP32 on CPU."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("mxfp4_moe_reference requires torch") from exc

    x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    packed = torch.as_tensor(packed_w, dtype=torch.uint8, device="cpu")
    scale_tensor = torch.as_tensor(scales, dtype=torch.uint8, device="cpu")
    if x.ndim != 2:
        raise ValueError("x must have [tokens, input_features] shape")
    if packed.ndim != 3 or scale_tensor.ndim != 3:
        raise ValueError("packed_w and scales must be rank-3")
    experts, outputs, packed_inputs = packed.shape
    logical_inputs = packed_inputs * 2
    if x.shape[1] != logical_inputs:
        raise ValueError(f"x K={x.shape[1]} does not match packed K={logical_inputs}")
    if logical_inputs % MXFP_BLOCK_SIZE:
        raise ValueError("MXFP4 input dimension must be divisible by 32")
    if scale_tensor.shape != (
        experts,
        outputs,
        logical_inputs // MXFP_BLOCK_SIZE,
    ):
        raise ValueError("scales must have [E, N, K/32] shape")

    expert_ids, route_weights = _routing_tensors(routing, torch)
    if expert_ids.shape[0] != x.shape[0]:
        raise ValueError("routing token count does not match x")
    if torch.any(expert_ids < 0) or torch.any(expert_ids >= experts):
        raise ValueError(f"routing contains an invalid expert id for E={experts}")

    result = torch.zeros((x.shape[0], outputs), dtype=torch.float32)
    cache: dict[int, object] = {}
    for token in range(x.shape[0]):
        for slot in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, slot])
            if expert not in cache:
                cache[expert] = _decode_expert(
                    packed,
                    scale_tensor,
                    expert,
                    torch,
                )
            result[token].add_(
                torch.mv(cache[expert], x[token]),
                alpha=float(route_weights[token, slot]),
            )
    return result
