"""Bit-exact software reference for NVIDIA/ONNX E4M3FN byte storage.

Kimi-K3 lanes 04/05 use this format only as a compact KV-cache storage
representation on SM80.  Arithmetic is performed after dequantization to
BF16; this module never depends on native FP8 instructions.

Policy:
  * round-to-nearest, ties-to-even (RNE)
  * finite overflow and infinities saturate to +/-448
  * NaN is encoded as canonical positive 0x7f
  * signed zero is preserved
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence


MAX_FINITE = 448.0
CANONICAL_NAN = 0x7F


def decode_e4m3fn(bits: int) -> float:
    """Decode one E4M3FN byte into a Python float."""
    if not isinstance(bits, int) or not 0 <= bits <= 0xFF:
        raise ValueError(f"E4M3FN byte must be in [0,255], got {bits!r}")
    sign = -1.0 if bits & 0x80 else 1.0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        magnitude = mantissa * 2.0**-9
    else:
        magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    if magnitude == 0.0:
        return math.copysign(0.0, sign)
    return sign * magnitude


_POSITIVE_CODES = tuple(range(0x7F))
_POSITIVE_VALUES = tuple(decode_e4m3fn(code) for code in _POSITIVE_CODES)


def _nearest_positive_code(value: float) -> int:
    if value >= MAX_FINITE:
        return 0x7E
    upper_index = bisect.bisect_left(_POSITIVE_VALUES, value)
    if upper_index == 0:
        return 0
    if upper_index >= len(_POSITIVE_VALUES):
        return 0x7E
    upper_code = _POSITIVE_CODES[upper_index]
    lower_code = _POSITIVE_CODES[upper_index - 1]
    lower_error = value - _POSITIVE_VALUES[upper_index - 1]
    upper_error = _POSITIVE_VALUES[upper_index] - value
    if lower_error < upper_error:
        return lower_code
    if upper_error < lower_error:
        return upper_code
    # Adjacent E4M3 values differ by one significand LSB.  An even stored
    # significand therefore corresponds to an even byte code.
    return lower_code if lower_code % 2 == 0 else upper_code


def encode_e4m3fn(x: float) -> int:
    """Encode one scalar using saturating E4M3FN RNE semantics."""
    value = float(x)
    if math.isnan(value):
        return CANONICAL_NAN
    negative = math.copysign(1.0, value) < 0.0
    magnitude = abs(value)
    positive = _nearest_positive_code(magnitude)
    return positive | (0x80 if negative else 0)


def pack_reference(
    values: Sequence[float],
    dst: memoryview,
    scale: float,
) -> None:
    """Quantize ``values / scale`` into a writable byte-oriented view."""
    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    if dst.readonly:
        raise ValueError("destination memoryview is read-only")
    if dst.nbytes < len(values):
        raise ValueError(
            f"destination has {dst.nbytes} bytes for {len(values)} values"
        )
    byte_view = dst.cast("B")
    for index, value in enumerate(values):
        byte_view[index] = encode_e4m3fn(float(value) / scale_value)
