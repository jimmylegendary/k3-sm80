# vendored from vllm-project/vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/utils/math_utils.py — Apache-2.0
# 변경점: 출처 헤더 + vllm.* import 상대화 재배선뿐(예외는 vendor/STUBS.md) — 본문 무수정.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Math utility functions for vLLM."""

from bisect import bisect_left

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)."""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)."""
    return 0 if n < 1 else 1 << (n.bit_length() - 1)


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    """Round down x to the nearest multiple of y."""
    return (x // y) * y


def largest_power_of_2_divisor(n: int) -> int:
    """Return the largest power-of-2 that divides *n* (isolate lowest set bit)."""
    return n & (-n)


def round_up_to(values: list[int], n: int) -> int | None:
    """Round `n` up to the smallest member of sorted `values` >= n, or None."""
    i = bisect_left(values, n)
    return values[i] if i < len(values) else None
