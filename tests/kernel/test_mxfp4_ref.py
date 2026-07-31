from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from kernel_kit_import import import_common


mxfp4_ref = import_common("mxfp4_ref")
decode_e2m1 = mxfp4_ref.decode_e2m1
decode_e8m0 = mxfp4_ref.decode_e8m0
mxfp4_moe_reference = mxfp4_ref.mxfp4_moe_reference


def packed_constant(code: int, count: int) -> list[int]:
    if count % 2:
        raise ValueError("count must be even")
    return [code | (code << 4)] * (count // 2)


class Mxfp4DecodeTest(unittest.TestCase):
    def test_all_e2m1_codes_are_defined(self) -> None:
        decoded = [decode_e2m1(bytes([code]))[0] for code in range(16)]

        self.assertEqual(
            decoded,
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        )
        self.assertTrue(all(math.isfinite(value) for value in decoded))
        self.assertEqual(math.copysign(1.0, decoded[8]), -1.0)

    def test_packed_nibbles_are_low_then_high(self) -> None:
        self.assertEqual(decode_e2m1(bytes([0x21, 0xF8])), [0.5, 1.0, -0.0, -6.0])

    def test_e8m0_boundary_exponents(self) -> None:
        decoded = decode_e8m0(bytes([0, 1, 126, 127, 128, 254, 255]))

        self.assertEqual(decoded[0], math.ldexp(1.0, -127))
        self.assertEqual(decoded[1], math.ldexp(1.0, -126))
        self.assertEqual(decoded[2:5], [0.5, 1.0, 2.0])
        self.assertEqual(decoded[5], math.ldexp(1.0, 127))
        self.assertTrue(math.isnan(decoded[6]))


@unittest.skipIf(torch is None, "torch fixture runs in the kernel-kit container")
class Mxfp4MoeReferenceTest(unittest.TestCase):
    def test_two_expert_routing_fixture(self) -> None:
        # [E=2, N=2, K/2=16]. Each row is constant in logical K=32.
        packed = torch.tensor(
            [
                [packed_constant(2, 32), packed_constant(4, 32)],
                [packed_constant(6, 32), packed_constant(1, 32)],
            ],
            dtype=torch.uint8,
        )
        scales = torch.full((2, 2, 1), 127, dtype=torch.uint8)
        x = torch.ones((1, 32), dtype=torch.float32)
        routing = (
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        )

        actual = mxfp4_moe_reference(x, packed, scales, routing)

        # expert0=[32,64], expert1=[128,16] -> .25*e0 + .75*e1
        torch.testing.assert_close(actual, torch.tensor([[104.0, 28.0]]))

    def test_block_scales_apply_per_32_values(self) -> None:
        packed = torch.tensor(
            [[packed_constant(2, 64)]],
            dtype=torch.uint8,
        )
        scales = torch.tensor([[[127, 126]]], dtype=torch.uint8)
        x = torch.ones((1, 64), dtype=torch.float32)
        routing = (
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1.0]], dtype=torch.float32),
        )

        actual = mxfp4_moe_reference(x, packed, scales, routing)

        self.assertEqual(actual.item(), 32.0 + 16.0)

    def test_rejects_nan_scale_and_invalid_expert(self) -> None:
        packed = torch.tensor([[packed_constant(2, 32)]], dtype=torch.uint8)
        x = torch.ones((1, 32), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "E8M0"):
            mxfp4_moe_reference(
                x,
                packed,
                torch.tensor([[[255]]], dtype=torch.uint8),
                (torch.tensor([[0]]), torch.tensor([[1.0]])),
            )
        with self.assertRaisesRegex(ValueError, "expert"):
            mxfp4_moe_reference(
                x,
                packed,
                torch.tensor([[[127]]], dtype=torch.uint8),
                (torch.tensor([[1]]), torch.tensor([[1.0]])),
            )


if __name__ == "__main__":
    unittest.main()
