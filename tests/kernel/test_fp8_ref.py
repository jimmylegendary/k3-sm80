from __future__ import annotations

import math
import unittest

from kernel_kit_import import import_common


fp8_ref = import_common("fp8_ref")


class E4M3FNReferenceTest(unittest.TestCase):
    def test_decode_is_defined_for_all_bytes(self) -> None:
        values = [fp8_ref.decode_e4m3fn(code) for code in range(256)]

        self.assertEqual(len(values), 256)
        self.assertTrue(math.isnan(values[0x7F]))
        self.assertTrue(math.isnan(values[0xFF]))
        self.assertEqual(values[0x7E], 448.0)
        self.assertEqual(values[0xFE], -448.0)

    def test_encode_roundtrip_for_all_finite_codes(self) -> None:
        for code in range(256):
            value = fp8_ref.decode_e4m3fn(code)
            if math.isfinite(value):
                encoded = fp8_ref.encode_e4m3fn(value)
                self.assertEqual(
                    fp8_ref.decode_e4m3fn(encoded),
                    value,
                    f"code=0x{code:02x} encoded=0x{encoded:02x}",
                )

    def test_round_to_nearest_even_midpoints(self) -> None:
        self.assertEqual(fp8_ref.encode_e4m3fn(2.0**-10), 0x00)
        self.assertEqual(fp8_ref.encode_e4m3fn(3.0 * 2.0**-10), 0x02)
        self.assertEqual(fp8_ref.encode_e4m3fn(1.0625), 0x38)
        self.assertEqual(fp8_ref.encode_e4m3fn(1.1875), 0x3A)

    def test_nan_infinity_and_saturation_policy(self) -> None:
        self.assertEqual(fp8_ref.encode_e4m3fn(math.nan), 0x7F)
        self.assertEqual(fp8_ref.encode_e4m3fn(math.inf), 0x7E)
        self.assertEqual(fp8_ref.encode_e4m3fn(-math.inf), 0xFE)
        self.assertEqual(fp8_ref.encode_e4m3fn(10_000.0), 0x7E)
        self.assertEqual(fp8_ref.encode_e4m3fn(-10_000.0), 0xFE)
        self.assertEqual(fp8_ref.encode_e4m3fn(-0.0), 0x80)

    def test_pack_reference_applies_inverse_scale(self) -> None:
        values = [0.0, 1.0, -2.0, 448.0]
        packed = bytearray(len(values))

        fp8_ref.pack_reference(values, memoryview(packed), scale=2.0)

        restored = [
            fp8_ref.decode_e4m3fn(code) * 2.0 for code in packed
        ]
        self.assertEqual(restored, values)

    def test_page_boundary_does_not_touch_canary(self) -> None:
        page = bytearray([0xAA] + [0x00] * 576 + [0x55])

        fp8_ref.pack_reference(
            [0.0] * 576, memoryview(page)[1:-1], scale=1.0
        )

        self.assertEqual((page[0], page[-1]), (0xAA, 0x55))

    def test_pack_rejects_invalid_destination_or_scale(self) -> None:
        with self.assertRaises(ValueError):
            fp8_ref.pack_reference([1.0, 2.0], memoryview(bytearray(1)), 1.0)
        for scale in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                fp8_ref.pack_reference([1.0], memoryview(bytearray(1)), scale)


if __name__ == "__main__":
    unittest.main()
