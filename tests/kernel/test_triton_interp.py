"""common/triton_interp.py — CPU Triton 인터프리터 보정 3종이 실제로 필요하고 실제로 고치는지 (torch+triton 필요).

보정이 없어도 우연히 통과하는 검사는 의미가 없으므로, 각 보정마다 '보정 전 오답/실패' 를 먼저 확인하고
'보정 후 정답' 을 확인한다. 각 검사는 TRITON_INTERPRET=1 이 triton import 이전에 필요하므로 서브프로세스에서 돈다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import unittest

from kernel_kit_import import KIT_ROOT

HAS_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("triton") is not None


def run_in_interpreter(body: str) -> subprocess.CompletedProcess:
    prog = textwrap.dedent("""
        import os, sys
        os.environ["TRITON_INTERPRET"] = "1"
        sys.path.insert(0, %r)
        import torch, triton, triton.language as tl
    """ % str(KIT_ROOT)) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", prog], text=True, capture_output=True, check=False, timeout=300)


@unittest.skipUnless(HAS_STACK, "torch+triton 필요")
class TritonInterpPatchTest(unittest.TestCase):
    def test_bf16_dot_is_wrong_without_patch_and_right_with_it(self) -> None:
        kernel = """
            @triton.jit
            def dot_k(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
                om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
                a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
                b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
                tl.store(o_ptr + om[:, None] * N + on[None, :], tl.dot(a, b))
            torch.manual_seed(0)
            a = torch.randn(16, 32).to(torch.bfloat16); b = torch.randn(32, 16).to(torch.bfloat16)
            c = torch.zeros(16, 16)
            dot_k[(1,)](a, b, c, 16, 16, 32)
            print("ERR", (c - a.float() @ b.float()).abs().max().item())
        """
        before = run_in_interpreter(kernel)
        self.assertEqual(before.returncode, 0, before.stderr[-1500:])
        self.assertGreater(float(before.stdout.split("ERR")[1]), 1.0)           # 보정 전: 쓰레기값
        after = run_in_interpreter("from common import triton_interp\ntriton_interp.install()\n" + textwrap.dedent(kernel))
        self.assertEqual(after.returncode, 0, after.stderr[-1500:])
        self.assertLess(float(after.stdout.split("ERR")[1]), 1e-5)

    def test_fp32_to_bf16_cast_is_rne_after_patch(self) -> None:
        kernel = """
            @triton.jit
            def cast_k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
                offs = tl.arange(0, BLOCK)
                x = tl.load(x_ptr + offs, mask=offs < n)
                tl.store(o_ptr + offs, x.to(tl.bfloat16).to(tl.float32), mask=offs < n)
            # 1 + 2^-8 은 bf16 의 정확한 중간점(짝수로 내림), 1 + 3·2^-9 는 올림 쪽 — RNE 와 RZ 가 갈리는 값들
            x = torch.tensor([1.0 + 2.0**-8, 1.0 + 1.5 * 2.0**-8, 1.0 + 2.0**-8 + 2.0**-16, 3.14159265, -2.71828], dtype=torch.float32)
            o = torch.empty_like(x)
            cast_k[(1,)](x, o, x.numel(), BLOCK=8)
            want = x.to(torch.bfloat16).float()
            print("RNE_MATCH", bool(torch.equal(o, want)))
        """
        before = run_in_interpreter(kernel)
        self.assertEqual(before.returncode, 0, before.stderr[-1500:])
        self.assertIn("RNE_MATCH False", before.stdout)                       # 보정 전: RZ(버림)
        after = run_in_interpreter("from common import triton_interp\ntriton_interp.install()\n" + textwrap.dedent(kernel))
        self.assertEqual(after.returncode, 0, after.stderr[-1500:])
        self.assertIn("RNE_MATCH True", after.stdout)

    def test_autotuned_kernel_fails_without_patch_and_runs_with_it(self) -> None:
        kernel = """
            @triton.autotune(configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["n"])
            @triton.jit
            def add_k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) + 1.0, mask=offs < n)
            x = torch.randn(100); o = torch.empty(100)
            add_k[lambda meta: (triton.cdiv(100, meta["BLOCK"]),)](x, o, 100)
            print("OK", bool(torch.allclose(o, x + 1.0)))
        """
        before = run_in_interpreter(kernel)
        self.assertNotEqual(before.returncode, 0)                              # 보정 전: 0 active drivers
        self.assertIn("active drivers", before.stderr)
        after = run_in_interpreter("from common import triton_interp\ntriton_interp.install()\n" + textwrap.dedent(kernel))
        self.assertEqual(after.returncode, 0, after.stderr[-1500:])
        self.assertIn("OK True", after.stdout)

    def test_install_requires_interpreter_env(self) -> None:
        res = subprocess.run([sys.executable, "-c", textwrap.dedent("""
            import os, sys
            os.environ.pop("TRITON_INTERPRET", None)
            sys.path.insert(0, %r)
            from common import triton_interp
            try:
                triton_interp.install()
            except RuntimeError as e:
                print("REFUSED", e)
        """ % str(KIT_ROOT))], text=True, capture_output=True, check=False, timeout=120)
        self.assertIn("REFUSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
