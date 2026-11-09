"""CPU Triton 인터프리터(`TRITON_INTERPRET=1`) 실행 보조 — k11/k12 `--interpret` 전용.

목적: GPU 없는 호스트에서 vendored DSA 커널(vendor/dsa_ops)을 **실제로 실행**해 fp32 참조와 대조한다.
이 대조는 PASS 근거가 아니다(정직한 상한) — 하네스는 결과를 evidence.interpret_cpu 에만 기록하고
게이트 판정은 올리지 않는다. 인터프리터가 GPU 와 의미가 다른 지점 3곳을 여기서 보정한다
(triton 3.9.0 `triton/runtime/interpreter.py` 실측 — 보정 없이 돌리면 오답/실행 불가):

  1) `tl.dot` bf16 입력: 인터프리터는 bf16 을 uint16 비트로 저장하는데 create_dot 은 fp8 만 변환해
     bf16 은 비트 패턴끼리 np.matmul 한다 → 쓰레기값(실측 4.5e10). bf16 비트를 fp32 로 올려 matmul.
  2) fp32→bf16 캐스트: cast_impl 이 rounding_mode=None 으로 `_convert_float` 를 불러 **버림(RZ)** 한다.
     GPU 는 cvt.rn(RNE) → p.to(bf16)·출력 bf16 반올림 오차가 비편향 RNE 가 되도록 RNE 로 보정.
  3) `@triton.autotune`: Autotuner 가 GPU 드라이버로 벤치마크하려다 "0 active drivers" 로 죽는다.
     vLLM 의 VLLM_TRITON_FORCE_FIRST_CONFIG(vllm/triton_utils/force_first_config.py)와 같은 방식으로
     첫 config 만 사용한다(커널 의미는 config(BLOCK_N·num_warps) 무관).

install() 은 멱등이고 `TRITON_INTERPRET=1` 이 triton import **이전**에 설정돼 있어야 의미가 있다.
"""

from __future__ import annotations

import os

_installed = False
PATCHES = (
    "bf16 tl.dot: raw uint16 bits -> fp32 matmul",
    "fp32->bf16 cast: RZ(interpreter default) -> RNE (GPU cvt.rn)",
    "Autotuner.run: always first config (no GPU benchmark)",
)


def interpreter_requested() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def install() -> dict:
    """인터프리터 보정 3종 설치. 반환: 버전·패치 목록(리포트 evidence 용)."""
    global _installed
    import numpy as np
    import triton
    import triton.language as tl
    from triton.runtime import autotuner as autotuner_mod
    from triton.runtime import interpreter as interp

    info = {"triton": triton.__version__, "patches": list(PATCHES)}
    if not interpreter_requested():
        raise RuntimeError("TRITON_INTERPRET=1 이 아니다 — triton import 이전에 환경변수를 설정할 것")
    if _installed:
        return info

    builder = interp.InterpreterBuilder
    orig_dot = builder.create_dot
    orig_cast = builder.cast_impl

    def _bf16_bits_to_f32(handle):
        return (handle.data.astype(np.uint32) << np.uint32(16)).view(np.float32)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype == tl.bfloat16 or b.dtype == tl.bfloat16:
            ad = _bf16_bits_to_f32(a) if a.dtype == tl.bfloat16 else a.data
            bd = _bf16_bits_to_f32(b) if b.dtype == tl.bfloat16 else b.data
            return interp.TensorHandle(np.matmul(ad, bd, dtype=d.data.dtype) + d.data, d.dtype.scalar)
        return orig_dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            x = np.ascontiguousarray(src.data, dtype=np.float32)
            bits = x.view(np.uint32)
            lsb = (bits >> np.uint32(16)) & np.uint32(1)
            rounded = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)
            rounded = np.where(np.isnan(x), np.uint16(0x7FC0), rounded).astype(np.uint16)
            return interp.TensorHandle(rounded.reshape(src.data.shape), dst_type.scalar)
        return orig_cast(self, src, dst_type)

    def run_first_config(self, *args, **kwargs):
        config = self.configs[0]
        self.nargs = dict(zip(self.arg_names, args))
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    builder.create_dot = create_dot
    builder.cast_impl = cast_impl
    autotuner_mod.Autotuner.run = run_first_config
    _installed = True
    return info
