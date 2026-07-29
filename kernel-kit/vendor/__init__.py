# k3-sm80 kernel-kit vendor — vLLM kimi-k3 핀 커밋의 Triton 커널 vendored 사본.
# 목적: k01~k04·k07 하네스를 vLLM 설치 없이 torch+triton 만으로 실행 (PLAN §S2 전달 모델).
# 원본 대응·스텁 목록: vendor/STUBS.md / vllm 미의존 게이트: vendor/check_no_vllm.py
"""하네스가 쓰는 커널 심볼을 원래 이름으로 노출한다 (PEP 562 lazy).

lazy 인 이유: 커널별 하네스가 자기 커널만 import 하므로, 무관한 서브모듈의
import 실패(예: numpy 부재가 k01 을 죽이는 상황)를 격리한다.

  from vendor import chunk_kda_with_fused_gate          # k01 (fla_kda)
  from vendor import fused_recurrent_kda_packed_decode  # k02 (fla_kda)
  from vendor import causal_conv1d_fn                   # k03 (mamba_ops)
  from vendor import decode_attention_fwd               # k04 (attn_ops)
  from vendor import situ_and_mul_native                # k07 (situ)
"""

import importlib

# 핀: vLLM kimi-k3 브랜치 (PR #50000 head) — 모든 vendored 파일 헤더와 동일
VENDOR_PIN = "0498dc7ea9204f1863b46515816f5d8d458e01e7"

# 심볼 → 서브모듈 (fla_kda 는 원본 kda/__init__.py 의 9개 공개 심볼 전부)
_EXPORTS = {
    "chunk_kda": ".fla_kda",
    "chunk_kda_fwd": ".fla_kda",
    "chunk_kda_with_fused_gate": ".fla_kda",
    "chunk_kda_with_fused_gate_fwd": ".fla_kda",
    "fused_kda_gate": ".fla_kda",
    "fused_kda_gate_chunk_cumsum": ".fla_kda",
    "fused_recurrent_kda": ".fla_kda",
    "fused_recurrent_kda_fwd": ".fla_kda",
    "fused_recurrent_kda_packed_decode": ".fla_kda",
    "causal_conv1d_fn": ".mamba_ops.causal_conv1d",
    "causal_conv1d_update": ".mamba_ops.causal_conv1d",
    "decode_attention_fwd": ".attn_ops.triton_decode_attention",
    "situ_and_mul_native": ".situ",
}

__all__ = ["VENDOR_PIN", *sorted(_EXPORTS)]


def __getattr__(name: str):
    mod_path = _EXPORTS.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_path, __name__)
    obj = getattr(mod, name)
    globals()[name] = obj  # 이후 접근은 캐시
    return obj


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
