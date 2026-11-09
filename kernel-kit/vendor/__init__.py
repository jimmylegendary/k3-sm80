# k3-sm80 kernel-kit vendor — vLLM main 핀 커밋(a3c14efb)의 Triton 커널 vendored 사본
# + 이미지에 실리는 SM80 DSA 커널 패치본(dsa_ops/ = a3c14efb + triton-mla-sparse-sm80.patch, 원천 PR #47629 head 064801dd).
# 목적: k01~k04·k07·k11·k12 하네스를 vLLM 설치 없이 torch+triton 만으로 실행 (PLAN §S2 전달 모델).
# 원본 대응·스텁 목록: vendor/STUBS.md / vllm 미의존 게이트: vendor/check_no_vllm.py
"""하네스가 쓰는 커널 심볼을 원래 이름으로 노출한다 (PEP 562 lazy).

lazy 인 이유: 커널별 하네스가 자기 커널만 import 하므로, 무관한 서브모듈의
import 실패(예: numpy 부재가 k01 을 죽이는 상황)를 격리한다.

  from vendor import chunk_kda_with_fused_gate          # k01 (fla_kda)
  from vendor import fused_recurrent_kda_packed_decode  # k02 (fla_kda)
  from vendor import causal_conv1d_fn                   # k03 (mamba_ops)
  from vendor import decode_attention_fwd               # k04 (attn_ops)
  from vendor import situ_and_mul_native                # k07 (situ)
  from vendor.dsa_ops import fp8_paged_mqa_logits_triton  # k11 (dsa_ops — 하네스가 직접 import)
"""

import importlib

# 핀: vLLM main 2026-11-09 (K3 코드가 vllm/models/kimi_k3 로 병합된 뒤, K3.1 split 3/6 포함) —
# 모든 vendored 파일 헤더와 동일(dsa_ops 는 같은 핀에 '+ 패치' 가 붙은 별도 헤더 형식). 이전 핀 0498dc7e(PR #50000 head)는 퇴역.
VENDOR_PIN = "a3c14efbc77c018e8bf579935095a87721349e05"
# dsa_ops/ 는 VENDOR_PIN 위에 아래 패치를 적용한 파일 2개(= 이미지에 실리는 커널). DSA_OPS_PIN 은 그 패치의 원천:
# vLLM PR #47629 head (원본 #38476 haosdent/thomaslwang) — 패치는 PR 원본에 int64 승격·workspace 주입을 더했다.
DSA_OPS_PIN = "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
DSA_OPS_PATCH = "docker/patches-k31/triton-mla-sparse-sm80.patch"
DSA_OPS_PATCH_SHA256 = "74675e998be0022aa4caed7df4467b62148fad09a6af9e47d13ea1b302fb7ed8"

# 심볼 → 서브모듈 (fla_kda 는 원본 kda/__init__.py 의 9개 공개 심볼 전부 — 핀 a3c14efb 재대조 일치)
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
