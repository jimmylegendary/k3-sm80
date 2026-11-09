# k3-sm80 kernel-kit vendor — SM80 DSA(DeepSeek Sparse Attention) Triton 커널 (k11 indexer / k12 sparse MLA).
# 출처: **이미지에 실리는 패치본** — vLLM a3c14efb + docker/patches-k31/triton-mla-sparse-sm80.patch 의 신규 파일 2개
# (원천 PR #47629 head 064801dd2bc6ac2e265dc3fa1f5d803d71bde25d, 원본 #38476 haosdent/thomaslwang).
# PR 원본 대비: int64 승격(packed hybrid KV row 의 `row*576` 오프셋이 2**31 을 넘는 구간) + split-KV 스크래치
# workspace 주입(mid_out_getter / max_split_mid_out_numel). 재생성·검증: vendor/revendor_dsa_ops.py,
# 계약 테스트: tests/kernel/test_dsa_ops_vendor.py. 대응표·근거: vendor/STUBS.md §4.
"""하네스(k11/k12)가 쓰는 DSA 커널 심볼을 원래 이름으로 노출한다 (PEP 562 lazy).

  from vendor.dsa_ops import fp8_paged_mqa_logits_triton    # k11 decode (paged, LUT 디코드)
  from vendor.dsa_ops import fp8_mqa_logits_triton          # k11 prefill (dense chunk logits)
  from vendor.dsa_ops import triton_mla_sparse_attention    # k12 sparse MLA (split-KV, mid_out_getter 지원)
  from vendor.dsa_ops import max_split_mid_out_numel        # k12 workspace 예약 상한 (백엔드의 예약식과 동일 함수)
"""

import importlib

_EXPORTS = {
    "fp8_mqa_logits_triton": ".mqa_logits_triton",
    "fp8_paged_mqa_logits_triton": ".mqa_logits_triton",
    "warmup_fp8_mqa_logits_triton": ".mqa_logits_triton",
    "warmup_fp8_paged_mqa_logits_triton": ".mqa_logits_triton",
    "triton_mla_sparse_attention": ".triton_mla_sparse_kernel",
    "max_split_mid_out_numel": ".triton_mla_sparse_kernel",
    "KV_SPLITS_CANDIDATES": ".triton_mla_sparse_kernel",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod_path = _EXPORTS.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(mod_path, __name__), name)
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
