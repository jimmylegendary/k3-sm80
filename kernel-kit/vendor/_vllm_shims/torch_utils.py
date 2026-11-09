# 스텁: vllm.utils.torch_utils.async_tensor_h2d 의 최소 대체 (k3-sm80 kernel-kit vendor).
# 원본: vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/utils/torch_utils.py L722-748.
# 차이: 원본은 PIN_MEMORY(=is_pin_memory_available()) 일 때 page-locked 버퍼에 한 번 스테이징해
# non_blocking 복사를 완전 비동기로 만든다 — 이 킷은 pinned 스테이징을 생략한다(페이지 가능
# 메모리에서의 non_blocking 은 동기 복사로 퇴화 — 값·의미 동일, 지연만 다름).
# 호출처: fla_common/index.py prepare_chunk_indices (varlen chunk 인덱스 H2D). vendor/STUBS.md #torch-utils.
"""host → device 텐서 복사 헬퍼 (pinned 스테이징 생략판)."""

import torch


def async_tensor_h2d(
    data,
    device=None,
    dtype=None,
    out=None,
) -> torch.Tensor:
    """Copy list/numpy array/tensor from host to device."""
    if dtype is None and out is not None:
        dtype = out.dtype
    if isinstance(data, torch.Tensor):
        t = data
    elif hasattr(data, "__array_interface__") or hasattr(data, "__array__"):
        import numpy as np  # allowlist: numpy

        t = torch.from_numpy(np.ascontiguousarray(data))
    else:
        t = torch.tensor(data, dtype=dtype, device="cpu")
    assert t.is_cpu

    if out is not None:
        assert out.dtype == dtype
        return out.copy_(t, non_blocking=True)
    assert device is not None, "must provide destination tensor or device"
    return t.to(device=device, dtype=dtype, non_blocking=True)
