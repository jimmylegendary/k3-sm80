# 스텁: vllm.utils.platform_utils 의 함수 2개 대체 (k3-sm80 kernel-kit vendor).
# 원본: vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/utils/platform_utils.py (74줄) 및
#   vllm/platforms/cuda.py L703-704 (CudaPlatformBase.num_compute_units).
#   - cuda_get_device_properties (원본 L26-37): CUDA 미초기화 시 fork 서브프로세스에서 질의해
#     부모의 CUDA 초기화를 피한다 — 이 킷의 하네스는 어차피 CUDA 를 초기화하므로 직접 질의한다.
#     (fla_common/utils.py check_shared_mem 에서 AMD 분기에서만 호출 — NVIDIA 경로는 안 탄다.)
#   - num_compute_units (원본: platform 위임): torch.cuda.get_device_properties(id).multi_processor_count.
#     dsa_ops/triton_mla_sparse_kernel.py 가 split 수 산정에 쓴다 (sm_count 인자 미지정 시).
# vendor/STUBS.md #platform-utils.
"""CUDA 디바이스 속성 질의 (직접 질의판)."""

from collections.abc import Sequence
from typing import Any

import torch


def cuda_get_device_properties(device, names: Sequence[str], init_cuda=False) -> tuple[Any, ...]:
    props = torch.cuda.get_device_properties(device)
    return tuple(getattr(props, name) for name in names)


def num_compute_units(device_id: int = 0) -> int:
    return torch.cuda.get_device_properties(device_id).multi_processor_count
