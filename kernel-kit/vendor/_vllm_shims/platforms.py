# 스텁: vllm.platforms.current_platform 의 최소 대체 (k3-sm80 kernel-kit vendor).
# vendored 커널이 실제 호출하는 메서드만 구현 — 의미는 핀 커밋
# vllm@a3c14efbc77c018e8bf579935095a87721349e05 의 원본과 1:1 대조 (vendor/STUBS.md #platforms):
#   is_cuda / is_rocm / is_cpu  interface.py L196 / L204 / L213 (플랫폼 enum 비교)
#   is_cuda_alike          interface.py L232 (CUDA|ROCM)
#   has_device_capability  interface.py L535-555 (int <major><minor> 또는 (major,minor), 현재 cap >= 요구)
#   get_device_capability  cuda.py L1009-1011 (NonNvml: torch.cuda.get_device_capability → DeviceCapability)
#   get_device_name        cuda.py L1014-1015 (torch.cuda.get_device_name)
#   is_arch_support_pdl    cuda.py L737-743 (torch.cuda capability major >= 9; 예외 시 False)
# 호출처: fla_common/utils.py(is_cuda·is_rocm·is_cuda_alike·get_device_name·get_device_capability(0).major) ·
# fla_kda/chunk_intra.py(is_cuda, has_device_capability(80)) · fla_kda/fused_recurrent.py,
# mamba_ops/causal_conv1d.py(is_arch_support_pdl) · attn_ops/triton_decode_attention.py(is_rocm).
"""torch 빌드 기준의 플랫폼 판별 — vLLM 의 플랫폼 해석 없이 CUDA/ROCm 티어만 지원."""

from typing import NamedTuple

import torch


class DeviceCapability(NamedTuple):
    """interface.py L93 `DeviceCapability` — (major, minor) 튜플과 직접 비교 가능."""

    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        return self.major * 10 + self.minor


class _TorchPlatformShim:
    """torch 빌드(+가시 디바이스) 기준 최소 판별. CPU/XPU 등 다른 플랫폼은 비지원."""

    @staticmethod
    def is_cuda() -> bool:
        return torch.version.hip is None and torch.version.cuda is not None

    @staticmethod
    def is_rocm() -> bool:
        return torch.version.hip is not None

    @staticmethod
    def is_cpu() -> bool:
        # 이 vendor 사본은 CUDA(triton) 티어 전용 — CPU 플랫폼 분기는 항상 비활성.
        return False

    def is_cuda_alike(self) -> bool:
        return self.is_cuda() or self.is_rocm()

    @staticmethod
    def get_device_capability(device_id: int = 0) -> DeviceCapability | None:
        try:
            if not torch.cuda.is_available():
                return None
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        except Exception:
            return None

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def has_device_capability(
        self, capability: tuple[int, int] | int, device_id: int = 0
    ) -> bool:
        current = self.get_device_capability(device_id)
        if current is None:
            return False
        if isinstance(capability, tuple):
            return current >= capability
        return current.to_int() >= capability

    @staticmethod
    def is_arch_support_pdl() -> bool:
        # PDL(Programmatic Dependent Launch) = SM90+ — SM80(A100)에서는 항상 False.
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9


current_platform = _TorchPlatformShim()
