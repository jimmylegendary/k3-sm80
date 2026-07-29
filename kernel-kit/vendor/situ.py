# vendored(발췌) from vllm-project/vllm@0498dc7ea9204f1863b46515816f5d8d458e01e7
#   vllm/model_executor/layers/activation.py L183-190 SituAndMul.forward_native — Apache-2.0
# 발췌 사유: 원본 activation.py 는 CustomOp/distributed/platform 인프라를 eager import
# 하는 레이어 파일 — SM80 배포 경로(forward_native, 순수 torch)만 함수로 추출.
# 수식은 원문 그대로 (변수명 포함). 퓨즈드 CUDA op(_C.situ_and_mul)는 C-확장이라
# vendor 불가 — k07 fused arm 은 vLLM 폴백 전용 (vendor/STUBS.md #situ).
"""SiTU-GLU 활성화 — vLLM SituAndMul.forward_native 의 함수형 발췌.

K3 값: beta=4.0, linear_beta=25.0 (config activation_situ_beta/…_linear_beta).
linear_beta=None 이면 up 은 소프트클립 없이 통과(원본 semantics).
"""

import torch


def situ_and_mul_native(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)
