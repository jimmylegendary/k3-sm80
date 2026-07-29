# 스텁: vllm.triton_utils 의 최소 대체 (k3-sm80 kernel-kit vendor).
# 원본(vllm/triton_utils/__init__.py @ 0498dc7e)은 triton 부재 시 Placeholder 로
# 바꿔치기하는 가드 + gluon/aggregate/use_tensor_descriptor 재수출 — vendored 커널이
# 실제 쓰는 것은 tl/triton/tldevice 뿐이라 직수입으로 대체 (vendor/STUBS.md #triton-utils).
# triton 미설치 환경에서는 import 자체가 실패한다(의도된 동작 — triton 티어 전제).
import triton  # noqa: F401
import triton.language as tl  # noqa: F401
import triton.language.extra.libdevice as tldevice  # noqa: F401

HAS_TRITON = True

# 원본과 동일 상수 (vllm/triton_utils/__init__.py L28-29)
LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453
