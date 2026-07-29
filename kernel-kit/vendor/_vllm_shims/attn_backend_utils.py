# 스텁: vllm.v1.attention.backends.utils 에서 상수 2개만 발췌 (k3-sm80 kernel-kit vendor).
# 원본 파일은 attention 백엔드 인프라 전체(config/kv_cache 의존) — vendored
# causal_conv1d.py 가 쓰는 것은 아래 상수뿐 (vendor/STUBS.md #attn-backend-utils).
# 값 원문 확인: vllm@0498dc7ea9204f1863b46515816f5d8d458e01e7
#   vllm/v1/attention/backends/utils.py L45-46.
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0
