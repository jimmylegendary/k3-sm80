# k3-sm80 kernel-kit vendor — vllm/third_party/flash_linear_attention/ops/ 중
# KDA 커널이 의존하는 6개 모듈만 수록 (chunk_delta_h, cumsum, index, l2norm, op, utils).
# 원본 ops/__init__.py 는 GDN 계열 커널 5종을 eager import 하므로 미러하지 않는다
# (kda 미사용 — vendor/STUBS.md #fla-common-init). 하위 모듈을 직접 import 할 것.
