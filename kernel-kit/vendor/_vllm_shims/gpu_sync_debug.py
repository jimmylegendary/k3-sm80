# 스텁: vllm.utils.gpu_sync_debug.gpu_sync_allowed 의 최소 대체 (k3-sm80 kernel-kit vendor).
# 원본(vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/utils/gpu_sync_debug.py, 403줄)은 GPU<->CPU
# 동기화 디버그 감시기 — `enable_gpu_sync_check()` 가 켜지기 전(_SYNC_CHECK_MODE is None)에는
# gpu_sync_allowed() 가 no-op 컨텍스트(_NOOP_CM = nullcontext(), 원본 L145)를 돌려준다 (원본 L369-383).
# 이 킷은 감시기를 켜지 않으므로 비활성 경로 = no-op 만 구현한다 (vendor/STUBS.md #gpu-sync-debug).
# 호출처: fla_common/index.py prepare_chunk_indices (`with gpu_sync_allowed():`).
"""GPU 동기화 감시 비활성 상태의 `gpu_sync_allowed` — 항상 no-op 컨텍스트."""

import contextlib


def gpu_sync_allowed(first_only: bool = False):
    """원본 시그니처 유지. 감시기가 없으므로 허용/비허용 구분이 없다."""
    return contextlib.nullcontext()
