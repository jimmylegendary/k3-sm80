# 스텁: vllm.model_executor.warmup.jit_warmup 에서 `WarmupChoices` 만 발췌 (k3-sm80 kernel-kit vendor).
# 원본: vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/model_executor/warmup/jit_warmup.py L66-73
# (전체 1145줄 = 서버 기동 시 JIT 커널 사전 컴파일 레지스트리/AST 추적기).
# 이 킷은 사전 컴파일(warmup)을 하지 않는다 — fla_common/l2norm.py 가 warmup 입력 정의
# (`_l2norm_fwd_block_warmup_inputs`)에서 이 클래스를 참조할 뿐, 그 함수는 호출되지 않는다.
# vendor/STUBS.md #jit-warmup.
"""warmup 입력 정의용 값 집합 마커 (shim 에서는 보관만)."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarmupChoices:
    """Expand an explicit finite set of values in a traced warmup method."""

    values: tuple[Any, ...]

    def __init__(self, *values: Any) -> None:
        object.__setattr__(self, "values", values)
