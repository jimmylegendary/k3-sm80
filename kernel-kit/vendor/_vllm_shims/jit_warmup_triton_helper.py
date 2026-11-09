# 스텁: vllm.model_executor.warmup.jit_warmup_triton_helper 의 **런타임 경로**만 이식 (k3-sm80 kernel-kit vendor).
# 원본: vllm@a3c14efbc77c018e8bf579935095a87721349e05 vllm/model_executor/warmup/jit_warmup_triton_helper.py (795줄).
# 원본은 (a) 서버 기동 시 Triton 커널 사전 컴파일(warmup) 키 도출·FakeTensor 실체화·병렬 컴파일 풀과
# (b) 그 위에 얹힌 "dispatch 함수 → (grid, launch kwargs) → kernel[grid](**kwargs)" 런타임 런처로 이뤄진다.
# warmup 은 컴파일만 앞당길 뿐 계산 결과를 바꾸지 않으므로, 이 킷은 (b) 만 이식한다:
#   - `_launch_binding_plan` L37-83 · `_prepare_launch_kwargs` L312-325 · `launch` L327-371(비-warming 분기) ·
#     `_DecoratedTritonJitKernel.__call__` L599-619(dispatch 분기) — 입력 이름 → 커널 인자 이름 바인딩 규칙
#     (`x`→`X`/`x_ptr`, `*_stride{dim}` 자동 바인딩)과 `kernel[grid](**kwargs)` 발사까지 동일.
#   - `TritonWarmupTensor` 는 L183 이후 필드·메서드만(warmup 전용 메타데이터 — 런타임에서는 쓰이지 않음).
# 호출처: fla_common/l2norm.py (`l2norm_fwd_block` 데코레이터). vendor/STUBS.md #jit-warmup-triton-helper.
"""Triton dispatcher 런처 (warmup 비활성판)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, cached_property, update_wrapper
from typing import Any

# Dispatchers select a concrete launch configuration for a Triton kernel. (원본 L31-33)
DispatchSpec = (
    tuple[tuple[int, ...], dict[str, Any]] | tuple[tuple[int, ...], dict[str, Any], Any]
)


@dataclass(frozen=True)
class _LaunchBindingPlan:
    input_targets: tuple[tuple[int, str], ...]
    stride_targets: tuple[tuple[str, str, int], ...]


@cache
def _launch_binding_plan(
    arg_names: tuple[str, ...],
    input_names: tuple[str, ...],
    launch_names: tuple[str, ...],
) -> _LaunchBindingPlan:
    arg_name_set = set(arg_names)
    launch_name_set = set(launch_names)

    def kernel_arg(input_name: str) -> str | None:
        pointer_name = f"{input_name}_ptr"
        candidates = (
            input_name,
            pointer_name,
            input_name.upper(),
            pointer_name.upper(),
        )
        return next(
            (candidate for candidate in candidates if candidate in arg_name_set),
            None,
        )

    input_targets = tuple(
        (index, target)
        for index, name in enumerate(input_names)
        if (target := kernel_arg(name)) is not None and target not in launch_name_set
    )
    bound_names = launch_name_set | {target for _, target in input_targets}
    stride_targets = []
    for target in arg_names:
        if target in bound_names:
            continue
        name, separator, dim = target.rpartition("_stride")
        if not separator:
            name, separator, dim = target.rpartition("_STRIDE")
        dim = dim.removeprefix("_")
        if separator and (not dim or dim.isdigit()):
            pointer = f"{name}_PTR" if separator == "_STRIDE" else f"{name}_ptr"
            source = name if name in bound_names else pointer
            if source in bound_names:
                stride_targets.append((target, source, int(dim or 0)))
    return _LaunchBindingPlan(input_targets, tuple(stride_targets))


@dataclass(frozen=True)
class TritonWarmupTensor:
    """Compile-only tensor metadata used by Triton warmup (런타임에서는 미사용)."""

    dtype: Any
    aligned: bool = True
    shape: tuple[int, ...] = (1,)
    strides: tuple[int, ...] | None = None
    init: Any = 0
    ptr_range_bytes: int = 0

    def data_ptr(self) -> int:
        return 0 if self.aligned else 1

    def ptr_range(self) -> int:
        return self.ptr_range_bytes

    def stride(self, dim: int | None = None) -> int | tuple[int, ...]:
        if self.strides is None:
            strides: list[int] = []
            stride = 1
            for size in reversed(self.shape):
                strides.append(stride)
                stride *= size
            result = tuple(reversed(strides))
        else:
            result = self.strides
        return result if dim is None else result[dim]


class _DispatchedTritonKernel:
    """dispatch 함수를 감싼 Triton 커널 런처 — 호출 시 dispatch → 바인딩 → kernel[grid](**kwargs)."""

    def __init__(
        self,
        kernel: Any,
        warmup_inputs: Callable[..., Any],
        dispatch: Callable[..., DispatchSpec],
    ) -> None:
        self.kernel = kernel
        self._warmup_inputs_fn = warmup_inputs  # 보관만 — shim 은 warmup 을 수행하지 않는다
        self._dispatch_fn = dispatch
        parameters = tuple(inspect.signature(dispatch).parameters.values())
        if any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        ):
            raise TypeError("Triton dispatchers require explicit parameters")
        self._dispatch_arg_names = tuple(parameter.name for parameter in parameters)
        self._dispatch_defaults = {
            parameter.name: parameter.default
            for parameter in parameters
            if parameter.default is not inspect.Parameter.empty
        }

    @cached_property
    def _kernel_arg_names(self) -> tuple[str, ...]:
        arg_names = getattr(self.kernel, "arg_names", None)
        if arg_names is not None:
            return tuple(arg_names)
        wrapped = getattr(self.kernel, "func", None)
        if wrapped is not None:
            return tuple(inspect.signature(wrapped).parameters)
        raise TypeError(
            f"Cannot inspect kernel parameters for {type(self.kernel).__name__}"
        )

    def _prepare_launch_kwargs(
        self,
        input_names: tuple[str, ...],
        input_values: tuple[Any, ...],
        launch_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        plan = _launch_binding_plan(
            self._kernel_arg_names, input_names, tuple(launch_kwargs)
        )
        for index, target in plan.input_targets:
            launch_kwargs[target] = input_values[index]
        for target, source, dim in plan.stride_targets:
            launch_kwargs[target] = launch_kwargs[source].stride(dim)
        return launch_kwargs

    def launch(self, launch_spec: Any, inputs: Mapping[str, Any]) -> Any:
        if len(launch_spec) == 3:
            grid, launch_kwargs, outputs = launch_spec
        else:
            grid, launch_kwargs = launch_spec
            outputs = None
        kwargs = self._prepare_launch_kwargs(
            tuple(inputs), tuple(inputs.values()), dict(launch_kwargs)
        )
        kernel = kwargs.pop("kernel", None) or self.kernel
        runtime_launcher = kwargs.pop("_runtime_launcher", None)
        runtime_launcher_arg_count = kwargs.pop("_runtime_launcher_arg_count", 0)
        if grid is None:
            return outputs
        if runtime_launcher is not None:
            regular_args = [
                kwargs.pop(name)
                for name in self._kernel_arg_names[:runtime_launcher_arg_count]
            ]
            runtime_launcher(kernel, grid, *regular_args, **kwargs)
        else:
            kernel[grid](**kwargs)
        return outputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        spec = self._dispatch_fn(*args, **kwargs)
        input_values = args
        if len(args) != len(self._dispatch_arg_names):
            input_values += tuple(
                kwargs[name] if name in kwargs else self._dispatch_defaults[name]
                for name in self._dispatch_arg_names[len(args) :]
            )
        return self.launch(spec, dict(zip(self._dispatch_arg_names, input_values)))

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., Any]:
        return self.kernel[grid]

    def __getattr__(self, name: str) -> Any:
        if name in ("kernel", "_dispatch_fn", "_warmup_inputs_fn"):
            raise AttributeError(name)
        return getattr(self.kernel, name)


def triton_kernel_dispatcher_with_warmup(
    *,
    warmup_inputs: Callable[..., Any],
    kernel: Any | None = None,
) -> Callable[[Any], Any]:
    """Decorate a dispatch function (원본 L728): shim 은 kernel+dispatch 형태만 지원한다."""
    if kernel is None:
        raise NotImplementedError(
            "shim 은 kernel 인자가 있는 dispatcher 만 지원한다 (l2norm.py 사용형)"
        )

    def decorate(dispatch: Any) -> Any:
        owner = _DispatchedTritonKernel(kernel, warmup_inputs, dispatch)
        update_wrapper(owner, dispatch, updated=())
        return owner

    return decorate
