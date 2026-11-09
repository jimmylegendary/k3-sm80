"""Strict, non-executing parser for a lane's single ``.env`` file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .lane_specs import LaneSpec


COMMON_KEYS = frozenset(
    {
        "K3_IMAGE",
        "K3_TARGET_MODEL",
        "K3_DRAFT_MODEL",
        "K3_HF_HOME",
        "K3_SERVING_ROOT",
        "K3_API_BIND",
        "K3_HEAD_NODE_IP",
        "K3_HEAD_SSH_HOST",
        "K3_HEAD_SSH_USER",
        "K3_WORKER1_NODE_IP",
        "K3_WORKER1_SSH_HOST",
        "K3_WORKER1_SSH_USER",
        "K3_WORKER2_NODE_IP",
        "K3_WORKER2_SSH_HOST",
        "K3_WORKER2_SSH_USER",
        "K3_WORKER3_NODE_IP",
        "K3_WORKER3_SSH_HOST",
        "K3_WORKER3_SSH_USER",
        "K3_NCCL_SOCKET_IFNAME",
        "K3_GLOO_SOCKET_IFNAME",
        "K3_NCCL_IB_HCA",
        "K3_NCCL_IB_GID_INDEX",
        "K3_ENGINE_READY_TIMEOUT_S",
        "K3_EXECUTE_MODEL_TIMEOUT_S",
    }
)

VLLM_KEYS = frozenset(
    {
        "VLLM_API_PORT",
        "VLLM_MASTER_PORT",
        "VLLM_DP_RPC_PORT",
        "VLLM_TENSOR_PARALLEL_SIZE",
        "VLLM_PIPELINE_PARALLEL_SIZE",
        "VLLM_DATA_PARALLEL_SIZE",
        "VLLM_DATA_PARALLEL_BACKEND",
        "VLLM_ENABLE_EXPERT_PARALLEL",
        "VLLM_ALL2ALL_BACKEND",
        "VLLM_MOE_BACKEND",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_ATTENTION_CONFIG",
        "VLLM_KDA_PREFILL_BACKEND",
        "VLLM_QUANTIZATION",
        "VLLM_DTYPE",
        "VLLM_KV_CACHE_DTYPE",
        "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS",
        "VLLM_K3_SM80_FP8_KV",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
        "VLLM_MARLIN_MXFP8_INPUT_QDQ",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_ENFORCE_EAGER",
        "VLLM_ENABLE_PREFIX_CACHING",
        "VLLM_LANGUAGE_MODEL_ONLY",
        "VLLM_COMPILATION_CONFIG",
        "VLLM_SPECULATIVE_CONFIG",
        "VLLM_SERVED_MODEL_NAME",
        "VLLM_TOOL_CALL_PARSER",
        "VLLM_REASONING_PARSER",
        "VLLM_EXTRA_ARGS",
    }
)

# lane 01-05(K3)의 필수 키 집합. K3 .env는 정확히 이 키들만 가진다.
ALLOWED_KEYS = COMMON_KEYS | VLLM_KEYS

# K3.1 lane에만 존재하는 키. K3 lane .env에 나타나면 안 된다(렌더 바이트 불변).
K31_KEYS = frozenset({"VLLM_SPARSE_INDEXER_TOPK_BACKEND"})
KNOWN_KEYS = ALLOWED_KEYS | K31_KEYS

# --sparse-indexer-topk-backend에 허용하는 값. SM80 P1은 persistent(auto도 SM80에서는
# persistent로 해석)이며 cooperative는 SM90+ 전용이라 SM80에서는 기동 시 RuntimeError다.
SPARSE_INDEXER_TOPK_BACKENDS = frozenset({"persistent", "auto", "cooperative", "torch"})

_INTEGER_KEYS = frozenset(
    {
        "K3_NCCL_IB_GID_INDEX",
        "K3_ENGINE_READY_TIMEOUT_S",
        "K3_EXECUTE_MODEL_TIMEOUT_S",
        "VLLM_API_PORT",
        "VLLM_MASTER_PORT",
        "VLLM_DP_RPC_PORT",
        "VLLM_TENSOR_PARALLEL_SIZE",
        "VLLM_PIPELINE_PARALLEL_SIZE",
        "VLLM_DATA_PARALLEL_SIZE",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
    }
)
_PORT_KEYS = frozenset({"VLLM_API_PORT", "VLLM_MASTER_PORT", "VLLM_DP_RPC_PORT"})
_BOOL_KEYS = frozenset(
    {
        "VLLM_ENABLE_EXPERT_PARALLEL",
        "VLLM_ENFORCE_EAGER",
        "VLLM_ENABLE_PREFIX_CACHING",
        "VLLM_LANGUAGE_MODEL_ONLY",
        "VLLM_K3_SM80_FP8_KV",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
        "VLLM_MARLIN_MXFP8_INPUT_QDQ",
    }
)
_JSON_KEYS = frozenset(
    {
        "VLLM_ATTENTION_CONFIG",
        "VLLM_COMPILATION_CONFIG",
        "VLLM_SPECULATIVE_CONFIG",
    }
)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


@dataclass
class LaneEnv:
    values: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "LaneEnv":
        values: dict[str, str] = {}
        errors: list[str] = []
        for line_number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not key:
                errors.append(f"line {line_number}: expected KEY=VALUE")
                continue
            if key in values:
                errors.append(f"duplicate key: {key}")
                continue
            values[key] = _unquote(value.strip())

        errors.extend(f"unknown key: {key}" for key in sorted(values.keys() - KNOWN_KEYS))
        return cls(values=values, errors=errors)

    @classmethod
    def load(cls, path: Path) -> "LaneEnv":
        return cls.parse(path.read_text(encoding="utf-8"))

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    def validate(self, spec: LaneSpec, *, require_site_values: bool = True) -> list[str]:
        errors = list(self.errors)
        required_keys = (ALLOWED_KEYS | K31_KEYS) if spec.is_k31 else ALLOWED_KEYS
        errors.extend(f"missing key: {key}" for key in sorted(required_keys - self.values.keys()))
        if not spec.is_k31:
            errors.extend(
                f"K3.1-only key is not allowed in K3 lane {spec.id}: {key}"
                for key in sorted(K31_KEYS & self.values.keys())
            )

        if require_site_values:
            errors.extend(
                f"unresolved site value: {key}"
                for key, value in sorted(self.values.items())
                if value.startswith("CHANGE_ME_")
            )

        parsed_ints: dict[str, int] = {}
        for key in sorted(_INTEGER_KEYS & self.values.keys()):
            try:
                parsed_ints[key] = int(self.values[key])
            except ValueError:
                errors.append(f"{key} must be an integer")
                continue
            if parsed_ints[key] < 0:
                errors.append(f"{key} must be non-negative")

        for key in sorted(_PORT_KEYS & parsed_ints.keys()):
            if not 1 <= parsed_ints[key] <= 65535:
                errors.append(f"{key} must be in [1, 65535]")

        for key in sorted(_BOOL_KEYS & self.values.keys()):
            if self.values[key] not in {"0", "1"}:
                errors.append(f"{key} must be 0 or 1")

        for key in sorted(_JSON_KEYS & self.values.keys()):
            value = self.values[key]
            if not value:
                continue
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                errors.append(f"{key} must be valid JSON: {exc.msg}")
            else:
                if not isinstance(parsed, dict):
                    errors.append(f"{key} must be a JSON object")

        topology_keys = (
            "VLLM_TENSOR_PARALLEL_SIZE",
            "VLLM_PIPELINE_PARALLEL_SIZE",
            "VLLM_DATA_PARALLEL_SIZE",
        )
        if all(key in parsed_ints for key in topology_keys):
            topology = tuple(parsed_ints[key] for key in topology_keys)
            expected = (spec.tp, spec.pp, spec.dp)
            if topology != expected:
                errors.append(f"topology {topology} does not match lane {expected}")
            if topology[0] * topology[1] * topology[2] != 32:
                errors.append("TP*PP*DP must equal 32")

        expected_pairs = {
            "VLLM_DATA_PARALLEL_BACKEND": "mp",
            "VLLM_ENABLE_EXPERT_PARALLEL": "1" if spec.ep else "0",
            "VLLM_MAX_MODEL_LEN": str(spec.max_len),
            "VLLM_KV_CACHE_DTYPE": "fp8" if spec.fp8_kv else "bfloat16",
            "VLLM_K3_SM80_FP8_KV": "1" if spec.fp8_kv else "0",
            "VLLM_KDA_PREFILL_BACKEND": "triton",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "0",
            "VLLM_MARLIN_MXFP8_INPUT_QDQ": "0",
            "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS": (
                "93 94 95 96 97" if spec.fp8_kv and spec.dspark else ""
            ),
        }
        hints: dict[str, str] = {}
        if spec.is_k31:
            # K3.1 전용 규칙: 모델/백엔드/KV dtype을 spec에 고정한다.
            expected_pairs.update(
                {
                    "K3_TARGET_MODEL": spec.target_model,
                    "K3_DRAFT_MODEL": spec.draft_model,
                    "VLLM_ATTENTION_BACKEND": spec.attention_backend,
                    "VLLM_KV_CACHE_DTYPE": "bfloat16",
                }
            )
            hints["VLLM_KV_CACHE_DTYPE"] = (
                " (K3.1은 dsa_config가 있어 auto/fp8*를 fp8_ds_mla로 재작성하므로"
                " bfloat16을 명시해야 한다)"
            )
            topk = self.values.get("VLLM_SPARSE_INDEXER_TOPK_BACKEND")
            if topk is not None and topk not in SPARSE_INDEXER_TOPK_BACKENDS:
                errors.append(
                    f"VLLM_SPARSE_INDEXER_TOPK_BACKEND={topk!r} must be one of "
                    f"{sorted(SPARSE_INDEXER_TOPK_BACKENDS)} for lane {spec.id}"
                )
        for key, expected in expected_pairs.items():
            actual = self.values.get(key)
            if actual is not None and actual != expected:
                errors.append(
                    f"{key}={actual!r} must be {expected!r} for lane {spec.id}"
                    f"{hints.get(key, '')}"
                )

        speculative = self.values.get("VLLM_SPECULATIVE_CONFIG", "")
        if spec.dspark and not speculative:
            errors.append("DSpark lane requires VLLM_SPECULATIVE_CONFIG")
        if not spec.dspark and speculative:
            errors.append("non-DSpark lane must leave VLLM_SPECULATIVE_CONFIG empty")

        return errors
