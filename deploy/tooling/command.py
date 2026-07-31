"""Build token-safe vLLM CLI arguments from a validated lane environment."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Sequence

from .lane_schema import LaneEnv


_MANAGED_LONG_FLAGS = frozenset(
    {
        "--model",
        "--host",
        "--port",
        "--distributed-executor-backend",
        "--master-addr",
        "--master-port",
        "--nnodes",
        "--node-rank",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-size-local",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--data-parallel-backend",
        "--enable-expert-parallel",
        "--all2all-backend",
        "--moe-backend",
        "--attention-backend",
        "--attention-config",
        "--kda-prefill-backend",
        "--quantization",
        "--dtype",
        "--kv-cache-dtype",
        "--kv-cache-dtype-skip-layers",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--gpu-memory-utilization",
        "--enforce-eager",
        "--no-enforce-eager",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--language-model-only",
        "--headless",
        "--compilation-config",
        "--speculative-config",
        "--served-model-name",
        "--tool-call-parser",
        "--reasoning-parser",
    }
)
_MANAGED_SHORT_FLAGS = frozenset(
    {
        "-n",
        "-r",
        "-tp",
        "-pp",
        "-dp",
        "-dpr",
        "-dpl",
        "-dpa",
        "-dpp",
        "-dpb",
        "-sc",
    }
)


def validate_extra_args(args: list[str]) -> list[str]:
    """Return safe extra tokens, rejecting overrides of lane-managed settings."""

    for token in args:
        flag = token.split("=", 1)[0]
        if flag in _MANAGED_LONG_FLAGS or flag in _MANAGED_SHORT_FLAGS:
            raise ValueError(f"VLLM_EXTRA_ARGS cannot override managed flag: {flag}")
        if "\0" in token:
            raise ValueError("VLLM_EXTRA_ARGS contains a NUL byte")
    return list(args)


def _add_bool(args: list[str], enabled: str, flag: str) -> None:
    if enabled == "1":
        args.append(flag)
    elif enabled != "0":
        raise ValueError(f"{flag} control must be 0 or 1, got {enabled!r}")


def build_vllm_args(env: LaneEnv, node_rank: int) -> list[str]:
    """Build the arguments following ``vllm serve`` for one of four nodes."""

    if env.errors:
        raise ValueError("invalid lane environment: " + "; ".join(env.errors))
    if not 0 <= node_rank < 4:
        raise ValueError(f"node_rank must be in [0, 3], got {node_rank}")

    v = env.values
    required = {
        "K3_TARGET_MODEL",
        "K3_API_BIND",
        "K3_HEAD_NODE_IP",
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
    missing = sorted(required - v.keys())
    if missing:
        raise ValueError(f"lane environment is missing keys: {missing}")

    args = [
        v["K3_TARGET_MODEL"],
        "--host",
        v["K3_API_BIND"],
        "--port",
        v["VLLM_API_PORT"],
        "--distributed-executor-backend",
        "mp",
        "--data-parallel-backend",
        v["VLLM_DATA_PARALLEL_BACKEND"],
        "--nnodes",
        "4",
        "--node-rank",
        str(node_rank),
        "--master-addr",
        v["K3_HEAD_NODE_IP"],
        "--master-port",
        v["VLLM_MASTER_PORT"],
        "--tensor-parallel-size",
        v["VLLM_TENSOR_PARALLEL_SIZE"],
        "--pipeline-parallel-size",
        v["VLLM_PIPELINE_PARALLEL_SIZE"],
        "--data-parallel-size",
        v["VLLM_DATA_PARALLEL_SIZE"],
        "--data-parallel-rpc-port",
        v["VLLM_DP_RPC_PORT"],
        "--quantization",
        v["VLLM_QUANTIZATION"],
        "--dtype",
        v["VLLM_DTYPE"],
        "--kv-cache-dtype",
        v["VLLM_KV_CACHE_DTYPE"],
        "--max-model-len",
        v["VLLM_MAX_MODEL_LEN"],
        "--max-num-seqs",
        v["VLLM_MAX_NUM_SEQS"],
        "--max-num-batched-tokens",
        v["VLLM_MAX_NUM_BATCHED_TOKENS"],
        "--gpu-memory-utilization",
        v["VLLM_GPU_MEMORY_UTILIZATION"],
        "--attention-backend",
        v["VLLM_ATTENTION_BACKEND"],
        "--attention-config",
        v["VLLM_ATTENTION_CONFIG"],
        "--kda-prefill-backend",
        v["VLLM_KDA_PREFILL_BACKEND"],
        "--moe-backend",
        v["VLLM_MOE_BACKEND"],
        "--compilation-config",
        v["VLLM_COMPILATION_CONFIG"],
        "--served-model-name",
        v["VLLM_SERVED_MODEL_NAME"],
        "--tool-call-parser",
        v["VLLM_TOOL_CALL_PARSER"],
        "--reasoning-parser",
        v["VLLM_REASONING_PARSER"],
    ]

    if v["VLLM_ENABLE_EXPERT_PARALLEL"] == "1":
        args.extend(["--enable-expert-parallel", "--all2all-backend", v["VLLM_ALL2ALL_BACKEND"]])
    elif v["VLLM_ENABLE_EXPERT_PARALLEL"] != "0":
        raise ValueError("VLLM_ENABLE_EXPERT_PARALLEL must be 0 or 1")

    _add_bool(args, v["VLLM_ENFORCE_EAGER"], "--enforce-eager")
    _add_bool(args, v["VLLM_ENABLE_PREFIX_CACHING"], "--enable-prefix-caching")
    _add_bool(args, v["VLLM_LANGUAGE_MODEL_ONLY"], "--language-model-only")
    if node_rank > 0:
        args.append("--headless")

    if v["VLLM_SPECULATIVE_CONFIG"]:
        args.extend(["--speculative-config", v["VLLM_SPECULATIVE_CONFIG"]])

    if v["VLLM_KV_CACHE_DTYPE_SKIP_LAYERS"]:
        skip_layers = shlex.split(v["VLLM_KV_CACHE_DTYPE_SKIP_LAYERS"])
        if not skip_layers or any(not layer.isdecimal() for layer in skip_layers):
            raise ValueError(
                "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS must contain decimal layer indices"
            )
        args.extend(["--kv-cache-dtype-skip-layers", *skip_layers])

    args.extend(validate_extra_args(shlex.split(v["VLLM_EXTRA_ARGS"])))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print token-safe vLLM serve commands without executing them."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--node-rank", type=int, choices=range(4))
    parser.add_argument("--all-ranks", action="store_true")
    args = parser.parse_args(argv)
    if args.all_ranks == (args.node_rank is not None):
        parser.error("choose exactly one of --all-ranks or --node-rank")
    env = LaneEnv.load(args.env_file)
    if env.errors:
        for error in env.errors[:20]:
            print(f"K3-CONFIG-FAIL {error}")
        return 2
    ranks = range(4) if args.all_ranks else (args.node_rank,)
    for rank in ranks:
        assert rank is not None
        command = ["vllm", "serve", *build_vllm_args(env, rank)]
        print(f"K3-SERVE-ARGV node_rank={rank} {shlex.join(command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
