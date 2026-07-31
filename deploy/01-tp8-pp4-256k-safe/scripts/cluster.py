#!/usr/bin/env python3
"""Container entrypoint for lane 01-tp8-pp4-256k-safe and host compatibility wrapper."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


def add_bool(args: list[str], env_key: str, flag: str) -> None:
    value = os.environ[env_key]
    if value == "1":
        args.append(flag)
    elif value != "0":
        raise ValueError(f"{env_key} must be 0 or 1")


def serve_args(node_rank: int) -> list[str]:
    value = os.environ.__getitem__
    args = [
        value("K3_TARGET_MODEL"),
        "--host", value("K3_API_BIND"),
        "--port", value("VLLM_API_PORT"),
        "--distributed-executor-backend", "mp",
        "--data-parallel-backend", value("VLLM_DATA_PARALLEL_BACKEND"),
        "--nnodes", "4",
        "--node-rank", str(node_rank),
        "--master-addr", value("K3_HEAD_NODE_IP"),
        "--master-port", value("VLLM_MASTER_PORT"),
        "--tensor-parallel-size", value("VLLM_TENSOR_PARALLEL_SIZE"),
        "--pipeline-parallel-size", value("VLLM_PIPELINE_PARALLEL_SIZE"),
        "--data-parallel-size", value("VLLM_DATA_PARALLEL_SIZE"),
        "--data-parallel-rpc-port", value("VLLM_DP_RPC_PORT"),
        "--quantization", value("VLLM_QUANTIZATION"),
        "--dtype", value("VLLM_DTYPE"),
        "--kv-cache-dtype", value("VLLM_KV_CACHE_DTYPE"),
        "--max-model-len", value("VLLM_MAX_MODEL_LEN"),
        "--max-num-seqs", value("VLLM_MAX_NUM_SEQS"),
        "--max-num-batched-tokens", value("VLLM_MAX_NUM_BATCHED_TOKENS"),
        "--gpu-memory-utilization", value("VLLM_GPU_MEMORY_UTILIZATION"),
        "--attention-backend", value("VLLM_ATTENTION_BACKEND"),
        "--attention-config", value("VLLM_ATTENTION_CONFIG"),
        "--kda-prefill-backend", value("VLLM_KDA_PREFILL_BACKEND"),
        "--moe-backend", value("VLLM_MOE_BACKEND"),
        "--compilation-config", value("VLLM_COMPILATION_CONFIG"),
        "--served-model-name", value("VLLM_SERVED_MODEL_NAME"),
        "--tool-call-parser", value("VLLM_TOOL_CALL_PARSER"),
        "--reasoning-parser", value("VLLM_REASONING_PARSER"),
    ]
    if value("VLLM_ENABLE_EXPERT_PARALLEL") == "1":
        args.extend([
            "--enable-expert-parallel",
            "--all2all-backend", value("VLLM_ALL2ALL_BACKEND"),
        ])
    elif value("VLLM_ENABLE_EXPERT_PARALLEL") != "0":
        raise ValueError("VLLM_ENABLE_EXPERT_PARALLEL must be 0 or 1")
    add_bool(args, "VLLM_ENFORCE_EAGER", "--enforce-eager")
    add_bool(args, "VLLM_ENABLE_PREFIX_CACHING", "--enable-prefix-caching")
    add_bool(args, "VLLM_LANGUAGE_MODEL_ONLY", "--language-model-only")
    if node_rank > 0:
        args.append("--headless")
    if value("VLLM_SPECULATIVE_CONFIG"):
        args.extend(["--speculative-config", value("VLLM_SPECULATIVE_CONFIG")])
    if value("VLLM_KV_CACHE_DTYPE_SKIP_LAYERS"):
        skip_layers = shlex.split(value("VLLM_KV_CACHE_DTYPE_SKIP_LAYERS"))
        if not skip_layers or any(not layer.isdecimal() for layer in skip_layers):
            raise ValueError(
                "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS must contain decimal layer indices"
            )
        args.extend(["--kv-cache-dtype-skip-layers", *skip_layers])
    args.extend(shlex.split(value("VLLM_EXTRA_ARGS")))
    return args


def host_action(action: str, dry_run: bool) -> int:
    lane_path = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from deploy.tooling.remote import main as remote_main

    argv = [action, "--lane", str(lane_path)]
    if dry_run:
        argv.append("--dry-run")
    return remote_main(argv)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "up", "down", "status", "logs"))
    parser.add_argument("--node-rank", type=int, choices=range(4))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action != "serve":
        return host_action(args.action, args.dry_run)
    if args.node_rank is None:
        parser.error("serve requires --node-rank")
    argv = ["vllm", "serve", *serve_args(args.node_rank)]
    print(
        f"K3-SERVE-EXEC lane=01-tp8-pp4-256k-safe node_rank={args.node_rank} "
        f"argv={shlex.join(argv)}",
        flush=True,
    )
    if args.dry_run:
        return 0
    os.execvp(argv[0], argv)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
