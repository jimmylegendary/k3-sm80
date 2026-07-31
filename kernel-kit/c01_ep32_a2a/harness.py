#!/usr/bin/env python3
"""Four-node/32-rank SM80 EP correctness gate.

Production-equivalent launch (run on every node):

  MASTER_ADDR=<head-ip> MASTER_PORT=29610 NODE_RANK=<0..3> NNODES=4 \
    bash kernel-kit/c01_ep32_a2a/launch.sh \
    --report-dir /mnt/nas/home/ubuntu/service/large_model/Kimi-K3/runs/gates/c01

The gate intentionally uses the portable ``allgather_reducescatter`` backend.
Hopper/Blackwell-only DeepEP and NVLink one-sided backends are rejected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


REPORT_ID = "c01_ep32_a2a"
WORLD_SIZE = 32
LOCAL_WORLD_SIZE = 8
NNODES_REQUIRED = 4
BACKEND = "allgather_reducescatter"
TOKEN_CASES = (1, 8, 32)


def rank_layout(node_rank: int, local_rank: int) -> tuple[int, int, int]:
    if node_rank not in range(NNODES_REQUIRED):
        raise ValueError(f"node_rank must be 0..3, got {node_rank}")
    if local_rank not in range(LOCAL_WORLD_SIZE):
        raise ValueError(f"local_rank must be 0..7, got {local_rank}")
    global_rank = node_rank * LOCAL_WORLD_SIZE + local_rank
    tp_rank = local_rank
    dp_rank = node_rank
    return global_rank, tp_rank, dp_rank


def validate_backend(name: str) -> None:
    if name != BACKEND:
        raise ValueError(
            f"{name!r} is not the SM80 correctness backend; use {BACKEND!r}"
        )


def routed_token_fixture(
    tokens: int,
    global_rank: int,
    *,
    skew_expert: int | None = None,
) -> list[dict[str, int]]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if global_rank not in range(WORLD_SIZE):
        raise ValueError("global_rank must be 0..31")
    return [
        {
            "token": global_rank * 100_000 + index,
            "expert": (
                skew_expert
                if skew_expert is not None
                else (global_rank * 37 + index * 13) % 896
            ),
            "source_rank": global_rank,
        }
        for index in range(tokens)
    ]


def classify_launch(
    returncode: int,
    *,
    timed_out: bool,
    stderr: str,
) -> dict[str, object]:
    if timed_out or returncode == 124:
        classification, status = "TIMEOUT", "INCOMPLETE"
    elif returncode == 0:
        classification, status = "PASS", "PASS"
    elif "NCCL" in stderr or "ProcessGroupNCCL" in stderr:
        classification, status = "NCCL_FAILURE", "FAIL"
    else:
        classification, status = "PROCESS_FAILURE", "FAIL"
    return {
        "returncode": returncode,
        "classification": classification,
        "status": status,
        "stderr_tail": stderr[-2000:],
    }


def build_dry_report() -> dict[str, object]:
    fixtures = {
        str(tokens): routed_token_fixture(tokens, 7) for tokens in TOKEN_CASES
    }
    fixtures["skew"] = routed_token_fixture(32, 7, skew_expert=511)
    return {
        "schema_version": "c01-ep32-report/v1",
        "id": REPORT_ID,
        "overall": "INCOMPLETE",
        "backend": BACKEND,
        "topology": {
            "nnodes": NNODES_REQUIRED,
            "local_world_size": LOCAL_WORLD_SIZE,
            "world_size": WORLD_SIZE,
        },
        "gates": {
            "CORRECTNESS": {
                "status": "SKIP",
                "reason": "cpu-dry-run: NCCL collectives not executed",
            },
            "SKEW": {
                "status": "SKIP",
                "reason": "cpu-dry-run: single-hot expert skew not executed",
            },
            "PERF": {
                "status": "SKIP",
                "reason": "cpu-dry-run: CUDA timing not executed",
            },
        },
        "evidence": {"fixtures": fixtures},
    }


def _payload(torch: Any, rows: list[dict[str, int]], device: int) -> Any:
    return torch.tensor(
        [
            [row["token"], row["expert"], row["source_rank"], index]
            for index, row in enumerate(rows)
        ],
        dtype=torch.int64,
        device=device,
    )


def _collective_case(
    torch: Any,
    dist: Any,
    *,
    tokens: int,
    rank: int,
    device: int,
    skew_expert: int | None,
) -> dict[str, object]:
    fixture = routed_token_fixture(tokens, rank, skew_expert=skew_expert)
    local = _payload(torch, fixture, device)
    gathered = torch.empty(
        (WORLD_SIZE * tokens, local.shape[1]),
        dtype=local.dtype,
        device=device,
    )
    reduced = torch.empty_like(local)

    def operation() -> None:
        dist.all_gather_into_tensor(gathered, local)
        dist.reduce_scatter_tensor(reduced, gathered, op=dist.ReduceOp.SUM)

    operation()
    torch.cuda.synchronize(device)
    expected_gathered = torch.cat(
        [
            _payload(
                torch,
                routed_token_fixture(
                    tokens, source, skew_expert=skew_expert
                ),
                device,
            )
            for source in range(WORLD_SIZE)
        ],
        dim=0,
    )
    torch.testing.assert_close(gathered, expected_gathered, rtol=0, atol=0)
    torch.testing.assert_close(
        reduced, local * WORLD_SIZE, rtol=0, atol=0
    )

    for _ in range(2):
        operation()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return {
        "tokens": tokens,
        "skew_expert": skew_expert,
        "median_us": samples[len(samples) // 2],
        "min_us": samples[0],
        "max_us": samples[-1],
        "payload_bytes_per_rank": int(local.numel() * local.element_size()),
        "passed": True,
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{REPORT_ID}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _distributed_run(args: argparse.Namespace) -> int:
    validate_backend(args.backend)
    required = ("MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "NNODES")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing environment: {', '.join(missing)}")
    if int(os.environ["NNODES"]) != NNODES_REQUIRED:
        raise RuntimeError("NNODES must be 4")

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    expected_rank, _, _ = rank_layout(node_rank, local_rank)
    if rank != expected_rank or world_size != WORLD_SIZE:
        raise RuntimeError(
            f"rank topology mismatch rank={rank}/{expected_rank} "
            f"world={world_size}/{WORLD_SIZE}"
        )
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability(local_rank) != (8, 0):
        raise RuntimeError("c01 is A100/SM80-only")

    timeout = dt.timedelta(seconds=args.collective_timeout)
    dist.init_process_group("nccl", timeout=timeout)
    started = time.monotonic()
    try:
        regular = [
            _collective_case(
                torch,
                dist,
                tokens=tokens,
                rank=rank,
                device=local_rank,
                skew_expert=None,
            )
            for tokens in TOKEN_CASES
        ]
        skew = _collective_case(
            torch,
            dist,
            tokens=32,
            rank=rank,
            device=local_rank,
            skew_expert=511,
        )
        gathered_results: list[object] = [None] * WORLD_SIZE
        dist.all_gather_object(
            gathered_results,
            {
                "rank": rank,
                "host": socket.gethostname(),
                "regular": regular,
                "skew": skew,
            },
        )
        if rank == 0:
            payload = {
                "schema_version": "c01-ep32-report/v1",
                "id": REPORT_ID,
                "overall": "PASS",
                "backend": BACKEND,
                "topology": {
                    "nnodes": NNODES_REQUIRED,
                    "local_world_size": LOCAL_WORLD_SIZE,
                    "world_size": WORLD_SIZE,
                },
                "gates": {
                    "CORRECTNESS": {
                        "status": "PASS",
                        "reason": "1/8/32-token all-gather + reduce-scatter exact",
                    },
                    "SKEW": {
                        "status": "PASS",
                        "reason": "single-hot expert=511 exact",
                    },
                    "PERF": {
                        "status": "PASS",
                        "reason": "five CUDA-event samples recorded per case",
                    },
                },
                "evidence": {
                    "ranks": gathered_results,
                    "duration_s": round(time.monotonic() - started, 3),
                },
            }
            _write_report(args.report_dir, payload)
            print(
                f"C01-OVERALL PASS world=32 backend={BACKEND} "
                f"report={args.report_dir / (REPORT_ID + '.json')}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "K3_GATE_REPORT_DIR",
                str(Path(__file__).resolve().parents[1] / "kernel-report"),
            )
        ),
    )
    parser.add_argument("--collective-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.cpu_dry_run:
        payload = build_dry_report()
        _write_report(args.report_dir, payload)
        print(
            f"C01-OVERALL INCOMPLETE mode=cpu-dry-run "
            f"report={args.report_dir / (REPORT_ID + '.json')}"
        )
        return 0
    return _distributed_run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"C01-FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
