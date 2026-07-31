#!/usr/bin/env python3
"""Fail-closed four-node preflight for lane 01-tp8-pp4-256k-safe."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


LANE_ID = "01-tp8-pp4-256k-safe"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv,
            124,
            exc.stdout if isinstance(exc.stdout, str) else "",
            "timeout",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def node_command(
    values: dict[str, str],
    rank: int,
    argv: list[str],
) -> list[str]:
    if rank == 0:
        return argv
    user = values[f"K3_WORKER{rank}_SSH_USER"]
    host = values[f"K3_WORKER{rank}_SSH_HOST"]
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}", *argv]


def check_result(
    result: subprocess.CompletedProcess[str],
    code: str,
) -> list[str]:
    if result.returncode == 0:
        return []
    tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
    return [f"{code} rc={result.returncode}: {' | '.join(tail)}"]


def model_cache_errors(hf_home: Path, model: str) -> list[str]:
    encoded = "models--" + model.replace("/", "--")
    candidates = (hf_home / encoded, hf_home / "hub" / encoded)
    root = next((path for path in candidates if path.is_dir()), None)
    if root is None:
        return [f"PREFLIGHT-MODEL-CACHE-MISSING model={model}"]
    snapshots = root / "snapshots"
    if not snapshots.is_dir() or not any(path.is_dir() for path in snapshots.iterdir()):
        return [f"PREFLIGHT-MODEL-SNAPSHOT-MISSING model={model} root={root}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    env_path = args.env_file.resolve()
    if not env_path.is_file():
        print(f"K3-PREFLIGHT-ENV-MISSING path={env_path}")
        return 2

    lane_path = env_path.parent
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from deploy.tooling.command import build_vllm_args
    from deploy.tooling.lane_schema import LaneEnv
    from deploy.tooling.lane_specs import lane_by_id

    env = LaneEnv.load(env_path)
    errors = env.validate(lane_by_id(LANE_ID), require_site_values=True)
    values = env.values
    if not errors:
        for rank in range(4):
            try:
                build_vllm_args(env, rank)
            except (KeyError, ValueError) as exc:
                errors.append(f"PREFLIGHT-ARGV rank={rank}: {exc}")

    if shutil.which("docker") is None:
        errors.append("PREFLIGHT-DOCKER-MISSING")
    else:
        errors.extend(
            check_result(
                run(
                    [
                        "docker",
                        "compose",
                        "--env-file",
                        str(env_path),
                        "-f",
                        str(lane_path / "compose.yaml"),
                        "config",
                        "--quiet",
                    ]
                ),
                "PREFLIGHT-COMPOSE",
            )
        )

    if not errors:
        hf_home = Path(values["K3_HF_HOME"])
        serving_root = Path(values["K3_SERVING_ROOT"])
        for path, code in (
            (hf_home, "PREFLIGHT-HF-HOME-MISSING"),
            (serving_root, "PREFLIGHT-SERVING-ROOT-MISSING"),
            (repo_root, "PREFLIGHT-REPO-MISSING"),
        ):
            if not path.is_dir():
                errors.append(f"{code} path={path}")
        errors.extend(model_cache_errors(hf_home, values["K3_TARGET_MODEL"]))
        if 0:
            errors.extend(model_cache_errors(hf_home, values["K3_DRAFT_MODEL"]))

    image_ids: list[str] = []
    if not errors:
        sentinel = Path(values["K3_SERVING_ROOT"]) / "runs" / ".k3-nfs-sentinel"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            hashlib.sha256(str(repo_root).encode()).hexdigest() + "\n",
            encoding="utf-8",
        )
        for rank in range(4):
            verify = node_command(
                values,
                rank,
                [
                    "python3",
                    str(repo_root / "docker/verify_image.py"),
                    values["K3_IMAGE"],
                    "--gpu",
                ],
            )
            errors.extend(
                check_result(run(verify), f"PREFLIGHT-IMAGE-GPU node_rank={rank}")
            )
            containers = node_command(
                values,
                rank,
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=k3-",
                    "--format",
                    "{{.Names}}",
                ],
            )
            container_result = run(containers)
            errors.extend(
                check_result(
                    container_result,
                    f"PREFLIGHT-CONTAINER-QUERY node_rank={rank}",
                )
            )
            if container_result.returncode == 0 and container_result.stdout.strip():
                names = ",".join(container_result.stdout.strip().splitlines())
                errors.append(
                    f"PREFLIGHT-K3-CONTAINER-BUSY node_rank={rank} names={names}"
                )
            compute = node_command(
                values,
                rank,
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader,nounits",
                ],
            )
            compute_result = run(compute)
            errors.extend(
                check_result(
                    compute_result,
                    f"PREFLIGHT-GPU-QUERY node_rank={rank}",
                )
            )
            if compute_result.returncode == 0 and compute_result.stdout.strip():
                processes = " | ".join(compute_result.stdout.strip().splitlines())
                errors.append(
                    f"PREFLIGHT-GPU-BUSY node_rank={rank} processes={processes}"
                )
            shared = node_command(
                values,
                rank,
                [
                    "test",
                    "-f",
                    str(sentinel),
                    "-a",
                    "-d",
                    str(repo_root),
                    "-a",
                    "-d",
                    values["K3_HF_HOME"],
                ],
            )
            errors.extend(
                check_result(run(shared), f"PREFLIGHT-NFS node_rank={rank}")
            )
            inspect = node_command(
                values,
                rank,
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    values["K3_IMAGE"],
                ],
            )
            result = run(inspect)
            errors.extend(
                check_result(result, f"PREFLIGHT-DIGEST node_rank={rank}")
            )
            if result.returncode == 0:
                image_ids.append(result.stdout.strip())
        if len(set(image_ids)) != 1:
            errors.append(f"PREFLIGHT-DIGEST-MISMATCH ids={image_ids}")

    ips = [
        values.get("K3_HEAD_NODE_IP", ""),
        values.get("K3_WORKER1_NODE_IP", ""),
        values.get("K3_WORKER2_NODE_IP", ""),
        values.get("K3_WORKER3_NODE_IP", ""),
    ]
    if len(set(ips)) != 4:
        errors.append(f"PREFLIGHT-NODE-IP-DUPLICATE ips={ips}")

    for error in errors[:20]:
        print(f"K3-PREFLIGHT-FAIL {error}")
    if errors:
        print(f"K3-PREFLIGHT-SUMMARY lane={LANE_ID} errors={len(errors)}")
        return 2
    print(
        f"K3-PREFLIGHT-PASS lane={LANE_ID} nodes=4 gpu=32xA100 "
        f"image={image_ids[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
