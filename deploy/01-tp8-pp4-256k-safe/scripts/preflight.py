#!/usr/bin/env python3
"""Fail-closed local preflight for lane 01-tp8-pp4-256k-safe."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


LANE_ID = "01-tp8-pp4-256k-safe"


def run_check(argv: list[str], code: str) -> list[str]:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False)
    except OSError as exc:
        return [f"{code} command error: {exc}"]
    if result.returncode == 0:
        return []
    tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
    return [f"{code} rc={result.returncode}: {' | '.join(tail)}"]


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
    if not errors:
        for rank in range(4):
            try:
                build_vllm_args(env, rank)
            except (KeyError, ValueError) as exc:
                errors.append(f"PREFLIGHT-ARGV rank={rank}: {exc}")
    values = env.values
    if not errors:
        for key in ("K3_HF_HOME", "K3_SERVING_ROOT"):
            path = Path(values[key])
            if not path.is_dir():
                errors.append(f"{key} directory missing: {path}")

    if shutil.which("docker") is None:
        errors.append("PREFLIGHT-DOCKER-MISSING")
    else:
        errors.extend(
            run_check(
                ["docker", "compose", "--env-file", str(env_path), "-f",
                 str(lane_path / "compose.yaml"), "config", "--quiet"],
                "PREFLIGHT-COMPOSE",
            )
        )
        if "K3_IMAGE" in values:
            errors.extend(
                run_check(
                    ["docker", "image", "inspect", values["K3_IMAGE"]],
                    "PREFLIGHT-IMAGE",
                )
            )

    if shutil.which("nvidia-smi") is None:
        errors.append("PREFLIGHT-NVIDIA-SMI-MISSING")
    else:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        names = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        if query.returncode != 0:
            errors.append(f"PREFLIGHT-GPU-QUERY rc={query.returncode}")
        elif len(names) != 8 or any("A100" not in name for name in names):
            errors.append(f"PREFLIGHT-GPU expected=8xA100 actual={names}")

    for error in errors[:20]:
        print(f"K3-PREFLIGHT-FAIL {error}")
    if errors:
        print(f"K3-PREFLIGHT-SUMMARY lane={LANE_ID} errors={len(errors)}")
        return 2
    print(f"K3-PREFLIGHT-PASS lane={LANE_ID} gpu=8xA100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
