"""Static repository validation for the five generated serving lanes."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .command import build_vllm_args
from .lane_schema import LaneEnv
from .lane_specs import LANES


_TOP_LEVEL_FILES = frozenset({".env", "compose.yaml", "Makefile", "RUNBOOK.md", "scripts"})
_SCRIPT_FILES = frozenset({"preflight.py", "cluster.py", "smoke.py"})
_PROFILES = ("head", "worker1", "worker2", "worker3")
_PORT_KEYS = ("VLLM_API_PORT", "VLLM_MASTER_PORT", "VLLM_DP_RPC_PORT")


def _duplicate_errors(name: str, owners: dict[str, list[str]]) -> list[str]:
    return [
        f"duplicate {name}={value}: {', '.join(lanes)}"
        for value, lanes in sorted(owners.items())
        if len(lanes) > 1
    ]


def _layout_errors(lane_dir: Path, lane_id: str) -> list[str]:
    errors: list[str] = []
    if not lane_dir.is_dir():
        return [f"{lane_id}: missing lane directory"]

    actual = {path.name for path in lane_dir.iterdir()}
    for name in sorted(_TOP_LEVEL_FILES - actual):
        errors.append(f"{lane_id}: missing {name}")
    for name in sorted(actual - _TOP_LEVEL_FILES):
        errors.append(f"{lane_id}: unexpected top-level entry {name}")

    scripts = lane_dir / "scripts"
    if scripts.is_dir():
        actual_scripts = {
            path.name
            for path in scripts.iterdir()
            if path.name != "__pycache__"
        }
        for name in sorted(_SCRIPT_FILES - actual_scripts):
            errors.append(f"{lane_id}: missing scripts/{name}")
        for name in sorted(actual_scripts - _SCRIPT_FILES):
            errors.append(f"{lane_id}: unexpected script {name}")
    return errors


def _compose_errors(path: Path, lane_id: str) -> tuple[list[str], str | None]:
    if not path.is_file():
        return [], None
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    profiles = re.findall(r'^\s*profiles:\s*\["([^"]+)"\]\s*$', text, re.MULTILINE)
    if sorted(profiles) != sorted(_PROFILES):
        errors.append(
            f"{lane_id}: compose profiles must be exactly {list(_PROFILES)}, got {profiles}"
        )
    project_match = re.search(r"^name:\s*([A-Za-z0-9_.-]+)\s*$", text, re.MULTILINE)
    if project_match is None:
        errors.append(f"{lane_id}: compose project name missing")
        return errors, None
    return errors, project_match.group(1)


def validate_repository(root: Path) -> list[str]:
    """Return static errors; site-specific ``CHANGE_ME`` values are allowed."""

    root = root.resolve()
    deploy = root / "deploy"
    errors: list[str] = []
    expected_ids = {lane.id for lane in LANES}
    actual_ids = {
        path.name
        for path in deploy.glob("0*-*")
        if path.is_dir()
    } if deploy.is_dir() else set()
    for lane_id in sorted(expected_ids - actual_ids):
        errors.append(f"missing lane: {lane_id}")
    for lane_id in sorted(actual_ids - expected_ids):
        errors.append(f"unexpected lane: {lane_id}")

    env_files = sorted(deploy.rglob(".env")) if deploy.is_dir() else []
    if len(env_files) != len(LANES):
        errors.append(f"expected exactly 5 .env files, got {len(env_files)}")
    override_envs = sorted(
        path
        for path in deploy.rglob("*")
        if path.is_file()
        and path.name != ".env"
        and (path.name.startswith(".env") or path.suffix == ".env")
    ) if deploy.is_dir() else []
    errors.extend(
        f"override env is forbidden: {path.relative_to(root)}"
        for path in override_envs
    )

    port_owners: dict[str, dict[str, list[str]]] = {
        key: defaultdict(list) for key in _PORT_KEYS
    }
    project_owners: dict[str, list[str]] = defaultdict(list)

    for spec in LANES:
        lane_dir = deploy / spec.id
        errors.extend(_layout_errors(lane_dir, spec.id))
        env_path = lane_dir / ".env"
        if env_path.is_file():
            env = LaneEnv.load(env_path)
            errors.extend(
                f"{spec.id}: {error}"
                for error in env.validate(spec, require_site_values=False)
            )
            if not env.errors:
                for rank in range(4):
                    try:
                        build_vllm_args(env, node_rank=rank)
                    except (KeyError, ValueError) as exc:
                        errors.append(f"{spec.id}: command rank {rank}: {exc}")
                for key in _PORT_KEYS:
                    value = env.values.get(key)
                    if value is not None:
                        port_owners[key][value].append(spec.id)

        compose_errors, project = _compose_errors(lane_dir / "compose.yaml", spec.id)
        errors.extend(compose_errors)
        if project is not None:
            project_owners[project].append(spec.id)

    for key, owners in port_owners.items():
        errors.extend(_duplicate_errors(key, owners))
    errors.extend(_duplicate_errors("compose project", project_owners))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root",
    )
    args = parser.parse_args()
    errors = validate_repository(args.root)
    for error in errors:
        print(f"ERROR {error}")
    env_files = list((args.root / "deploy").rglob(".env"))
    print(f"lanes={len(LANES)} env_files={len(env_files)} errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
