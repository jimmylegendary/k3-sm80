"""Execute and validate fail-closed A100 gate stamps for serving lanes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .lane_schema import LaneEnv
from .lane_specs import LaneSpec, lane_by_id
from .remote import (
    _acquire_cluster_lock,
    _release_cluster_lock,
    lane_lock,
)
from .stamps import StampKey, sha256_file, validate_stamp, write_stamp


GATE_MATRIX: dict[str, tuple[str, ...]] = {
    "01": ("k08", "kda", "pp"),
    "02": ("k08", "c01", "ep"),
    "03": ("k08", "c01", "k10", "ep", "dspark"),
    "04": ("k08", "c01", "k09", "ep", "1m"),
    "05": ("k08", "c01", "k09", "k10", "ep", "1m", "dspark"),
}

# These gates can run before a 32-GPU vLLM service owns the cluster.  The
# remaining matrix entries are recorded only by explicit integration targets.
PREREQUISITE_GATES: dict[str, tuple[str, ...]] = {
    "01": ("k08", "kda"),
    "02": ("k08", "c01"),
    "03": ("k08", "c01"),
    "04": ("k08", "c01", "k09"),
    "05": ("k08", "c01", "k09"),
}

_REPORT_NAMES = {
    "k08": "k08_mxfp4_marlin_moe.json",
    "k09": "k09_mla_fp8_kv_sm80.json",
    "k10": "k10_k3_dspark_sm80.json",
    "c01": "c01_ep32_a2a.json",
    "k01": "k01_kda_chunk.json",
    "k02": "k02_kda_recurrent.json",
}


@dataclass(frozen=True)
class GateContext:
    lane_path: Path
    repo_root: Path
    spec: LaneSpec
    env: LaneEnv
    run_root: Path

    @property
    def number(self) -> str:
        return f"{self.spec.number:02d}"

    @property
    def matrix(self) -> tuple[str, ...]:
        return GATE_MATRIX[self.number]

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return PREREQUISITE_GATES[self.number]

    def report_dir(self, gate: str) -> Path:
        return self.run_root / "gates" / gate

    def stamp_path(self, gate: str) -> Path:
        return self.run_root / "stamps" / f"{gate}.json"


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(argv, 124, stdout, stderr + "\ntimeout")
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def load_context(lane_path: Path, *, require_site_values: bool = True) -> GateContext:
    lane_path = lane_path.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    spec = lane_by_id(lane_path.name)
    env_path = lane_path / ".env"
    env = LaneEnv.load(env_path)
    errors = env.validate(spec, require_site_values=require_site_values)
    if errors:
        raise ValueError("; ".join(errors[:20]))
    run_root = Path(env["K3_SERVING_ROOT"]) / "runs" / spec.id
    return GateContext(lane_path, repo_root, spec, env, run_root)


def _repo_commit(context: GateContext) -> str:
    result = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=context.repo_root,
    )
    if result.returncode:
        raise RuntimeError(f"git rev-parse rc={result.returncode}")
    return result.stdout.strip()


def _image_digest(context: GateContext) -> str:
    result = _run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            context.env["K3_IMAGE"],
        ],
        cwd=context.repo_root,
    )
    if result.returncode:
        raise RuntimeError(
            f"docker image inspect rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    digest = result.stdout.strip()
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"unexpected image digest: {digest!r}")
    return digest


def stamp_key(context: GateContext, gate: str) -> StampKey:
    return StampKey(
        repo_commit=_repo_commit(context),
        image_digest=_image_digest(context),
        env_sha256=sha256_file(context.lane_path / ".env"),
        gate_id=gate,
    )


def _report_overall(path: Path) -> tuple[str, dict[str, object]]:
    if not path.is_file():
        return "MISSING", {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "INVALID", {}
    if not isinstance(payload, dict):
        return "INVALID", {}
    overall = payload.get("overall")
    return (overall if isinstance(overall, str) else "INVALID"), payload


def _write_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"$ {shlex.join([str(item) for item in result.args])}\n"
        f"rc={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}\n",
        encoding="utf-8",
    )


def _relay(result: subprocess.CompletedProcess[str], *, gate: str) -> None:
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lines = [line for line in combined.splitlines() if line.strip()]
    for line in lines[-20:]:
        print(f"K3-GATE-LOG gate={gate} {line}")


def _gate_timeout(context: GateContext, gate: str) -> int:
    configured = int(context.env["K3_EXECUTE_MODEL_TIMEOUT_S"])
    minimums = {
        "k08": 21_600,
        "kda": 7_200,
        "k09": 7_200,
        "c01": 3_600,
    }
    return max(configured, minimums[gate])


def _generic_gate_commands(
    context: GateContext,
    gate: str,
    report_dir: Path,
    *,
    cpu_dry_run: bool,
) -> list[tuple[str, list[str], Path]]:
    dry = ["--cpu-dry-run"] if cpu_dry_run else []
    container = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--cap-add",
        "SYS_PTRACE",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--env-file",
        str(context.lane_path / ".env"),
        "--volume",
        f"{context.repo_root}:/work:ro",
        "--volume",
        f"{context.env['K3_SERVING_ROOT']}:{context.env['K3_SERVING_ROOT']}",
        "--workdir",
        "/work",
        "--entrypoint",
        "python3",
        context.env["K3_IMAGE"],
    ]
    if gate == "k08":
        return [
            (
                "k08",
                [
                    *container,
                    "/work/kernel-kit/k08_mxfp4_marlin_moe/harness.py",
                    "--report-dir",
                    str(report_dir),
                    *dry,
                ],
                report_dir / _REPORT_NAMES["k08"],
            )
        ]
    if gate == "k09":
        return [
            (
                "k09",
                [
                    *container,
                    "/work/kernel-kit/k09_mla_fp8_kv_sm80/harness.py",
                    "--report-dir",
                    str(report_dir),
                    *dry,
                ],
                report_dir / _REPORT_NAMES["k09"],
            )
        ]
    if gate == "kda":
        return [
            (
                subgate,
                [
                    *container,
                    f"/work/kernel-kit/{directory}/harness.py",
                    "--report-dir",
                    str(report_dir / subgate),
                    *dry,
                ],
                report_dir / subgate / _REPORT_NAMES[subgate],
            )
            for subgate, directory in (
                ("k01", "k01_kda_chunk"),
                ("k02", "k02_kda_recurrent"),
            )
        ]
    raise ValueError(f"gate {gate} is not a local standalone gate")


def _c01_command(
    context: GateContext,
    profile: str,
    node_rank: int,
    report_dir: Path,
) -> list[str]:
    values = context.env.values
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"k3-gate-c01-{node_rank}",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--cap-add",
        "SYS_PTRACE",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--env-file",
        str(context.lane_path / ".env"),
        "--env",
        f"MASTER_ADDR={values['K3_HEAD_NODE_IP']}",
        "--env",
        f"MASTER_PORT={values['VLLM_MASTER_PORT']}",
        "--env",
        f"NODE_RANK={node_rank}",
        "--env",
        "NNODES=4",
        "--volume",
        f"{context.repo_root}:/work:ro",
        "--volume",
        f"{values['K3_SERVING_ROOT']}:{values['K3_SERVING_ROOT']}",
        "--workdir",
        "/work",
        "--entrypoint",
        "bash",
        values["K3_IMAGE"],
        "/work/kernel-kit/c01_ep32_a2a/launch.sh",
        "--report-dir",
        str(report_dir),
    ]
    if node_rank == 0:
        return command
    user = values[f"K3_WORKER{node_rank}_SSH_USER"]
    host = values[f"K3_WORKER{node_rank}_SSH_HOST"]
    remote = f"cd {shlex.quote(str(context.repo_root))} && "
    remote += shlex.join(command)
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}", remote]


def _run_c01(
    context: GateContext,
    report_dir: Path,
    *,
    cpu_dry_run: bool,
) -> tuple[bool, dict[str, object]]:
    if cpu_dry_run:
        for rank, profile in enumerate(("head", "worker1", "worker2", "worker3")):
            print(
                f"K3-GATE-DRY gate=c01 profile={profile} "
                f"argv={shlex.join(_c01_command(context, profile, rank, report_dir))}"
            )
        return False, {"overall": "INCOMPLETE", "mode": "cpu-dry-run"}

    timeout = _gate_timeout(context, "c01")
    commands = [
        (profile, _c01_command(context, profile, rank, report_dir))
        for rank, profile in enumerate(("head", "worker1", "worker2", "worker3"))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _run,
                argv,
                cwd=context.repo_root,
                timeout=timeout,
            ): (profile, argv)
            for profile, argv in commands
        }
        results = [
            (profile, future.result())
            for future, (profile, _argv) in futures.items()
        ]
    results.sort(key=lambda item: item[0])
    for profile, result in results:
        _write_log(report_dir / f"{profile}.log", result)
        _relay(result, gate=f"c01/{profile}")
    overall, payload = _report_overall(report_dir / _REPORT_NAMES["c01"])
    return (
        all(result.returncode == 0 for _, result in results) and overall == "PASS",
        payload,
    )


def _run_gate_locked(
    context: GateContext,
    gate: str,
    *,
    cpu_dry_run: bool = False,
) -> bool:
    if gate not in context.prerequisites:
        raise ValueError(f"{gate} is not a prerequisite for lane {context.number}")
    report_dir = context.report_dir(gate)
    if not cpu_dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
    print(f"K3-GATE-START lane={context.spec.id} gate={gate}")

    if gate == "c01":
        passed, evidence = _run_c01(
            context,
            report_dir,
            cpu_dry_run=cpu_dry_run,
        )
    else:
        passed = True
        reports: dict[str, object] = {}
        commands = _generic_gate_commands(
            context,
            gate,
            report_dir,
            cpu_dry_run=cpu_dry_run,
        )
        for name, argv, report_path in commands:
            if cpu_dry_run:
                print(f"K3-GATE-DRY gate={name} argv={shlex.join(argv)}")
                passed = False
                continue
            result = _run(
                argv,
                cwd=context.repo_root,
                timeout=_gate_timeout(context, gate),
            )
            _write_log(report_dir / f"{name}.log", result)
            _relay(result, gate=name)
            overall, payload = _report_overall(report_path)
            reports[name] = payload
            passed = passed and result.returncode == 0 and overall == "PASS"
        evidence = {
            "overall": "PASS" if passed else "INCOMPLETE" if cpu_dry_run else "FAIL",
            "reports": reports,
        }

    if cpu_dry_run:
        print(f"K3-GATE-INCOMPLETE lane={context.spec.id} gate={gate} mode=cpu-dry-run")
        return False
    if not passed:
        print(f"K3-GATE-FAIL lane={context.spec.id} gate={gate}")
        return False
    evidence["overall"] = "PASS"
    write_stamp(context.stamp_path(gate), stamp_key(context, gate), evidence)
    print(
        f"K3-GATE-PASS lane={context.spec.id} gate={gate} "
        f"stamp={context.stamp_path(gate)}"
    )
    return True


def run_gate(
    context: GateContext,
    gate: str,
    *,
    cpu_dry_run: bool = False,
) -> bool:
    """Run a GPU gate while exclusively owning the four-node cluster."""

    if cpu_dry_run:
        return _run_gate_locked(context, gate, cpu_dry_run=True)

    run_id = f"gate-{gate}-{uuid.uuid4().hex}"
    with lane_lock(context.lane_path):
        _acquire_cluster_lock(context.lane_path, context.env, run_id)
        try:
            return _run_gate_locked(context, gate, cpu_dry_run=False)
        finally:
            _release_cluster_lock(context.lane_path, context.env)


def verify_gates(context: GateContext, gates: Sequence[str]) -> list[str]:
    errors: list[str] = []
    for gate in gates:
        try:
            expected = stamp_key(context, gate)
        except RuntimeError as exc:
            errors.append(f"{gate}: {exc}")
            continue
        for error in validate_stamp(context.stamp_path(gate), expected):
            errors.append(f"{gate}: {error}")
    return errors


def record_gate(context: GateContext, gate: str, evidence_path: Path) -> bool:
    if gate not in context.matrix:
        raise ValueError(f"gate {gate} is not required by lane {context.number}")
    overall, payload = _report_overall(evidence_path)
    if overall != "PASS":
        print(
            f"K3-GATE-RECORD-FAIL lane={context.spec.id} gate={gate} "
            f"evidence={overall}"
        )
        return False
    write_stamp(context.stamp_path(gate), stamp_key(context, gate), payload)
    print(
        f"K3-GATE-RECORD-PASS lane={context.spec.id} gate={gate} "
        f"stamp={context.stamp_path(gate)}"
    )
    return True


def _print_errors(prefix: str, errors: Sequence[str]) -> None:
    for error in list(errors)[:20]:
        print(f"{prefix} {error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("run", "verify-prerequisites", "verify-all", "record", "status"),
    )
    parser.add_argument("--lane", type=Path, required=True)
    parser.add_argument("--gate")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--cpu-dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        context = load_context(
            args.lane,
            require_site_values=not args.cpu_dry_run,
        )
        if args.action == "run":
            gates = (args.gate,) if args.gate else context.prerequisites
            success = True
            for gate in gates:
                success = run_gate(
                    context,
                    gate,
                    cpu_dry_run=args.cpu_dry_run,
                ) and success
            return 0 if success or args.cpu_dry_run else 1
        if args.action == "record":
            if not args.gate or args.evidence is None:
                parser.error("record requires --gate and --evidence")
            return 0 if record_gate(context, args.gate, args.evidence) else 1
        gates = (
            context.prerequisites
            if args.action == "verify-prerequisites"
            else context.matrix
        )
        errors = verify_gates(context, gates)
        if errors:
            _print_errors("K3-GATE-VERIFY-FAIL", errors)
            return 1
        print(
            f"K3-GATE-VERIFY-PASS lane={context.spec.id} "
            f"scope={'prerequisites' if args.action == 'verify-prerequisites' else 'all'}"
        )
        return 0
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"K3-GATE-CONFIG-FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
