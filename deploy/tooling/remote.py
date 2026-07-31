"""Four-node native-vLLM MP orchestration over existing SSH trust."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .lane_schema import LaneEnv
from .lane_specs import lane_by_id


_PROFILE_TO_SERVICE = {
    "head": "k3-head",
    "worker1": "k3-worker1",
    "worker2": "k3-worker2",
    "worker3": "k3-worker3",
}
_PROFILE_TO_RANK = {profile: rank for rank, profile in enumerate(_PROFILE_TO_SERVICE)}


@dataclass(frozen=True)
class RemoteNode:
    profile: str
    ssh_user: str
    ssh_host: str

    @property
    def node_rank(self) -> int:
        return _PROFILE_TO_RANK[self.profile]

    @property
    def service(self) -> str:
        return _PROFILE_TO_SERVICE[self.profile]

    @property
    def is_local(self) -> bool:
        return self.profile == "head"

    @property
    def target(self) -> str:
        return "local" if self.is_local else f"{self.ssh_user}@{self.ssh_host}"


@dataclass(frozen=True)
class RemoteLane:
    shared_directory: str
    workers: tuple[RemoteNode, ...]


@dataclass(frozen=True)
class PlannedCommand:
    node: RemoteNode
    action: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()


def load_remote_lane(lane_path: Path, env: LaneEnv) -> RemoteLane:
    v = env.values
    nodes = (
        RemoteNode("head", v["K3_HEAD_SSH_USER"], v["K3_HEAD_SSH_HOST"]),
        RemoteNode("worker1", v["K3_WORKER1_SSH_USER"], v["K3_WORKER1_SSH_HOST"]),
        RemoteNode("worker2", v["K3_WORKER2_SSH_USER"], v["K3_WORKER2_SSH_HOST"]),
        RemoteNode("worker3", v["K3_WORKER3_SSH_USER"], v["K3_WORKER3_SSH_HOST"]),
    )
    return RemoteLane(str(lane_path.resolve()), nodes)


def _compose_tokens(node: RemoteNode, action: str) -> list[str]:
    base = [
        "docker",
        "compose",
        "--env-file",
        ".env",
        "--profile",
        node.profile,
    ]
    if action == "up":
        return [*base, "up", "-d", node.service]
    if action == "down":
        return [*base, "rm", "-s", "-f", node.service]
    if action == "status":
        return [*base, "ps", node.service]
    if action == "logs":
        return [*base, "logs", "--tail", "200", node.service]
    if action == "running":
        return [*base, "ps", "--status", "running", "--services", node.service]
    if action == "existing":
        return [*base, "ps", "-q", node.service]
    raise ValueError(f"unsupported action: {action}")


def plan_command(
    lane: RemoteLane,
    node: RemoteNode,
    action: str,
    *,
    run_id: str = "",
) -> PlannedCommand:
    compose = _compose_tokens(node, action)
    environment = (("K3_RUN_ID", run_id),) if run_id else ()
    if node.is_local:
        return PlannedCommand(node, action, tuple(compose), environment)

    remote_command = f"cd {shlex.quote(lane.shared_directory)} && "
    if run_id:
        remote_command += f"K3_RUN_ID={shlex.quote(run_id)} "
    remote_command += shlex.join(compose)
    argv = (
        "ssh",
        "-o",
        "BatchMode=yes",
        node.target,
        remote_command,
    )
    return PlannedCommand(node, action, argv)


def format_plan(command: PlannedCommand) -> str:
    return (
        f"K3-REMOTE-CMD profile={command.node.profile} "
        f"node_rank={command.node.node_rank} target={command.node.target} "
        f"action={command.action} argv={shlex.join(command.argv)}"
    )


def _run(command: PlannedCommand, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(command.environment)
    try:
        return subprocess.run(
            command.argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            env=environment,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command.argv, 127, "", str(exc))


def _run_many(
    commands: Sequence[PlannedCommand],
    *,
    capture: bool = False,
) -> list[tuple[PlannedCommand, subprocess.CompletedProcess[str]]]:
    if not commands:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {executor.submit(_run, command, capture=capture): command for command in commands}
        results = [(futures[future], future.result()) for future in concurrent.futures.as_completed(futures)]
    return sorted(results, key=lambda item: item[0].node.node_rank)


@contextlib.contextmanager
def lane_lock(lane_path: Path) -> Iterator[None]:
    lock_path = lane_path / ".k3-cluster.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"lane is locked: {lock_path}") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _print_failures(
    results: Sequence[tuple[PlannedCommand, subprocess.CompletedProcess[str]]],
) -> int:
    failures = 0
    for command, result in results:
        if result.returncode == 0:
            continue
        failures += 1
        summary = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        print(
            f"K3-REMOTE-FAIL profile={command.node.profile} "
            f"action={command.action} rc={result.returncode}",
            file=sys.stderr,
        )
        for line in summary:
            print(f"  {line}", file=sys.stderr)
    return failures


def _assert_empty(lane: RemoteLane) -> None:
    commands = [plan_command(lane, node, "existing") for node in lane.workers]
    results = _run_many(commands, capture=True)
    failures = _print_failures(results)
    occupied = [
        command.node.profile
        for command, result in results
        if result.returncode == 0 and result.stdout.strip()
    ]
    if failures:
        raise RuntimeError(f"could not inspect {failures} node(s)")
    if occupied:
        raise RuntimeError(f"lane already has containers on profiles: {occupied}")


def _wait_for_launchers(lane: RemoteLane, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        commands = [plan_command(lane, node, "running") for node in lane.workers]
        results = _run_many(commands, capture=True)
        if all(
            result.returncode == 0 and command.node.service in result.stdout.split()
            for command, result in results
        ):
            print("K3-LAUNCHERS ready=4 local_world_size=8 expected_ranks=32")
            return
        time.sleep(2)
    raise RuntimeError("timed out waiting for four running launcher containers")


def _wait_for_health(env: LaneEnv, timeout_s: int) -> None:
    url = f"http://{env.values['K3_HEAD_NODE_IP']}:{env.values['VLLM_API_PORT']}/health"
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(
                        "K3-HEALTH PASS "
                        f"url={url} expected_ranks=32 inference=all-ranks-initialized"
                    )
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"health timeout: {last_error}")


def _rollback(lane: RemoteLane, started: Sequence[RemoteNode], run_id: str) -> None:
    commands = [
        plan_command(lane, node, "down", run_id=run_id)
        for node in reversed(started)
    ]
    _print_failures(_run_many(commands, capture=True))


def execute_up(lane_path: Path, lane: RemoteLane, env: LaneEnv) -> int:
    run_id = uuid.uuid4().hex
    started: list[RemoteNode] = []
    with lane_lock(lane_path):
        try:
            _assert_empty(lane)
            head = lane.workers[0]
            head_command = plan_command(lane, head, "up", run_id=run_id)
            head_result = _run(head_command, capture=True)
            if head_result.returncode:
                _print_failures([(head_command, head_result)])
                raise RuntimeError("head startup failed")
            started.append(head)

            worker_commands = [
                plan_command(lane, node, "up", run_id=run_id)
                for node in lane.workers[1:]
            ]
            worker_results = _run_many(worker_commands, capture=True)
            for command, result in worker_results:
                if result.returncode == 0:
                    started.append(command.node)
            if _print_failures(worker_results):
                raise RuntimeError("worker startup failed")

            timeout_s = int(env.values["K3_ENGINE_READY_TIMEOUT_S"])
            _wait_for_launchers(lane, timeout_s)
            _wait_for_health(env, timeout_s)
        except Exception as exc:
            print(f"K3-UP-FAIL run_id={run_id} error={exc}", file=sys.stderr)
            _rollback(lane, started, run_id)
            return 1
    print(f"K3-UP-PASS run_id={run_id} launchers=4 expected_ranks=32")
    return 0


def execute_simple(lane_path: Path, lane: RemoteLane, action: str) -> int:
    commands = [plan_command(lane, node, action) for node in lane.workers]
    lock_context = lane_lock(lane_path) if action == "down" else contextlib.nullcontext()
    with lock_context:
        results = _run_many(commands, capture=False)
    failures = _print_failures(results)
    print(f"K3-{action.upper()} profiles=4 failures={failures}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("up", "down", "status", "logs"))
    parser.add_argument("--lane", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    lane_path = args.lane.resolve()
    env_path = lane_path / ".env"
    if not env_path.is_file():
        print(f"K3-REMOTE-ENV-MISSING path={env_path}", file=sys.stderr)
        return 2
    try:
        spec = lane_by_id(lane_path.name)
        env = LaneEnv.load(env_path)
        errors = env.validate(spec, require_site_values=not args.dry_run)
        if errors:
            for error in errors[:20]:
                print(f"K3-REMOTE-CONFIG-ERROR {error}", file=sys.stderr)
            return 2
        lane = load_remote_lane(lane_path, env)
    except (KeyError, ValueError) as exc:
        print(f"K3-REMOTE-CONFIG-ERROR {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for node in lane.workers:
            print(format_plan(plan_command(lane, node, args.action, run_id="DRY_RUN")))
        print(f"K3-REMOTE-DRY-RUN lane={lane_path.name} action={args.action} mutation=0")
        return 0

    if args.action == "up":
        return execute_up(lane_path, lane, env)
    return execute_simple(lane_path, lane, args.action)


if __name__ == "__main__":
    raise SystemExit(main())
