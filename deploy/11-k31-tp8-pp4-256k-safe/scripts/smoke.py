#!/usr/bin/env python3
"""OpenAI/load/context smoke gate for lane 11-k31-tp8-pp4-256k-safe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LANE_ID = "11-k31-tp8-pp4-256k-safe"
MAX_MODEL_LEN = 262144
FP8_KV = bool(0)
DSPARK = bool(0)
BASELINE_LANE_ID = "11-k31-tp8-pp4-256k-safe"
CONTEXT_LENGTHS = (
    (262_144, 524_288, 786_432, 1_048_576)
    if MAX_MODEL_LEN == 1_048_576
    else (8_192, 131_072, 262_144)
)
BASELINE_PROMPTS = (
    "The capital of France is",
    "Write the next integer after 41:",
    "Return exactly the word OK.",
)
DSPARK_MAX_ROUNDS = 8
GREEDY_REPRO_PROMPT = "Return OK."
GREEDY_REPRO_MAX_TOKENS = 16
REASONING_EFFORT = "high"
REASONING_PROMPT = "Return exactly the word OK."
REASONING_MAX_TOKENS = 64


def _load_env(path: Path) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from deploy.tooling.lane_schema import LaneEnv
    from deploy.tooling.lane_specs import lane_by_id

    env = LaneEnv.load(path)
    errors = env.validate(lane_by_id(LANE_ID), require_site_values=True)
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return env.values


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"response from {url} is not an object")
    return body


def _request_text(url: str, *, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _completion(
    base_url: str,
    model: str,
    prompt: str | list[int],
    *,
    timeout: int,
    max_tokens: int = 1,
) -> dict[str, Any]:
    return _request(
        f"{base_url}/v1/completions",
        method="POST",
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1,
        },
        timeout=timeout,
    )


def _probe_token(base_url: str, model: str, timeout: int) -> int:
    body = _request(
        f"{base_url}/tokenize",
        method="POST",
        payload={"model": model, "prompt": "x"},
        timeout=timeout,
    )
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], int):
        raise ValueError("/tokenize did not return a non-empty integer token list")
    return tokens[0]


def _choice_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        raise ValueError("completion choice has no text")
    return first["text"]


def _greedy_repro(
    base_url: str,
    model: str,
    timeout: int,
) -> dict[str, object]:
    """Same greedy /v1/completions request twice must return identical text."""

    texts = [
        _choice_text(
            _completion(
                base_url,
                model,
                GREEDY_REPRO_PROMPT,
                timeout=timeout,
                max_tokens=GREEDY_REPRO_MAX_TOKENS,
            )
        )
        for _ in range(2)
    ]
    if texts[0] != texts[1]:
        raise RuntimeError("greedy completion was not reproducible across 2 requests")
    return {
        "prompt": GREEDY_REPRO_PROMPT,
        "max_tokens": GREEDY_REPRO_MAX_TOKENS,
        "text": texts[0],
        "reproducible": True,
    }


def _chat_reasoning(
    base_url: str,
    model: str,
    timeout: int,
) -> dict[str, object]:
    """One greedy chat call with the top-level ``reasoning_effort`` request field."""

    body = _request(
        f"{base_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": REASONING_PROMPT}],
            "max_tokens": REASONING_MAX_TOKENS,
            "temperature": 0,
            "seed": 1,
            "reasoning_effort": REASONING_EFFORT,
        },
        timeout=timeout,
    )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("chat completion response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat completion choice has no message")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if not isinstance(reasoning, str) or not isinstance(content, str):
        raise ValueError("chat message reasoning/content must be strings")
    return {
        "reasoning": reasoning,
        "content": content,
        "finish_reason": choices[0].get("finish_reason"),
    }


def _reasoning_case(
    base_url: str,
    model: str,
    timeout: int,
) -> dict[str, object]:
    """``reasoning_effort: high`` must yield a reasoning channel, reproducibly."""

    first = _chat_reasoning(base_url, model, timeout)
    second = _chat_reasoning(base_url, model, timeout)
    if not first["reasoning"]:
        raise RuntimeError(
            "reasoning_effort=high returned an empty reasoning channel "
            "(kimi_k3 reasoning parser or chat template not applied)"
        )
    if first != second:
        raise RuntimeError("greedy reasoning request was not reproducible")
    return {
        "reasoning_effort": REASONING_EFFORT,
        "max_tokens": REASONING_MAX_TOKENS,
        "reasoning_chars": len(str(first["reasoning"])),
        "reasoning_head": str(first["reasoning"])[:120],
        "content_head": str(first["content"])[:120],
        "finish_reason": first["finish_reason"],
        "reproducible": True,
    }


def _gpu_query_argv(values: dict[str, str], rank: int) -> list[str]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    if rank == 0:
        return query
    user = values[f"K3_WORKER{rank}_SSH_USER"]
    host = values[f"K3_WORKER{rank}_SSH_HOST"]
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}", *query]


def _gpu_memory(values: dict[str, str]) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    for rank in range(4):
        result = subprocess.run(
            _gpu_query_argv(values, rank),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"GPU query node_rank={rank} rc={result.returncode}")
        node_rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            index, name, total, used = fields
            node_rows.append(
                {
                    "node_rank": rank,
                    "gpu_index": int(index),
                    "name": name,
                    "total_mib": int(total),
                    "used_mib": int(used),
                }
            )
        if len(node_rows) != 8 or any("A100" not in str(row["name"]) for row in node_rows):
            raise RuntimeError(f"GPU query node_rank={rank} expected 8xA100")
        rows.extend(node_rows)
    minimum_free = min(
        100.0 * (int(row["total_mib"]) - int(row["used_mib"]))
        / int(row["total_mib"])
        for row in rows
    )
    return rows, minimum_free


def _context_cases(
    base_url: str,
    model: str,
    timeout: int,
) -> list[dict[str, object]]:
    token = _probe_token(base_url, model, timeout)
    cases: list[dict[str, object]] = []
    for requested in CONTEXT_LENGTHS:
        prompt_tokens = max(1, requested - 1)
        started = time.monotonic()
        body = _completion(
            base_url,
            model,
            [token] * prompt_tokens,
            timeout=timeout,
        )
        cases.append(
            {
                "requested_context": requested,
                "prompt_tokens": prompt_tokens,
                "elapsed_s": round(time.monotonic() - started, 3),
                "text": _choice_text(body)[:120],
                "passed": True,
            }
        )
        print(
            f"K3-SMOKE-CONTEXT requested={requested} status=PASS "
            f"elapsed_s={cases[-1]['elapsed_s']}",
            flush=True,
        )
    return cases


def _baseline(
    base_url: str,
    model: str,
    timeout: int,
) -> list[dict[str, object]]:
    rows = []
    for prompt in BASELINE_PROMPTS:
        body = _completion(
            base_url,
            model,
            prompt,
            timeout=timeout,
            max_tokens=16,
        )
        rows.append({"prompt": prompt, "text": _choice_text(body)})
    return rows


def _calibrate(
    base_url: str,
    model: str,
    timeout: int,
    concurrency: int,
) -> list[dict[str, object]]:
    token = _probe_token(base_url, model, timeout)
    prompt = [token] * (MAX_MODEL_LEN - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _completion,
                base_url,
                model,
                prompt,
                timeout=timeout,
            )
            for _ in range(concurrency)
        ]
        return [
            {
                "request": index,
                "text": _choice_text(future.result())[:120],
                "passed": True,
            }
            for index, future in enumerate(futures)
        ]


def _load_report(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or body.get("overall") != "PASS":
        raise ValueError(f"required PASS report missing: {path}")
    return body


def _metric_total(metrics: str, name: str) -> float:
    total = 0.0
    for line in metrics.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sample_name = line.split(None, 1)[0].split("{", 1)[0]
        if sample_name not in {
            name,
            f"{name}_total",
            f"vllm:{name}",
            f"vllm:{name}_total",
        }:
            continue
        try:
            total += float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return total


def _accepted_positions(metrics: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in metrics.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sample_name = line.split(None, 1)[0].split("{", 1)[0]
        if sample_name not in {
            "spec_decode_num_accepted_tokens_per_pos",
            "spec_decode_num_accepted_tokens_per_pos_total",
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        }:
            continue
        match = re.search(r'position="(\d+)"', line)
        if match is None:
            continue
        try:
            position = int(match.group(1))
            values[position] = values.get(position, 0.0) + float(
                line.rsplit(None, 1)[1]
            )
        except (IndexError, ValueError):
            continue
    return values


def _position_delta(
    after: dict[int, float],
    before: dict[int, float],
) -> dict[int, float]:
    return {
        position: max(0.0, value - before.get(position, 0.0))
        for position, value in after.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--mode",
        choices=("smoke", "load", "context", "baseline", "calibrate", "dspark"),
        default="smoke",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        values = _load_env(args.env_file.resolve())
        base_url = (
            f"http://{values['K3_HEAD_NODE_IP']}:{values['VLLM_API_PORT']}"
        )
        timeout = int(values["K3_EXECUTE_MODEL_TIMEOUT_S"])
        model = values["VLLM_SERVED_MODEL_NAME"]
        models = _request(f"{base_url}/v1/models", timeout=30)
        evidence: dict[str, object] = {
            "mode": args.mode,
            "models": models,
            "fp8_kv": FP8_KV,
            "dspark": DSPARK,
        }
        if args.mode in ("smoke", "load"):
            body = _completion(base_url, model, "Return OK.", timeout=timeout)
            evidence["completion"] = _choice_text(body)
            if args.mode == "smoke":
                evidence["greedy_repro"] = _greedy_repro(base_url, model, timeout)
                evidence["reasoning"] = _reasoning_case(base_url, model, timeout)
        elif args.mode == "context":
            evidence["contexts"] = _context_cases(base_url, model, timeout)
        elif args.mode == "baseline":
            evidence["baseline"] = _baseline(base_url, model, timeout)
        elif args.mode == "calibrate":
            concurrency = int(values["VLLM_MAX_NUM_SEQS"])
            evidence["concurrency"] = concurrency
            evidence["requests"] = _calibrate(
                base_url,
                model,
                timeout,
                concurrency,
            )
        elif args.mode == "dspark":
            if not DSPARK:
                raise ValueError("dspark mode is valid only for DSpark lanes")
            baseline_path = (
                Path(values["K3_SERVING_ROOT"])
                / "runs"
                / BASELINE_LANE_ID
                / "integration"
                / "baseline.json"
            )
            context_path = (
                Path(values["K3_SERVING_ROOT"])
                / "runs"
                / LANE_ID
                / "integration"
                / "context.json"
            )
            baseline_report = _load_report(baseline_path)
            context_report = _load_report(context_path)
            baseline_rows = baseline_report["evidence"]["baseline"]
            metrics_before = _request_text(f"{base_url}/metrics", timeout=30)
            drafts_before = _metric_total(metrics_before, "spec_decode_num_drafts")
            accepted_before = _metric_total(
                metrics_before,
                "spec_decode_num_accepted_tokens",
            )
            positions_before = _accepted_positions(metrics_before)
            speculative_rows = _baseline(base_url, model, timeout)
            parity = baseline_rows == speculative_rows
            rounds = 1
            while True:
                metrics_after = _request_text(f"{base_url}/metrics", timeout=30)
                drafts = max(
                    0.0,
                    _metric_total(metrics_after, "spec_decode_num_drafts")
                    - drafts_before,
                )
                accepted = max(
                    0.0,
                    _metric_total(
                        metrics_after,
                        "spec_decode_num_accepted_tokens",
                    )
                    - accepted_before,
                )
                positions = _position_delta(
                    _accepted_positions(metrics_after),
                    positions_before,
                )
                position_coverage = all(
                    positions.get(index, 0.0) > 0 for index in range(7)
                )
                if (drafts > 0.0 and position_coverage) or rounds >= DSPARK_MAX_ROUNDS:
                    break
                next_rows = _baseline(base_url, model, timeout)
                parity = parity and baseline_rows == next_rows
                rounds += 1
            evidence.update(
                {
                    "baseline_lane": BASELINE_LANE_ID,
                    "baseline": baseline_rows,
                    "speculative": speculative_rows,
                    "greedy_parity": parity,
                    "spec_decode_num_drafts": drafts,
                    "spec_decode_num_accepted_tokens": accepted,
                    "accepted_positions": positions,
                    "metric_rounds": rounds,
                    "state_contract_pass": (
                        drafts > 0.0 and accepted > 0.0 and position_coverage
                    ),
                    "context_report": str(context_path),
                    "context_overall": context_report["overall"],
                    "baseline_gpus": baseline_report["evidence"]["gpus"],
                }
            )
            if not parity:
                raise RuntimeError("target-only and DSpark greedy outputs differ")
            if not evidence["state_contract_pass"]:
                raise RuntimeError(
                    "DSpark metrics lack drafts or accepted-position coverage 0..6"
                )
        gpus, minimum_free = _gpu_memory(values)
        evidence["gpus"] = gpus
        evidence["minimum_free_hbm_pct"] = round(minimum_free, 3)
        if args.mode == "dspark":
            baseline_used = sum(
                int(row["used_mib"])
                for row in evidence["baseline_gpus"]
            )
            current_used = sum(int(row["used_mib"]) for row in gpus)
            evidence["draft_memory_delta_mib"] = current_used - baseline_used
            if current_used <= baseline_used:
                raise RuntimeError(
                    "DSpark draft memory delta was not positive; "
                    "baseline/current evidence may be stale"
                )
        minimum_required = 5.0 if args.mode in ("context", "calibrate") else 0.0
        if minimum_free < minimum_required:
            raise RuntimeError(
                f"free HBM {minimum_free:.3f}% < required {minimum_required:.1f}%"
            )
        payload = {
            "schema_version": "k3-lane-smoke/v1",
            "id": f"{LANE_ID}:{args.mode}",
            "overall": "PASS",
            "evidence": evidence,
        }
    except (
        OSError,
        ValueError,
        RuntimeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        payload = {
            "schema_version": "k3-lane-smoke/v1",
            "id": f"{LANE_ID}:{args.mode}",
            "overall": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    report = args.report
    if report is None:
        serving_root = (
            values.get("K3_SERVING_ROOT", ".")
            if "values" in locals()
            else "."
        )
        report = (
            Path(serving_root)
            / "runs"
            / LANE_ID
            / "integration"
            / f"{args.mode}.json"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"K3-SMOKE-{payload['overall']} lane={LANE_ID} mode={args.mode} "
        f"report={report}"
    )
    if payload["overall"] != "PASS":
        print(f"K3-SMOKE-ERROR {payload.get('error', 'unknown')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
