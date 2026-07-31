#!/usr/bin/env python3
"""Kimi-K3 DSpark state and service-integration gate for SM80.

The CPU mode validates only the acceptance/state-transition contract.  A full
PASS additionally requires two already-running OpenAI-compatible endpoints:
one baseline target and one target+``Inferact/Kimi-K3-DSpark`` service.  This
keeps a 32-GPU service launch outside a single-process "kernel" test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import shapes  # noqa: E402
from common.dspark_ref import (  # noqa: E402
    DSPARK_BLOCK_SIZE,
    TARGET_AUX_LAYER_IDS,
    accepted_block_length,
    accepted_state_prefix,
    commit_state_bundle,
)


KID = "k10_k3_dspark_sm80"
TARGET_DEFAULT = "moonshotai/Kimi-K3"
DRAFT_DEFAULT = "Inferact/Kimi-K3-DSpark"


def build_dry_report() -> dict[str, object]:
    states = tuple(f"state-{index}" for index in range(DSPARK_BLOCK_SIZE + 1))
    transitions = {
        str(accepted): list(accepted_state_prefix(states, accepted))
        for accepted in range(DSPARK_BLOCK_SIZE + 1)
    }
    rejection = accepted_block_length(
        (0.9, 0.8, 0.7, 0.6), (0.1, 0.2, 0.95, 0.0)
    )
    return {
        "schema_version": "k10-dspark-report/v1",
        "id": KID,
        "overall": "INCOMPLETE",
        "gates": {
            "STATE": {
                "status": "SKIP",
                "reason": "cpu-dry-run: reference contract only",
            },
            "GREEDY_PARITY": {
                "status": "SKIP",
                "reason": "baseline/speculative endpoints not queried",
            },
            "BLOCK_REJECTION": {
                "status": "SKIP",
                "reason": "vLLM acceptance telemetry not supplied",
            },
            "CONTEXT": {
                "status": "SKIP",
                "reason": "256k/1M service requests not executed",
            },
            "MEMORY": {
                "status": "SKIP",
                "reason": "draft GPU memory delta not measured",
            },
        },
        "evidence": {
            "target": TARGET_DEFAULT,
            "draft": DRAFT_DEFAULT,
            "attention_backend": "TRITON_MLA",
            "target_aux_layer_ids": TARGET_AUX_LAYER_IDS,
            "block_size": DSPARK_BLOCK_SIZE,
            "required_cases": shapes.DSPARK_REQUIRED_CASES,
            "state_prefixes": transitions,
            "reference_rejection_length": rejection,
        },
    }


def _request(url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 20260731,
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def _integration_run(args: argparse.Namespace) -> int:
    if not args.baseline_url or not args.speculative_url:
        print(
            "K10-PRECOND baseline/speculative URLs required; start lane services "
            "first and pass --baseline-url/--speculative-url",
            file=sys.stderr,
        )
        return 2
    prompts = (
        "Return exactly the word alpha.",
        "Compute 17*19 and return only the integer.",
        "Write three comma-separated prime numbers.",
    )
    parity: list[dict[str, object]] = []
    try:
        for prompt in prompts:
            baseline = _request(
                args.baseline_url, args.served_model_name, prompt, 16
            )
            speculative = _request(
                args.speculative_url, args.served_model_name, prompt, 16
            )
            same = _content(baseline) == _content(speculative)
            parity.append({"prompt": prompt, "same": same})
    except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
        print(f"K10-FAIL endpoint request: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    parity_pass = all(row["same"] for row in parity)
    # Acceptance/state coverage and long-context/memory evidence come from the
    # structured lane smoke result; absence remains INCOMPLETE, never PASS.
    telemetry: dict[str, Any] | None = None
    if args.telemetry and args.telemetry.is_file():
        telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    required_accepts = set(range(DSPARK_BLOCK_SIZE + 1))
    observed = set((telemetry or {}).get("accepted_counts", []))
    state_pass = observed >= required_accepts
    context_pass = bool((telemetry or {}).get("ctx256k_pass")) and bool(
        (telemetry or {}).get("ctx1m_pass")
    )
    memory_delta = (telemetry or {}).get("draft_memory_delta_bytes")
    memory_pass = isinstance(memory_delta, int) and memory_delta > 0
    statuses = {
        "STATE": "PASS" if state_pass else "SKIP",
        "GREEDY_PARITY": "PASS" if parity_pass else "FAIL",
        "BLOCK_REJECTION": "PASS" if state_pass else "SKIP",
        "CONTEXT": "PASS" if context_pass else "SKIP",
        "MEMORY": "PASS" if memory_pass else "SKIP",
    }
    overall = (
        "FAIL"
        if "FAIL" in statuses.values()
        else "PASS"
        if all(value == "PASS" for value in statuses.values())
        else "INCOMPLETE"
    )
    payload = {
        "schema_version": "k10-dspark-report/v1",
        "id": KID,
        "overall": overall,
        "gates": {
            name: {"status": status, "reason": ""}
            for name, status in statuses.items()
        },
        "evidence": {
            "target": args.target,
            "draft": args.draft,
            "attention_backend": "TRITON_MLA",
            "parity": parity,
            "telemetry": telemetry,
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"{KID}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"K10-OVERALL {overall} report={path}")
    return 1 if overall == "FAIL" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--target", default=TARGET_DEFAULT)
    parser.add_argument("--draft", default=DRAFT_DEFAULT)
    parser.add_argument("--baseline-url")
    parser.add_argument("--speculative-url")
    parser.add_argument("--served-model-name", default="kimi-k3")
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument(
        "--report-dir", type=Path, default=KIT / "kernel-report"
    )
    args = parser.parse_args()
    if args.cpu_dry_run:
        payload = build_dry_report()
        args.report_dir.mkdir(parents=True, exist_ok=True)
        path = args.report_dir / f"{KID}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"K10-OVERALL INCOMPLETE mode=cpu-dry-run report={path}")
        return 0
    return _integration_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
