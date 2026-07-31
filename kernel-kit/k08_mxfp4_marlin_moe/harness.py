#!/usr/bin/env python3
"""Original-checkpoint MXFP4 Marlin MoE gate (Task 6: CPU contract only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import report, shapes  # noqa: E402
from common.mxfp4_ref import decode_e2m1, decode_e8m0  # noqa: E402


KID = "k08_mxfp4_marlin_moe"
PINS = {
    "vllm_kimi_k3": "0498dc7ea9204f1863b46515816f5d8d458e01e7",
    "hf_kimi_k3": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
}


def build_dry_report() -> dict[str, object]:
    decoded = [decode_e2m1(bytes([code]))[0] for code in range(16)]
    scale_probe = decode_e8m0(bytes([0, 127, 254, 255]))
    selfcheck = (
        decoded == list(shapes.MXFP4_E2M1_VALUES)
        and scale_probe[:3] == [2.0**-127, 1.0, 2.0**127]
        and scale_probe[3] != scale_probe[3]
    )
    if not selfcheck:
        raise RuntimeError("MXFP4 decoder self-check failed")

    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    result.add_evidence("layout_arms", ["tp8", "ep32"])
    result.add_evidence(
        "checkpoint_layout",
        {
            "packed_w": "[E,N,K/2], low nibble first",
            "scales": "[E,N,K/32], E8M0FNU",
            "block_size": 32,
        },
    )
    result.add_evidence("required_shapes", shapes.REQUIRED_SHAPES[KID])
    result.add_evidence("decoder_selfcheck", "PASS")
    result.set_gate(
        "NUMERIC",
        "SKIP",
        "cpu-dry-run: decoder/reference contract only; Marlin kernel not executed",
    )
    result.set_gate(
        "SANITIZER",
        "SKIP",
        "cpu-dry-run: compute-sanitizer requires A100 execution",
    )
    result.set_gate(
        "SHAPES",
        "SKIP",
        "cpu-dry-run: TP8 and EP32 metadata registered but GPU arms not executed",
    )
    result.set_gate(
        "PERF",
        "SKIP",
        "cpu-dry-run: no GPU timing",
    )
    return result.as_dict()


def write_payload(payload: dict[str, object], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{KID}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_dir / f"{KID}.md").write_text(
        report.render_md(payload),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-verdict")
    parser.add_argument("--twin")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=KIT / "kernel-report",
    )
    parser.add_argument("--cpu-dry-run", action="store_true")
    args = parser.parse_args()

    if not args.cpu_dry_run:
        print(
            "[k08] GPU arms are implemented in Task 7; use --cpu-dry-run for now",
            file=sys.stderr,
        )
        return report.EXIT_PRECOND
    payload = build_dry_report()
    write_payload(payload, args.report_dir)
    print(
        f"[k08] dry-run report={args.report_dir / (KID + '.json')} "
        f"overall={payload['overall']} layout_arms=tp8,ep32"
    )
    return report.EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[k08] internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(report.EXIT_ERR)
