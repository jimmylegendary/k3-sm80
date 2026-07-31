#!/usr/bin/env python3
"""Standalone SM80 gate for Kimi-K3 software E4M3FN MLA KV storage.

The harness calls the Triton pack/load kernels directly.  It does not start
vLLM and it never performs native FP8 arithmetic.  Large logical context cases
exercise paged addressing, while a small exhaustive probe prevents correlated
pack/decode bugs from passing numeric validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import env_check, report, shapes  # noqa: E402
from common.fp8_ref import decode_e4m3fn  # noqa: E402


KID = "k09_mla_fp8_kv_sm80"
PINS = {
    "hf_kimi_k3": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    "storage": "software E4M3FN uint8",
    "compute": "BF16",
}
CHILD_SCHEMA = "k09-child/v1"
CASE_ORDER = tuple(shapes.MLA_FP8_REQUIRED_CASES)
GUARD_BYTES = 4096
SCALE = 1.0 / 128.0


def build_dry_report() -> dict[str, object]:
    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    result.add_evidence("storage_dtype", "uint8/e4m3fn")
    result.add_evidence("compute_dtype", "bfloat16")
    result.add_evidence("native_fp8_arithmetic", False)
    result.add_evidence("cache_dim", shapes.MLA_CACHE_D)
    result.add_evidence("page_size", shapes.MLA_FP8_PAGE_SIZE)
    result.add_evidence("required_shapes", shapes.REQUIRED_SHAPES[KID])
    result.set_gate(
        "NUMERIC",
        "SKIP",
        "cpu-dry-run: bit reference loaded; Triton conversion not executed",
    )
    result.set_gate(
        "SANITIZER",
        "SKIP",
        "cpu-dry-run: compute-sanitizer requires A100 execution",
    )
    result.set_gate(
        "SHAPES",
        "SKIP",
        "cpu-dry-run: 8k/256k/1M page cases registered but not executed",
    )
    result.set_gate(
        "PERF",
        "SKIP",
        "cpu-dry-run: HBM allocation and CUDA timing not measured",
    )
    return result.as_dict()


def _write_payload(payload: dict[str, object], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{KID}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_dir / f"{KID}.md").write_text(
        report.render_md(payload), encoding="utf-8"
    )


def _logical_codes(torch: Any, physical: Any, table: Any, n: int) -> Any:
    logical = torch.arange(n, device="cuda", dtype=torch.int64)
    token = logical // shapes.MLA_CACHE_D
    channel = logical % shapes.MLA_CACHE_D
    logical_page = token // shapes.MLA_FP8_PAGE_SIZE
    page_offset = token % shapes.MLA_FP8_PAGE_SIZE
    physical_page = table[logical_page].to(torch.int64)
    physical_offset = (
        (physical_page * shapes.MLA_FP8_PAGE_SIZE + page_offset)
        * shapes.MLA_CACHE_D
        + channel
    )
    return physical[physical_offset]


def _event_time_ms(torch: Any, operation: Any, reps: int = 3) -> float:
    operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]


def _run_exact_probe(torch: Any, kernels: Any) -> dict[str, object]:
    finite_codes = [
        code for code in range(256)
        if math.isfinite(decode_e4m3fn(code))
    ]
    values = [
        decode_e4m3fn(code) * SCALE for code in finite_codes
    ]
    src = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
    # One token is enough, but wrappers use K3's real 576-wide cache row.
    padded = torch.zeros(
        (1, shapes.MLA_CACHE_D), dtype=torch.bfloat16, device="cuda"
    )
    padded.view(-1)[: len(values)] = src
    physical = torch.zeros(
        shapes.MLA_FP8_PAGE_SIZE * shapes.MLA_CACHE_D,
        dtype=torch.uint8,
        device="cuda",
    )
    table = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernels.pack_bf16_to_e4m3(
        padded,
        physical,
        SCALE,
        block_table=table,
        n_tokens=1,
    )
    torch.cuda.synchronize()
    actual = physical[: len(finite_codes)].cpu().tolist()
    mismatches = [
        {
            "index": index,
            "expected": expected,
            "actual": got,
        }
        for index, (expected, got) in enumerate(zip(finite_codes, actual))
        if expected != got
    ]
    return {
        "finite_codes": len(finite_codes),
        "mismatch_count": len(mismatches),
        "mismatches_head": mismatches[:16],
    }


def _run_child(case_name: str, child_out: Path) -> int:
    import torch
    from vendor import mla_fp8_kv_sm80 as kernels

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        raise RuntimeError(f"k09 requires SM80, got {capability}")
    if case_name not in shapes.MLA_FP8_REQUIRED_CASES:
        raise ValueError(f"unknown case: {case_name}")

    cfg = shapes.MLA_FP8_REQUIRED_CASES[case_name]
    capacity_tokens = int(cfg["tokens"])
    active_tokens = int(cfg["active_tokens"])
    page_size = shapes.MLA_FP8_PAGE_SIZE
    cache_dim = shapes.MLA_CACHE_D
    physical_pages = (capacity_tokens + page_size - 1) // page_size
    active_pages = (active_tokens + page_size - 1) // page_size
    seed = shapes.stable_seed(KID, case_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(
        f"K09-CHILD start case={case_name} capacity={capacity_tokens} "
        f"active={active_tokens} pages={physical_pages}",
        flush=True,
    )
    if cfg["block_table"] == "contiguous":
        table = torch.arange(
            active_pages, dtype=torch.int32, device="cuda"
        )
    else:
        table = torch.randperm(
            physical_pages, dtype=torch.int64, device="cuda"
        )[:active_pages].to(torch.int32)

    source = torch.randn(
        (active_tokens, cache_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    source.mul_(0.5)
    physical_elements = physical_pages * page_size * cache_dim
    guarded = torch.full(
        (physical_elements + 2 * GUARD_BYTES,),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    cache = guarded[GUARD_BYTES:-GUARD_BYTES]
    cache.zero_()
    restored = torch.empty_like(source)

    def round_trip() -> None:
        kernels.pack_bf16_to_e4m3(
            source,
            cache,
            SCALE,
            block_table=table,
            n_tokens=active_tokens,
            page_size=page_size,
            cache_dim=cache_dim,
        )
        kernels.load_e4m3_to_bf16(
            cache,
            SCALE,
            restored,
            block_table=table,
            n_tokens=active_tokens,
            page_size=page_size,
            cache_dim=cache_dim,
        )

    median_ms = _event_time_ms(torch, round_trip)
    canary_ok = bool(
        torch.all(guarded[:GUARD_BYTES] == 0xA5).item()
        and torch.all(guarded[-GUARD_BYTES:] == 0xA5).item()
    )
    tail_ok = True
    tail_values = 0
    if active_tokens % page_size:
        final_page = int(table[active_pages - 1].item())
        first_tail = (
            final_page * page_size + active_tokens % page_size
        ) * cache_dim
        tail_values = (page_size - active_tokens % page_size) * cache_dim
        tail_ok = bool(
            torch.all(cache[first_tail:first_tail + tail_values] == 0).item()
        )

    difference = restored.float() - source.float()
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(
        torch.sqrt(torch.mean(source.float().square())).item()
    )
    rmse_rel = rmse / max(reference_rms, 1.0e-12)
    max_abs = float(torch.max(torch.abs(difference)).item())
    finite = bool(torch.isfinite(restored).all().item())
    exact_probe = _run_exact_probe(torch, kernels)
    numeric_pass = (
        finite
        and exact_probe["mismatch_count"] == 0
        and rmse_rel <= 0.04
        and max_abs <= 0.25
    )
    allocated_bytes = int(cache.numel())
    bf16_bytes = capacity_tokens * cache_dim * 2
    memory_ratio = allocated_bytes / bf16_bytes
    memory_pass = (
        allocated_bytes == capacity_tokens * cache_dim
        and memory_ratio == 0.5
    )
    passed = numeric_pass and canary_ok and tail_ok and memory_pass
    payload = {
        "schema_version": CHILD_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "case": case_name,
        "shape": dict(cfg),
        "storage_dtype": "uint8/e4m3fn",
        "compute_dtype": "bfloat16",
        "native_fp8_arithmetic": False,
        "numeric": {
            "passed": numeric_pass,
            "rmse_rel": rmse_rel,
            "max_abs": max_abs,
            "finite": finite,
            "exact_probe": exact_probe,
        },
        "canary_ok": canary_ok,
        "partial_tail_ok": tail_ok,
        "partial_tail_values": tail_values,
        "memory": {
            "allocated_cache_bytes": allocated_bytes,
            "bf16_equivalent_bytes": bf16_bytes,
            "ratio": memory_ratio,
            "passed": memory_pass,
        },
        "timing": {
            "round_trip_median_ms": median_ms,
            "logical_gib_per_s": (
                (active_tokens * cache_dim * 4)
                / (median_ms / 1000.0)
                / (1024.0**3)
            ),
        },
    }
    child_out.parent.mkdir(parents=True, exist_ok=True)
    child_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"K09-CHILD done case={case_name} status={payload['status']} "
        f"rmse_rel={rmse_rel:.6f} median_ms={median_ms:.3f}",
        flush=True,
    )
    return 0 if passed else 1


def _spawn_case(
    case_name: str, report_dir: Path, timeout_s: int
) -> dict[str, object]:
    child_out = report_dir / "children" / f"{case_name}.json"
    child_out.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-case",
        case_name,
        "--child-out",
        str(child_out),
    ]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        )
        returncode = process.returncode
        combined = (process.stderr or "") + "\n" + (process.stdout or "")
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        combined = (
            exc.stderr if isinstance(exc.stderr, str) else ""
        ) + "\nK09 child timeout"
    try:
        payload = json.loads(child_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    memory_error = bool(
        re.search(
            r"illegal memory access|invalid (?:read|write)|misaligned|"
            r"out of bounds",
            combined,
            flags=re.IGNORECASE,
        )
    )
    classification = (
        "MEMORY_VIOLATION"
        if memory_error
        else "PASS"
        if returncode == 0
        and isinstance(payload, dict)
        and payload.get("status") == "PASS"
        else "FAIL"
    )
    return {
        "case": case_name,
        "status": "PASS" if classification == "PASS" else "FAIL",
        "classification": classification,
        "returncode": returncode,
        "duration_s": round(time.monotonic() - started, 1),
        "result": payload,
        "log_tail": combined[-3000:],
    }


def _sanitizer_gate(
    report_dir: Path, timeout_s: int
) -> tuple[str, str, dict[str, object]]:
    sanitizer = env_check.find_compute_sanitizer()
    if not sanitizer:
        return (
            "SKIP",
            "compute-sanitizer 미발견 — full/OpenAI image에서 실행 필요",
            {},
        )
    with tempfile.TemporaryDirectory(dir=report_dir) as tempdir:
        child_out = Path(tempdir) / "sanitizer.json"
        command = [
            sanitizer,
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-case",
            "contiguous_8k",
            "--child-out",
            str(child_out),
        ]
        try:
            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_s,
                env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            )
            combined = (process.stderr or "") + "\n" + (process.stdout or "")
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            combined = (
                exc.stderr if isinstance(exc.stderr, str) else ""
            ) + "\ncompute-sanitizer timeout"
    match = re.search(r"ERROR SUMMARY:\s*(\d+)\s+error", combined)
    error_count = int(match.group(1)) if match else None
    evidence: dict[str, object] = {
        "command": command,
        "returncode": returncode,
        "error_count": error_count,
        "sanitizer_lines": [
            line for line in combined.splitlines()
            if line.startswith("=========")
        ][-80:],
    }
    if returncode == 0 and error_count == 0:
        return "PASS", "memcheck ERROR SUMMARY: 0", evidence
    if error_count is None:
        return (
            "FAIL",
            f"memcheck summary 없음 또는 비정상 종료 rc={returncode}",
            evidence,
        )
    return "FAIL", f"memcheck 오류 {error_count}건", evidence


def _run_parent(report_dir: Path, timeout_s: int) -> int:
    environment = env_check.collect_env()
    ready, checks = env_check.check_preconditions(environment)
    if not ready:
        print("[k09] GPU 사전조건 미충족", file=sys.stderr)
        for check in checks:
            if check["status"] == "FAIL":
                print(
                    f"  FAIL {check['check']}: {check['detail']}",
                    file=sys.stderr,
                )
        return report.EXIT_PRECOND

    cases: list[dict[str, object]] = []
    for case_name in CASE_ORDER:
        print(f"K09-PARENT case={case_name} launch", flush=True)
        case = _spawn_case(case_name, report_dir, timeout_s)
        cases.append(case)
        print(
            f"K09-PARENT case={case_name} status={case['status']} "
            f"class={case['classification']} duration={case['duration_s']}s",
            flush=True,
        )

    result = report.KernelReport(KID, pins=PINS)
    result.add_evidence("storage_dtype", "uint8/e4m3fn")
    result.add_evidence("compute_dtype", "bfloat16")
    result.add_evidence("native_fp8_arithmetic", False)
    result.add_evidence("cases", cases)
    valid_results = [
        case["result"] for case in cases
        if case["status"] == "PASS" and isinstance(case["result"], dict)
    ]
    numeric_ok = len(valid_results) == len(CASE_ORDER) and all(
        child["numeric"]["passed"] for child in valid_results
    )
    result.set_gate(
        "NUMERIC",
        "PASS" if numeric_ok else "FAIL",
        "전 케이스 exhaustive-code probe + BF16 round-trip 통과"
        if numeric_ok
        else "일부 케이스 numeric/child 실패",
        {"cases": [case["case"] for case in cases if case["status"] != "PASS"]},
    )
    covered = [
        str(case["case"]) for case in cases if case["status"] == "PASS"
    ]
    shape_status, shape_reason, shape_evidence = shapes.shapes_gate(
        KID, covered
    )
    result.set_gate(
        "SHAPES", shape_status, shape_reason, shape_evidence
    )
    memory_ok = len(valid_results) == len(CASE_ORDER) and all(
        child["memory"]["passed"] for child in valid_results
    )
    timing_ok = len(valid_results) == len(CASE_ORDER) and all(
        child["timing"]["round_trip_median_ms"] > 0.0
        for child in valid_results
    )
    result.set_gate(
        "PERF",
        "PASS" if memory_ok and timing_ok else "FAIL",
        "uint8 cache=BF16 대비 0.5× 및 CUDA timing 유효"
        if memory_ok and timing_ok
        else "HBM footprint 또는 timing 검증 실패",
        {
            child["case"]: {
                "memory": child["memory"],
                "timing": child["timing"],
            }
            for child in valid_results
        },
    )
    sanitizer_status, sanitizer_reason, sanitizer_evidence = _sanitizer_gate(
        report_dir, timeout_s
    )
    result.set_gate(
        "SANITIZER",
        sanitizer_status,
        sanitizer_reason,
        sanitizer_evidence,
    )
    json_path, _ = result.write(str(report_dir))
    print(
        f"K09-OVERALL {result.overall()} report={json_path}",
        flush=True,
    )
    return result.exit_code()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=KIT / "kernel-report")
    parser.add_argument("--phase0-verdict")  # run_all compatibility
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--child-case", choices=CASE_ORDER)
    parser.add_argument("--child-out", type=Path)
    args = parser.parse_args()

    if args.child_case:
        if args.child_out is None:
            parser.error("--child-case requires --child-out")
        return _run_child(args.child_case, args.child_out)
    if args.cpu_dry_run:
        payload = build_dry_report()
        _write_payload(payload, args.report_dir)
        print(
            f"K09-OVERALL INCOMPLETE mode=cpu-dry-run "
            f"report={args.report_dir / (KID + '.json')}"
        )
        return 0
    return _run_parent(args.report_dir, args.timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
