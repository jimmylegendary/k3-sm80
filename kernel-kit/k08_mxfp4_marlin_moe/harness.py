#!/usr/bin/env python3
"""Original Kimi-K3 MXFP4 Marlin MoE gate for A100 (SM80).

This is a standalone kernel harness.  It does not start a vLLM server and does
not load the 1.56 TB checkpoint.  Each GPU case synthesizes tensors in the
original checkpoint layout, repacks them with vLLM's K3 MXFP4 path, and calls
``fused_marlin_moe`` directly.  Cases run in isolated child processes so a
CUDA illegal access cannot poison the remaining crash ladder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import env_check, report, shapes, timing, tolerance  # noqa: E402
from common.mxfp4_ref import decode_e2m1, decode_e8m0  # noqa: E402


KID = "k08_mxfp4_marlin_moe"
PINS = {
    "vllm_kimi_k3": "0498dc7ea9204f1863b46515816f5d8d458e01e7",
    "hf_kimi_k3": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    "marlin_fix": "vllm#36889 adapted to libtorch_stable",
}
ARMS: dict[str, dict[str, int]] = {
    # TP8: all 896 experts, one eighth of the intermediate width.
    "tp8": {
        "local_experts": shapes.N_EXPERTS,
        "global_experts": shapes.N_EXPERTS,
        "local_n": shapes.MOE_INTER // 8,
    },
    # EP32: 28 complete experts per rank.
    "ep32": {
        "local_experts": shapes.N_EXPERTS // 32,
        "global_experts": shapes.N_EXPERTS,
        "local_n": shapes.MOE_INTER,
    },
}
CASE_ORDER = ("m1", "m8", "m32", "prefill")
NUMERIC_CASES = frozenset(("m1", "m8", "m32"))
PERF_CASE = "m32"
CHILD_SCHEMA = "k08-child/v1"
GUARD_ELEMENTS = 32

_ILLEGAL_PATTERNS = (
    r"\billegal memory access\b",
    r"\binvalid (?:__global__ |__shared__ )?(?:read|write|access)\b",
    r"\bmisaligned address\b",
    r"\bout of bounds\b",
)
_OOM_PATTERN = re.compile(
    r"cudaErrorMemoryAllocation|CUDA out of memory|out of memory",
    re.IGNORECASE,
)


def build_dry_report() -> dict[str, object]:
    """Validate the checkpoint-format/reference contract without claiming GPU PASS."""
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
    result.add_evidence("layout_arms", list(ARMS))
    result.add_evidence("arm_shapes", ARMS)
    result.add_evidence(
        "checkpoint_layout",
        {
            "weight_packed": "[E,N,K/2], low nibble first, E2M1",
            "weight_scale": "[E,N,K/32], E8M0FNU",
            "block_size": 32,
            "activation": "BF16 (weight-only Marlin)",
            "requantized": False,
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
    result.set_gate("PERF", "SKIP", "cpu-dry-run: no GPU timing")
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


def _read_child_json(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict) or value.get("schema_version") != CHILD_SCHEMA:
        return None, "schema mismatch"
    return value, None


def classify_child(
    case: str,
    returncode: int,
    child_json: str,
    stderr: str,
) -> dict[str, object]:
    """Classify a child without conflating allocator OOM and memory corruption."""
    combined = stderr or ""
    memory_violation_count = sum(
        len(re.findall(pattern, combined, flags=re.IGNORECASE))
        for pattern in _ILLEGAL_PATTERNS
    )
    allocator_oom = bool(_OOM_PATTERN.search(combined)) and memory_violation_count == 0
    child, child_error = _read_child_json(child_json)

    if memory_violation_count:
        classification = "MEMORY_VIOLATION"
    elif allocator_oom:
        classification = "OOM"
    elif returncode < 0:
        classification = "SIGNAL"
    elif returncode != 0:
        classification = "CRASH"
    elif child is None:
        classification = "MISSING_RESULT"
    elif child.get("status") != "PASS":
        classification = "CHILD_FAIL"
    elif not child.get("canary_ok"):
        classification = "CANARY_FAIL"
    else:
        classification = "PASS"

    return {
        "case": case,
        "returncode": returncode,
        "classification": classification,
        "status": "PASS" if classification == "PASS" else "FAIL",
        "allocator_oom": allocator_oom,
        "memory_violation_count": memory_violation_count,
        "child_json_valid": child is not None,
        "child_json_error": child_error,
        "result": child,
        "stderr_tail": combined[-2000:],
    }


def _balanced_mxfp4_packed(torch: Any, shape: tuple[int, ...]) -> Any:
    """Generate E2M1 bytes using only +/-0.5 and +/-1.0, low nibble first."""
    raw = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
    low = (raw & 1) + 1 + ((raw >> 1) & 1) * 8
    high = ((raw >> 2) & 1) + 1 + ((raw >> 3) & 1) * 8
    return (low | (high << 4)).contiguous()


def _make_checkpoint_tensors(torch: Any, arm: str) -> tuple[Any, Any, Any, Any]:
    cfg = ARMS[arm]
    e = cfg["local_experts"]
    n = cfg["local_n"]
    k = shapes.LATENT
    # Names intentionally match the compressed-tensors checkpoint contract.
    weight_packed_w13 = _balanced_mxfp4_packed(torch, (e, 2 * n, k // 2))
    weight_packed_w2 = _balanced_mxfp4_packed(torch, (e, k, n // 2))
    # 121 encodes 2^-6.  This keeps the two-layer reference in a stable range.
    weight_scale_w13 = torch.full(
        (e, 2 * n, k // 32), 121, dtype=torch.uint8, device="cuda"
    )
    weight_scale_w2 = torch.full(
        (e, k, n // 32), 121, dtype=torch.uint8, device="cuda"
    )
    return (
        weight_packed_w13,
        weight_packed_w2,
        weight_scale_w13,
        weight_scale_w2,
    )


def _routing(torch: Any, arm: str, m: int) -> tuple[Any, Any, Any | None]:
    cfg = ARMS[arm]
    topk = shapes.TOPK
    if cfg["local_experts"] < topk:
        raise RuntimeError(f"{arm}: local experts smaller than top-k")
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(m, 1)
    weights = torch.full(
        (m, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    if arm == "ep32":
        expert_map = torch.full(
            (cfg["global_experts"],), -1, dtype=torch.int32, device="cuda"
        )
        expert_map[: cfg["local_experts"]] = torch.arange(
            cfg["local_experts"], dtype=torch.int32, device="cuda"
        )
    else:
        expert_map = None
    return ids, weights, expert_map


def _dequant_mxfp4(torch: Any, packed: Any, scales: Any) -> Any:
    lut = torch.tensor(
        shapes.MXFP4_E2M1_VALUES, dtype=torch.float32, device=packed.device
    )
    out = torch.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,),
        dtype=torch.float32,
        device=packed.device,
    )
    out[..., 0::2] = lut[(packed & 0x0F).long()]
    out[..., 1::2] = lut[(packed >> 4).long()]
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return out * scale


def _situ_reference(torch: Any, projected: Any) -> Any:
    n = projected.shape[-1] // 2
    gate = projected[..., :n]
    up = projected[..., n:]
    beta = float(shapes.SITU_BETA)
    linear_beta = float(shapes.SITU_LINEAR_BETA)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _bf16_reference(
    torch: Any,
    hidden_states: Any,
    logical_w13: Any,
    logical_w2: Any,
    logical_s13: Any,
    logical_s2: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> Any:
    """FP32 accumulation over the exact logical MXFP4 weights used by Marlin."""
    output = torch.zeros(
        hidden_states.shape, dtype=torch.float32, device=hidden_states.device
    )
    for expert in range(shapes.TOPK):
        positions = torch.nonzero(topk_ids == expert, as_tuple=False)
        token_ids = positions[:, 0]
        slots = positions[:, 1]
        w13 = _dequant_mxfp4(
            torch, logical_w13[expert], logical_s13[expert]
        )
        projected = hidden_states[token_ids].float() @ w13.T
        activated = _situ_reference(torch, projected)
        del projected, w13
        w2 = _dequant_mxfp4(torch, logical_w2[expert], logical_s2[expert])
        partial = activated @ w2.T
        partial *= topk_weights[token_ids, slots].unsqueeze(1)
        output.index_add_(0, token_ids, partial)
        del activated, w2, partial
    return output


def _situ_activation(
    activation: Any,
    output: Any,
    input: Any,
    **_: Any,
) -> None:
    import torch

    torch.ops._C.situ_and_mul(
        output,
        input,
        float(shapes.SITU_BETA),
        float(shapes.SITU_LINEAR_BETA),
    )


def _run_child(arm: str, case: str, child_out: Path) -> int:
    import torch
    from vllm import _custom_ops  # noqa: F401 - registers torch custom ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError(
            f"k08 requires SM80, got {torch.cuda.get_device_capability(0)}"
        )
    if arm not in ARMS or case not in CASE_ORDER:
        raise ValueError(f"unknown arm/case: {arm}/{case}")

    torch.manual_seed(shapes.stable_seed(KID, f"{arm}:{case}"))
    torch.cuda.manual_seed_all(shapes.stable_seed(KID, f"{arm}:{case}"))
    cfg = ARMS[arm]
    case_cfg = shapes.REQUIRED_SHAPES[KID][case]
    m = case_cfg["m"]
    k = case_cfg["k"]
    n = cfg["local_n"]

    print(
        f"K08-CHILD start arm={arm} case={case} "
        f"M={m} K={k} Nlocal={n} Elocal={cfg['local_experts']}",
        flush=True,
    )
    logical_w13, logical_w2, logical_s13, logical_s2 = (
        _make_checkpoint_tensors(torch, arm)
    )
    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    w13, w2, w13_scale, w2_scale, _, _ = (
        prepare_moe_mxfp4_layer_for_marlin(
            layer,
            logical_w13,
            logical_w2,
            logical_s13,
            logical_s2,
            None,
            None,
        )
    )
    topk_ids, topk_weights, expert_map = _routing(torch, arm, m)
    hidden_states = (
        torch.randn((m, k), dtype=torch.float32, device="cuda") * 0.25
    ).to(torch.bfloat16)

    guard = torch.full(
        (m * k + 2 * GUARD_ELEMENTS,),
        123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guard[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(m, k)
    output.fill_(float("nan"))
    workspace13 = torch.empty(
        (m * shapes.TOPK, max(2 * n, k)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    workspace2 = torch.empty(
        (m * shapes.TOPK, n),
        dtype=torch.bfloat16,
        device="cuda",
    )

    def kernel() -> Any:
        return fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            bias1=None,
            bias2=None,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=False,
            global_num_experts=(
                cfg["global_experts"] if arm == "ep32" else -1
            ),
            activation=MoEActivation.SITU,
            activation_func=_situ_activation,
            expert_map=expert_map,
            intermediate_cache13=workspace13,
            intermediate_cache2=workspace2,
            output=output,
        )

    kernel()
    torch.cuda.synchronize()
    canary_ok = bool(
        torch.all(guard[:GUARD_ELEMENTS] == 123.0).item()
        and torch.all(guard[-GUARD_ELEMENTS:] == 123.0).item()
    )
    finite = bool(torch.isfinite(output).all().item())

    numeric: dict[str, Any] | None = None
    if case in NUMERIC_CASES:
        reference = _bf16_reference(
            torch,
            hidden_states,
            logical_w13,
            logical_w2,
            logical_s13,
            logical_s2,
            topk_ids,
            topk_weights,
        )
        compared = tolerance.compare(
            KID, reference, output, case=f"{arm}:{case}"
        )
        numeric = {
            "metrics": compared.metrics,
            "violations": compared.violations,
            "passed": compared.passed,
        }
        del reference

    warmup, reps = ((1, 3) if case == "prefill" else (3, 10))
    stats = timing.measure(kernel, warmup=warmup, reps=reps)
    passed = canary_ok and finite and (
        numeric is None or bool(numeric["passed"])
    )
    payload = {
        "schema_version": CHILD_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "arm": arm,
        "case": case,
        "shape": {
            "m": m,
            "k": k,
            "n_local": n,
            "local_experts": cfg["local_experts"],
            "global_experts": cfg["global_experts"],
            "topk": shapes.TOPK,
        },
        "checkpoint": {
            "weight_packed": "E2M1 low-nibble-first",
            "weight_scale": "E8M0FNU block32",
            "requantized": False,
        },
        "numeric": numeric,
        "canary_ok": canary_ok,
        "finite": finite,
        "timing": stats,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    child_out.parent.mkdir(parents=True, exist_ok=True)
    child_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"K08-CHILD done arm={arm} case={case} status={payload['status']} "
        f"median_us={stats['median_us']:.1f} canary={canary_ok}",
        flush=True,
    )
    return 0 if passed else 1


def _spawn_child(
    arm: str,
    case: str,
    report_dir: Path,
    timeout_s: int,
) -> dict[str, object]:
    child_out = report_dir / "children" / f"{arm}-{case}.json"
    child_out.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_LAUNCH_BLOCKING": "1",
            "VLLM_MXFP4_USE_MARLIN": "1",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "0",
            "VLLM_MARLIN_MXFP8_INPUT_QDQ": "0",
        }
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-arm",
        arm,
        "--child-case",
        case,
        "--child-out",
        str(child_out),
    ]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        returncode = process.returncode
        stderr = (process.stderr or "") + "\n" + (process.stdout or "")
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stderr = (
            (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        ) + "\nK08 child timeout"
    record = classify_child(case, returncode, str(child_out), stderr)
    record.update(
        {
            "arm": arm,
            "duration_s": round(time.monotonic() - started, 1),
            "command": command,
        }
    )
    return record


def _sanitizer_gate(
    report_dir: Path,
    timeout_s: int,
) -> tuple[str, str, dict[str, object]]:
    sanitizer = env_check.find_compute_sanitizer()
    if not sanitizer:
        return (
            "SKIP",
            "compute-sanitizer 미발견 — full/OpenAI image에서 실행 필요",
            {},
        )

    child_out = report_dir / "children" / "sanitizer-ep32-m1.json"
    child_out.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_LAUNCH_BLOCKING": "1",
            "VLLM_MXFP4_USE_MARLIN": "1",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "0",
            "VLLM_MARLIN_MXFP8_INPUT_QDQ": "0",
        }
    )
    command = [
        sanitizer,
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-arm",
        "ep32",
        "--child-case",
        "m1",
        "--child-out",
        str(child_out),
    ]
    try:
        process = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        combined = (process.stderr or "") + "\n" + (process.stdout or "")
        returncode = process.returncode
    except subprocess.TimeoutExpired as exc:
        combined = (
            (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        ) + "\ncompute-sanitizer timeout"
        returncode = 124

    classified = classify_child(
        "sanitizer:ep32:m1", returncode, str(child_out), combined
    )
    summary = re.search(r"ERROR SUMMARY:\s*(\d+)\s*error", combined)
    error_count = int(summary.group(1)) if summary else None
    evidence: dict[str, object] = {
        "command": command,
        "returncode": returncode,
        "error_count": error_count,
        "memory_violation_count": classified["memory_violation_count"],
        "allocator_oom": classified["allocator_oom"],
        "classification": classified["classification"],
        "sanitizer_lines": [
            line
            for line in combined.splitlines()
            if line.startswith("=========")
        ][:120],
    }
    if classified["memory_violation_count"]:
        return (
            "FAIL",
            f"memcheck memory violation "
            f"{classified['memory_violation_count']}건",
            evidence,
        )
    if classified["allocator_oom"]:
        return (
            "SKIP",
            "memcheck allocator OOM — memory violation 0건이나 clean 실행 미완료",
            evidence,
        )
    child, _ = _read_child_json(str(child_out))
    if returncode == 0 and error_count == 0 and child and child.get("status") == "PASS":
        return ("PASS", "memcheck clean: EP32 m1, ERROR SUMMARY 0", evidence)
    if returncode == 124:
        return ("SKIP", "compute-sanitizer timeout — 재실행 필요", evidence)
    return (
        "FAIL",
        f"memcheck 비정상 종료 rc={returncode}, "
        f"ERROR SUMMARY={error_count}",
        evidence,
    )


def _normal_gpu_run(args: argparse.Namespace) -> int:
    env = env_check.collect_env()
    ok, checks = env_check.check_preconditions(env, cpu_dry_run=False)
    if not ok or not env.get("vllm"):
        print("[k08] GPU/vLLM 사전조건 미충족", file=sys.stderr)
        for item in checks:
            print(
                f"  {item['status']:4s} {item['check']:20s} {item['detail']}",
                file=sys.stderr,
            )
        if not env.get("vllm"):
            print("  FAIL vllm                patched OpenAI image required", file=sys.stderr)
        return report.EXIT_PRECOND

    selected_arms = [arm for arm in args.arms.split(",") if arm]
    selected_cases = [case for case in args.cases.split(",") if case]
    if any(arm not in ARMS for arm in selected_arms):
        raise ValueError(f"unknown --arms: {selected_arms}")
    if any(case not in CASE_ORDER for case in selected_cases):
        raise ValueError(f"unknown --cases: {selected_cases}")

    result = report.KernelReport(KID, dry_run=False, pins=PINS)
    result.add_evidence("layout_arms", selected_arms)
    result.add_evidence("arm_shapes", ARMS)
    result.add_evidence("checkpoint_requantized", False)
    records: list[dict[str, object]] = []
    for arm in selected_arms:
        for case in selected_cases:
            record = _spawn_child(arm, case, args.report_dir, args.case_timeout)
            records.append(record)
            print(
                f"K08-CASE {arm}/{case} {record['classification']} "
                f"rc={record['returncode']} t={record['duration_s']}s",
                flush=True,
            )

    numeric_results: list[tolerance.NumericResult] = []
    for record in records:
        child = record.get("result")
        numeric = child.get("numeric") if isinstance(child, dict) else None
        if numeric is not None:
            numeric_results.append(
                tolerance.NumericResult(
                    metrics=numeric["metrics"],
                    violations=numeric["violations"],
                    passed=bool(numeric["passed"]),
                )
            )
        elif record["case"] in NUMERIC_CASES:
            numeric_results.append(
                tolerance.NumericResult(
                    metrics={"case": f"{record['arm']}:{record['case']}", "n": 0},
                    violations=[
                        f"{record['classification']} rc={record['returncode']}"
                    ],
                    passed=False,
                )
            )
    status, reason, evidence = tolerance.numeric_gate(KID, numeric_results)
    result.set_gate("NUMERIC", status, reason, evidence)

    all_requested = set(selected_arms) == set(ARMS) and set(selected_cases) == set(
        CASE_ORDER
    )
    covered = [
        case
        for case in CASE_ORDER
        if all(
            any(
                record["arm"] == arm
                and record["case"] == case
                and record["classification"] == "PASS"
                for record in records
            )
            for arm in ARMS
        )
    ]
    if not all_requested:
        result.set_gate(
            "SHAPES",
            "SKIP",
            "부분 실행(--arms/--cases): TP8×EP32 전체 crash ladder 미판정",
            {"covered_both_arms": covered},
        )
    else:
        status, reason, evidence = shapes.shapes_gate(KID, covered)
        evidence["required_arms"] = list(ARMS)
        result.set_gate("SHAPES", status, reason, evidence)

    if args.skip_sanitizer:
        result.set_gate(
            "SANITIZER", "SKIP", "--skip-sanitizer 지정 (부분 진단 실행)"
        )
    else:
        status, reason, evidence = _sanitizer_gate(
            args.report_dir, args.sanitizer_timeout
        )
        result.set_gate("SANITIZER", status, reason, evidence)

    perf_evidence: dict[str, object] = {"arms": {}}
    perf_statuses: list[str] = []
    perf_reasons: list[str] = []
    for arm in ARMS:
        record = next(
            (
                item
                for item in records
                if item["arm"] == arm
                and item["case"] == PERF_CASE
                and item["classification"] == "PASS"
            ),
            None,
        )
        child = record.get("result") if record else None
        measured = (
            child["timing"]["median_us"] if isinstance(child, dict) else None
        )
        stats = child.get("timing") if isinstance(child, dict) else None
        cfg = ARMS[arm]
        model = shapes.perf_bytes_model(
            KID,
            m=shapes.REQUIRED_SHAPES[KID][PERF_CASE]["m"],
            k=shapes.LATENT,
            n=cfg["local_n"],
            e=cfg["local_experts"],
            topk=shapes.TOPK,
        )
        arm_status, arm_reason, arm_evidence = timing.perf_gate(
            KID,
            measured,
            model,
            args.phase0_verdict,
            args.twin,
            stats=stats,
        )
        perf_evidence["arms"][arm] = {
            "status": arm_status,
            "reason": arm_reason,
            "evidence": arm_evidence,
        }
        perf_statuses.append(arm_status)
        perf_reasons.append(f"{arm}={arm_status}")
    if "FAIL" in perf_statuses:
        perf_status = "FAIL"
    elif all(item == "PASS" for item in perf_statuses):
        perf_status = "PASS"
    else:
        perf_status = "SKIP"
    result.set_gate(
        "PERF",
        perf_status,
        ", ".join(perf_reasons),
        perf_evidence,
    )

    result.add_evidence(
        "ladder",
        [
            {
                "arm": item["arm"],
                "case": item["case"],
                "classification": item["classification"],
                "returncode": item["returncode"],
                "duration_s": item["duration_s"],
                "allocator_oom": item["allocator_oom"],
                "memory_violation_count": item["memory_violation_count"],
                "median_us": (
                    (item.get("result") or {}).get("timing") or {}
                ).get("median_us"),
            }
            for item in records
        ],
    )
    result.add_evidence(
        "failure_tails",
        {
            f"{item['arm']}:{item['case']}": item["stderr_tail"]
            for item in records
            if item["classification"] != "PASS"
        },
    )
    json_path, _ = result.write(str(args.report_dir))
    for gate in report.GATES:
        gate_result = result.gates[gate]
        print(
            f"K08-GATE {gate}={gate_result['status']} "
            f"{gate_result['reason'][:140]}",
            flush=True,
        )
    print(f"K08-OVERALL {result.overall()} report={json_path}", flush=True)
    return result.exit_code()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-verdict")
    parser.add_argument("--twin")
    parser.add_argument(
        "--report-dir", type=Path, default=KIT / "kernel-report"
    )
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--arms", default="tp8,ep32")
    parser.add_argument("--cases", default=",".join(CASE_ORDER))
    parser.add_argument("--skip-sanitizer", action="store_true")
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--sanitizer-timeout", type=int, default=3600)
    parser.add_argument("--child-arm", choices=tuple(ARMS), help=argparse.SUPPRESS)
    parser.add_argument("--child-case", choices=CASE_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.child_arm or args.child_case or args.child_out:
        if not (args.child_arm and args.child_case and args.child_out):
            raise ValueError("all child arguments are required together")
        return _run_child(args.child_arm, args.child_case, args.child_out)
    if args.cpu_dry_run:
        payload = build_dry_report()
        write_payload(payload, args.report_dir)
        print(
            f"[k08] dry-run report={args.report_dir / (KID + '.json')} "
            f"overall={payload['overall']} layout_arms=tp8,ep32"
        )
        return report.EXIT_OK
    return _normal_gpu_run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"[k08] internal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(report.EXIT_ERR)
