"""Render the committed serving lane directories deterministically."""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .lane_specs import (
    FAMILY_K3,
    FAMILY_K31,
    LANES,
    LaneSpec,
    lane_by_number,
    lanes_in_family,
)

_TOOLING_DIR = Path(__file__).resolve().parent
_DEPLOY_DIR = _TOOLING_DIR.parent
_TEMPLATE_DIR = _TOOLING_DIR / "templates"


_CONTROL_LINE = re.compile(r"^@@(IF|ELSE|ENDIF)(?: ([A-Z][A-Z0-9_]*))?@@$")


def _truthy(value: object) -> bool:
    return value not in (None, "", 0, False, "0")


def _apply_conditionals(name: str, text: str, context: dict[str, object]) -> str:
    """Resolve whole-line ``@@IF NAME@@`` / ``@@ELSE@@`` / ``@@ENDIF@@`` blocks.

    제어 줄은 줄 전체를 차지해야 하며 개행까지 함께 제거된다. 비활성 분기의 줄도 통째로
    제거되므로, K3 lane에서는 K3.1 전용 블록이 있던 자리가 템플릿에 블록을 넣기 전과
    byte-identical이다. 조건 이름은 ``context``의 키이며 값이 비었거나 ``0``이면 거짓이다.
    블록은 중첩할 수 있다.
    """

    output: list[str] = []
    # 프레임: (바깥 분기가 활성인지, IF 조건의 진릿값, ELSE를 이미 만났는지)
    stack: list[tuple[bool, bool, bool]] = []
    active = True
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        match = _CONTROL_LINE.match(line.rstrip("\n"))
        if match is None:
            if active:
                output.append(line)
            continue
        kind, key = match.groups()
        where = f"{name}:{number}"
        if (kind == "IF") != (key is not None):
            raise ValueError(f"{where}: malformed control line {line.strip()!r}")
        if kind == "IF":
            assert key is not None
            if key not in context:
                raise ValueError(f"{where}: unknown condition {key!r}")
            condition = _truthy(context[key])
            stack.append((active, condition, False))
            active = active and condition
        elif kind == "ELSE":
            if not stack or stack[-1][2]:
                raise ValueError(f"{where}: @@ELSE@@ without an open @@IF@@ block")
            parent_active, condition, _ = stack[-1]
            stack[-1] = (parent_active, condition, True)
            active = parent_active and not condition
        else:
            if not stack:
                raise ValueError(f"{where}: @@ENDIF@@ without an open @@IF@@ block")
            active = stack.pop()[0]
    if stack:
        raise ValueError(f"{name}: unterminated @@IF@@ block")
    return "".join(output)


def render_template(name: str, context: dict[str, object]) -> str:
    rendered = (_TEMPLATE_DIR / name).read_text(encoding="utf-8")
    rendered = _apply_conditionals(name, rendered, context)
    for key, value in sorted(context.items()):
        rendered = rendered.replace(f"@@{key}@@", str(value))
    unresolved = sorted(
        token.split("@@", 1)[0]
        for token in rendered.split("@@")[1::2]
        if token
    )
    if unresolved:
        raise ValueError(f"unresolved template keys in {name}: {unresolved}")
    return rendered


def _quote_env(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() for char in value) or value.startswith(("{", "[")):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def _lane_values(spec: LaneSpec) -> dict[str, str]:
    number = spec.number
    speculative = ""
    if spec.dspark:
        speculative = json.dumps(
            {
                "method": "dspark",
                "model": spec.draft_model,
                "num_speculative_tokens": 7,
                # draft(dense MLA)의 backend다. target의 spec.attention_backend와 다르다.
                "attention_backend": "TRITON_MLA",
                "draft_sample_method": "probabilistic",
                "rejection_sample_method": "block",
            },
            separators=(",", ":"),
        )
    return {
        "K3_IMAGE": spec.image,
        "K3_TARGET_MODEL": spec.target_model,
        "K3_DRAFT_MODEL": spec.draft_model,
        "K3_HF_HOME": "/mnt/nas/home/models",
        "K3_SERVING_ROOT": spec.serving_root,
        "K3_API_BIND": "0.0.0.0",
        "K3_HEAD_NODE_IP": "CHANGE_ME_HEAD_NODE_IP",
        "K3_HEAD_SSH_HOST": "CHANGE_ME_HEAD_SSH_HOST",
        "K3_HEAD_SSH_USER": "CHANGE_ME_HEAD_SSH_USER",
        "K3_WORKER1_NODE_IP": "CHANGE_ME_WORKER1_NODE_IP",
        "K3_WORKER1_SSH_HOST": "CHANGE_ME_WORKER1_SSH_HOST",
        "K3_WORKER1_SSH_USER": "CHANGE_ME_WORKER1_SSH_USER",
        "K3_WORKER2_NODE_IP": "CHANGE_ME_WORKER2_NODE_IP",
        "K3_WORKER2_SSH_HOST": "CHANGE_ME_WORKER2_SSH_HOST",
        "K3_WORKER2_SSH_USER": "CHANGE_ME_WORKER2_SSH_USER",
        "K3_WORKER3_NODE_IP": "CHANGE_ME_WORKER3_NODE_IP",
        "K3_WORKER3_SSH_HOST": "CHANGE_ME_WORKER3_SSH_HOST",
        "K3_WORKER3_SSH_USER": "CHANGE_ME_WORKER3_SSH_USER",
        "K3_NCCL_SOCKET_IFNAME": "CHANGE_ME_NCCL_SOCKET_IFNAME",
        "K3_GLOO_SOCKET_IFNAME": "CHANGE_ME_GLOO_SOCKET_IFNAME",
        "K3_NCCL_IB_HCA": "CHANGE_ME_NCCL_IB_HCA",
        "K3_NCCL_IB_GID_INDEX": "0",
        "K3_ENGINE_READY_TIMEOUT_S": "3600",
        "K3_EXECUTE_MODEL_TIMEOUT_S": "1800",
        "VLLM_API_PORT": str(8100 + number),
        "VLLM_MASTER_PORT": str(29600 + number),
        "VLLM_DP_RPC_PORT": str(29700 + number),
        "VLLM_TENSOR_PARALLEL_SIZE": str(spec.tp),
        "VLLM_PIPELINE_PARALLEL_SIZE": str(spec.pp),
        "VLLM_DATA_PARALLEL_SIZE": str(spec.dp),
        "VLLM_DATA_PARALLEL_BACKEND": "mp",
        "VLLM_ENABLE_EXPERT_PARALLEL": "1" if spec.ep else "0",
        "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter",
        "VLLM_MOE_BACKEND": "marlin",
        "VLLM_ATTENTION_BACKEND": spec.attention_backend,
        "VLLM_ATTENTION_CONFIG": json.dumps(
            {"use_prefill_query_quantization": False},
            separators=(",", ":"),
        ),
        # K3.1 전용 키: 값이 비면 키 자체를 쓰지 않아 K3 lane .env는 바이트가 바뀌지 않는다.
        **(
            {"VLLM_SPARSE_INDEXER_TOPK_BACKEND": spec.sparse_indexer_topk_backend}
            if spec.sparse_indexer_topk_backend
            else {}
        ),
        "VLLM_KDA_PREFILL_BACKEND": "triton",
        "VLLM_QUANTIZATION": "mxfp4",
        "VLLM_DTYPE": "bfloat16",
        "VLLM_KV_CACHE_DTYPE": "fp8" if spec.fp8_kv else "bfloat16",
        "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS": (
            "93 94 95 96 97" if spec.fp8_kv and spec.dspark else ""
        ),
        "VLLM_K3_SM80_FP8_KV": "1" if spec.fp8_kv else "0",
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "0",
        "VLLM_MARLIN_MXFP8_INPUT_QDQ": "0",
        "VLLM_MAX_MODEL_LEN": str(spec.max_len),
        "VLLM_MAX_NUM_SEQS": "1",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.92",
        "VLLM_ENFORCE_EAGER": "1",
        "VLLM_ENABLE_PREFIX_CACHING": "0",
        "VLLM_LANGUAGE_MODEL_ONLY": "1",
        "VLLM_COMPILATION_CONFIG": "{}",
        "VLLM_SPECULATIVE_CONFIG": speculative,
        "VLLM_SERVED_MODEL_NAME": spec.served_name,
        "VLLM_TOOL_CALL_PARSER": "kimi_k3",
        "VLLM_REASONING_PARSER": "kimi_k3",
        "VLLM_EXTRA_ARGS": "--trust-remote-code --disable-custom-all-reduce",
    }


def render_env(context: dict[str, object]) -> str:
    spec = context["SPEC"]
    if not isinstance(spec, LaneSpec):
        raise TypeError("context['SPEC'] must be LaneSpec")
    values = _lane_values(spec)
    groups = (
        (
            "Shared image, models, and filesystem",
            (
                "K3_IMAGE",
                "K3_TARGET_MODEL",
                "K3_DRAFT_MODEL",
                "K3_HF_HOME",
                "K3_SERVING_ROOT",
                "K3_API_BIND",
            ),
        ),
        (
            "Four-node cluster (replace every CHANGE_ME value)",
            tuple(key for key in values if "_NODE_IP" in key or "_SSH_" in key)
            + (
                "K3_NCCL_SOCKET_IFNAME",
                "K3_GLOO_SOCKET_IFNAME",
                "K3_NCCL_IB_HCA",
                "K3_NCCL_IB_GID_INDEX",
                "K3_ENGINE_READY_TIMEOUT_S",
                "K3_EXECUTE_MODEL_TIMEOUT_S",
            ),
        ),
        (
            "vLLM serving controls",
            tuple(key for key in values if key.startswith("VLLM_")),
        ),
    )
    lines = [
        f"# Generated lane: {spec.id}",
        "# This is the only .env used by head/worker1/worker2/worker3 for this lane.",
    ]
    emitted: set[str] = set()
    for title, keys in groups:
        lines.extend(("", f"# {title}"))
        for key in keys:
            if key in emitted:
                continue
            emitted.add(key)
            lines.append(f"{key}={_quote_env(values[key])}")
    missing = values.keys() - emitted
    if missing:
        raise AssertionError(f"env renderer omitted keys: {sorted(missing)}")
    return "\n".join(lines) + "\n"


# 같은 family의 마지막 lane을 통과한 뒤 가리키는 다음 단계.
_FAMILY_TERMINAL_LANE = {
    FAMILY_K3: "production-candidate",
    FAMILY_K31: "K3.1 P2",
}


def _next_lane(spec: LaneSpec) -> str:
    later = [
        lane.number
        for lane in lanes_in_family(spec.family)
        if lane.number > spec.number
    ]
    if later:
        return f"{min(later):02d}"
    return _FAMILY_TERMINAL_LANE[spec.family]


def _context(spec: LaneSpec) -> dict[str, Any]:
    from .gates import GATE_MATRIX, PREREQUISITE_GATES

    number = f"{spec.number:02d}"
    load_gate = "pp" if spec.pp > 1 else "ep"
    context_gate = "1m" if spec.fp8_kv else load_gate
    if spec.fp8_kv:
        context_checkpoints = "262144 → 524288 → 786432 → 1048576"
        calibrate_instruction = (
            "`make calibrate`로 현재 `VLLM_MAX_NUM_SEQS` 후보를 검증한다."
        )
    else:
        context_checkpoints = "8192 → 131072 → 262144"
        calibrate_instruction = "이 lane에는 calibration을 적용하지 않는다."
    dspark_instruction = (
        "`make gate-dspark`를 실행한다."
        if spec.dspark
        else "이 lane에는 DSpark gate를 적용하지 않는다."
    )
    baseline_note = (
        "`make baseline`으로 다음 DSpark lane의 target-only fixture를 남긴다."
        if spec.number in (2, 4)
        else "이 lane은 다음 DSpark 비교용 baseline fixture를 만들지 않는다."
    )
    if spec.number == 5:
        kv_layout_note = (
            "lane 05의 target MLA는 SM80 software FP8 cache를 사용하지만, "
            "RoPE를 쓰는 DSpark draft 5개 layer(93–97)는 "
            "`VLLM_KV_CACHE_DTYPE_SKIP_LAYERS='93 94 95 96 97'`로 BF16을 "
            "유지한다. 이 값은 pinned 93-layer target에 묶인 fail-closed 계약이다."
        )
    elif spec.is_k31:
        kv_layout_note = (
            "K3.1 DSA 층(24개)의 MLA KV cache는 BF16이며 `--kv-cache-dtype bfloat16`을 "
            "명시한다. `auto`/`fp8*`는 vLLM `KimiK3ForConditionalGenerationConfig`가 "
            "`fp8_ds_mla`로 재작성하므로 쓰지 않는다. DSA indexer-K cache는 vLLM 기본 "
            "형식(E4M3 128 B + fp32 scale 4 B)이고 SM80 Triton MQA logits가 읽는다. "
            "software FP8 KV 패치는 이 lane에서 적용하지 않는다."
        )
    elif spec.fp8_kv:
        kv_layout_note = (
            "target MLA cache는 SM80 software FP8 storage를 사용한다. "
            "native FP8 arithmetic은 사용하지 않는다."
        )
    else:
        kv_layout_note = "MLA KV cache는 BF16이며 software FP8 경로는 비활성이다."
    return {
        "SPEC": spec,
        "LANE_ID": spec.id,
        "LANE_NUMBER": number,
        "TP": spec.tp,
        "PP": spec.pp,
        "DP": spec.dp,
        "EP": int(spec.ep),
        "MAX_LEN": spec.max_len,
        "DSPARK": int(spec.dspark),
        "FP8_KV": int(spec.fp8_kv),
        "REQUIRED_GATES": " ".join(GATE_MATRIX[number]),
        "PREREQUISITE_GATES": " ".join(PREREQUISITE_GATES[number]),
        "LOAD_GATE": load_gate,
        "CONTEXT_GATE": context_gate,
        "BASELINE_LANE_ID": (
            lane_by_number(spec.number - 1, spec.family).id
            if spec.dspark
            else spec.id
        ),
        "NEXT_LANE": _next_lane(spec),
        "TARGET_MODEL": spec.target_model,
        "TARGET_CACHE_DIR": spec.target_model.replace("/", "--"),
        "DRAFT_CACHE_DIR": spec.draft_model.replace("/", "--"),
        "SERVING_ROOT": spec.serving_root,
        "IMAGE_VERIFIER": spec.image_verifier,
        # 템플릿의 @@IF 조건. K3 lane에서는 전부 거짓이라 해당 블록이 통째로 빠진다.
        "K31": int(spec.is_k31),
        "SPARSE_TOPK": int(bool(spec.sparse_indexer_topk_backend)),
        "HAS_DRAFT": int(bool(spec.draft_model)),
        "CONTEXT_CHECKPOINTS": context_checkpoints,
        "CALIBRATE_INSTRUCTION": calibrate_instruction,
        "DSPARK_INSTRUCTION": dspark_instruction,
        "BASELINE_NOTE": baseline_note,
        "KV_LAYOUT_NOTE": kv_layout_note,
    }


def render_lane(spec: LaneSpec, root: Path) -> list[Path]:
    lane = root / spec.id
    scripts = lane / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    context = _context(spec)
    outputs = {
        lane / ".env": render_env(context),
        lane / "compose.yaml": render_template("compose.yaml.tmpl", context),
        lane / "Makefile": render_template("Makefile.tmpl", context),
        lane / "RUNBOOK.md": render_template("RUNBOOK.md.tmpl", context),
        scripts / "preflight.py": render_template("preflight.py.tmpl", context),
        scripts / "cluster.py": render_template("cluster.py.tmpl", context),
        scripts / "smoke.py": render_template("smoke.py.tmpl", context),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return sorted(outputs)


def render_all(root: Path = _DEPLOY_DIR) -> list[Path]:
    outputs: list[Path] = []
    for spec in LANES:
        outputs.extend(render_lane(spec, root))
    return outputs


def _tree_diff(expected: Path, actual: Path) -> list[str]:
    errors: list[str] = []
    for spec in LANES:
        relative = Path(spec.id)
        comparison = filecmp.dircmp(expected / relative, actual / relative)
        errors.extend(f"missing: {relative / name}" for name in comparison.left_only)
        errors.extend(f"unexpected: {relative / name}" for name in comparison.right_only)
        errors.extend(f"changed: {relative / name}" for name in comparison.diff_files)
        for subdir in comparison.common_dirs:
            nested = filecmp.dircmp(
                expected / relative / subdir,
                actual / relative / subdir,
            )
            errors.extend(
                f"missing: {relative / subdir / name}" for name in nested.left_only
            )
            errors.extend(
                f"unexpected: {relative / subdir / name}" for name in nested.right_only
            )
            errors.extend(
                f"changed: {relative / subdir / name}" for name in nested.diff_files
            )
    return errors


def check_all(root: Path = _DEPLOY_DIR) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="k3-lanes-") as tempdir:
        expected = Path(tempdir)
        render_all(expected)
        return _tree_diff(expected, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on generated drift")
    parser.add_argument("--root", type=Path, default=_DEPLOY_DIR)
    args = parser.parse_args()
    if args.check:
        errors = check_all(args.root)
        for error in errors:
            print(error)
        print(f"lanes={len(LANES)} drift={len(errors)}")
        return 1 if errors else 0
    outputs = render_all(args.root)
    print(f"lanes={len(LANES)} files={len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
