#!/usr/bin/env python3
"""Verify the immutable K3.1 (Kimi-K3.1-Pro) OpenAI image: metadata, patch set, CPU/GPU contracts.

K3 이미지용 verify_image.py 와 별개다(그 파일은 그대로). 검사는 세 구역으로 나뉘어 출력된다.

  [static]  GPU 불필요 -- 이미지 라벨/ENV, 핀 sha, overlay 매니페스트·패치셋 sha256, 설치된 .so, torch/CUDA 계약
  [import]  GPU 불필요 -- 컨테이너 안에서 CPU 로 import 가능한 것(API 서버·파서·DSA 모델·TRITON_MLA_SPARSE 백엔드·
            Triton 커널 모듈·Marlin MoE op 등록)과 `vllm serve --help` 플래그
  [gpu]     --gpu 필요 -- A100(SM80) 장치 수/세대, 드라이버(r580+), Triton JIT 스모크 (사내 A100 노드에서 실행)

GPU 없는 빌드 호스트에서는 [gpu] 구역이 SKIP 으로 표시된다 -- 그 경우 GPU 동작은 검증되지 않은 것이다.
이 스크립트는 "이미지가 계약대로 만들어졌는가" 를 본다. 커널 수치/성능은 kernel-kit gate(k08/k11/k12)가 판정한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "k31-sm80-openai:a3c14efb-sm80-v1"
COMMIT = "a3c14efbc77c018e8bf579935095a87721349e05"
PATCH_NAMES = (
    "mxfp4-marlin-moe-sm80-main",
    "triton-mla-sparse-sm80",
    "k31-sm80-dsa-glue",
)
PATCHSET = "+".join(PATCH_NAMES)
DEFAULT_PATCH_DIR = REPO_ROOT / "docker" / "patches-k31"
EXPECTED_LABELS = {
    "org.opencontainers.image.revision": COMMIT,
    "io.k3.cuda_arch": "sm80",
    "io.k3.patchset": PATCHSET,
    "io.k31.vllm_commit": COMMIT,
}
EXPECTED_ENV = {
    "K3_PATCHSET": PATCHSET,
    "K31_VLLM_COMMIT": COMMIT,
    "TORCH_CUDA_ARCH_LIST": "8.0",
}
BASE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

REQUIRED_SERVE_FLAGS = (
    "--distributed-executor-backend",
    "--data-parallel-backend",
    "--nnodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--headless",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--data-parallel-size",
    "--data-parallel-rpc-port",
    "--enable-expert-parallel",
    "--all2all-backend",
    "--moe-backend",
    "--attention-backend",
    "--attention-config",
    "--kda-prefill-backend",
    "--quantization",
    "--kv-cache-dtype",
    "--kv-cache-dtype-skip-layers",
    "--speculative-config",
    "--language-model-only",
    "--reasoning-parser",
    "--tool-call-parser",
    "--served-model-name",
)

SCOPE_TITLES = {
    "static": "[static] GPU 불필요 -- 메타데이터·패치셋·설치물 계약",
    "import": "[import] GPU 불필요 -- 컨테이너 CPU import",
    "cli": "[cli]    GPU 불필요 -- vllm serve --help 플래그",
    "gpu": "[gpu]    --gpu 필요 -- A100 노드에서만 실행 가능",
}
CHECK_PREFIX = "K31-CHECK|"
PATCH_SHA_PREFIX = "K31-PATCH-SHA|"
PROBE_END = "K31-PROBE-END"

# 컨테이너 안에서 실행하는 probe 는 파일(/tmp/k31_probe.py)로 저장한 뒤 실행한다:
# Triton @jit 는 함수 소스를 파일에서 읽으므로 `python3 -c` 로는 JIT 스모크를 돌릴 수 없다.
PROBE_PRELUDE = r"""
import os
import pathlib

EXPECTED = "a3c14efbc77c018e8bf579935095a87721349e05"
BUILD = pathlib.Path("/opt/k31-openai-build")


def check(scope, name, fn):
    try:
        out = fn()
        status, detail = out if isinstance(out, tuple) else ("PASS", out or "")
    except BaseException as exc:  # noqa: BLE001 - probe 는 어떤 실패도 한 줄로 보고한다
        status = "FAIL"
        detail = (type(exc).__name__ + ": " + str(exc)).replace("\n", " ")[:400]
    print(f"K31-CHECK|{scope}|{name}|{status}|{detail}", flush=True)
"""

CPU_PROBE = (
    PROBE_PRELUDE
    + r"""
import hashlib
import importlib
import shutil
import sysconfig

PATCHES = pathlib.Path("/opt/k31-patches")
REQUIRED_OVERLAY = (
    "vllm/v1/attention/backends/registry.py",
    "vllm/v1/attention/ops/mqa_logits_triton.py",
    "vllm/v1/attention/ops/triton_mla_sparse_kernel.py",
)
IMPORTS = (
    "vllm.entrypoints.openai.api_server",
    "vllm.tool_parsers.kimi_k3_tool_parser",
    "vllm.reasoning.kimi_k3_reasoning_parser",
    "vllm.models.kimi_k3.nvidia.dsa",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.triton_mla_sparse_kernel",
)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def vllm_root():
    import vllm

    return pathlib.Path(vllm.__file__).parent


def provenance():
    for path in ("/opt/vllm-build/VLLM_COMMIT.txt", str(BUILD / "VLLM_COMMIT.txt")):
        got = pathlib.Path(path).read_text().strip()
        assert got == EXPECTED, f"{path}={got}"
    assert os.environ.get("K31_VLLM_COMMIT") == EXPECTED, "ENV K31_VLLM_COMMIT"
    return "VLLM_COMMIT.txt x2 + ENV match the pin"


def torch_contract():
    import torch

    assert torch.__version__.startswith("2.14.0"), torch.__version__
    assert (torch.version.cuda or "").startswith("13.0"), torch.version.cuda
    assert "sm_80" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
    return f"torch={torch.__version__} cuda={torch.version.cuda} arch={torch.cuda.get_arch_list()}"


def overlay():
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    assert vllm_root() == site / "vllm", (str(vllm_root()), str(site))
    rows = [
        line.split("\t")
        for line in (BUILD / "overlay-manifest.tsv").read_text().splitlines()[1:]
        if line
    ]
    assert rows, "empty overlay manifest"
    bad = [rel for _s, rel, _p, patched in rows if sha(site / rel) != patched]
    assert not bad, f"installed != patched: {bad}"
    missing = set(REQUIRED_OVERLAY) - {rel for _s, rel, _p, _h in rows}
    assert not missing, f"not in overlay: {sorted(missing)}"
    return f"{len(rows)} overlay files match the manifest"


def shipped_patches():
    lines = [
        line.split()
        for line in (BUILD / "patches.sha256").read_text().splitlines()
        if line.strip()
    ]
    for digest, path in lines:
        name = pathlib.Path(path).name
        print(f"K31-PATCH-SHA|{digest}|{name}", flush=True)
        assert sha(PATCHES / name) == digest, f"/opt/k31-patches/{name} != patches.sha256"
    return f"{len(lines)} patches shipped in /opt/k31-patches and match patches.sha256"


def moe_so():
    want = (BUILD / "moe-so.sha256").read_text().split()[0]
    got = sha(vllm_root() / "_moe_C_stable_libtorch.abi3.so")
    assert got == want, "installed _moe_C_stable_libtorch != built one"
    return "installed .so == the one built with the marlin patch"


def deepgemm_absent():
    dg = vllm_root() / "third_party" / "deep_gemm"
    hits = sorted(str(p) for p in dg.glob("_C_extension*.so")) if dg.is_dir() else []
    assert not hits, hits
    return "no DeepGEMM _C_extension (SM80-only build)"


def kimi_k3_code():
    root = vllm_root()
    assert (root / "models" / "kimi_k3" / "nvidia" / "dsa.py").is_file(), "dsa.py missing"
    return "models/kimi_k3/nvidia/dsa.py present (K3.1 DSA code)"


def sanitizer():
    found = shutil.which("compute-sanitizer")
    if not found:
        import glob

        hits = glob.glob("/usr/local/cuda*/bin/compute-sanitizer")
        assert hits, "compute-sanitizer not found"
        found = hits[0]
    return found


def import_module(name):
    def run():
        importlib.import_module(name)
        return ""

    return run


def marlin_op():
    import torch

    try:
        importlib.import_module("vllm._moe_C_stable_libtorch")
    except ImportError as exc:
        # GPU 없는 호스트에서는 .so 의 libcuda.so.1 요구가 풀리지 않을 수 있다 -- 그 경우만 --gpu 로 미룬다.
        if "libcuda" in str(exc):
            return ("WARN", f"deferred to --gpu (libcuda unresolved without a GPU): {exc}")
        raise
    assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"), "op not registered"
    return "torch.ops._moe_C.moe_wna16_marlin_gemm registered"


def sparse_backend():
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cls = AttentionBackendEnum.TRITON_MLA_SPARSE.get_class()
    assert cls.is_mla() and cls.is_sparse(), (cls.is_mla(), cls.is_sparse())
    assert cls.supports_compute_capability(DeviceCapability(8, 0)), "SM80 not supported"
    return f"{cls.__module__}.{cls.__name__} (mla+sparse, sm80)"


check("static", "provenance", provenance)
check("static", "torch-cuda", torch_contract)
check("static", "overlay-manifest", overlay)
check("static", "shipped-patches", shipped_patches)
check("static", "moe-so", moe_so)
check("static", "deepgemm-absent", deepgemm_absent)
check("static", "kimi-k3-dsa-code", kimi_k3_code)
check("static", "compute-sanitizer", sanitizer)
for module in IMPORTS:
    check("import", module, import_module(module))
check("import", "moe_wna16_marlin_gemm", marlin_op)
check("import", "AttentionBackendEnum.TRITON_MLA_SPARSE", sparse_backend)
print("K31-PROBE-END", flush=True)
"""
)

GPU_PROBE = (
    PROBE_PRELUDE
    + r"""
import subprocess

EXPECT_GPUS = int(os.environ.get("K31_EXPECT_GPUS", "8"))

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _double(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

except Exception as exc:  # noqa: BLE001
    _double = None
    TRITON_ERROR = repr(exc)


def devices():
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    count = torch.cuda.device_count()
    assert count == EXPECT_GPUS, f"devices={count} != {EXPECT_GPUS}"
    for index in range(count):
        name = torch.cuda.get_device_name(index)
        cap = torch.cuda.get_device_capability(index)
        assert "A100" in name, (index, name)
        assert cap == (8, 0), (index, cap)
    return f"{count}x {torch.cuda.get_device_name(0)} (sm_80)"


def driver():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).split()[0]
    if int(out.split(".")[0]) < 580:
        return ("WARN", f"driver {out} < r580 (CUDA 13.0 userland needs r580+ unless cuda-compat applies)")
    return f"driver {out}"


def triton_jit():
    import torch

    assert _double is not None, TRITON_ERROR
    x = torch.arange(2048, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    _double[(2,)](x, y, x.numel(), BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2.0), "triton kernel result mismatch"
    return f"triton {triton.__version__} JIT compile+launch ok (needs gcc + Python.h at runtime)"


def marlin_op():
    import importlib

    import torch

    importlib.import_module("vllm._moe_C_stable_libtorch")
    assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"), "op not registered"
    return "torch.ops._moe_C.moe_wna16_marlin_gemm registered with the driver present"


check("gpu", "cuda-devices", devices)
check("gpu", "driver", driver)
check("gpu", "triton-jit-smoke", triton_jit)
check("gpu", "marlin-op-registered", marlin_op)
print("K31-PROBE-END", flush=True)
"""
)

CLI_HELP_PROBE = r"""
import sys
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform

# `vllm serve --help` materializes DeviceConfig defaults before parsing any
# arguments.  A CPU-only build host therefore resolves UnspecifiedPlatform and
# exits before printing help, even though the CUDA image itself is valid.  Use
# vLLM's CPU platform only for parser construction; the A100 runtime still
# performs the independent GPU probe when --gpu is requested.
platforms._current_platform = CpuPlatform()
sys.argv = ["vllm", "serve", "--help=all"]
from vllm.entrypoints.cli.main import main
main()
"""


def run(
    argv: list[str], input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, text=True, input=input_text, capture_output=True, check=False
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def inspect_image(image: str) -> tuple[dict[str, object] | None, list[str]]:
    result = run(["docker", "image", "inspect", image])
    if result.returncode:
        return None, [
            f"docker image inspect rc={result.returncode}: {result.stderr.strip()}"
        ]
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, [f"docker inspect returned invalid JSON: {exc}"]
    if not isinstance(values, list) or len(values) != 1:
        return None, ["docker inspect must return exactly one image"]
    return values[0], []


def metadata_findings(
    metadata: dict[str, object], base_image_id: str | None = None
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for labels, ENV, entrypoint and healthcheck."""
    errors: list[str] = []
    warnings: list[str] = []
    config = metadata.get("Config")
    if not isinstance(config, dict):
        return ["image Config missing"], warnings
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        labels = {}
    for key, expected in EXPECTED_LABELS.items():
        if labels.get(key) != expected:
            errors.append(f"label {key} mismatch")
    digest = labels.get("io.k3.base_digest")
    if digest == "unpinned":
        warnings.append(
            "label io.k3.base_digest is 'unpinned' (built without build-k31.sh)"
        )
    elif not isinstance(digest, str) or not BASE_DIGEST_RE.match(digest):
        errors.append("label io.k3.base_digest is not a sha256:<64 hex> value")
    elif base_image_id is not None and digest != base_image_id:
        errors.append("label io.k3.base_digest does not match --base-image id")
    env_items = config.get("Env")
    env: dict[str, str] = {}
    if isinstance(env_items, list):
        for item in env_items:
            if isinstance(item, str) and "=" in item:
                key, _, value = item.partition("=")
                env[key] = value
    for key, expected in EXPECTED_ENV.items():
        if env.get(key) != expected:
            errors.append(f"ENV {key} mismatch")
    if any(key.startswith("VLLM_") and key != "VLLM_API_PORT" for key in env):
        errors.append(
            "image ENV defines VLLM_* variables (vLLM warns on unknown VLLM_* env)"
        )
    if config.get("Entrypoint") != ["vllm", "serve"]:
        errors.append("entrypoint mismatch")
    if not config.get("Healthcheck"):
        errors.append("healthcheck missing")
    return errors, warnings


def parse_sha256_listing(text: str) -> list[tuple[str, str]]:
    """Parse `sha256sum` style lines into [(digest, file name)]."""
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append((parts[0], Path(parts[1]).name))
    return rows


def patch_errors(image_rows: list[tuple[str, str]], patch_dir: Path) -> list[str]:
    """Compare the image's patches.sha256 with the repo's docker/patches-k31 files."""
    errors: list[str] = []
    expected_names = [f"{name}.patch" for name in PATCH_NAMES]
    if [name for _, name in image_rows] != expected_names:
        errors.append(
            "image patch list/order mismatch: "
            f"{[name for _, name in image_rows]} != {expected_names}"
        )
    for digest, name in image_rows:
        local = patch_dir / name
        if not local.is_file():
            errors.append(f"repo patch missing for comparison: {local}")
            continue
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            errors.append(f"patch {name}: image sha256 differs from {local}")
    return errors


def required_cli_errors(help_text: str) -> list[str]:
    return [
        f"vllm serve --help missing flag: {flag}"
        for flag in REQUIRED_SERVE_FLAGS
        if flag not in help_text
    ]


def parse_checks(output: str) -> list[tuple[str, str, str, str]]:
    """Parse probe stdout lines `K31-CHECK|scope|name|STATUS|detail`."""
    checks: list[tuple[str, str, str, str]] = []
    for line in output.splitlines():
        if not line.startswith(CHECK_PREFIX):
            continue
        parts = line[len(CHECK_PREFIX) :].split("|", 3)
        if len(parts) == 4:
            checks.append((parts[0], parts[1], parts[2], parts[3]))
    return checks


def parse_patch_rows(output: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        if line.startswith(PATCH_SHA_PREFIX):
            parts = line[len(PATCH_SHA_PREFIX) :].split("|", 1)
            if len(parts) == 2:
                rows.append((parts[0], parts[1]))
    return rows


def container_probe(
    image: str, source: str, *, gpu: bool, expect_gpus: int
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]], list[str]]:
    """Run a probe script inside the image; return (checks, patch rows, errors)."""
    argv = ["docker", "run", "--rm", "-i", "-e", f"K31_EXPECT_GPUS={expect_gpus}"]
    if gpu:
        argv.extend(["--gpus", "all"])
    argv.extend(
        [
            "--entrypoint",
            "bash",
            image,
            "-c",
            "cat > /tmp/k31_probe.py && python3 /tmp/k31_probe.py",
        ]
    )
    result = run(argv, input_text=source)
    checks = parse_checks(result.stdout)
    rows = parse_patch_rows(result.stdout)
    errors: list[str] = []
    if PROBE_END not in result.stdout:
        tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        errors.append(
            f"{'GPU' if gpu else 'CPU'} probe did not finish rc={result.returncode}: "
            + " | ".join(tail)
        )
    return checks, rows, errors


def render(
    checks: list[tuple[str, str, str, str]], *, gpu_requested: bool
) -> list[str]:
    lines: list[str] = []
    for scope in ("static", "import", "cli", "gpu"):
        lines.append(SCOPE_TITLES[scope])
        scoped = [item for item in checks if item[0] == scope]
        if not scoped and scope == "gpu" and not gpu_requested:
            lines.append(
                "  SKIP  (CPU 모드: GPU 동작은 검증되지 않음 -- A100 노드에서 --gpu 로 재실행)"
            )
        for _scope, name, status, detail in scoped:
            suffix = f" -- {detail}" if detail else ""
            lines.append(f"  {status:<4}  {name}{suffix}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image",
        metavar="IMAGE",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=f"image to verify (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also run the [gpu] checks (docker run --gpus all): A100 devices, driver, Triton JIT",
    )
    parser.add_argument(
        "--expect-gpus",
        type=int,
        default=8,
        help="number of A100 devices expected with --gpu (default: 8)",
    )
    parser.add_argument(
        "--base-image",
        metavar="TAG",
        help="local full-tier image whose id must equal the io.k3.base_digest label",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=DEFAULT_PATCH_DIR,
        help="repo directory holding the three patches (default: docker/patches-k31)",
    )
    parser.add_argument(
        "--skip-cli-help",
        action="store_true",
        help="skip the `vllm serve --help=all` flag check",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[tuple[str, str, str, str]] = []
    metadata, found = inspect_image(args.image)
    errors.extend(found)
    if metadata is not None:
        base_id = None
        if args.base_image:
            base_meta, base_errors = inspect_image(args.base_image)
            errors.extend(base_errors)
            if base_meta is not None:
                base_id = str(base_meta.get("Id"))
        meta_errors, meta_warnings = metadata_findings(metadata, base_id)
        errors.extend(meta_errors)
        warnings.extend(meta_warnings)

        cpu_checks, rows, probe_errors = container_probe(
            args.image, CPU_PROBE, gpu=False, expect_gpus=args.expect_gpus
        )
        checks.extend(cpu_checks)
        errors.extend(probe_errors)
        errors.extend(patch_errors(rows, args.patch_dir))

        if not args.skip_cli_help:
            help_result = run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python3",
                    args.image,
                    "-c",
                    CLI_HELP_PROBE,
                ]
            )
            if help_result.returncode:
                tail = (
                    (help_result.stderr or help_result.stdout).strip().splitlines()[-8:]
                )
                errors.append(
                    f"vllm serve --help rc={help_result.returncode}: {' | '.join(tail)}"
                )
                checks.append(("cli", "serve-help", "FAIL", "see error above"))
            else:
                flag_errors = required_cli_errors(help_result.stdout)
                errors.extend(flag_errors)
                status = "FAIL" if flag_errors else "PASS"
                detail = f"{len(REQUIRED_SERVE_FLAGS)} required flags"
                checks.append(("cli", "serve-help", status, detail))
        if args.gpu:
            gpu_checks, _rows, gpu_errors = container_probe(
                args.image, GPU_PROBE, gpu=True, expect_gpus=args.expect_gpus
            )
            checks.extend(gpu_checks)
            errors.extend(gpu_errors)

    for _scope, name, status, detail in checks:
        if status == "FAIL":
            errors.append(f"check {name}: {detail}")
        elif status == "WARN":
            warnings.append(f"check {name}: {detail}")

    print("\n".join(render(checks, gpu_requested=args.gpu)))
    for warning in warnings:
        print(f"K31-OPENAI-VERIFY-WARN {warning}")
    for error in errors:
        print(f"K31-OPENAI-VERIFY-FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    image_id = metadata.get("Id", "unknown") if metadata else "unknown"
    mode = "gpu" if args.gpu else "cpu"
    print(
        f"K31-OPENAI-VERIFY-PASS image={args.image} id={image_id} mode={mode} "
        f"checks={len(checks)} warnings={len(warnings)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
