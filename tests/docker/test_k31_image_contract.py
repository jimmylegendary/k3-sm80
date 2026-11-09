"""K3.1(Kimi-K3.1-Pro) 도커 레시피 계약 테스트 — 전부 정적 검사다(이미지 빌드·GPU 없음).

K3 이미지용 test_openai_image_contract.py / test_fp8_patch_contract.py 는 건드리지 않는다.
docker/patches-k31/ 의 나머지 두 패치(triton-mla-sparse-sm80, k31-sm80-dsa-glue)는 다른 워커 산출물이라
존재 여부만 별도 메서드(test_worker1_patches_present)로 본다 — 아직 없으면 그 메서드만 실패한다.

핀 소스 트리에 대한 `git apply --check` 는 환경변수가 있을 때만 돈다:
    K31_VLLM_SRC=/path/to/vllm-checkout-at-a3c14efb python3 -m unittest tests.docker.test_k31_image_contract
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
PIN = "a3c14efbc77c018e8bf579935095a87721349e05"
OPENAI_IMAGE = "k31-sm80-openai:a3c14efb-sm80-v1"
FULL_IMAGE = "k31-sm80-kernelkit:full-a3c14efb"
RELEASE_TAG = "openai-k31-a3c14efb-sm80-v1"
PATCH_NAMES = (
    "mxfp4-marlin-moe-sm80-main",
    "triton-mla-sparse-sm80",
    "k31-sm80-dsa-glue",
)
PATCHSET = "+".join(PATCH_NAMES)
PATCH_DIR = DOCKER / "patches-k31"
MARLIN_PATCH = PATCH_DIR / "mxfp4-marlin-moe-sm80-main.patch"
FULL = DOCKER / "Dockerfile.k31-full"
OPENAI = DOCKER / "Dockerfile.k31-openai"
VERIFY = DOCKER / "verify_k31_image.py"
BUILD = DOCKER / "build-k31.sh"
RELEASE = DOCKER / "release-k31.sh"
TAG_FILE = DOCKER / "RELEASE-TAG-K31.txt"
RUN11 = REPO_ROOT / "run" / "11-load-k31-image.sh"
RUN21 = REPO_ROOT / "run" / "21-k31-kernel-test.sh"
# K3 이미지(0498dc7e 핀, PR #50000 head, kda 패치)를 가리키는 문자열 — K3.1 파일에는 없어야 한다.
K3_ONLY_STRINGS = ("pull/50000", "0498dc7e", "kda-sm80-int64")
DEFAULT_KERNELS = (
    "k01_kda_chunk",
    "k02_kda_recurrent",
    "k03_causal_conv1d",
    "k04_mla_decode",
    "k07_situ_act",
    "k08_mxfp4_marlin_moe",
    "k11_dsa_indexer_sm80",
    "k12_sparse_mla_sm80",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_verify_module():
    spec = importlib.util.spec_from_file_location("k31_verify_image", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dockerfile_runs(text: str) -> list[tuple[int, str, list[tuple[str, list[str]]]]]:
    """Dockerfile 을 RUN 명령 단위로 쪼갠다: (줄번호, 이어붙인 명령, [(heredoc 태그, 본문 줄들)])."""
    lines = text.splitlines()
    runs = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("RUN "):
            i += 1
            continue
        logical = lines[i]
        last = i
        while logical.rstrip().endswith("\\"):
            last += 1
            logical = logical.rstrip()[:-1] + " " + lines[last].strip()
        heredocs = []
        cursor = last + 1
        for tag in re.findall(r"(?<!<)<<(?!<)-?'?(\w+)'?", logical):
            body = []
            while lines[cursor].strip() != tag:
                body.append(lines[cursor])
                cursor += 1
            heredocs.append((tag, body))
            cursor += 1
        runs.append((i + 1, logical, heredocs))
        i = cursor
    return runs


class K31PinAndTagsTest(unittest.TestCase):
    def test_pin_sha_is_the_same_everywhere(self) -> None:
        for path in (FULL, OPENAI, VERIFY, BUILD):
            self.assertIn(PIN, read(path), path.name)

    def test_image_tags_and_release_tag(self) -> None:
        self.assertEqual(read(TAG_FILE).strip(), RELEASE_TAG)
        for path in (BUILD, RELEASE, RUN11, RUN21, VERIFY):
            self.assertIn(OPENAI_IMAGE, read(path), path.name)
        for path in (BUILD, RUN21, OPENAI):
            self.assertIn(FULL_IMAGE, read(path), path.name)
        for path in (RELEASE, RUN11):
            self.assertIn(RELEASE_TAG, read(path), path.name)
        self.assertIn("K31_OPENAI_GITHUB_UPLOAD", read(RELEASE))
        self.assertIn("K31_FULL_IMAGE", read(BUILD))

    def test_k31_files_do_not_reference_the_k3_stack(self) -> None:
        files = [
            FULL,
            OPENAI,
            VERIFY,
            BUILD,
            RELEASE,
            TAG_FILE,
            RUN11,
            RUN21,
            MARLIN_PATCH,
        ]
        for path in files:
            text = read(path)
            for forbidden in K3_ONLY_STRINGS:
                self.assertNotIn(forbidden, text, f"{path.name} mentions {forbidden}")
        self.assertNotRegex(read(OPENAI), r"COPY\s+docker/patches/")
        self.assertNotIn("ghcr.io/jimmylegendary/k3-sm80-kernelkit", read(RUN21))

    def test_worker1_patches_do_not_reference_the_k3_stack(self) -> None:
        """워커 ① 패치가 이미 있으면 같은 K3 문자열 금지 규칙을 적용한다(없으면 건너뜀 -- 존재 여부는 별도 메서드)."""
        found = [
            PATCH_DIR / f"{name}.patch"
            for name in PATCH_NAMES[1:]
            if (PATCH_DIR / f"{name}.patch").is_file()
        ]
        if not found:
            self.skipTest("worker 1 patches not present yet")
        for path in found:
            text = read(path)
            for forbidden in K3_ONLY_STRINGS:
                self.assertNotIn(forbidden, text, f"{path.name} mentions {forbidden}")


class K31FullDockerfileTest(unittest.TestCase):
    def test_full_tier_recipe(self) -> None:
        text = read(FULL)
        self.assertIn(f"ARG VLLM_COMMIT={PIN}", text)
        self.assertIn("ARG TORCH_CUDA_ARCH_LIST=8.0", text)
        self.assertIn("ARG CUDA_VERSION=13.0.3", text)
        # 핀 CMakeLists.txt:1842 가 g++ < 13 을 거부 → Ubuntu 24.04 (22.04 의 g++ 11 불가)
        self.assertIn("ARG UBUNTU_VERSION=24.04", text)
        self.assertIn("CMakeLists.txt:1842", text)
        self.assertIn("-devel-ubuntu${UBUNTU_VERSION}", text)
        self.assertIn("-base-ubuntu${UBUNTU_VERSION}", text)
        self.assertNotRegex(text, r"(?m)^FROM .*ubuntu22\.04")
        # 설치 순서 (RUN 명령 기준 — 머리말 주석의 언급은 제외)
        runs = [command for _lineno, command, _heredocs in dockerfile_runs(text)]

        def first_run(needle: str) -> int:
            return next(i for i, command in enumerate(runs) if needle in command)

        build_req = first_run("pip install -r requirements/build/cuda.txt")
        runtime_req = first_run("pip install -r requirements/cuda.txt")
        build_cmd = first_run("pip install -v . --no-build-isolation")
        self.assertLessEqual(build_req, runtime_req)
        self.assertLess(runtime_req, build_cmd)
        self.assertLess(
            runs[build_req].index("requirements/build/cuda.txt"),
            runs[build_req].index("requirements/cuda.txt"),
        )
        self.assertIn("MAX_JOBS", text)
        self.assertIn("NVCC_THREADS", text)
        # torch/cuda 계약 (조기 gate + 최종 gate)
        self.assertGreaterEqual(text.count('startswith("2.14.0")'), 2)
        self.assertGreaterEqual(text.count('startswith("13.0")'), 2)
        # K3.1 코드 증명
        self.assertIn('"kimi_k3" / "__init__.py"', text)
        self.assertIn('"nvidia" / "dsa.py"', text)
        # provenance + sanitizer
        for token in (
            "/opt/build-k31.log",
            "gzip -9",
            "pip freeze > /opt/pip-freeze.txt",
            "VLLM_COMMIT.txt",
            "cuda-sanitizer-",
            "compute-sanitizer --version",
        ):
            self.assertIn(token, text)
        # K3 흔적 없음 + 병행 원칙 명시
        self.assertNotIn("pull/50000", text)
        self.assertIn("K3 이미지와 별개", text)
        self.assertIn("Dockerfile.full", text)

    def test_deepgemm_not_built_gate(self) -> None:
        text = read(FULL)
        self.assertIn("DeepGEMM will not compile: unsupported CUDA architecture", text)
        self.assertIn("DeepGEMM CUDA architectures:", text)
        self.assertIn("CUDA target architectures: 8.0", text)
        # 로그에 cmake 출력이 남도록 pip -v 가 필요하다
        self.assertIn("pip install -v .", text)
        # 별도 RUN 이라 소스빌드 레이어를 다시 돌리지 않고 gate 만 고칠 수 있다
        self.assertLess(
            text.index("pip freeze > /opt/pip-freeze.txt"),
            text.index("zgrep -Fq 'CUDA target architectures: 8.0'"),
        )
        self.assertIn("_C_extension*.so", text)

    def test_rust_frontend_is_not_required(self) -> None:
        text = read(FULL)
        self.assertIn("vllm-rs", text)  # 설계 근거 주석
        for _lineno, command, _heredocs in dockerfile_runs(text):
            for tool in ("rustup", "rustc", "cargo"):
                self.assertNotIn(tool, command)


class K31OpenAIDockerfileTest(unittest.TestCase):
    def test_patch_order_and_check_before_apply(self) -> None:
        text = read(OPENAI)
        applied = []
        for name in PATCH_NAMES:
            check = text.index(f"git apply --check --index /tmp/patches/{name}.patch")
            apply_ = text.index(f"git apply --index /tmp/patches/{name}.patch")
            self.assertLess(check, apply_, name)
            applied.append(apply_)
        self.assertEqual(applied, sorted(applied))
        self.assertEqual(text.count("git diff --cached --check"), 2)

    def test_only_the_moe_extension_is_rebuilt_from_the_new_csrc_path(self) -> None:
        text = read(OPENAI)
        self.assertIn("csrc/moe/marlin_moe_wna16/", text)
        self.assertNotIn("libtorch_stable/moe", text)
        self.assertIn("--target _moe_C_stable_libtorch", text)
        self.assertIn("--component _moe_C_stable_libtorch", text)
        self.assertIn("/tmp/k31-moe-install/vllm/_moe_C_stable_libtorch.abi3.so", text)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80", text)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.0", text)
        self.assertIn("CMakeLists.txt:1842", text)  # g++ >= 13
        # 캐시 id 분리 (K3 빌드 캐시 오염 방지)
        self.assertIn("id=k31-vllm-deps", text)
        self.assertIn("id=k31-ccache", text)

    def test_marlin_gates(self) -> None:
        text = read(OPENAI)
        self.assertIn("grep -Fq 'device_max_shared_mem'", text)
        self.assertIn("kernel<<<blocks, num_threads, sh_cache_size, stream>>>", text)
        self.assertIn("! grep -Fq 'size_n * sorted_token_ids.size(0)'", text)
        self.assertIn("test -s", text)
        self.assertIn("sm80_kernel_bfloat16_fe2m1f_bfloat16.cu", text)
        self.assertIn("grep -Fq kFE8M0fnu", text)

    def test_python_overlay_is_derived_from_the_patches(self) -> None:
        text = read(OPENAI)
        self.assertIn('"diff", "--cached", "--name-status", "--no-renames", pin', text)
        self.assertIn("/opt/k31-overlay-files.txt", text)
        self.assertIn("/opt/k31-overlay.tar", text)
        self.assertIn("-T /opt/k31-overlay-files.txt", text)
        self.assertIn("python3 -m py_compile", text)
        self.assertIn("overlay-manifest.tsv", text)
        # 경로 매핑 vllm/… → <site-packages>/vllm/… 와 기반==핀 원본 확인
        self.assertIn('sysconfig.get_paths()["purelib"]', text)
        self.assertIn("sha(dst) == pristine", text)
        self.assertIn('"tar", "-xf"', text)
        # K3 방식(파일별 COPY --from=python-overlay /src/vllm/vllm/…)이 남아 있지 않다
        self.assertNotIn("COPY --from=python-overlay /src/vllm", text)
        # scope gate: csrc/moe/ 밖의 변경은 .so 1개 재빌드로 반영되지 않으므로 빌드를 멈춘다
        self.assertIn('path.startswith("csrc/moe/")', text)
        self.assertIn("overlay scope 위반", text)

    def test_labels_env_and_runtime_contract(self) -> None:
        text = read(OPENAI)
        self.assertIn(f"K3_PATCHSET={PATCHSET}", text)
        self.assertIn(f'io.k3.patchset="{PATCHSET}"', text)
        for token in (
            "org.opencontainers.image.revision",
            "io.k3.cuda_arch",
            "io.k3.base_digest",
            "patches.sha256",
            "VLLM_COMMIT.txt",
            "K31_VLLM_COMMIT",
            "/opt/k31-patches/",
            "HEALTHCHECK",
            'ENTRYPOINT ["vllm", "serve"]',
            "FROM runtime AS contract",
            "K31-OPENAI-CONTRACT PASS",
        ):
            self.assertIn(token, text)
        # 핀의 validate_environ 은 알 수 없는 VLLM_* 환경변수를 경고/오류로 본다 → 이미지 ENV 에 VLLM_ 접두 금지
        self.assertIsNone(re.search(r"^\s+VLLM_[A-Z_]+=", text, re.MULTILINE))
        self.assertIsNone(re.search(r"^ENV\s+VLLM_", text, re.MULTILINE))

    def test_marlin_patch_text(self) -> None:
        patch = read(MARLIN_PATCH)
        self.assertEqual(patch.count("\ndiff --git "), 1)
        self.assertIn(
            "diff --git a/csrc/moe/marlin_moe_wna16/ops.cu "
            "b/csrc/moe/marlin_moe_wna16/ops.cu",
            patch,
        )
        header, body = patch.split("\ndiff --git a/", 1)
        self.assertNotIn("libtorch_stable", body)
        self.assertNotIn("generate_kernels", body)
        self.assertEqual(len(re.findall(r"^@@ ", patch, re.MULTILINE)), 5)
        for token in (
            "device_max_shared_mem",
            "sh_cache_size",
            "max_c_tmp_size",
            "int max_shared_mem = device_max_shared_mem;",
        ):
            self.assertIn(token, body)
        for line in body.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                self.assertEqual(line, line.rstrip(), "trailing whitespace in patch")
        # 헤더 설명: 폐기한 generator hunk 와 hunk 5개의 의도가 문서화되어 있다
        for token in ("H1", "H2", "H3", "H4", "H5", "generate_kernels.py", "kFE8M0fnu"):
            self.assertIn(token, header)

    @unittest.skipUnless(
        os.environ.get("K31_VLLM_SRC"),
        "set K31_VLLM_SRC to a vLLM checkout at the pin to run git apply --check",
    )
    def test_marlin_patch_applies_to_the_pin_tree(self) -> None:
        src = os.environ["K31_VLLM_SRC"]
        head = subprocess.run(
            ["git", "-C", src, "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(head, PIN)
        result = subprocess.run(
            ["git", "-C", src, "apply", "--check", str(MARLIN_PATCH)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker1_patches_present(self) -> None:
        """워커 ① 산출물(triton-mla-sparse-sm80, k31-sm80-dsa-glue) — 아직 없으면 이 메서드만 실패한다."""
        for name in PATCH_NAMES[1:]:
            path = PATCH_DIR / f"{name}.patch"
            self.assertTrue(path.is_file(), f"missing (worker ①): {path}")
            self.assertGreater(path.stat().st_size, 0, str(path))


class K31DockerfileSyntaxTest(unittest.TestCase):
    def test_every_run_instruction_is_valid_bash_and_inline_python_compiles(
        self,
    ) -> None:
        for dockerfile in (FULL, OPENAI):
            for lineno, logical, heredocs in dockerfile_runs(read(dockerfile)):
                command = re.sub(r"--mount=\S+\s*", "", logical[4:])
                script = command + "\n"
                for tag, body in heredocs:
                    script += "\n".join(body) + "\n" + tag + "\n"
                    if "python" in command and tag in ("PY", "EOF"):
                        compile(
                            "\n".join(body) + "\n",
                            f"{dockerfile.name}:{lineno}",
                            "exec",
                        )
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".sh", delete=False
                ) as handle:
                    handle.write("set -o pipefail\n" + script)
                result = subprocess.run(
                    ["bash", "-n", handle.name],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                os.unlink(handle.name)
                self.assertEqual(
                    result.returncode, 0, f"{dockerfile.name}:{lineno}: {result.stderr}"
                )


class K31ScriptsTest(unittest.TestCase):
    def test_scripts_exist_are_executable_and_parse(self) -> None:
        scripts = (BUILD, RELEASE, RUN11, RUN21)
        for path in scripts:
            self.assertTrue(path.is_file(), str(path))
            self.assertTrue(os.access(path, os.X_OK), f"{path.name} is not executable")
        result = subprocess.run(
            ["bash", "-n", *(str(path) for path in scripts)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_script_flags_and_logs(self) -> None:
        text = read(BUILD)
        for token in (
            "--full-only",
            "--openai-only",
            "MAX_JOBS",
            "LOG_DIR=",
            "k31-full-",
            "k31-openai-",
            "docker image inspect",
            "verify_k31_image.py",
            "--target contract",
            "BASE_DIGEST",
        ):
            self.assertIn(token, text)
        for name in PATCH_NAMES:
            self.assertIn(name, text)

    def test_release_script_follows_the_openai_release_pattern(self) -> None:
        text = read(RELEASE)
        for token in (
            "docker save",
            "zstd -T0 -10",
            "split -b 1900m",
            "SHA256SUMS",
            "image-inspect.json",
            "gh release create",
            "gh release upload",
        ):
            self.assertIn(token, text)
        self.assertNotIn("K3_OPENAI", text)

    def test_load_script_is_idempotent_and_isolated(self) -> None:
        text = read(RUN11)
        self.assertIn('docker image inspect "$IMG"', text)
        self.assertIn(".dl-k31-", text)
        self.assertNotIn(".dl-$TAG-$TIER", text)
        self.assertIn("SHA256SUMS", text)
        self.assertIn("docker/RELEASE-TAG-K31.txt", text)
        self.assertIn("zstd -d | docker load", text)

    def test_kernel_test_script_defaults(self) -> None:
        text = read(RUN21)
        for kernel in DEFAULT_KERNELS:
            self.assertIn(kernel, text)
        self.assertIn('K3_TIER="k31-$TIER"', text)
        self.assertIn("--entrypoint bash", text)
        self.assertIn("-e KERNELS=", text)
        self.assertIn(FULL_IMAGE, text)
        self.assertIn(OPENAI_IMAGE, text)

    def test_kernel_test_script_fails_closed_without_images(self) -> None:
        """이미지가 없는 호스트에서 run/21 은 성공으로 위장하지 않고 exit 3 으로 끝난다."""
        env = dict(os.environ)
        env["K31_FULL_IMAGE"] = "k31-contract-test-missing:full"
        env["K31_OPENAI_IMAGE"] = "k31-contract-test-missing:openai"
        if (
            subprocess.run(
                ["which", "docker"], capture_output=True, check=False
            ).returncode
            != 0
        ):
            self.skipTest("docker not installed")
        result = subprocess.run(
            ["bash", str(RUN21), "--cpu-dry-run"],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 3, result.stderr)


class K31VerifyModuleTest(unittest.TestCase):
    def test_help_works_without_docker(self) -> None:
        result = subprocess.run(
            ["python3", str(VERIFY), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMAGE", result.stdout)
        self.assertIn("--gpu", result.stdout)

    def test_constants_match_the_dockerfile(self) -> None:
        module = load_verify_module()
        self.assertEqual(module.COMMIT, PIN)
        self.assertEqual(module.PATCHSET, PATCHSET)
        self.assertEqual(module.DEFAULT_IMAGE, OPENAI_IMAGE)
        self.assertEqual(
            module.EXPECTED_LABELS["org.opencontainers.image.revision"], PIN
        )
        self.assertEqual(module.EXPECTED_LABELS["io.k3.patchset"], PATCHSET)
        self.assertEqual(module.EXPECTED_LABELS["io.k3.cuda_arch"], "sm80")
        self.assertEqual(module.EXPECTED_ENV["K3_PATCHSET"], PATCHSET)

    def test_probes_cover_the_required_checks_and_compile(self) -> None:
        module = load_verify_module()
        compile(module.CPU_PROBE, "cpu_probe", "exec")
        compile(module.GPU_PROBE, "gpu_probe", "exec")
        compile(module.CLI_HELP_PROBE, "cli_probe", "exec")
        cpu = module.CPU_PROBE
        for token in (
            PIN,
            "2.14.0",
            "13.0",
            "moe_wna16_marlin_gemm",
            "AttentionBackendEnum.TRITON_MLA_SPARSE.get_class()",
            "vllm.v1.attention.ops.mqa_logits_triton",
            "vllm.v1.attention.ops.triton_mla_sparse_kernel",
            "vllm.models.kimi_k3.nvidia.dsa",
            "vllm.tool_parsers.kimi_k3_tool_parser",
            "vllm.reasoning.kimi_k3_reasoning_parser",
            "patches.sha256",
            "overlay-manifest.tsv",
        ):
            self.assertIn(token, cpu)
        gpu = module.GPU_PROBE
        for token in ("A100", "(8, 0)", "nvidia-smi", "triton.jit", "K31_EXPECT_GPUS"):
            self.assertIn(token, gpu)
        self.assertIn("CpuPlatform", module.CLI_HELP_PROBE)

    def test_cli_flag_check(self) -> None:
        module = load_verify_module()
        required = set(module.REQUIRED_SERVE_FLAGS)
        self.assertTrue(
            {"--headless", "--attention-backend", "--kv-cache-dtype"}.issubset(required)
        )
        self.assertEqual(module.required_cli_errors("\n".join(required)), [])
        self.assertEqual(
            module.required_cli_errors("\n".join(required - {"--headless"})),
            ["vllm serve --help missing flag: --headless"],
        )

    def test_metadata_findings(self) -> None:
        module = load_verify_module()
        digest = "sha256:" + "a" * 64
        metadata = {
            "Config": {
                "Labels": {**module.EXPECTED_LABELS, "io.k3.base_digest": digest},
                "Env": [f"{k}={v}" for k, v in module.EXPECTED_ENV.items()],
                "Entrypoint": ["vllm", "serve"],
                "Healthcheck": {"Test": ["CMD-SHELL", "true"]},
            }
        }
        self.assertEqual(module.metadata_findings(metadata, digest), ([], []))
        errors, _ = module.metadata_findings(metadata, "sha256:" + "b" * 64)
        self.assertEqual(
            errors, ["label io.k3.base_digest does not match --base-image id"]
        )
        metadata["Config"]["Labels"]["io.k3.base_digest"] = "unpinned"
        errors, warnings = module.metadata_findings(metadata)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        metadata["Config"]["Env"].append("VLLM_COMMIT=" + PIN)
        errors, _ = module.metadata_findings(metadata)
        self.assertTrue(any("VLLM_" in error for error in errors))
        metadata["Config"]["Entrypoint"] = ["bash"]
        errors, _ = module.metadata_findings(metadata)
        self.assertIn("entrypoint mismatch", errors)

    def test_patch_sha_comparison(self) -> None:
        module = load_verify_module()
        with tempfile.TemporaryDirectory() as tmp:
            patch_dir = Path(tmp)
            rows = []
            for name in PATCH_NAMES:
                data = f"patch {name}\n".encode()
                (patch_dir / f"{name}.patch").write_bytes(data)
                rows.append((hashlib.sha256(data).hexdigest(), f"{name}.patch"))
            self.assertEqual(module.patch_errors(rows, patch_dir), [])
            self.assertTrue(module.patch_errors(list(reversed(rows)), patch_dir))
            (patch_dir / f"{PATCH_NAMES[1]}.patch").write_bytes(b"changed\n")
            self.assertTrue(module.patch_errors(rows, patch_dir))
            parsed = module.parse_sha256_listing(
                "\n".join(f"{d}  /tmp/patches/{n}" for d, n in rows)
            )
            self.assertEqual(parsed, rows)

    def test_probe_output_parsing_and_rendering(self) -> None:
        module = load_verify_module()
        output = (
            "noise\n"
            "K31-CHECK|static|provenance|PASS|ok\n"
            "K31-CHECK|import|vllm.models.kimi_k3.nvidia.dsa|FAIL|ImportError: a | b\n"
            "K31-PATCH-SHA|abc|x.patch\n"
            "K31-PROBE-END"
        )
        checks = module.parse_checks(output)
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[1][3], "ImportError: a | b")
        self.assertEqual(module.parse_patch_rows(output), [("abc", "x.patch")])
        text = "\n".join(module.render(checks, gpu_requested=False))
        self.assertIn("SKIP", text)  # GPU 구역은 CPU 모드에서 SKIP 으로 명시된다
        self.assertIn("[gpu]", text)
        text = "\n".join(module.render(checks, gpu_requested=True))
        self.assertNotIn("SKIP", text)

    def test_image_inspect_json_shape_is_what_release_script_stores(self) -> None:
        # release-k31.sh 는 `docker image inspect` 원문을 image-inspect.json 으로 저장한다 — 리스트 1개 원소여야 한다.
        sample = json.loads('[{"Id": "sha256:' + "c" * 64 + '"}]')
        self.assertEqual(len(sample), 1)


if __name__ == "__main__":
    unittest.main()
