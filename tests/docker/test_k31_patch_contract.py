"""K3.1 vLLM 패치 세트(`docker/patches-k31/`)의 정적 계약 테스트.

호스트 python만으로 돈다(torch·vllm·GPU 없음): 패치 텍스트의 대상 파일 집합, 금지
경로(csrc 등), 핵심 토큰, 새 파일의 문법·클래스 구조를 확인한다. 핀에 실제로
적용되는지(`git apply --check`)와 GPU 수치는 이 테스트의 범위 밖이다 -- 적용 순서
계약은 marlin(csrc) -> triton-mla-sparse-sm80 -> k31-sm80-dsa-glue 이다.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO_ROOT / "docker" / "patches-k31"
TRITON_PATCH = PATCH_DIR / "triton-mla-sparse-sm80.patch"
GLUE_PATCH = PATCH_DIR / "k31-sm80-dsa-glue.patch"
MARLIN_PATCH = PATCH_DIR / "mxfp4-marlin-moe-sm80-main.patch"

PROVENANCE = (
    "# rebased from vllm-project/vllm PR #47629 @ 064801dd (origin #38476 haosdent)"
)

TRITON_FILES = {
    "tests/kernels/attention/test_mqa_logits_triton.py",
    "tests/kernels/attention/test_triton_mla_sparse_kernel.py",
    "tests/v1/attention/test_triton_mla_sparse_backend.py",
    "vllm/model_executor/layers/sparse_attn_indexer.py",
    "vllm/model_executor/models/deepseek_v2.py",
    "vllm/platforms/cuda.py",
    "vllm/v1/attention/backends/mla/indexer.py",
    "vllm/v1/attention/backends/mla/triton_mla_sparse.py",
    "vllm/v1/attention/backends/registry.py",
    "vllm/v1/attention/ops/mqa_logits_triton.py",
    "vllm/v1/attention/ops/triton_mla_sparse_kernel.py",
}
GLUE_FILES = {
    "tests/models/kimi_k3/test_dsa_sm80.py",
    "tests/models/kimi_k3/test_dsa_sm80_indexer.py",
    "vllm/envs.py",
    "vllm/models/kimi_k3/nvidia/dsa.py",
}
BACKEND = "vllm/v1/attention/backends/mla/triton_mla_sparse.py"
LOGITS_KERNEL = "vllm/v1/attention/ops/mqa_logits_triton.py"
SPARSE_KERNEL = "vllm/v1/attention/ops/triton_mla_sparse_kernel.py"
SPARSE_INDEXER = "vllm/model_executor/layers/sparse_attn_indexer.py"


class FilePatch:
    """`diff --git` 한 구간: 경로, 신규 여부, 추가/삭제된 줄."""

    def __init__(self, section: str) -> None:
        header, _, body = section.partition("\n")
        match = re.match(r"a/(?P<a>\S+) b/(?P<b>\S+)$", header)
        assert match, f"unexpected diff header: {header!r}"
        self.path = match["b"]
        self.is_new = body.startswith("new file mode")
        self.added: list[str] = []
        self.removed: list[str] = []
        for line in body.split("\n"):
            if line.startswith(("+++ ", "--- ")):
                continue
            if line.startswith("+"):
                self.added.append(line[1:])
            elif line.startswith("-"):
                self.removed.append(line[1:])

    @property
    def added_text(self) -> str:
        return "\n".join(self.added)

    @property
    def removed_text(self) -> str:
        return "\n".join(self.removed)


def split_preamble(path: Path) -> tuple[str, list[str]]:
    """머리글(첫 `diff --git` 앞, `git apply`가 무시)과 파일별 구간으로 나눈다."""
    text = path.read_text(encoding="utf-8")
    preamble, *sections = re.split(r"(?m)^diff --git ", text)
    return preamble, sections


def parse_patch(path: Path) -> dict[str, FilePatch]:
    _, sections = split_preamble(path)
    patches = [FilePatch(section) for section in sections]
    return {patch.path: patch for patch in patches}


def new_file_source(patch: FilePatch) -> str:
    assert patch.is_new, patch.path
    return patch.added_text + "\n"


class K31PatchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.triton = parse_patch(TRITON_PATCH)
        cls.glue = parse_patch(GLUE_PATCH)

    def test_patch_files_exist_and_are_plain_git_diffs(self) -> None:
        for path in (TRITON_PATCH, GLUE_PATCH):
            self.assertTrue(path.is_file(), path)
            raw = path.read_bytes()
            self.assertIn(b"\ndiff --git a/", raw, path.name)
            self.assertTrue(raw.endswith(b"\n"), path.name)
            self.assertNotIn(b"\r\n", raw, path.name)

    def test_preambles_document_pin_and_apply_order(self) -> None:
        pin = "a3c14efbc77c018e8bf579935095a87721349e05"
        triton, _ = split_preamble(TRITON_PATCH)
        glue, _ = split_preamble(GLUE_PATCH)
        for preamble in (triton, glue):
            self.assertIn(pin, preamble)
            self.assertIn("git apply", preamble)
            self.assertIn("mxfp4-marlin-moe-sm80-main.patch", preamble)
        self.assertIn("PR #47629", triton)
        self.assertIn("VLLM_K31_SM80_DSA", glue)

    def test_target_files_match_the_expected_sets(self) -> None:
        self.assertEqual(set(self.triton), TRITON_FILES)
        self.assertEqual(set(self.glue), GLUE_FILES)

    def test_python_only_no_csrc_no_build_files(self) -> None:
        for name, patches in (("triton", self.triton), ("glue", self.glue)):
            for path in patches:
                self.assertTrue(path.endswith(".py"), (name, path))
                for forbidden in (
                    "csrc/",
                    "cmake/",
                    "CMakeLists",
                    "docker/",
                    "deploy/",
                ):
                    self.assertFalse(path.startswith(forbidden), (name, path))
        # PR #47629의 csrc hunk(nvfp4_quant_entry.cu SM8x 링크 스텁)는 핀에서
        # 불필요하다: mxfp4_experts_quant 바인딩이 SM100 전용 소스로 옮겨졌다.
        text = TRITON_PATCH.read_text(encoding="utf-8")
        self.assertNotIn("nvfp4_quant_entry", text)
        self.assertNotIn("libtorch_stable", text)

    def test_patches_are_independent_and_disjoint_from_marlin(self) -> None:
        # 적용 순서(marlin -> triton -> glue)가 파일 충돌 없이 성립한다.
        self.assertEqual(set(self.triton) & set(self.glue), set())
        if MARLIN_PATCH.is_file():
            marlin = set(
                re.findall(
                    r"(?m)^diff --git a/(\S+) b/",
                    MARLIN_PATCH.read_text(encoding="utf-8"),
                )
            )
            self.assertEqual(marlin & (set(self.triton) | set(self.glue)), set())

    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_git_apply_understands_both_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for path, count in ((TRITON_PATCH, len(TRITON_FILES)), (GLUE_PATCH, 4)):
                result = subprocess.run(
                    ["git", "apply", "--numstat", str(path)],
                    cwd=tmp,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout.strip().splitlines()), count)

    def test_new_python_files_parse(self) -> None:
        for patches in (self.triton, self.glue):
            for path, patch in patches.items():
                if patch.is_new:
                    ast.parse(new_file_source(patch), filename=path)

    def test_backend_class_structure(self) -> None:
        tree = ast.parse(new_file_source(self.triton[BACKEND]))
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        bases = {
            name: {ast.unparse(base).split("[")[0] for base in node.bases}
            for name, node in classes.items()
        }
        self.assertEqual(bases["TritonMLASparseBackend"], {"AttentionBackend"})
        self.assertEqual(bases["TritonMLASparseMetadata"], {"SparseMLACommonMetadata"})
        self.assertEqual(
            bases["TritonMLASparseMetadataBuilder"], {"SparseMLACommonMetadataBuilder"}
        )
        self.assertEqual(bases["TritonMLASparseImpl"], {"SparseMLACommonImpl"})
        impl_methods = {
            n.name
            for n in classes["TritonMLASparseImpl"].body
            if isinstance(n, ast.FunctionDef)
        }
        # 유일한 추상 훅 + workspace 예약 훅
        self.assertTrue({"_forward_mqa_kernel", "reserve_workspace"} <= impl_methods)
        # PR의 XPUMLASparse* 상속 구조는 쓰지 않는다(템플릿 flashattn_mla_sparse 골격).
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("xpu_mla_sparse", node.module or "")
        for node in classes.values():
            self.assertFalse(
                any("XPU" in ast.unparse(base) for base in node.bases), node.name
            )

    def test_backend_contract_tokens(self) -> None:
        text = self.triton[BACKEND].added_text
        for token in (
            'return "TRITON_MLA_SPARSE"',
            "capability.major == 8",
            "supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]",
            '"auto",\n        "float16",\n        "bfloat16",',
            "return [_DIM_QK]",
            "[MultipleOf(64)]",
            "align_blocks_to_rows(spec)",
            "flat_kv_row_view(kv_cache, block_size)",
            "AttentionCGSupport.NEVER",
            "_init_reorder_batch_threshold(1, supports_spec_as_decode=True)",
            "group_conversion_decode_only: ClassVar[bool] = True",
            "mid_out_getter",
            "max_split_mid_out_numel",
            "current_workspace_manager().get_simultaneous",
            "--kv-cache-dtype bfloat16",
        ):
            self.assertIn(token, text)
        # 양자화 KV는 P3: 허용 목록에 fp8 계열이 없어야 한다.
        listing = re.search(
            r"supported_kv_cache_dtypes: ClassVar\[list\[CacheDType\]\] = \[(.*?)\]",
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(listing)
        self.assertNotIn("fp8", listing.group(1))
        self.assertIn("NotImplementedError", text)

    def test_registry_cuda_priority_and_fused_q_gate(self) -> None:
        registry = self.triton["vllm/v1/attention/backends/registry.py"].added_text
        self.assertIn("TRITON_MLA_SPARSE = (", registry)
        self.assertIn(
            "vllm.v1.attention.backends.mla.triton_mla_sparse.TritonMLASparseBackend",
            registry,
        )
        cuda = self.triton["vllm/platforms/cuda.py"].added
        # SM8x/9x sparse tail 맨 뒤 한 곳에만 추가한다(SM12x 분기에는 넣지 않는다).
        self.assertEqual(
            [
                line.strip()
                for line in cuda
                if "TRITON_MLA_SPARSE" in line and "Attention" in line
            ],
            ["AttentionBackendEnum.TRITON_MLA_SPARSE,"],
        )
        v2 = self.triton["vllm/model_executor/models/deepseek_v2.py"].added_text
        self.assertIn("current_platform.has_device_capability(89)", v2)

    def test_sparse_indexer_dispatch_tokens(self) -> None:
        patch = self.triton[SPARSE_INDEXER]
        text = patch.added_text
        for token in (
            "use_triton_logits = "
            "current_platform.is_cuda() and not is_deep_gemm_supported()",
            "fp8_mqa_logits_triton(",
            "fp8_paged_mqa_logits_triton(",
            "clean_logits=False",
            "from vllm.v1.attention.ops.mqa_logits_triton import (",
            "does not support the MXFP4 indexer cache / FP4 queries",
            "does not support DCP/PCP",
            "does not support two-level candidate selection",
        ):
            self.assertIn(token, text)
        # prefill/decode 두 분기가 XPU 분기와 같은 자리에서 갈라진다.
        self.assertEqual(text.count("elif use_triton_logits:"), 2)
        # SparseAttnIndexer.__init__의 하드 에러는 경고로 낮춘다.
        self.assertIn("    has_deep_gemm,", patch.removed)
        self.assertIn("logger.warning_once(", text)
        indexer = self.triton["vllm/v1/attention/backends/mla/indexer.py"].added_text
        self.assertIn("is_deep_gemm_supported,", indexer)
        # has_deep_gemm() 가드는 두고 arch 판정만 더한다(기존 테스트 monkeypatch 보존).
        self.assertEqual(indexer.count("and has_deep_gemm()"), 2)
        self.assertEqual(indexer.count("and is_deep_gemm_supported()"), 2)

    def test_ported_kernels_keep_provenance_and_promote_offsets_to_int64(self) -> None:
        for path in (LOGITS_KERNEL, SPARSE_KERNEL):
            patch = self.triton[path]
            self.assertTrue(patch.is_new)
            self.assertIn(PROVENANCE, patch.added)
            self.assertEqual(patch.added[0], "# SPDX-License-Identifier: Apache-2.0")
            self.assertIn("tl.int64", patch.added_text)
        sparse = self.triton[SPARSE_KERNEL].added_text
        # packed hybrid KV row id * 576 이 2**31을 넘는 구간의 int64 승격
        self.assertIn("indices = indices.to(tl.int64)", sparse)
        self.assertEqual(sparse.count("cur_q = tl.program_id(0).to(tl.int64)"), 3)
        self.assertIn("def max_split_mid_out_numel(", sparse)
        logits = self.triton[LOGITS_KERNEL].added_text
        self.assertIn("token_id = tl.program_id(0).to(tl.int64)", logits)
        self.assertIn("m = tl.program_id(0).to(tl.int64)", logits)
        self.assertIn("n_start = n_block.to(tl.int64) * BLOCK_N", logits)

    def test_tests_cover_backend_int64_and_workspace(self) -> None:
        kernel_test = self.triton[
            "tests/kernels/attention/test_triton_mla_sparse_kernel.py"
        ].added_text
        self.assertIn("def test_row_ids_beyond_int32_offset", kernel_test)
        self.assertIn("2**31", kernel_test)
        backend_test = self.triton[
            "tests/v1/attention/test_triton_mla_sparse_backend.py"
        ].added_text
        for token in (
            "supports_compute_capability",
            "lock_workspace()",
            "test_workspace_bound_covers_every_split_the_kernel_can_request",
            "test_sm80_auto_selection_lands_on_triton_mla_sparse",
        ):
            self.assertIn(token, backend_test)

    def test_glue_switches_off_the_fp8e4nv_paths_on_sm80(self) -> None:
        dsa = self.glue["vllm/models/kimi_k3/nvidia/dsa.py"]
        text = dsa.added_text
        for token in (
            "def use_sm80_dsa() -> bool:",
            "envs.VLLM_K31_SM80_DSA",
            "self.use_sm80_dsa = use_sm80_dsa()",
            "and not self.use_sm80_dsa",
            "per_token_group_quant_fp8",
            "use_ue8m0=True",
            "elif self.use_sm80_dsa:",
            "def _sm80_q_quant_fold_weights(",
            "assert not indexer.use_sm80_dsa",
            "MXFP4 indexer keys (indexer_kv_dtype='mxfp4') need SM10x",
            "does not support DCP/PCP",
        ):
            self.assertIn(token, text)
        self.assertRegex(text, r"has_device_capability\(\s*89\s*\)")
        # fused 전방 경로와 Triton fp8e4nv 커널 정의는 건드리지 않는다.
        self.assertNotIn("ops/dsa_indexer.py", "\n".join(self.glue))
        self.assertNotIn("def dsa_fused_prologue", text)

    def test_glue_registers_the_override_env_with_vllm_envs(self) -> None:
        envs = self.glue["vllm/envs.py"].added_text
        self.assertIn("VLLM_K31_SM80_DSA: bool | None = None", envs)
        self.assertIn('"VLLM_K31_SM80_DSA": lambda: {"1": True, "0": False}.get(', envs)

    def test_glue_tests_cover_env_override_and_indexer_path(self) -> None:
        unit = self.glue["tests/models/kimi_k3/test_dsa_sm80.py"].added_text
        self.assertIn("test_env_forces_on_and_off", unit)
        self.assertIn("test_env_is_registered_with_vllm_envs", unit)
        e2e = self.glue["tests/models/kimi_k3/test_dsa_sm80_indexer.py"].added_text
        for token in (
            "test_decode_topk_matches_reference",
            "test_prefill_topk_matches_reference",
            "use_fused_k_insert=False",
            "ROW_STRIDE = 2**21",
        ):
            self.assertIn(token, e2e)


if __name__ == "__main__":
    unittest.main()
