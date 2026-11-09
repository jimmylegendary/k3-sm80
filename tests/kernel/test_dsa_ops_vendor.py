"""vendor/dsa_ops == 이미지에 실리는 SM80 DSA 커널 — 계약 테스트 (kernel-kit 통합 결함 B 의 재발 방지).

결함: k11/k12 가 PR #47629 원본(head 064801dd)을 판정하고 있었다. 이미지에 실리는 커널은 그 PR 을 핀 a3c14efb 에 rebase 한
`docker/patches-k31/triton-mla-sparse-sm80.patch` 의 신규 파일 2개이고, PR 원본 대비 ① int64 승격(packed hybrid KV row 의
`row*576` 오프셋 — 물리 row id ≥ 3,728,270 에서 PR 원본은 int32 wrap 으로 잘못된 주소를 읽는다) ② split-KV 스크래치 workspace
주입(`mid_out_getter` / `max_split_mid_out_numel`)을 더했다. 이 테스트가 고정하는 것:

  1. 헤더가 핀·패치·sha256·원천(PR/원저자)을 정확히 적는다.
  2. 두 파일에 int64 승격 마커와 workspace 심볼이 있다 (마커 검사기 자체가 PR 원본·변형본을 잡는지 음성 대조로 확인).
  3. 본문 == 패치 신규 파일 본문 (재배선한 `from vllm.*` 줄만 다르다) — 패치 파일 기준으로 항상, `K31_VLLM_SRC`(패치 적용 vLLM 트리)가
     주어지면 그 트리의 파일과도 비교. 재생성 도구(vendor/revendor_dsa_ops.py)가 디스크 사본을 바이트 단위로 재현한다.
  4. workspace 예약식(`max_split_mid_out_numel`)이 커널의 실제 요청을 덮고 빈틈이 없다.
  5. (torch+triton, CPU 인터프리터) 음성 대조: int64 승격을 뺀 변종·PR 원본은 packed-row 실기하(row id ≥ 3.7M)에서 SIGSEGV 하거나
     오답이고, 같은 코드 경로의 작은 기하(대조군)에서는 통과한다 → 배포 패치본만 통과. 스크래치를 안 쓰는 변종은 더러운 workspace
     (NaN 오염)에서 NaN 으로 드러난다.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT

REPO_ROOT = KIT_ROOT.parent
VENDOR = KIT_ROOT / "vendor"
DSA = VENDOR / "dsa_ops"
PATCH_REL = "docker/patches-k31/triton-mla-sparse-sm80.patch"
PATCH = REPO_ROOT / PATCH_REL
PIN = "a3c14efbc77c018e8bf579935095a87721349e05"
PR_HEAD = "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
PIN_CLONE = Path("/home/jimmy/repos/vllm-pin-k31")
KERNELS = {  # 패치의 신규 파일 경로 -> vendor/dsa_ops 안의 파일명
    "vllm/v1/attention/ops/mqa_logits_triton.py": "mqa_logits_triton.py",
    "vllm/v1/attention/ops/triton_mla_sparse_kernel.py": "triton_mla_sparse_kernel.py",
}
# 재배선 역방향: vendored 상대 import 접두 -> 원본 vllm import 접두 (정방향 규칙은 vendor/STUBS.md §4)
INVERSE_REWIRE = {
    "from .._vllm_shims.triton_utils import": "from vllm.triton_utils import",
    "from .._vllm_shims.platform_utils import": "from vllm.utils.platform_utils import",
}
HEADER_LINES = 3
HEADER1 = re.compile(
    r"^# vendored from vllm-project/vllm@(?P<pin>[0-9a-f]{40}) \+ k3-sm80 "
    r"(?P<patch>docker/patches-k31/triton-mla-sparse-sm80\.patch) \(sha256 (?P<sha>[0-9a-f]{64})\) "
    r"(?P<orig>vllm/v1/attention/ops/\S+\.py) — Apache-2\.0$")

# 마커 규칙: 파일명 -> {정확히 이 줄(strip 후)이 이 횟수만큼 있어야 한다}. 패치가 PR 원본에 더한 int64 승격 + (sparse) workspace 주입.
MARKERS = {
    "triton_mla_sparse_kernel.py": {
        "indices = indices.to(tl.int64)": 1,
        "cur_q = tl.program_id(0).to(tl.int64)": 3,
        "mid_out = mid_out_getter(mid_shape)": 1,
        "def max_split_mid_out_numel(num_heads_q: int, index_topk: int, sm_count: int) -> int:": 1,
        "mid_out_getter: Callable[[tuple[int, ...]], torch.Tensor] | None = None,": 1,
    },
    "mqa_logits_triton.py": {
        "token_id = tl.program_id(0).to(tl.int64)": 1,
        "m = tl.program_id(0).to(tl.int64)": 1,
        "n_start = n_block.to(tl.int64) * BLOCK_N": 1,
    },
}
INT64_FLOOR = {"triton_mla_sparse_kernel.py": 4, "mqa_logits_triton.py": 4}   # `to(tl.int64)` 개수 하한 (패치본 실측)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def has_torch_and_triton() -> bool:
    return torch is not None and importlib.util.find_spec("triton") is not None


def header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[:HEADER_LINES]


def body(path: Path) -> str:
    """헤더 3줄을 뗀 본문 (원본의 첫 줄부터)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return "\n".join(lines[HEADER_LINES:])


def unwire(text: str) -> str:
    """vendored 본문의 상대 import 줄을 원본 `from vllm.*` 줄로 되돌린다 (재배선 외 변환은 하지 않는다)."""
    out = []
    for line in text.split("\n"):
        for rewired, original in INVERSE_REWIRE.items():
            if line.startswith(rewired):
                line = original + line[len(rewired):]
                break
        out.append(line)
    return "\n".join(out)


def patch_bodies(patch: Path) -> dict[str, str]:
    """패치의 신규 파일 본문 복원 — revendor_dsa_ops.py 와 별개로 구현한 줄 스캐너 (두 구현이 일치해야 한다)."""
    bodies: dict[str, list[str]] = {}
    current = None
    in_hunk = False
    for line in patch.read_text(encoding="utf-8").split("\n"):
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(\S+) b/(\S+)$", line)
            current = match.group(2) if match and match.group(2) in KERNELS else None
            in_hunk = False
            if current:
                bodies[current] = []
        elif current is not None:
            if line.startswith("@@ "):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                bodies[current].append(line[1:])
    return {path: "\n".join(lines) + "\n" for path, lines in bodies.items()}


def marker_problems(name: str, text: str) -> list[str]:
    """MARKERS/INT64_FLOOR 규칙 위반 목록 (빈 리스트 = 배포 패치본과 같은 승격·주입을 갖춤)."""
    problems = []
    stripped = [ln.strip() for ln in text.split("\n")]
    for marker, expected in MARKERS[name].items():
        got = stripped.count(marker)
        if got != expected:
            problems.append(f"{name}: {marker!r} x{got} (기대 {expected})")
    n64 = text.count("to(tl.int64)")
    if n64 < INT64_FLOOR[name]:
        problems.append(f"{name}: to(tl.int64) {n64}개 < 하한 {INT64_FLOOR[name]}")
    return problems


class DsaOpsHeaderTest(unittest.TestCase):
    def test_headers_state_pin_patch_sha_and_origin(self) -> None:
        for name in KERNELS.values():
            lines = header(DSA / name)
            match = HEADER1.match(lines[0])
            self.assertIsNotNone(match, f"{name}: 출처 헤더 1행 형식 위반: {lines[0]!r}")
            self.assertEqual(match["pin"], PIN, name)
            self.assertIn(match["orig"], KERNELS, name)
            self.assertEqual(KERNELS[match["orig"]], name)
            for token in ("PR #47629", PR_HEAD, "#38476", "haosdent/thomaslwang"):
                self.assertIn(token, lines[1], f"{name}: 2행에 {token!r} 없음")
            self.assertIn("본문 무수정", lines[2], name)
            self.assertIn("Apache-2.0", lines[2], name)
            # 원본 SPDX 헤더는 헤더 3줄 바로 아래에 그대로 남아 있어야 한다
            tail = (DSA / name).read_text(encoding="utf-8").splitlines()[HEADER_LINES:HEADER_LINES + 2]
            self.assertEqual(tail, ["# SPDX-License-Identifier: Apache-2.0",
                                    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"])

    @unittest.skipUnless(PATCH.is_file(), "docker/patches-k31 패치 파일 필요")
    def test_header_sha256_is_the_sha256_of_the_patch_file(self) -> None:
        actual = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        for name in KERNELS.values():
            self.assertEqual(HEADER1.match(header(DSA / name)[0])["sha"], actual,
                             f"{name}: 패치가 바뀌었다 — `python3 kernel-kit/vendor/revendor_dsa_ops.py --write` 로 재vendor 하고 "
                             "vendor/__init__.py DSA_OPS_PATCH_SHA256 · vendor/STUBS.md §4 의 sha 도 갱신")

    def test_vendor_package_constants_agree_with_the_headers(self) -> None:
        tree = ast.parse((VENDOR / "__init__.py").read_text(encoding="utf-8"))
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual(consts["VENDOR_PIN"], PIN)
        self.assertEqual(consts["DSA_OPS_PIN"], PR_HEAD)               # 패치의 원천(PR head)
        self.assertEqual(consts["DSA_OPS_PATCH"], PATCH_REL)
        for name in KERNELS.values():
            self.assertEqual(HEADER1.match(header(DSA / name)[0])["sha"], consts["DSA_OPS_PATCH_SHA256"], name)


class DsaOpsMarkerTest(unittest.TestCase):
    def test_vendored_files_carry_the_int64_promotions_and_workspace_injection(self) -> None:
        for name in KERNELS.values():
            self.assertEqual(marker_problems(name, (DSA / name).read_text(encoding="utf-8")), [], name)

    @unittest.skipUnless(PATCH.is_file(), "docker/patches-k31 패치 파일 필요")
    def test_int64_marker_counts_are_at_least_those_of_the_patch(self) -> None:
        for path, text in patch_bodies(PATCH).items():
            name = KERNELS[path]
            vendored = (DSA / name).read_text(encoding="utf-8").count("to(tl.int64)")
            self.assertGreaterEqual(vendored, text.count("to(tl.int64)"), name)
            self.assertGreaterEqual(text.count("to(tl.int64)"), INT64_FLOOR[name], f"{name}: 패치의 마커가 하한보다 적다")

    def test_marker_checker_catches_the_pr_original_shape_and_stripped_variants(self) -> None:
        sparse = (DSA / "triton_mla_sparse_kernel.py").read_text(encoding="utf-8")
        stripped = sparse.replace("indices = indices.to(tl.int64)", "pass")
        self.assertTrue(any("indices.to(tl.int64)" in p for p in marker_problems("triton_mla_sparse_kernel.py", stripped)))
        no_cur_q = sparse.replace("tl.program_id(0).to(tl.int64)", "tl.program_id(0)")
        self.assertTrue(marker_problems("triton_mla_sparse_kernel.py", no_cur_q))
        no_getter = sparse.replace("mid_out_getter(mid_shape)", "None")
        self.assertTrue(marker_problems("triton_mla_sparse_kernel.py", no_getter))
        logits = (DSA / "mqa_logits_triton.py").read_text(encoding="utf-8")
        self.assertTrue(marker_problems("mqa_logits_triton.py", logits.replace("n_block.to(tl.int64) * BLOCK_N", "n_block * BLOCK_N")))

    @unittest.skipUnless((PIN_CLONE / ".git").exists(), "핀 클론 필요")
    def test_marker_checker_rejects_the_actual_pr_original(self) -> None:
        for path, name in KERNELS.items():
            shown = subprocess.run(["git", "-C", str(PIN_CLONE), "show", f"pr-47629:{path}"], capture_output=True, text=True)
            if shown.returncode != 0:
                self.skipTest("핀 클론에 pr-47629 ref 없음 (`git fetch origin pull/47629/head:pr-47629`)")
            self.assertTrue(marker_problems(name, shown.stdout), f"{name}: PR 원본이 마커 검사를 통과 — 검사가 공회전")
            self.assertLess(shown.stdout.count("to(tl.int64)"), INT64_FLOOR[name], name)

    def test_symbols_are_exposed_with_the_backend_call_shape(self) -> None:
        tree = ast.parse((DSA / "triton_mla_sparse_kernel.py").read_text(encoding="utf-8"))
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        attn = funcs["triton_mla_sparse_attention"].args
        names = [a.arg for a in attn.args]
        self.assertEqual(names, ["q", "kv", "indices", "sm_scale", "num_kv_splits", "sm_count", "mid_out_getter"])
        self.assertIsInstance(attn.defaults[-1], ast.Constant)
        self.assertIsNone(attn.defaults[-1].value)                        # mid_out_getter 기본 None = 호출마다 할당
        self.assertEqual([a.arg for a in funcs["max_split_mid_out_numel"].args.args],
                         ["num_heads_q", "index_topk", "sm_count"])
        init = ast.parse((DSA / "__init__.py").read_text(encoding="utf-8"))
        exports = next(ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in n.targets))
        self.assertEqual(exports["max_split_mid_out_numel"], ".triton_mla_sparse_kernel")
        self.assertEqual(exports["triton_mla_sparse_attention"], ".triton_mla_sparse_kernel")

    def test_no_vllm_import_gate_is_green(self) -> None:
        spec = importlib.util.spec_from_file_location("check_no_vllm", VENDOR / "check_no_vllm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.main(["--quiet"]), 0)
        for name in KERNELS.values():
            for node in ast.walk(ast.parse((DSA / name).read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotEqual((node.module or "").split(".")[0], "vllm", name)


@unittest.skipUnless(PATCH.is_file(), "docker/patches-k31 패치 파일 필요")
class DsaOpsBodyEqualsPatchTest(unittest.TestCase):
    """'재배선 외 본문 수정 0' 증명 — 패치 신규 파일 본문과 줄 단위로 같다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bodies = patch_bodies(PATCH)

    def test_patch_has_both_kernel_files(self) -> None:
        self.assertEqual(set(self.bodies), set(KERNELS))

    def test_body_equals_patch_body_modulo_import_rewiring(self) -> None:
        for path, name in KERNELS.items():
            vendored = unwire(body(DSA / name))
            self.assertEqual(vendored, self.bodies[path], f"{name}: 패치 본문과 다르다 (재배선 줄 외 변경 또는 패치 갱신)")

    def test_only_the_expected_import_lines_were_rewired(self) -> None:
        expected_rewired = {"mqa_logits_triton.py": 1, "triton_mla_sparse_kernel.py": 2}
        for path, name in KERNELS.items():
            vendored_lines = body(DSA / name).split("\n")
            patch_lines = self.bodies[path].split("\n")
            changed = [(a, b) for a, b in zip(patch_lines, vendored_lines) if a != b]
            self.assertEqual(len(patch_lines), len(vendored_lines), name)       # 줄 수 불변 (삽입/삭제 없음)
            self.assertEqual(len(changed), expected_rewired[name], f"{name}: {changed}")
            for original, rewired in changed:
                self.assertTrue(original.startswith("from vllm."), original)
                self.assertTrue(rewired.startswith("from .._vllm_shims."), rewired)
            self.assertEqual(sum(1 for ln in patch_lines if ln.startswith("from vllm.")), expected_rewired[name], name)
            self.assertFalse([ln for ln in vendored_lines if "vllm." in ln and ln.lstrip().startswith(("from", "import"))], name)

    def test_regeneration_tool_reproduces_the_vendored_files_byte_for_byte(self) -> None:
        spec = importlib.util.spec_from_file_location("revendor_dsa_ops", VENDOR / "revendor_dsa_ops.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        rendered = tool.render(PATCH)
        self.assertEqual(set(rendered), set(KERNELS.values()))
        for name, content in rendered.items():
            self.assertEqual((DSA / name).read_text(encoding="utf-8"), content, f"{name}: 재생성 결과와 다름")
        # 도구의 패치 파서와 이 테스트의 줄 스캐너가 같은 본문을 복원한다 (독립 구현 교차 검증)
        tool_bodies = tool.new_file_bodies(PATCH.read_text(encoding="utf-8"))
        self.assertEqual(tool_bodies, self.bodies)

    def test_revendor_check_cli_is_green(self) -> None:
        res = subprocess.run([sys.executable, str(VENDOR / "revendor_dsa_ops.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_revendor_check_detects_a_tampered_copy(self) -> None:
        """도구 CLI 로 직접: 임시 디렉터리에 재생성 -> 일치(rc 0) -> int64 승격 한 줄만 바꾸면 불일치(rc 1)이고 마커 검사기도 잡는다."""
        tool = [sys.executable, str(VENDOR / "revendor_dsa_ops.py")]
        name = "triton_mla_sparse_kernel.py"
        with tempfile.TemporaryDirectory() as tmp:
            written = subprocess.run([*tool, "--write", "--dst", tmp], capture_output=True, text=True)
            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertEqual((Path(tmp) / name).read_text(encoding="utf-8"), (DSA / name).read_text(encoding="utf-8"))
            self.assertEqual(subprocess.run([*tool, "--check", "--dst", tmp], capture_output=True, text=True).returncode, 0)
            target = Path(tmp) / name
            tampered = target.read_text(encoding="utf-8").replace("indices = indices.to(tl.int64)", "indices = indices")
            target.write_text(tampered, encoding="utf-8")
            checked = subprocess.run([*tool, "--check", "--dst", tmp], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 1, checked.stdout + checked.stderr)
            self.assertIn(name, checked.stderr)
            self.assertTrue(marker_problems(name, tampered))                 # 마커 검사기도 같은 변조를 잡는다
            body_path = next(p for p, n in KERNELS.items() if n == name)
            self.assertNotEqual(unwire("\n".join(tampered.split("\n")[HEADER_LINES:])), self.bodies[body_path])

    def test_revendor_check_reports_a_missing_patch_as_tool_error(self) -> None:
        res = subprocess.run([sys.executable, str(VENDOR / "revendor_dsa_ops.py"), "--check", "--patch", "/nonexistent.patch"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 3, res.stdout + res.stderr)


class DsaOpsAppliedTreeTest(unittest.TestCase):
    """K31_VLLM_SRC = 패치가 적용된 vLLM 트리(예: ~/repos/vllm-pin-k31-wt-lead) — 실제 적용 결과와 vendored 사본이 같은가."""

    @unittest.skipUnless(os.environ.get("K31_VLLM_SRC"), "K31_VLLM_SRC(패치 적용 vLLM 트리) 미지정 — skip")
    def test_body_equals_the_applied_tree_files_modulo_import_rewiring(self) -> None:
        tree = Path(os.environ["K31_VLLM_SRC"])
        for path, name in KERNELS.items():
            source = tree / path
            self.assertTrue(source.is_file(), f"{source} 없음 — K31_VLLM_SRC 가 패치 적용 트리인가")
            self.assertEqual(unwire(body(DSA / name)), source.read_text(encoding="utf-8"),
                             f"{name}: 패치 적용 트리의 {path} 와 다름")


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class DsaOpsWorkspaceBoundTest(unittest.TestCase):
    """예약식 `max_split_mid_out_numel` 이 `triton_mla_sparse_attention` 의 실제 스크래치 요청을 덮고(안전) 빈틈이 없다(타이트).

    잠근 workspace 는 증설이 안 되므로 상한이 토큰 수 하나라도 모자라면 첫 decode 에서 죽는다 — 패치의 백엔드 테스트
    (test_workspace_bound_covers_every_split_the_kernel_can_request)와 같은 불변식을 vendored 사본으로 다시 확인한다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if str(KIT_ROOT) not in sys.path:
            sys.path.insert(0, str(KIT_ROOT))
        import vendor.dsa_ops.triton_mla_sparse_kernel as kernel
        cls.k = kernel

    def test_bound_covers_every_split_and_is_tight(self) -> None:
        k = self.k
        for heads in (4, 12, 16, 24, 96):
            for sm_count in (84, 108, 132):
                for topk in (256, 2048):
                    bound = k.max_split_mid_out_numel(heads, topk, sm_count)
                    groups = -(-heads // min(k._BLOCK_H, heads))
                    biggest = 0
                    for tokens in range(1, 4 * sm_count):
                        splits = k._choose_num_kv_splits(tokens, groups, topk, sm_count)
                        if splits > 1:
                            need = tokens * heads * splits * (k._BLOCK_DV + 1)
                            self.assertLessEqual(need, bound, (heads, sm_count, topk, tokens))
                            biggest = max(biggest, need)
                        else:
                            self.assertTrue(tokens * groups * k._SPLIT_MAX_OCCUPANCY >= sm_count or topk // k._MIN_TOPK_PER_SPLIT < 2,
                                            (heads, sm_count, topk, tokens))
                    self.assertEqual(bound, biggest, (heads, sm_count, topk))

    def test_k31_tp8_reservation_is_a_few_mib(self) -> None:
        bound = self.k.max_split_mid_out_numel(12, 2048, 108)
        self.assertEqual(bound, 640_224)                  # 12 heads · topk 2048 · A100(108 SM)
        self.assertLess(bound * 4, 4 * 2**20)

    def test_wrapper_requests_stay_within_the_bound(self) -> None:
        """실제 래퍼를 부르되 getter 에서 중단해(커널 미실행) 요청 shape 를 수집 -> 예약 상한 이내."""
        k = self.k

        class Stop(Exception):
            pass

        heads, topk, sm_count = 12, 2048, 108
        bound = k.max_split_mid_out_numel(heads, topk, sm_count)
        kv = torch.zeros(8, 1, 576, dtype=torch.bfloat16)
        seen = []
        for tokens in range(1, sm_count + 1):
            if k._choose_num_kv_splits(tokens, 1, topk, sm_count) == 1:
                continue                                      # single-pass: 스크래치 요청 없음(커널이 바로 뜬다)
            q = torch.zeros(tokens, heads, 576, dtype=torch.bfloat16)
            idx = torch.zeros(tokens, 1, topk, dtype=torch.int32)

            def getter(shape):
                seen.append((tokens, tuple(shape)))
                raise Stop

            with self.assertRaises(Stop):
                k.triton_mla_sparse_attention(q, kv, idx, 0.1, sm_count=sm_count, mid_out_getter=getter)
        self.assertTrue(seen)
        sizes = [int(torch.Size(shape).numel()) for _, shape in seen]
        self.assertLessEqual(max(sizes), bound)
        self.assertEqual(max(sizes), bound)                   # 가장 큰 요청이 예약과 정확히 같다 (빈틈 없음)


# ---------------------------------------------------------------------------------------------------------------------
# 음성 대조: 변종 커널을 CPU 인터프리터에서 k12 하네스 run_case 로 실행 (서브프로세스 — 변종은 SIGSEGV 할 수 있다)
# ---------------------------------------------------------------------------------------------------------------------
DRIVER = r'''
import importlib.util, json, os, resource, sys, types
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))          # SIGSEGV 변종이 코어 덤프를 남기지 않게
os.environ["TRITON_INTERPRET"] = "1"
kit, harness, variant, variant_src, cases_json = sys.argv[1:6]
sys.path.insert(0, kit)
from common import triton_interp
triton_interp.install()
import vendor.dsa_ops as dsa_ops
import vendor.dsa_ops.triton_mla_sparse_kernel as real
if variant != "patched":
    spec = importlib.util.spec_from_file_location("vendor.dsa_ops._control_kernel", variant_src)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    if variant == "pr_original":                           # PR 원본: mid_out_getter / max_split_mid_out_numel 이 없다 -> 어댑터
        original = control
        adapter = types.ModuleType("vendor.dsa_ops.triton_mla_sparse_kernel")
        adapter._choose_num_kv_splits = original._choose_num_kv_splits
        adapter.max_split_mid_out_numel = real.max_split_mid_out_numel
        def triton_mla_sparse_attention(q, kv, indices, sm_scale, num_kv_splits=None, sm_count=None, mid_out_getter=None):
            return original.triton_mla_sparse_attention(q, kv, indices, sm_scale, num_kv_splits, sm_count)
        adapter.triton_mla_sparse_attention = triton_mla_sparse_attention
        control = adapter
    sys.modules["vendor.dsa_ops.triton_mla_sparse_kernel"] = control
spec = importlib.util.spec_from_file_location("k12_harness", harness)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
for label, cfg in json.loads(cases_json).items():
    try:
        payload = h.run_case(cfg, label, "cpu", False, 0, 0)
    except RuntimeError as exc:
        if "can't allocate memory" in str(exc):
            print("ALLOC_FAIL", label, flush=True)
            sys.exit(0)
        raise
    bad = [e["violations"] for e in payload["entries"] if not e["passed"]]
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print("RESULT", label, payload["status"], "peak_rss_mb=%d" % peak_mb, json.dumps(bad, ensure_ascii=False)[:300], flush=True)
'''


def load_harness():
    spec = importlib.util.spec_from_file_location("k12_harness_for_vendor_test", KIT_ROOT / "k12_sparse_mla_sm80" / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_variant(variant: str, source: str | None, cases: dict, timeout: int = 900) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        src = ""
        if source is not None:
            src = str(Path(tmp) / "control_kernel.py")
            Path(src).write_text(source, encoding="utf-8")
        driver = Path(tmp) / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8")
        return subprocess.run([sys.executable, str(driver), str(KIT_ROOT), str(KIT_ROOT / "k12_sparse_mla_sm80" / "harness.py"),
                               variant, src, json.dumps(cases)], capture_output=True, text=True, timeout=timeout)


def results(proc: subprocess.CompletedProcess) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r"^RESULT (\S+) (\S+)", proc.stdout, flags=re.M)}


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class DsaOpsInt32NegativeControlTest(unittest.TestCase):
    """packed-row 실기하(물리 row id ≥ 3,728,270 · stride 576 · 풀 ≈ 4.3 GB 가상 메모리)에서 배포 패치본만 통과한다.

    대조군(small) = 같은 케이스·같은 코드 경로에서 풀을 40 블록으로 줄여 `row*576 < 2^31` 로 만든 것 — 변종이 이 대조군을 못 통과하면
    (로더·어댑터 오류 등) 큰 기하의 실패를 int32 wrap 탓으로 돌릴 수 없으므로 테스트가 실패한다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        h = load_harness()
        big = dict(h.INTERPRET_CASES["int_decode_b1_ctx330_packed"])
        cls.cases = {"small": {**big, "pool_blocks": 40, "require_past_int32": False}, "big": big}
        cls.sparse_src = (DSA / "triton_mla_sparse_kernel.py").read_text(encoding="utf-8")

    def assert_control_ok(self, proc: subprocess.CompletedProcess) -> None:
        if "ALLOC_FAIL" in proc.stdout:
            self.skipTest("호스트가 4.3 GB 가상 메모리 예약을 허용하지 않음")
        self.assertEqual(results(proc).get("small"), "PASS", proc.stdout + proc.stderr[-1500:])

    def test_deployed_patch_passes_at_row_ids_beyond_the_int32_limit(self) -> None:
        proc = run_variant("patched", None, self.cases)
        self.assert_control_ok(proc)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(results(proc), {"small": "PASS", "big": "PASS"}, proc.stdout)
        peak = int(re.search(r"^RESULT big \S+ peak_rss_mb=(\d+)", proc.stdout, flags=re.M)[1])
        self.assertLess(peak, 3000, "풀이 가상 메모리로만 잡혀야 한다(사용 블록 행만 채움)")

    def test_variant_without_int64_promotion_fails_only_at_the_big_geometry(self) -> None:
        mutated = self.sparse_src.replace("indices = indices.to(tl.int64)", "indices = indices")
        mutated = mutated.replace("tl.program_id(0).to(tl.int64)", "tl.program_id(0)")
        self.assertEqual(self.sparse_src.count("to(tl.int64)") - mutated.count("to(tl.int64)"), 4)
        proc = run_variant("no_int64", mutated, self.cases)
        self.assert_control_ok(proc)
        self.assertNotEqual(results(proc).get("big"), "PASS", f"int64 승격이 없는데 통과 — 케이스가 int32 wrap 을 못 건드림\n{proc.stdout}")
        crashed_or_wrong = proc.returncode != 0 or results(proc).get("big") == "FAIL"
        self.assertTrue(crashed_or_wrong, proc.stdout + proc.stderr[-1000:])

    @unittest.skipUnless((PIN_CLONE / ".git").exists(), "핀 클론 필요")
    def test_pr_original_kernel_fails_only_at_the_big_geometry(self) -> None:
        shown = subprocess.run(["git", "-C", str(PIN_CLONE), "show", "pr-47629:vllm/v1/attention/ops/triton_mla_sparse_kernel.py"],
                               capture_output=True, text=True)
        if shown.returncode != 0:
            self.skipTest("핀 클론에 pr-47629 ref 없음 (`git fetch origin pull/47629/head:pr-47629`)")
        source = (shown.stdout.replace("from vllm.triton_utils import", "from .._vllm_shims.triton_utils import")
                  .replace("from vllm.utils.platform_utils import", "from .._vllm_shims.platform_utils import"))
        self.assertNotIn("from vllm", source)
        proc = run_variant("pr_original", source, self.cases)
        self.assert_control_ok(proc)
        self.assertNotEqual(results(proc).get("big"), "PASS", proc.stdout)
        self.assertTrue(proc.returncode != 0 or results(proc).get("big") == "FAIL", proc.stdout + proc.stderr[-1000:])


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class DsaOpsDirtyWorkspaceControlTest(unittest.TestCase):
    """auto 변종의 예약 스크래치는 매 호출 NaN 으로 오염된다 — 스크래치를 다 쓰지 않는 커널은 NaN 출력으로 드러나야 한다."""

    @classmethod
    def setUpClass(cls) -> None:
        h = load_harness()
        cls.cases = {"ws": dict(h.INTERPRET_CASES["int_decode_b2_ctx512"])}
        cls.sparse_src = (DSA / "triton_mla_sparse_kernel.py").read_text(encoding="utf-8")

    def test_patched_kernel_passes_with_a_poisoned_reserved_workspace(self) -> None:
        proc = run_variant("patched", None, self.cases)
        self.assertEqual(results(proc), {"ws": "PASS"}, proc.stdout + proc.stderr[-1500:])

    def test_kernel_that_skips_the_lse_store_is_caught(self) -> None:
        marker = "    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)\n"
        self.assertEqual(self.sparse_src.count(marker), 1)
        broken = self.sparse_src.replace(marker, "    pass  # (control) lse 스크래치를 쓰지 않는다\n")
        proc = run_variant("skips_lse_store", broken, self.cases)
        self.assertEqual(results(proc), {"ws": "FAIL"}, proc.stdout + proc.stderr[-1500:])
        self.assertIn("NaN", proc.stdout)


if __name__ == "__main__":
    unittest.main()
