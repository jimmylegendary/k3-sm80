"""vendor 재생성(핀 a3c14efb) 계약 — GPU/torch 없이 AST·해시·문자열로 검증한다.

잡으려는 것: ① 핀 표기 드리프트 ② vendored 본문에 옛 인라인 패치(SM80-FIX) 잔존 ③ `_vllm_shims` 가 vendored 커널이
쓰는 심볼을 덮는지(재vendor 때 가장 먼저 깨지는 곳) ④ k01~k04·k07 하네스의 커널 호출이 vendored 시그니처와 맞는지
⑤ k04 의 sha 핀 ⑥ k01 long-T(int64 퇴역 확정용) 케이스 ⑦ `_EXPORTS` 가 실재 심볼을 가리키는지.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common

shapes = import_common("shapes")

VENDOR = KIT_ROOT / "vendor"
SHIMS = VENDOR / "_vllm_shims"
PIN = "a3c14efbc77c018e8bf579935095a87721349e05"
PR_HEAD = "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
OLD_PINS = ("0498dc7e", "d223c900")
HEADER = re.compile(r"^# vendored from vllm-project/vllm@([0-9a-f]{40}) (\S+) — Apache-2\.0$")


def vendored_files() -> list[Path]:
    return sorted(p for p in VENDOR.rglob("*.py") if "__pycache__" not in p.parts)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in parse(path).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((a.asname or a.name) for a in node.names)
    return names


class VendorPinTest(unittest.TestCase):
    def test_vendor_constants(self) -> None:
        init = parse(VENDOR / "__init__.py")
        consts = {t.id: ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual(consts["VENDOR_PIN"], PIN)
        self.assertEqual(consts["DSA_OPS_PIN"], PR_HEAD)

    def test_every_vendored_source_header_carries_the_new_pin(self) -> None:
        headed = []
        for path in vendored_files():
            if path.parts[-2] == "dsa_ops" or path.name in ("check_no_vllm.py", "mla_fp8_kv_sm80.py",
                                                            "revendor_dsa_ops.py"):
                continue                     # dsa_ops 는 '핀 + 패치' 별도 헤더(tests/kernel/test_dsa_ops_vendor.py), 나머지는 킷 도구/저작물
            if path.name == "__init__.py" and path.parent.name != "fla_kda":
                continue                     # fla_kda/__init__.py 만 원본 사본(나머지 __init__ 은 킷 저작물)
            if path.parent == SHIMS and path.name != "math_utils.py":
                continue                     # 스텁은 킷 저작물 — 아래 별도 테스트
            first = path.read_text(encoding="utf-8").splitlines()
            line = first[0]
            if path.name == "situ.py":
                self.assertIn(PIN, line + first[1], path)
                continue
            m = HEADER.match(line)
            self.assertIsNotNone(m, f"{path}: 출처 헤더 형식 위반: {line!r}")
            self.assertEqual(m.group(1), PIN, path)
            headed.append(path.relative_to(VENDOR).as_posix())
        self.assertEqual(len(headed), 14, headed)       # fla_kda 5 + fla_common 6 + mamba 1 + attn 1 + math_utils 1

    def test_no_stale_pin_strings_in_vendor_sources(self) -> None:
        for path in vendored_files():
            text = path.read_text(encoding="utf-8")
            for stale in OLD_PINS:
                if (path.name == "__init__.py" and path.parent == VENDOR) or path.name == "situ.py":
                    continue                 # 주석에서 '이전 핀 0498dc7e' 대비를 설명하는 곳만 예외
                self.assertNotIn(stale, text, f"{path}: 옛 핀 {stale} 잔존")

    def test_old_inline_int64_patches_are_retired(self) -> None:
        # 핀 a3c14efb 는 kda-sm80-int64 패치를 upstream 에 포함 — vendor 본문에 인라인 SM80-FIX 가 남으면 안 된다
        for rel in ("fla_kda/chunk.py", "fla_common/chunk_delta_h.py"):
            text = (VENDOR / rel).read_text(encoding="utf-8")
            self.assertNotIn("SM80-FIX", text, rel)
        chunk = (VENDOR / "fla_kda/chunk.py").read_text(encoding="utf-8")
        self.assertNotIn("make_block_ptr", chunk)       # 핀에서 포인터 연산으로 재작성됨
        delta = (VENDOR / "fla_common/chunk_delta_h.py").read_text(encoding="utf-8")
        self.assertIn("to(tl.int64)", delta)             # upstream 의 cast-before-multiply 수정

    def test_conv1d_cpu_override_block_is_the_only_body_change(self) -> None:
        text = (VENDOR / "mamba_ops/causal_conv1d.py").read_text(encoding="utf-8")
        self.assertNotIn("if current_platform.is_cpu():", text)
        self.assertIn("vendor/STUBS.md #cpu-block", text)


KDA = "vllm/models/kimi_k3/nvidia/ops/third_party/kda"
FLA = "vllm/third_party/flash_linear_attention/ops"
ORIGINALS = {
    "fla_kda/__init__.py": f"{KDA}/__init__.py", "fla_kda/chunk.py": f"{KDA}/chunk.py",
    "fla_kda/chunk_intra.py": f"{KDA}/chunk_intra.py",
    "fla_kda/chunk_intra_token_parallel.py": f"{KDA}/chunk_intra_token_parallel.py",
    "fla_kda/fused_recurrent.py": f"{KDA}/fused_recurrent.py",
    "fla_common/chunk_delta_h.py": f"{FLA}/chunk_delta_h.py", "fla_common/cumsum.py": f"{FLA}/cumsum.py",
    "fla_common/index.py": f"{FLA}/index.py", "fla_common/l2norm.py": f"{FLA}/l2norm.py",
    "fla_common/op.py": f"{FLA}/op.py", "fla_common/utils.py": f"{FLA}/utils.py",
    "mamba_ops/causal_conv1d.py": "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "attn_ops/triton_decode_attention.py": "vllm/v1/attention/ops/triton_decode_attention.py",
    "_vllm_shims/math_utils.py": "vllm/utils/math_utils.py",
}
PIN_CLONE = Path("/home/jimmy/repos/vllm-pin-k31")


@unittest.skipUnless(PIN_CLONE.is_dir(), "핀 클론(/home/jimmy/repos/vllm-pin-k31) 필요")
class VendorBodyEqualsPinTest(unittest.TestCase):
    """'본문 무수정' 증명: 핀 원본과 vendor 사본의 차이는 import 행(+ conv1d CPU 블록 제거)뿐이어야 한다."""

    def test_only_import_lines_differ_from_the_pinned_originals(self) -> None:
        import difflib
        import subprocess
        for vend, orig in ORIGINALS.items():
            blob = subprocess.run(["git", "-C", str(PIN_CLONE), "show", f"{PIN}:{orig}"], capture_output=True, text=True)
            if blob.returncode != 0:
                self.skipTest(f"핀 클론에서 {PIN} 를 못 읽음")
            original = blob.stdout.splitlines()
            vendored = (VENDOR / vend).read_text(encoding="utf-8").splitlines()
            self.assertTrue(HEADER.match(vendored[0]), vend)
            self.assertTrue(vendored[1].startswith("# 변경점:"), vend)
            body = vendored[2:]
            changed = []
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, body, autojunk=False).get_opcodes():
                if tag == "equal":
                    continue
                changed += [("-", ln) for ln in original[i1:i2]] + [("+", ln) for ln in body[j1:j2]]
            for sign, line in changed:
                if vend == "mamba_ops/causal_conv1d.py" and (
                        sign == "-" or line.startswith("# [vendor 변경]") or line.startswith("# ")):
                    continue                  # CPU 오버라이드 블록 제거 + 그 자리의 설명 주석(STUBS.md #cpu-block)
                self.assertRegex(line.strip(), r"^(from\b|import\b|\)|[A-Za-z_]\w*,$|\(|$)",
                                 f"{vend}: import 외 변경 {sign} {line!r}")


class ShimCoverageTest(unittest.TestCase):
    """_vllm_shims 가 vendored 커널이 쓰는 심볼·플랫폼 메서드를 전부 덮는지 (재vendor 때 가장 먼저 깨지는 곳)."""

    def test_every_shim_import_resolves_to_a_defined_name(self) -> None:
        seen = 0
        for path in vendored_files():
            if SHIMS in path.parents:
                continue
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.level == 2 and (node.module or "").startswith("_vllm_shims."):
                    shim = SHIMS / (node.module.split(".", 1)[1] + ".py")
                    self.assertTrue(shim.is_file(), f"{path.name}: 스텁 모듈 없음 {node.module}")
                    defined = top_level_names(shim)
                    for alias in node.names:
                        seen += 1
                        self.assertIn(alias.name, defined, f"{path.name}: {node.module}.{alias.name} 미정의")
        self.assertGreater(seen, 20)

    def test_current_platform_attributes_used_are_implemented(self) -> None:
        cls = next(n for n in parse(SHIMS / "platforms.py").body
                   if isinstance(n, ast.ClassDef) and n.name == "_TorchPlatformShim")
        implemented = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        used: dict[str, set[str]] = {}
        for path in vendored_files():
            if SHIMS in path.parents:
                continue
            for node in ast.walk(parse(path)):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "current_platform"):
                    used.setdefault(node.attr, set()).add(path.name)
        self.assertTrue({"is_cuda", "is_rocm", "is_cuda_alike", "has_device_capability", "is_arch_support_pdl",
                         "get_device_name", "get_device_capability"} <= set(used), used)
        for attr, where in used.items():
            self.assertIn(attr, implemented, f"current_platform.{attr} 를 {sorted(where)} 가 쓰는데 스텁에 없음")

    def test_shims_do_not_import_vllm(self) -> None:
        for path in SHIMS.glob("*.py"):
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotEqual((node.module or "").split(".")[0], "vllm", path)
                if isinstance(node, ast.Import):
                    self.assertTrue(all(a.name.split(".")[0] != "vllm" for a in node.names), path)

    def test_no_vllm_gate_is_green(self) -> None:
        spec = importlib.util.spec_from_file_location("check_no_vllm", VENDOR / "check_no_vllm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.main(["--quiet"]), 0)


class ExportsTest(unittest.TestCase):
    def test_exports_point_at_real_symbols_and_match_kda_public_api(self) -> None:
        init = parse(VENDOR / "__init__.py")
        exports = next(ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in n.targets))
        for symbol, mod in exports.items():
            target = VENDOR / (mod.lstrip(".").replace(".", "/") + ".py")
            if not target.is_file():
                target = VENDOR / mod.lstrip(".") / "__init__.py"
            self.assertTrue(target.is_file(), f"{symbol}: {mod}")
            self.assertIn(symbol, top_level_names(target), f"{symbol} not in {target}")
        kda_all = ast.literal_eval(next(n.value for n in parse(VENDOR / "fla_kda/__init__.py").body
                                        if isinstance(n, ast.Assign) and n.targets[0].id == "__all__"))
        self.assertEqual({s for s, m in exports.items() if m == ".fla_kda"}, set(kda_all))
        self.assertEqual(len(kda_all), 9)                 # 핀 a3c14efb kda/__init__.py 공개 심볼 9개 재대조


def signature_of(path: Path, name: str) -> dict:
    fn = next(n for n in parse(path).body if isinstance(n, ast.FunctionDef) and n.name == name)
    a = fn.args
    pos = [x.arg for x in a.posonlyargs + a.args]
    n_def = len(a.defaults)
    return {"pos": pos, "kwonly": [x.arg for x in a.kwonlyargs],
            "required": pos[: len(pos) - n_def] + [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]}


# 하네스 → {호출에 쓰는 지역 이름: (vendored 파일, 함수)}
CALLEES = {
    "k01_kda_chunk": {"chunk_fn": ("fla_kda/chunk.py", "chunk_kda_with_fused_gate"),
                      "chunk_kda_with_fused_gate": ("fla_kda/chunk.py", "chunk_kda_with_fused_gate")},
    "k02_kda_recurrent": {
        "chunk_fn": ("fla_kda/chunk.py", "chunk_kda_with_fused_gate"),
        "fused_recurrent_kda_fwd": ("fla_kda/fused_recurrent.py", "fused_recurrent_kda_fwd"),
        "fused_recurrent_kda_packed_decode": ("fla_kda/fused_recurrent.py", "fused_recurrent_kda_packed_decode")},
    "k03_causal_conv1d": {
        "fn": ("mamba_ops/causal_conv1d.py", "causal_conv1d_fn"),
        "causal_conv1d_fn": ("mamba_ops/causal_conv1d.py", "causal_conv1d_fn"),
        "update_fn": ("mamba_ops/causal_conv1d.py", "causal_conv1d_update"),
        "causal_conv1d_update": ("mamba_ops/causal_conv1d.py", "causal_conv1d_update")},
    "k04_mla_decode": {"decode_attention_fwd": ("attn_ops/triton_decode_attention.py", "decode_attention_fwd")},
}


class HarnessCallCompatTest(unittest.TestCase):
    def test_harness_kernel_calls_match_vendored_signatures(self) -> None:
        for harness, callees in CALLEES.items():
            tree = parse(KIT_ROOT / harness / "harness.py")
            found: dict[str, int] = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name not in callees:
                    continue
                path, func = callees[name]
                sig = signature_of(VENDOR / path, func)
                found[name] = found.get(name, 0) + 1
                kws = [k.arg for k in node.keywords if k.arg is not None]
                self.assertEqual(len(kws), len(set(kws)), f"{harness}:{node.lineno} 중복 키워드")
                for kw in kws:
                    self.assertIn(kw, sig["pos"] + sig["kwonly"], f"{harness}:{node.lineno} {func}({kw}=…) 신 시그니처에 없음")
                self.assertLessEqual(len(node.args), len(sig["pos"]), f"{harness}:{node.lineno} 위치 인자 초과")
                supplied = set(sig["pos"][: len(node.args)]) | set(kws)
                missing = [r for r in sig["required"] if r not in supplied]
                self.assertFalse(missing, f"{harness}:{node.lineno} {func} 필수 인자 누락 {missing}")
            # 이름 표가 낡아 호출 지점을 놓치면 검증이 조용히 공회전한다 → 함수마다 최소 1개 호출 지점을 요구
            self.assertEqual({callees[n][1] for n in found}, {func for _, func in callees.values()},
                             f"{harness}: 호출 지점을 못 찾은 커널 함수가 있음(이름 표 갱신 필요)")

    def test_new_pin_signature_changes_are_backward_compatible_for_the_harness(self) -> None:
        # 핀에서 추가된 인자는 전부 기본값 뒤쪽 — 하네스가 안 쓰는 신규 인자(out / sliding_window)가 필수가 아님을 확인
        chunk = signature_of(VENDOR / "fla_kda/chunk.py", "chunk_kda_with_fused_gate")
        self.assertIn("out", chunk["pos"])
        self.assertNotIn("out", chunk["required"])
        decode = signature_of(VENDOR / "attn_ops/triton_decode_attention.py", "decode_attention_fwd")
        self.assertIn("sliding_window", decode["pos"])
        self.assertNotIn("sliding_window", decode["required"])
        conv = signature_of(VENDOR / "mamba_ops/causal_conv1d.py", "causal_conv1d_fn")
        for removed in ("block_idx_first_scheduled_token", "block_idx_last_scheduled_token", "initial_state_idx",
                        "num_computed_tokens", "block_size_to_align"):
            self.assertNotIn(removed, conv["pos"])     # 핀에서 제거된 APC 인자 — 하네스가 쓰면 위 테스트가 잡는다


class PinsDictTest(unittest.TestCase):
    def test_harness_pins_name_the_new_vllm_pin(self) -> None:
        for kernel_dir in ("k01_kda_chunk", "k02_kda_recurrent", "k03_causal_conv1d", "k04_mla_decode",
                           "k05_marlin_moe_w4a16", "k06_dequant_bench", "k07_situ_act", "k08_mxfp4_marlin_moe",
                           "k09_mla_fp8_kv_sm80", "k11_dsa_indexer_sm80", "k12_sparse_mla_sm80"):
            tree = parse(KIT_ROOT / kernel_dir / "harness.py")
            pins = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "PINS" for t in n.targets))
            self.assertIn("vllm_pin", pins, kernel_dir)
            self.assertTrue(PIN.startswith(pins["vllm_pin"]) and len(pins["vllm_pin"]) >= 8, (kernel_dir, pins))
            for value in pins.values():
                for stale in OLD_PINS:
                    self.assertNotIn(stale, str(value), (kernel_dir, pins))

    def test_k04_sha_pin_matches_vendored_file(self) -> None:
        text = (KIT_ROOT / "k04_mla_decode" / "harness.py").read_text(encoding="utf-8")
        vendored = re.search(r'VENDORED_KERNEL_SHA256 = "([0-9a-f]{64})"', text).group(1)
        pinned = re.search(r'PINNED_KERNEL_SHA256 = "([0-9a-f]{64})"', text).group(1)
        actual = hashlib.sha256((VENDOR / "attn_ops/triton_decode_attention.py").read_bytes()).hexdigest()
        self.assertEqual(vendored, actual, "vendored 사본이 바뀌었다 — k04 VENDORED_KERNEL_SHA256 갱신")
        self.assertNotEqual(vendored, pinned)           # vendor 사본 = 원본 + 헤더 2줄 + import 상대화
        orig = Path("/home/jimmy/repos/vllm-pin-k31")
        if orig.is_dir():                                # 핀 클론이 있으면 원본 해시도 대조
            import subprocess
            blob = subprocess.run(["git", "-C", str(orig), "show", f"{PIN}:vllm/v1/attention/ops/triton_decode_attention.py"],
                                  capture_output=True, check=False)
            if blob.returncode == 0:
                self.assertEqual(pinned, hashlib.sha256(blob.stdout).hexdigest())


class K01LongTTest(unittest.TestCase):
    def test_int32_overflow_threshold_derivation(self) -> None:
        h, k, v = shapes.KDA_HEADS, shapes.KDA_HEAD_DIM, shapes.KDA_HEAD_DIM
        # h 버퍼 오프셋 (i_tg·H + i_h)·K·V 가 int32 로 wrap 하는 첫 청크: 1365 는 일부 head 만, 1366 부터는 전 head
        self.assertLess((1365 * h) * k * v, 2**31)
        self.assertGreaterEqual((1365 * h + (h - 1)) * k * v, 2**31)
        self.assertGreaterEqual((1366 * h) * k * v, 2**31)
        self.assertLess((1365 * h + 30) * k * v, 2**31)
        self.assertEqual(1366 * 64, 87_424)

    def test_k01_required_shapes_include_a_case_past_the_threshold(self) -> None:
        spec = importlib.util.spec_from_file_location("k01_harness", KIT_ROOT / "k01_kda_chunk" / "harness.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.INT32_OVERFLOW_MIN_TOKENS, 87_424)
        required = shapes.REQUIRED_SHAPES["k01_kda_chunk"]
        self.assertIn(module.LONG_T_CASE, required)
        self.assertGreater(required[module.LONG_T_CASE]["T"], module.INT32_OVERFLOW_MIN_TOKENS)
        self.assertEqual(sum(module.CASE_LENGTHS[module.LONG_T_CASE]), required[module.LONG_T_CASE]["T"])


if __name__ == "__main__":
    unittest.main()
