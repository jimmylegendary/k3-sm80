"""k12_sparse_mla_sm80 계약 테스트 — 등록 지점 · vendor 소스 · 케이스 구성 · 판정 로직 · dry-run 상한 · (선택) CPU 인터프리터 대조."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from kernel_kit_import import KIT_ROOT, import_common

shapes = import_common("shapes")
timing = import_common("timing")
tolerance = import_common("tolerance")
dsa_gate = import_common("dsa_gate")

KID = "k12_sparse_mla_sm80"
HARNESS = KIT_ROOT / "k12_sparse_mla_sm80" / "harness.py"
VENDOR = KIT_ROOT / "vendor" / "dsa_ops" / "triton_mla_sparse_kernel.py"

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_harness():
    spec = importlib.util.spec_from_file_location("k12_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k12 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_torch_and_triton() -> bool:
    return torch is not None and importlib.util.find_spec("triton") is not None


class K12RegistrationTest(unittest.TestCase):
    def test_k31_sparse_mla_shapes_are_registered(self) -> None:
        required = shapes.REQUIRED_SHAPES[KID]
        self.assertEqual(shapes.MLA_CACHE_D, 576)
        self.assertEqual(shapes.DSA_HEADS_LOCAL, 12)            # 96 heads / TP8
        self.assertEqual(shapes.DSA_TOPK, 2048)
        self.assertEqual(
            set(required),
            {"dec_b1_ctx8k", "dec_b8_ctx8k", "dec_b1_ctx128k", "dec_b8_ctx128k", "dec_b1_ctx256k",
             "dec_b8_ctx256k", "dec_b1_ctx160k_packed", "pre_t256_ctx8k", "pre_t256_ctx128k", "pre_t256_ctx256k",
             "pre_short_t64"})
        self.assertEqual({c["ctx"] for c in required.values() if c["mode"] == "decode"},
                         {8192, 131072, 262144, 163840})
        self.assertTrue(required["pre_short_t64"]["short"])      # -1 sentinel 경로 (PR 회귀 테스트와 동형)
        packed = required["dec_b1_ctx160k_packed"]               # packed hybrid KV row 흉내: 블록 stride 확대 + int32 한계 너머 row id
        self.assertEqual((packed["mode"], packed["b"], packed["block_stride_rows"]),
                         ("decode", 1, shapes.DSA_PACKED_BLOCK_STRIDE_ROWS))
        self.assertGreater(shapes.DSA_PACKED_BLOCK_STRIDE_ROWS, shapes.DSA_BLOCK_SIZE)
        self.assertEqual(shapes.DSA_ROW_INT32_LIMIT, 2**31 // shapes.MLA_CACHE_D)
        model = shapes.perf_bytes_model(KID, **next(iter(required.values())))
        self.assertGreater(model["bytes"], 0)

    def test_perf_bytes_model_is_gather_dominated(self) -> None:
        one = shapes.perf_bytes_model(KID, mode="decode", b=1, ctx=8192)
        gather = shapes.DSA_TOPK * shapes.MLA_CACHE_D * 2
        self.assertGreaterEqual(one["bytes"], gather)
        self.assertLess(one["bytes"], gather * 1.1)
        eight = shapes.perf_bytes_model(KID, mode="decode", b=8, ctx=8192)
        self.assertAlmostEqual(eight["bytes"] / one["bytes"], 8.0, places=9)    # 토큰 수에 선형, ctx 무관
        self.assertEqual(shapes.perf_bytes_model(KID, mode="decode", b=1, ctx=262144)["bytes"], one["bytes"])
        self.assertEqual(one["macs"], shapes.DSA_HEADS_LOCAL * shapes.DSA_TOPK * (576 + 512))

    def test_registered_in_framework_tables_and_runner(self) -> None:
        self.assertIn(KID, shapes.KERNEL_IDS)
        self.assertIn(KID, shapes.KERNEL_TITLES)
        policy = timing.PERF_POLICY[KID]
        self.assertEqual(policy["mode"], "measure_only")
        self.assertLessEqual(policy["cv_max"], 0.15)
        pol = tolerance.TOL[KID]
        self.assertEqual((pol.rmse_rel, pol.max_abs), (5e-3, 5e-2))
        self.assertIn(KID, (KIT_ROOT / "run_all.sh").read_text(encoding="utf-8"))
        self.assertTrue((KIT_ROOT / KID / "harness.py").is_file())      # run_all 이 찾는 경로


class K12VendorSourceTest(unittest.TestCase):
    def test_vendor_is_the_deployed_patch_copy_with_no_vllm_import(self) -> None:
        # 이미지에 실리는 커널 = 핀 a3c14efb + triton-mla-sparse-sm80.patch (PR 원본이 아니다) — 계약 전체는 test_dsa_ops_vendor.py
        source = VENDOR.read_text(encoding="utf-8")
        head = "\n".join(source.splitlines()[:4])
        for token in ("a3c14efbc77c018e8bf579935095a87721349e05", "docker/patches-k31/triton-mla-sparse-sm80.patch",
                      "sha256 74675e998be0022aa4caed7df4467b62148fad09a6af9e47d13ea1b302fb7ed8", "PR #47629",
                      "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d", "#38476", "haosdent/thomaslwang", "Apache-2.0"):
            self.assertIn(token, head)
        self.assertGreaterEqual(source.count("to(tl.int64)"), 4)         # PR 원본에는 0개 — indices·cur_q×3
        for symbol in ("def triton_mla_sparse_attention", "_sparse_mla_kernel_final", "_sparse_mla_kernel_split",
                       "_sparse_mla_merge_kernel", "def _choose_num_kv_splits", "KV_SPLITS_CANDIDATES",
                       "mid_out_getter", "def max_split_mid_out_numel"):
            self.assertIn(symbol, source)
        self.assertIn("_BLOCK_DMODEL = 512", source)
        self.assertIn("_BLOCK_DPE = 64", source)                          # latent 576 = 512 + 64
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                self.assertNotEqual((node.module or "").split(".")[0], "vllm")
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name.split(".")[0] != "vllm" for a in node.names))


class K12ContractTest(unittest.TestCase):
    def test_dry_run_never_claims_gpu_pass(self) -> None:
        payload = load_harness().build_dry_report()
        self.assertEqual(payload["id"], KID)
        self.assertEqual(payload["overall"], "INCOMPLETE")
        self.assertTrue(all(g["status"] == "SKIP" for g in payload["gates"].values()))
        self.assertEqual(payload["evidence"]["dry_run"]["selfcheck"], "PASS")
        checks = payload["evidence"]["dry_run"]["checks"]
        for name in ("packed_case_registered", "packed_case_pool_crosses_int32_offset",
                     "packed_case_gpu_memory_within_budget", "int32_row_limit_is_2p31_over_576"):
            self.assertTrue(checks[name], name)
        kernel = payload["evidence"]["kernel"]                              # 리포트가 '어느 커널을 판정했는지'를 스스로 말한다
        self.assertIn("triton-mla-sparse-sm80.patch", kernel["module"])
        entry = kernel["files"]["triton_mla_sparse_kernel.py"]
        self.assertGreaterEqual(entry["to_int64_markers"], 4)
        self.assertRegex(entry["header_patch_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["sha256"], __import__("hashlib").sha256(VENDOR.read_bytes()).hexdigest())
        pool = payload["evidence"]["packed_pool"]["cases"]["dec_b1_ctx160k_packed"]    # GPU 메모리 요구량 산출이 evidence 에 남는다
        self.assertLessEqual(pool["est_gpu_bytes"], 8 * 2**30)
        self.assertGreater(pool["max_offset_elems"], 2**31)
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--cpu-dry-run", "--report-dir", tmp],
                                 text=True, capture_output=True, check=False)
            self.assertEqual(res.returncode, 0, res.stderr)
            written = json.loads((Path(tmp) / f"{KID}.json").read_text())
        self.assertIn("K12-OVERALL INCOMPLETE", res.stdout)
        self.assertEqual(written["overall"], "INCOMPLETE")

    def test_gpu_mode_without_gpu_exits_precondition_not_fail(self) -> None:
        if torch is not None and torch.cuda.is_available():
            self.skipTest("GPU 가 있는 호스트")
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--report-dir", tmp], text=True,
                                 capture_output=True, check=False)
        self.assertEqual(res.returncode, 2, res.stdout + res.stderr)

    def test_run_parent_end_to_end_with_fake_children_and_sanitizer(self) -> None:
        """GPU 없이 parent 오케스트레이션 전체(사전조건→child 순차→sanitizer→게이트 조립→리포트 기록)를 가짜 child 로 실행."""
        import os
        from unittest import mock
        h = load_harness()
        env_check = import_common("env_check")
        child = (
            "import argparse, json, os\nap = argparse.ArgumentParser(); ap.add_argument('--child-case')\n"
            "ap.add_argument('--child-out'); ap.add_argument('--warmup'); ap.add_argument('--reps'); a = ap.parse_args()\n"
            "os.makedirs(os.path.dirname(a.child_out), exist_ok=True)\n"
            "json.dump({'status': 'PASS', 'shape': {}, 'entries': [{'case': a.child_case + '/out', 'metrics': {'rmse_rel': 1e-4, 'max_abs': 1e-4},"
            " 'violations': [], 'passed': True}], 'timing': {'median_us': 50.0, 'mean_us': 50.0, 'min_us': 49.0, 'p90_us': 51.0,"
            " 'cv': 0.01, 'reps': 5}, 'extra': {}}, open(a.child_out, 'w'))\n")
        fake_san = "#!/bin/sh\nshift 2\n\"$@\"; rc=$?\necho \"========= ERROR SUMMARY: 0 errors\"\nexit $rc\n"
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "child.py"
            harness.write_text(child)
            san = Path(tmp) / "compute-sanitizer"
            san.write_text(fake_san)
            san.chmod(0o755)
            out_dir = Path(tmp) / "report"
            args = types.SimpleNamespace(report_dir=str(out_dir), cases=None, warmup=0, reps=1, timeout_s=120,
                                         phase0_verdict=str(KIT_ROOT.parent / "phase0" / "verdict.json"), twin=None)
            with mock.patch.object(env_check, "check_preconditions", return_value=(True, [])), \
                 mock.patch.object(env_check, "find_compute_sanitizer", return_value=str(san)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = dsa_gate.run_parent(KID, harness, h.PINS, h.CASE_ORDER, h.PERF_CASE,
                                             list(h.SANITIZER_CASES), args, {"kernel": "fake"})
            written = json.loads((out_dir / f"{KID}.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(written["overall"], "PASS", {g: v["status"] for g, v in written["gates"].items()})
        self.assertEqual(written["evidence"]["kernel"], "fake")
        perf = written["gates"]["PERF"]["evidence"]
        self.assertEqual(perf["perf_case"], h.PERF_CASE)
        self.assertEqual(len(perf["all_cases"]), len(h.CASE_ORDER))
        self.assertIn("roofline_eff", perf["all_cases"][h.PERF_CASE])        # 트윈 해석 경로까지 탄다 (정보용)

    def test_case_construction_matches_k31_decode_prefill_semantics(self) -> None:
        h = load_harness()
        dec = h.query_lengths(dict(mode="decode", b=8, ctx=8192))
        self.assertEqual(len(dec), 8)
        self.assertTrue(all(length % 64 != 0 for length in dec))        # 마지막 블록이 항상 부분 블록
        self.assertEqual(dec[0], 8191)
        pre = h.query_lengths(dict(mode="prefill", t=256, ctx=262144))
        self.assertEqual((pre[0], pre[-1]), (262144 - 256 + 1, 262144))   # 마지막 t 토큰, 인과 [0, p]
        short = h.query_lengths(dict(mode="prefill", t=64, ctx=64, short=True))
        self.assertEqual(short, list(range(1, 65)))                    # 유효 후보 1..64 ≪ topk 2048
        self.assertEqual(h.variants_for(dict(mode="decode")), [None, 1, 4])
        self.assertEqual(h.variants_for(dict(mode="prefill")), [None, 2])   # auto 와 반대 경로를 덮는다
        self.assertEqual(h.SCALE, 192 ** -0.5)

    def test_assemble_report_gates_from_child_payloads(self) -> None:
        h = load_harness()
        args = types.SimpleNamespace(phase0_verdict=str(KIT_ROOT.parent / "phase0" / "verdict.json"), twin=None)

        def child(name, ok=True):
            return {"case": name, "status": "PASS" if ok else "FAIL", "classification": "PASS" if ok else "FAIL",
                    "returncode": 0 if ok else 1, "duration_s": 1.0, "log_tail": "",
                    "result": {"shape": shapes.REQUIRED_SHAPES[KID][name], "status": "PASS" if ok else "FAIL",
                               "entries": [{"case": f"{name}/out[auto]",
                                            "metrics": {"rmse_rel": 2e-3 if ok else 0.2, "max_abs": 3e-3},
                                            "violations": [] if ok else ["rmse_rel 0.2 > 5e-3"], "passed": ok}],
                               "timing": {"median_us": 80.0, "mean_us": 80.0, "min_us": 79.0, "p90_us": 81.0,
                                          "cv": 0.03, "reps": 50}, "extra": {}}}

        san = ("PASS", "memcheck ERROR SUMMARY: 0", {})
        rep = dsa_gate.assemble_report(KID, h.PINS, [child(n) for n in h.CASE_ORDER], san, h.PERF_CASE, args, {})
        self.assertEqual(rep.overall(), "PASS")
        bad = dsa_gate.assemble_report(KID, h.PINS, [child(n, n != "pre_short_t64") for n in h.CASE_ORDER],
                                       san, h.PERF_CASE, args, {})
        self.assertEqual(bad.gates["NUMERIC"]["status"], "FAIL")
        self.assertEqual(bad.overall(), "FAIL")


@unittest.skipUnless(torch is not None, "torch 필요")
class K12JudgeNegativeControlTest(unittest.TestCase):
    def test_tolerance_accepts_bf16_noise_and_rejects_real_errors(self) -> None:
        g = torch.Generator().manual_seed(11)
        ref = torch.randn(8, 12, 512, generator=g) * 0.1
        bf16_like = ref.to(torch.bfloat16).float()                         # 출력 bf16 반올림 수준
        self.assertTrue(tolerance.compare(KID, ref, bf16_like, "bf16").passed)
        self.assertFalse(tolerance.compare(KID, ref, ref * 1.05, "scaled").passed)
        wrong_row = ref.clone()
        wrong_row[3] = torch.randn(12, 512, generator=g) * 0.1               # 한 질의가 다른 행을 gather
        self.assertFalse(tolerance.compare(KID, ref, wrong_row, "wrong_row").passed)
        nan = ref.clone()
        nan[0, 0, 0] = float("nan")
        self.assertFalse(tolerance.compare(KID, ref, nan, "nan").passed)


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K12InterpretCpuTest(unittest.TestCase):
    def test_interpret_runs_vendored_kernel_and_keeps_gates_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--interpret", "--report-dir", tmp],
                                 text=True, capture_output=True, check=False, timeout=1800)
            if res.returncode != 0 and "can't allocate memory" in res.stdout + res.stderr:
                self.skipTest("호스트가 packed-row 케이스의 4 GiB 가상 메모리 예약을 허용하지 않음")
            self.assertEqual(res.returncode, 0, res.stdout[-2000:] + res.stderr[-2000:])
            rep = json.loads((Path(tmp) / f"{KID}.json").read_text())
        self.assertEqual(rep["overall"], "INCOMPLETE")
        self.assertTrue(all(g["status"] == "SKIP" for g in rep["gates"].values()))
        ic = rep["evidence"]["interpret_cpu"]
        self.assertEqual(ic["overall"], "PASS")
        self.assertEqual(len(ic["cases"]), 4)
        labels = {e["case"].split("/")[-1] for c in ic["cases"] for e in c["entries"]}
        self.assertTrue({"out[auto]", "out[splits=1]", "out[splits=4]", "out[splits=2]"} <= labels)
        short = next(c for c in ic["cases"] if "short" in c["case"])
        self.assertTrue(all(e["metrics"].get("padding_row_exact_zero") for e in short["entries"] if "out[" in e["case"]))
        # auto 변종이 split 경로를 골라 예약 workspace(max_split_mid_out_numel)에서 스크래치를 받는 경로까지 돈다
        ws_case = next(c for c in ic["cases"] if c["case"] == "int_decode_b2_ctx512")
        self.assertEqual(ws_case["extra"]["auto_num_kv_splits"], 2)
        self.assertTrue(ws_case["extra"]["workspace"]["auto_requests"])
        self.assertTrue(ws_case["extra"]["workspace"]["bound_covers_requests"])
        # packed-row 케이스: 선택 행 전부가 int32 오프셋 한계(row*576 ≥ 2^31) 너머 — 실기하(풀 4 GiB+)를 CPU 에서 읽는다
        packed = next(c for c in ic["cases"] if c["case"].endswith("_packed"))
        zone = packed["extra"]["int32_zone"]
        self.assertEqual(zone["selected_rows_past_int32_limit"], zone["selected_rows"])
        self.assertGreaterEqual(zone["max_selected_offset_elems"], 2**31)
        self.assertTrue(any(e["case"].endswith("/int32_zone") and e["passed"] for e in packed["entries"]))
        self.assertGreater(packed["extra"]["pool"]["pool_gib"], 4.0)


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K12KernelOnlyModeTest(unittest.TestCase):
    def test_kernel_only_executes_all_split_variants(self) -> None:
        prog = f"""
import os, sys, importlib.util
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, {str(KIT_ROOT)!r})
from common import triton_interp
triton_interp.install()
spec = importlib.util.spec_from_file_location("h", {str(HARNESS)!r})
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
for cfg in (dict(mode="decode", b=2, ctx=70, heads=4, topk=64), dict(mode="prefill", t=4, ctx=4, short=True, heads=4, topk=64)):
    p = h.run_case(cfg, "kernel_only", "cpu", False, 0, 0, kernel_only=True)
    print("KERNEL_ONLY", p["status"], p["entries"], p["extra"]["kernel_only"])
"""
        res = subprocess.run([sys.executable, "-c", prog], text=True, capture_output=True, check=False, timeout=900)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertEqual(res.stdout.count("KERNEL_ONLY PASS [] True"), 2, res.stdout)


class K12PackedGeometryTest(unittest.TestCase):
    """packed-row REQUIRED 케이스의 기하·GPU 메모리 산출 (순수 python) — shapes.py 주석의 숫자와 같아야 한다."""

    def setUp(self) -> None:
        self.h = load_harness()
        self.cfg = shapes.SPARSE_MLA_REQUIRED_CASES["dec_b1_ctx160k_packed"]
        self.geo = self.h.packed_geometry(self.cfg)

    def test_required_packed_case_crosses_the_int32_offset_limit(self) -> None:
        g = self.geo
        self.assertEqual((g["used_blocks"], g["pool_blocks"], g["pool_rows"]), (2560, 2562, 4_386_144))
        self.assertEqual(g["first_zone_block"] * shapes.DSA_BLOCK_SIZE, 139_392)       # 약 139k 토큰부터 row*576 이 2^31 을 넘는다
        self.assertEqual(g["blocks_past_limit"], 2562 - 2178)
        self.assertGreater(g["max_row"], shapes.DSA_ROW_INT32_LIMIT)
        self.assertGreater(g["max_offset_elems"], 2**31)
        limit = shapes.DSA_ROW_INT32_LIMIT
        self.assertLess(limit * shapes.MLA_CACHE_D + shapes.MLA_CACHE_D - 1 - 448, 2**31)     # 한계 row 의 앞 lane 은 int32 안
        self.assertGreaterEqual((limit + 1) * shapes.MLA_CACHE_D, 2**31)                      # 한계+1 row 는 통째로 밖

    def test_gpu_memory_estimate_is_within_the_8_gib_target(self) -> None:
        g = self.geo
        self.assertAlmostEqual(g["pool_gib"], 4.706, places=3)
        self.assertEqual(g["pool_bytes"], 4_386_144 * 576 * 2)
        self.assertGreater(g["est_gpu_bytes"], g["pool_bytes"])                       # 작업 버퍼가 풀 위에 얹힌다
        self.assertLessEqual(g["est_gpu_bytes"], 8 * 2**30)
        self.assertLess(g["est_gpu_gib"], 5.5)

    def test_no_dense_case_reaches_the_int32_offset_limit(self) -> None:
        """기존 dense REQUIRED 케이스는 행 수가 한계에 못 미친다 — 그래서 이 구간은 packed 케이스만 덮는다."""
        for name, cfg in shapes.SPARSE_MLA_REQUIRED_CASES.items():
            if "block_stride_rows" in cfg:
                continue
            self.assertFalse(self.h.packed_geometry(cfg)["crosses_int32"], name)

    def test_a_short_context_does_not_reach_the_limit_even_with_a_wide_block_stride(self) -> None:
        small = self.h.packed_geometry({**self.cfg, "ctx": 4096})
        self.assertFalse(small["crosses_int32"])                                      # ctx 4k 는 stride 를 벌려도 한계에 못 미친다


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K12WorkspaceTest(unittest.TestCase):
    """auto 변종의 예약 스크래치(ReservedScratch) — 모자라면 실패하는 음성 대조 + 예약식이 실제 요청을 덮는 양성 대조."""

    @classmethod
    def setUpClass(cls) -> None:
        if str(KIT_ROOT) not in sys.path:
            sys.path.insert(0, str(KIT_ROOT))
        import vendor.dsa_ops.triton_mla_sparse_kernel as kernel
        cls.k = kernel
        cls.h = load_harness()

    def test_reserved_scratch_hands_out_poisoned_views_of_one_reservation(self) -> None:
        h = self.h
        ws = h.ReservedScratch(torch, 100, "cpu")
        first = ws((2, 5, 3))
        self.assertEqual((tuple(first.shape), first.dtype), ((2, 5, 3), torch.float32))
        self.assertTrue(bool(torch.isnan(first).all()))                       # 더러운 workspace: 쓰기 전엔 NaN
        first.fill_(1.0)
        second = ws((4, 25))
        self.assertTrue(bool(torch.isnan(second).all()))                      # 매 호출 다시 오염
        self.assertEqual(first.data_ptr(), second.data_ptr())                 # 한 덩어리 예약의 앞부분 view
        with self.assertRaises(h.ScratchReservationError):
            ws((101,))
        self.assertEqual(ws.requests, [30, 100, 101])
        clean = h.ReservedScratch(torch, 8, "cpu", poison=False)              # 성능 측정용: fill 이 타이밍에 끼지 않는다
        clean.buf.fill_(7.0)
        self.assertTrue(bool((clean((2, 4)) == 7.0).all()))

    def test_one_element_short_reservation_fails_through_the_real_wrapper(self) -> None:
        """예약식이 한 원소라도 모자라면(잠긴 workspace 증설 불가와 동치) 가장 큰 split 요청에서 실패한다. 커널은 뜨기 전에 중단된다."""
        k, h = self.k, self.h
        heads, topk, sm_count = 12, 2048, 108
        bound = k.max_split_mid_out_numel(heads, topk, sm_count)

        class Stop(Exception):
            pass

        def call(reservation: int, tokens: int):
            ws = h.ReservedScratch(torch, reservation, "cpu")

            def getter(shape):
                ws(shape)                       # ScratchReservationError 는 그대로 전파
                raise Stop                      # 예약이 요청을 받아들였다 -> 커널 실행 전에 중단

            q = torch.zeros(tokens, heads, 576, dtype=torch.bfloat16)
            kv = torch.zeros(8, 1, 576, dtype=torch.bfloat16)
            idx = torch.zeros(tokens, 1, topk, dtype=torch.int32)
            k.triton_mla_sparse_attention(q, kv, idx, 0.1, sm_count=sm_count, mid_out_getter=getter)

        def scratch(tokens: int) -> int:
            splits = k._choose_num_kv_splits(tokens, 1, topk, sm_count)
            return tokens * heads * splits * (k._BLOCK_DV + 1) if splits > 1 else 0

        worst = max(range(1, sm_count + 1), key=scratch)
        self.assertEqual(scratch(worst), bound)
        with self.assertRaises(h.ScratchReservationError):
            call(bound - 1, worst)                                            # 음성 대조: 1 원소 부족
        with self.assertRaises(Stop):
            call(bound, worst)                                                # 양성 대조: 정확한 예약은 받아들여진다
        for tokens in range(1, sm_count + 1):                                 # 어떤 토큰 수에서도 정확한 예약이 요청을 덮는다
            if scratch(tokens):
                with self.assertRaises(Stop):
                    call(bound, tokens)

    def test_harness_reservation_uses_the_vendored_bound_for_the_case_shape(self) -> None:
        text = HARNESS.read_text(encoding="utf-8")
        self.assertIn("reserved = max_split_mid_out_numel(heads, topk, sm_count)", text)
        self.assertIn("mid_out_getter=ws if nks is None else None", text)     # auto(프로덕션 디스패치)만 예약 workspace 를 쓴다


def run_cfg_on_cpu_interpreter(cfg: dict, case: str) -> tuple[subprocess.CompletedProcess, dict | None, int | None]:
    """하네스 run_case 를 CPU Triton 인터프리터에서 서브프로세스로 실행 -> (프로세스, payload 요약, peak RSS MB)."""
    prog = """
import os, sys, json, importlib.util, resource
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, __KIT__)
from common import triton_interp
triton_interp.install()
spec = importlib.util.spec_from_file_location("h", __HARNESS__)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
try:
    p = h.run_case(json.loads(__CFG__), __CASE__, "cpu", False, 0, 0)
except RuntimeError as exc:
    if "can't allocate memory" in str(exc):
        print("ALLOC_FAIL"); sys.exit(0)
    raise
print("PAYLOAD", json.dumps({"status": p["status"], "zone": p["extra"]["int32_zone"], "ws": p["extra"]["workspace"],
                             "auto": p["extra"]["auto_num_kv_splits"], "pool": p["extra"]["pool"],
                             "entries": [(e["case"], e["passed"], e["violations"]) for e in p["entries"]]}))
print("PEAK_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
""".replace("__KIT__", repr(str(KIT_ROOT))).replace("__HARNESS__", repr(str(HARNESS))) \
   .replace("__CFG__", repr(json.dumps(cfg))).replace("__CASE__", repr(case))
    res = subprocess.run([sys.executable, "-c", prog], text=True, capture_output=True, check=False, timeout=900)
    payload = next((json.loads(ln[len("PAYLOAD "):]) for ln in res.stdout.splitlines() if ln.startswith("PAYLOAD ")), None)
    peak = next((int(ln[len("PEAK_MB "):]) for ln in res.stdout.splitlines() if ln.startswith("PEAK_MB ")), None)
    return res, payload, peak


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K12PackedRequiredCaseCpuTest(unittest.TestCase):
    """REQUIRED `dec_b1_ctx160k_packed` 를 GPU 대신 CPU 인터프리터로 **그대로**(heads 12 · topk 2048 · 풀 4.7 GiB 가상 메모리) 실행.

    GPU 실행(A100)을 대체하지 않는다 — 케이스 구성(블록 배치·row id·오라클·int32 구간 판정)이 내부적으로 맞는지 호스트에서 미리 본다.
    GPU 전용 분기는 `pool.fill_(POISON)` 뿐이다(CPU 는 가상 메모리를 건드리지 않으려 생략).
    """

    def test_required_case_runs_on_the_cpu_interpreter_and_reads_rows_past_the_limit(self) -> None:
        res, payload, peak = run_cfg_on_cpu_interpreter(dict(shapes.SPARSE_MLA_REQUIRED_CASES["dec_b1_ctx160k_packed"]),
                                                        "dec_b1_ctx160k_packed")
        if "ALLOC_FAIL" in res.stdout:
            self.skipTest("호스트가 4.7 GiB 가상 메모리 예약을 허용하지 않음")
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertEqual(payload["status"], "PASS", payload)
        zone = payload["zone"]
        self.assertGreater(zone["selected_rows_past_int32_limit"], 100)       # top-k 2048 중 상당수가 한계 너머 (평균 ≈ 300)
        self.assertGreaterEqual(zone["max_selected_offset_elems"], 2**31)
        self.assertEqual(payload["auto"], 16)                                  # decode b1 -> 16 splits -> 예약 workspace 경로
        self.assertEqual(len(payload["ws"]["auto_requests"]), 1)
        self.assertTrue(payload["ws"]["bound_covers_requests"])
        self.assertEqual([c for c, ok, _ in payload["entries"] if not ok], [])
        self.assertEqual(sorted(c.split("/")[-1] for c, _, _ in payload["entries"]),
                         ["int32_zone", "out[auto]", "out[splits=1]", "out[splits=4]"])
        self.assertLess(peak, 3000)                                            # 풀은 가상 메모리, 사용 행만 채운다

    def test_a_packed_configuration_that_never_reads_past_the_limit_is_reported_vacuous(self) -> None:
        """블록 stride 만 벌리고 풀이 작아 int32 한계 너머를 하나도 안 읽는 구성 -> 수치가 맞아도 케이스는 FAIL (공회전 방지)."""
        cfg = dict(mode="decode", b=1, ctx=330, heads=4, topk=64, block_stride_rows=shapes.DSA_PACKED_BLOCK_STRIDE_ROWS)
        res, payload, _ = run_cfg_on_cpu_interpreter(cfg, "vacuous_packed")
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertEqual(payload["status"], "FAIL", payload)
        zone_entry = next(e for e in payload["entries"] if e[0].endswith("/int32_zone"))
        self.assertFalse(zone_entry[1])
        self.assertTrue(any("공회전" in v for v in zone_entry[2]), zone_entry)
        numeric = [e for e in payload["entries"] if "/out[" in e[0]]
        self.assertTrue(numeric and all(ok for _, ok, _ in numeric), numeric)   # 수치 자체는 맞다 — 실패 원인은 구간 미도달뿐
        control = dict(cfg, require_past_int32=False)                          # 대조군: 점검을 끄면 같은 구성이 PASS
        res, payload, _ = run_cfg_on_cpu_interpreter(control, "vacuous_packed_control")
        self.assertEqual(payload["status"], "PASS", payload)


if __name__ == "__main__":
    unittest.main()
