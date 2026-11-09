"""k11_dsa_indexer_sm80 계약 테스트 — 등록 지점 · vendor 소스 · 판정 로직 · dry-run 상한 · (선택) CPU 인터프리터 대조.

호스트 python 만으로 도는 항목이 기본이고, torch(+triton) 이 있는 환경(/home/jimmy/.venvs/k31-cpu)에서는
판정 음성 대조와 `--interpret` 서브프로세스 대조가 추가로 돈다(없으면 skip).
"""

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
dsa_ref = import_common("dsa_ref")
dsa_gate = import_common("dsa_gate")

KID = "k11_dsa_indexer_sm80"
HARNESS = KIT_ROOT / "k11_dsa_indexer_sm80" / "harness.py"
VENDOR = KIT_ROOT / "vendor" / "dsa_ops" / "mqa_logits_triton.py"

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_harness():
    spec = importlib.util.spec_from_file_location("k11_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load k11 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_torch_and_triton() -> bool:
    return torch is not None and importlib.util.find_spec("triton") is not None


class K11RegistrationTest(unittest.TestCase):
    def test_k31_dsa_constants_come_from_recon(self) -> None:
        self.assertEqual(
            (shapes.DSA_INDEX_HEADS, shapes.DSA_INDEX_HEAD_DIM, shapes.DSA_TOPK, shapes.DSA_LAYERS,
             shapes.DSA_INDEX_K_BYTES, shapes.DSA_INDEX_SKIP_TOPK_OFFSET, shapes.DSA_INDEX_ROPE_DIM),
            (32, 128, 2048, 24, 132, 1, 0))
        self.assertEqual(shapes.DSA_LAYERS, shapes.MLA_LAYERS)
        self.assertEqual(shapes.DSA_INDEX_K_BYTES, shapes.DSA_INDEX_HEAD_DIM + 4)

    def test_required_shapes_cover_decode_and_prefill_at_8k_128k_256k(self) -> None:
        required = shapes.REQUIRED_SHAPES[KID]
        self.assertEqual(
            set(required),
            {"dec_b1_ctx8k", "dec_b8_ctx8k", "dec_b1_ctx128k", "dec_b8_ctx128k", "dec_b1_ctx256k",
             "dec_b8_ctx256k", "dec_b8_ctx256k_pool27x", "pre_m256_n8k", "pre_m256_n128k", "pre_m256_n256k",
             "pre_varlen_m256_n8k"})
        self.assertEqual({c["ctx"] for c in required.values() if c["mode"] == "decode"}, {8192, 131072, 262144})
        self.assertEqual({c["b"] for c in required.values() if c["mode"] == "decode"}, {1, 8})
        first = next(iter(required.values()))
        model = shapes.perf_bytes_model(KID, **first)          # 첫 케이스가 perf 모델 입력 키를 가져야 한다
        self.assertGreater(model["bytes"], 0)

    def test_perf_bytes_model_streams_132_bytes_per_token(self) -> None:
        small = shapes.perf_bytes_model(KID, mode="decode", b=1, ctx=8192, next_n=1)
        big = shapes.perf_bytes_model(KID, mode="decode", b=1, ctx=262144, next_n=1)
        per_token = (big["bytes"] - small["bytes"]) / (262144 - 8192)
        # 토큰당 증가분 = index-K 132 B + logits fp32 쓰기 4 B
        self.assertAlmostEqual(per_token, shapes.DSA_INDEX_K_BYTES + 4, places=6)
        self.assertEqual(shapes.perf_bytes_model(KID, mode="decode", b=8, ctx=8192, next_n=1)["macs"],
                         8 * 8192 * shapes.DSA_INDEX_HEADS * shapes.DSA_INDEX_HEAD_DIM)

    def test_registered_in_framework_tables_and_runner(self) -> None:
        self.assertIn(KID, shapes.KERNEL_IDS)
        self.assertIn(KID, shapes.KERNEL_TITLES)
        policy = timing.PERF_POLICY[KID]
        self.assertEqual(policy["mode"], "measure_only")          # P1 정직한 상한: 속도 미보증
        self.assertLessEqual(policy["cv_max"], 0.15)
        self.assertIn(KID, tolerance.TOL)
        self.assertIn(KID, (KIT_ROOT / "run_all.sh").read_text(encoding="utf-8"))

    def test_relay_filter_includes_k10_plus_reports(self) -> None:
        relay = import_common("relay")
        self.assertTrue(relay._REPORT_FILE.match("k11_dsa_indexer_sm80.json"))
        self.assertTrue(relay._REPORT_FILE.match("k12_sparse_mla_sm80.json"))
        self.assertFalse(relay._REPORT_FILE.match("env.json"))
        self.assertFalse(relay._REPORT_FILE.match("summary.json"))


class K11VendorSourceTest(unittest.TestCase):
    def test_vendor_is_the_deployed_patch_copy_with_no_vllm_import(self) -> None:
        # 이미지에 실리는 커널 = 핀 a3c14efb + triton-mla-sparse-sm80.patch (PR 원본이 아니다) — 계약 전체는 test_dsa_ops_vendor.py
        source = VENDOR.read_text(encoding="utf-8")
        head = "\n".join(source.splitlines()[:4])
        for token in ("a3c14efbc77c018e8bf579935095a87721349e05", "docker/patches-k31/triton-mla-sparse-sm80.patch",
                      "sha256 74675e998be0022aa4caed7df4467b62148fad09a6af9e47d13ea1b302fb7ed8", "PR #47629",
                      "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d", "#38476", "haosdent/thomaslwang", "Apache-2.0"):
            self.assertIn(token, head)
        self.assertGreaterEqual(source.count("to(tl.int64)"), 4)         # PR 원본(1개)에 더한 int64 승격(token_id·m·n_start)
        self.assertIn("def fp8_paged_mqa_logits_triton", source)
        self.assertIn("def fp8_mqa_logits_triton", source)
        self.assertIn("_decode_e4m3fn_bf16_lut", source)         # K 를 uint8 LUT 로 in-kernel 디코드
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                self.assertNotEqual((node.module or "").split(".")[0], "vllm")
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name.split(".")[0] != "vllm" for a in node.names))


class K11ContractTest(unittest.TestCase):
    def test_dry_run_never_claims_gpu_pass(self) -> None:
        payload = load_harness().build_dry_report()
        self.assertEqual(payload["id"], KID)
        self.assertEqual(payload["overall"], "INCOMPLETE")
        self.assertTrue(all(g["status"] == "SKIP" for g in payload["gates"].values()))
        self.assertEqual(payload["evidence"]["dry_run"]["selfcheck"], "PASS")
        self.assertEqual(payload["evidence"]["topk_policy"]["agreement_min"], dsa_ref.TOPK_AGREEMENT_MIN)
        kernel = payload["evidence"]["kernel"]                              # 리포트가 '어느 커널을 판정했는지'를 스스로 말한다
        self.assertIn("triton-mla-sparse-sm80.patch", kernel["module"])
        entry = kernel["files"]["mqa_logits_triton.py"]
        self.assertGreaterEqual(entry["to_int64_markers"], 4)
        self.assertRegex(entry["header_patch_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["sha256"], __import__("hashlib").sha256(VENDOR.read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--cpu-dry-run", "--report-dir", tmp],
                                 text=True, capture_output=True, check=False)
            self.assertEqual(res.returncode, 0, res.stderr)
            written = json.loads((Path(tmp) / f"{KID}.json").read_text())
        self.assertIn("K11-OVERALL INCOMPLETE", res.stdout)
        self.assertEqual(written["overall"], "INCOMPLETE")

    def test_gpu_mode_without_gpu_exits_precondition_not_fail(self) -> None:
        if torch is not None and torch.cuda.is_available():
            self.skipTest("GPU 가 있는 호스트")
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--report-dir", tmp], text=True,
                                 capture_output=True, check=False)
        self.assertEqual(res.returncode, 2, res.stdout + res.stderr)   # EXIT_PRECOND — FAIL(1) 이 아니다

    def test_docstring_pins_topk_threshold_and_band(self) -> None:
        doc = load_harness().__doc__
        for token in ("TOPK_AGREEMENT_MIN", "TOPK_TIE_BAND_REL", "경계 동률", "측정 전용", "PASS 근거 아님"):
            self.assertIn(token, doc)
        self.assertEqual(dsa_ref.TOPK_TIE_BAND_REL, 1e-3)
        self.assertEqual(dsa_ref.TOPK_AGREEMENT_MIN, 1.0)

    def test_assemble_report_gates_from_child_payloads(self) -> None:
        h = load_harness()
        import types
        verdict = str(KIT_ROOT.parent / "phase0" / "verdict.json")
        args = types.SimpleNamespace(phase0_verdict=verdict, twin=None)

        def child(name, ok=True, cv=0.02):
            return {"case": name, "status": "PASS" if ok else "FAIL", "classification": "PASS" if ok else "FAIL",
                    "returncode": 0 if ok else 1, "duration_s": 1.0, "log_tail": "",
                    "result": {"shape": shapes.REQUIRED_SHAPES[KID][name], "status": "PASS" if ok else "FAIL",
                               "entries": [{"case": f"{name}/logits", "metrics": {"rmse_rel": 1e-7 if ok else 0.3,
                                                                                  "max_abs": 1e-6},
                                            "violations": [] if ok else ["rmse_rel 0.3 > 1e-4"], "passed": ok}],
                               "timing": {"median_us": 40.0, "mean_us": 40.0, "min_us": 39.0, "p90_us": 41.0,
                                          "cv": cv, "reps": 50}, "extra": {}}}

        san_ok = ("PASS", "memcheck ERROR SUMMARY: 0", {})
        all_ok = [child(n) for n in h.CASE_ORDER]
        rep = dsa_gate.assemble_report(KID, h.PINS, all_ok, san_ok, h.PERF_CASE, args, {})
        self.assertEqual(rep.overall(), "PASS", {g: v["status"] for g, v in rep.gates.items()})
        # 케이스 누락 → SHAPES FAIL, 수치 위반 → NUMERIC FAIL, 불안정 측정 → PERF FAIL, sanitizer SKIP → INCOMPLETE
        missing = dsa_gate.assemble_report(KID, h.PINS, all_ok[:-1], san_ok, h.PERF_CASE, args, {})
        self.assertEqual(missing.gates["SHAPES"]["status"], "FAIL")
        bad = [child(n, ok=(n != "dec_b1_ctx8k")) for n in h.CASE_ORDER]
        self.assertEqual(dsa_gate.assemble_report(KID, h.PINS, bad, san_ok, h.PERF_CASE, args, {})
                         .gates["NUMERIC"]["status"], "FAIL")
        noisy = [child(n, cv=0.5 if n == h.PERF_CASE else 0.02) for n in h.CASE_ORDER]
        self.assertEqual(dsa_gate.assemble_report(KID, h.PINS, noisy, san_ok, h.PERF_CASE, args, {})
                         .gates["PERF"]["status"], "FAIL")
        skipped = dsa_gate.assemble_report(KID, h.PINS, all_ok, ("SKIP", "sanitizer 미발견", {}), h.PERF_CASE, args, {})
        self.assertEqual(skipped.overall(), "INCOMPLETE")
        crashed = [dict(c) for c in all_ok]
        crashed[0] = {"case": h.CASE_ORDER[0], "status": "FAIL", "classification": "MEMORY_VIOLATION",
                      "returncode": 1, "duration_s": 1.0, "log_tail": "illegal memory access", "result": None}
        rc = dsa_gate.assemble_report(KID, h.PINS, crashed, san_ok, h.PERF_CASE, args, {})
        self.assertEqual((rc.gates["NUMERIC"]["status"], rc.gates["SHAPES"]["status"]), ("FAIL", "FAIL"))

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

    def test_spawn_case_classifies_memory_violation_and_pass(self) -> None:
        script = (
            "import argparse, json, sys\n"
            "ap = argparse.ArgumentParser(); ap.add_argument('--child-case'); ap.add_argument('--child-out')\n"
            "ap.add_argument('--warmup'); ap.add_argument('--reps'); a = ap.parse_args()\n"
            "if a.child_case == 'boom':\n"
            "    print('Invalid __global__ read of size 16 bytes', file=sys.stderr); sys.exit(1)\n"
            "import os; os.makedirs(os.path.dirname(a.child_out), exist_ok=True)\n"
            "open(a.child_out, 'w').write(json.dumps({'status': 'PASS', 'entries': []}))\n")
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_harness.py"
            fake.write_text(script)
            ok = dsa_gate.spawn_case(KID, fake, "fine", Path(tmp), 60, ["--warmup", "0", "--reps", "0"])
            bad = dsa_gate.spawn_case(KID, fake, "boom", Path(tmp), 60, ["--warmup", "0", "--reps", "0"])
        self.assertEqual((ok["status"], ok["classification"]), ("PASS", "PASS"))
        self.assertEqual((bad["status"], bad["classification"]), ("FAIL", "MEMORY_VIOLATION"))


class K11SanitizerGateTest(unittest.TestCase):
    """compute-sanitizer 게이트 배선: 가짜 compute-sanitizer 로 PASS / 위반 FAIL / 정상 allocator 재시도 PASS / 미발견 SKIP."""

    FAKE = ("#!/bin/sh\n# fake: compute-sanitizer --tool memcheck <cmd...>\nshift 2\n\"$@\"; rc=$?\n"
            "[ -n \"$FAKE_PRELUDE\" ] && echo \"$FAKE_PRELUDE\"\n"
            "echo \"========= ERROR SUMMARY: ${FAKE_ERRORS:-0} errors\"\nexit $rc\n")
    CHILD = ("import argparse, os, json\nap = argparse.ArgumentParser(); ap.add_argument('--child-case')\n"
             "ap.add_argument('--child-out'); ap.add_argument('--warmup'); ap.add_argument('--reps'); a = ap.parse_args()\n"
             "os.makedirs(os.path.dirname(a.child_out), exist_ok=True)\n"
             "open(a.child_out, 'w').write(json.dumps({'status': 'PASS', 'entries': []}))\n")

    def run_gate(self, errors="0", prelude="", sanitizer_present=True):
        import os
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "compute-sanitizer"
            fake.write_text(self.FAKE)
            fake.chmod(0o755)
            harness = Path(tmp) / "child.py"
            harness.write_text(self.CHILD)
            env_check = import_common("env_check")
            found = str(fake) if sanitizer_present else None
            with mock.patch.object(env_check, "find_compute_sanitizer", return_value=found), \
                 mock.patch.dict(os.environ, {"FAKE_ERRORS": errors, "FAKE_PRELUDE": prelude}):
                return dsa_gate.sanitizer_gate(KID, harness, ["san_a", "san_b"], Path(tmp), 60,
                                               passthrough=["--warmup", "0", "--reps", "0"])

    def test_clean_run_passes_all_cases(self) -> None:
        status, reason, ev = self.run_gate()
        self.assertEqual(status, "PASS", reason)
        self.assertEqual([r["case"] for r in ev["runs"]], ["san_a", "san_b"])

    def test_memcheck_errors_fail(self) -> None:
        status, reason, _ = self.run_gate(errors="3")
        self.assertEqual(status, "FAIL")
        self.assertIn("san_a", reason)

    def test_allocator_oom_retry_is_not_a_memory_violation(self) -> None:
        # 사내 실측(2026-07-30): sanitizer 오버헤드 하 cudaMalloc OOM 을 PyTorch allocator 가 재시도하는 것은 위반이 아니다
        status, reason, _ = self.run_gate(
            errors="2", prelude="========= Program hit cudaErrorMemoryAllocation (error 2) on CUDA API call to cudaMalloc.")
        self.assertEqual(status, "PASS", reason)
        status, reason, _ = self.run_gate(
            errors="2", prelude="========= Invalid __global__ read of size 16 bytes\n=========     at kernel+0x10")
        self.assertEqual(status, "FAIL", reason)

    def test_missing_sanitizer_is_skip_with_reason_not_pass(self) -> None:
        status, reason, _ = self.run_gate(sanitizer_present=False)
        self.assertEqual(status, "SKIP")
        self.assertTrue(reason)


@unittest.skipUnless(torch is not None, "torch 필요")
class K11JudgeNegativeControlTest(unittest.TestCase):
    """판정 로직이 살아있는지: 통과 입력은 통과, 주입한 오류는 반드시 잡힌다 (GPU/triton 불필요)."""

    def setUp(self) -> None:
        self.h = load_harness()
        g = torch.Generator().manual_seed(5)
        self.ref = torch.randn(6, 300, generator=g)
        self.ref[:, 260:] = float("-inf")
        self.limit = torch.full((6,), 300)

    def test_logits_compare_accepts_fp32_noise_and_rejects_injected_errors(self) -> None:
        h = self.h
        self.assertTrue(h._compare_logits(torch, self.ref, self.ref.clone(), None, "same").passed)
        noise = self.ref.clone()
        noise[torch.isfinite(noise)] *= 1 + 1e-7
        self.assertTrue(h._compare_logits(torch, self.ref, noise, None, "noise").passed)
        off = self.ref.clone()
        off[0, 3] += 0.5                                           # 한 칸 오차 → max_abs 위반
        self.assertFalse(h._compare_logits(torch, self.ref, off, None, "off").passed)
        leak = self.ref.clone()
        leak[1, 270] = 0.0                                         # 마스크(-inf) 자리에 값 → 마스크 불일치
        res = h._compare_logits(torch, self.ref, leak, None, "leak")
        self.assertFalse(res.passed)
        self.assertEqual(res.metrics["mask_diff"], 1)
        nan = self.ref.clone()
        nan[2, 5] = float("nan")
        self.assertFalse(h._compare_logits(torch, self.ref, nan, None, "nan").passed)
        # limit 밖(미정의 tail)은 보지 않는다 — decode clean_logits=False 계약
        garbage_tail = self.ref.clone()
        garbage_tail[:, 280:] = 1e30
        lim = torch.full((6,), 270)
        self.assertTrue(h._compare_logits(torch, self.ref, garbage_tail, lim, "tail").passed)

    def test_topk_entry_detects_definite_swaps_but_tolerates_ties(self) -> None:
        h = self.h
        ok = h._topk_entry(torch, self.ref, self.ref.clone(), None, 50, "same")
        self.assertTrue(ok["passed"])
        bad = self.ref.clone()
        top = int(torch.argmax(torch.where(torch.isfinite(bad[0]), bad[0], torch.tensor(-1e9))))
        bad[0, top] = float("-inf")                                # 최상위 하나를 놓침 → 확정 불일치
        entry = h._topk_entry(torch, self.ref, bad, None, 50, "swap")
        self.assertFalse(entry["passed"])
        self.assertGreaterEqual(entry["metrics"]["mismatch_total"], 1)


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K11InterpretCpuTest(unittest.TestCase):
    def test_interpret_runs_vendored_kernels_and_keeps_gates_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, str(HARNESS), "--interpret", "--report-dir", tmp],
                                 text=True, capture_output=True, check=False, timeout=1200)
            self.assertEqual(res.returncode, 0, res.stdout[-2000:] + res.stderr[-2000:])
            rep = json.loads((Path(tmp) / f"{KID}.json").read_text())
        self.assertEqual(rep["overall"], "INCOMPLETE")                       # CPU 대조는 PASS 근거가 아니다
        self.assertTrue(all(g["status"] == "SKIP" for g in rep["gates"].values()))
        ic = rep["evidence"]["interpret_cpu"]
        self.assertEqual(ic["overall"], "PASS")
        self.assertEqual(len(ic["cases"]), 5)
        for case in ic["cases"]:
            self.assertEqual(case["status"], "PASS", case)
            for entry in case["entries"]:
                self.assertTrue(entry["passed"], entry)


@unittest.skipUnless(has_torch_and_triton(), "torch+triton 필요")
class K11KernelOnlyModeTest(unittest.TestCase):
    """sanitizer child 가 쓰는 kernel_only 경로가 (오라클 없이) 커널 두 모드를 실행하는지 — 인터프리터로 확인."""

    def test_kernel_only_executes_both_clean_logits_modes(self) -> None:
        prog = f"""
import os, sys, importlib.util
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, {str(KIT_ROOT)!r})
from common import triton_interp
triton_interp.install()
spec = importlib.util.spec_from_file_location("h", {str(HARNESS)!r})
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
for cfg in (dict(mode="decode", b=1, ctx=130, next_n=2, topk=32), dict(mode="prefill", m=4, n=70, varlen=True, topk=32)):
    p = h.run_case(cfg, "kernel_only", "cpu", False, 0, 0, kernel_only=True)
    print("KERNEL_ONLY", p["status"], p["entries"], p["extra"]["kernel_only"])
"""
        res = subprocess.run([sys.executable, "-c", prog], text=True, capture_output=True, check=False, timeout=600)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertEqual(res.stdout.count("KERNEL_ONLY PASS [] True"), 2, res.stdout)


if __name__ == "__main__":
    unittest.main()
