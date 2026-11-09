from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.tooling.gates import (
    GATE_MATRIX,
    PREREQUISITE_GATES,
    _REPORT_NAMES,
    _c01_command,
    _gate_timeout,
    _generic_gate_commands,
    load_context,
    run_gate,
)
from deploy.tooling.lane_specs import LANES
from deploy.tooling.render_lanes import render_all


EXPECTED = {
    "01": ("k08", "kda", "pp"),
    "02": ("k08", "c01", "ep"),
    "03": ("k08", "c01", "k10", "ep", "dspark"),
    "04": ("k08", "c01", "k09", "ep", "1m"),
    "05": ("k08", "c01", "k09", "k10", "ep", "1m", "dspark"),
    "11": ("k08", "kda", "k11", "k12", "pp"),
}
K31_LANE_DIR = "11-k31-tp8-pp4-256k-safe"


class GateMatrixTest(unittest.TestCase):
    def test_gate_matrix_is_exact(self) -> None:
        self.assertEqual(GATE_MATRIX, EXPECTED)

    def test_prerequisite_subset_contains_only_pre_service_gates(self) -> None:
        for lane, prerequisites in PREREQUISITE_GATES.items():
            self.assertTrue(set(prerequisites) <= set(GATE_MATRIX[lane]))
            self.assertFalse(
                set(prerequisites) & {"pp", "ep", "1m", "dspark"},
                lane,
            )

    def test_makefiles_expose_gate_lifecycle_and_exact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                makefile = (root / spec.id / "Makefile").read_text()
                expected = " ".join(EXPECTED[f"{spec.number:02d}"])
                self.assertIn(f"REQUIRED_GATES := {expected}", makefile)
                for target in (
                    "gates:",
                    "gate-load:",
                    "gate-context:",
                    "gate-dspark:",
                    "calibrate:",
                ):
                    self.assertIn(target, makefile, spec.id)
                for target in ("gate-k11:", "gate-k12:"):
                    if spec.is_k31:
                        self.assertIn(target, makefile, spec.id)
                    else:
                        self.assertNotIn(target, makefile, spec.id)

    def test_fp8_lanes_opt_in_and_all_lanes_force_triton_kda(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for spec in LANES:
                env = (root / spec.id / ".env").read_text()
                expected = "1" if spec.fp8_kv else "0"
                self.assertIn(f"VLLM_K3_SM80_FP8_KV={expected}\n", env)
                self.assertIn("VLLM_KDA_PREFILL_BACKEND=triton\n", env)
                self.assertIn("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0\n", env)
                self.assertIn("VLLM_MARLIN_MXFP8_INPUT_QDQ=0\n", env)
                self.assertIn(
                    "VLLM_ATTENTION_CONFIG="
                    '\'{"use_prefill_query_quantization":false}\'\n',
                    env,
                )
                expected_skip = (
                    "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS='93 94 95 96 97'\n"
                    if spec.number == 5
                    else "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS=\n"
                )
                self.assertIn(expected_skip, env)

    def test_standalone_gpu_gates_run_inside_serving_image(self) -> None:
        lane = (
            Path(__file__).resolve().parents[2]
            / "deploy"
            / "01-tp8-pp4-256k-safe"
        )
        context = load_context(lane, require_site_values=False)
        report_dir = context.report_dir("k08")

        for gate in ("k08", "kda"):
            for _name, argv, _report in _generic_gate_commands(
                context,
                gate,
                report_dir,
                cpu_dry_run=True,
            ):
                self.assertEqual(argv[:3], ["docker", "run", "--rm"])
                self.assertIn("--gpus", argv)
                self.assertIn(context.env["K3_IMAGE"], argv)
                self.assertEqual(argv[argv.index("--entrypoint") + 1], "python3")
                self.assertTrue(any(item.startswith("/work/kernel-kit/") for item in argv))

    def test_k31_prerequisites_are_the_two_new_kernel_gates_after_k08_and_kda(self) -> None:
        self.assertEqual(PREREQUISITE_GATES["11"], ("k08", "kda", "k11", "k12"))
        self.assertEqual(GATE_MATRIX["11"][: len(PREREQUISITE_GATES["11"])], PREREQUISITE_GATES["11"])
        self.assertEqual(GATE_MATRIX["11"][-1], "pp")
        self.assertEqual(
            {key: _REPORT_NAMES[key] for key in ("k11", "k12")},
            {
                "k11": "k11_dsa_indexer_sm80.json",
                "k12": "k12_sparse_mla_sm80.json",
            },
        )

    def test_k31_kernel_gates_run_kernel_kit_harness_in_the_k31_image(self) -> None:
        lane = Path(__file__).resolve().parents[2] / "deploy" / K31_LANE_DIR
        context = load_context(lane, require_site_values=False)
        harness = {
            "k11": "k11_dsa_indexer_sm80",
            "k12": "k12_sparse_mla_sm80",
        }
        for gate, directory in harness.items():
            with self.subTest(gate=gate):
                report_dir = context.report_dir(gate)
                ((name, argv, report),) = _generic_gate_commands(
                    context,
                    gate,
                    report_dir,
                    cpu_dry_run=False,
                )
                self.assertEqual(name, gate)
                self.assertEqual(argv[:3], ["docker", "run", "--rm"])
                self.assertIn("--gpus", argv)
                self.assertEqual(argv[argv.index("--entrypoint") + 1], "python3")
                self.assertIn(context.env["K3_IMAGE"], argv)
                self.assertEqual(
                    context.env["K3_IMAGE"], "k31-sm80-openai:a3c14efb-sm80-v1"
                )
                self.assertIn(f"/work/kernel-kit/{directory}/harness.py", argv)
                self.assertEqual(
                    argv[argv.index("--report-dir") + 1], str(report_dir)
                )
                self.assertNotIn("--cpu-dry-run", argv)
                self.assertEqual(report, report_dir / f"{directory}.json")

                ((_, dry_argv, _),) = _generic_gate_commands(
                    context,
                    gate,
                    report_dir,
                    cpu_dry_run=True,
                )
                self.assertEqual(dry_argv[-1], "--cpu-dry-run")

    def test_k08_and_k09_harness_paths_are_unchanged_by_the_k11_k12_refactor(self) -> None:
        lane = Path(__file__).resolve().parents[2] / "deploy" / K31_LANE_DIR
        context = load_context(lane, require_site_values=False)
        for gate, directory in (
            ("k08", "k08_mxfp4_marlin_moe"),
            ("k09", "k09_mla_fp8_kv_sm80"),
        ):
            with self.subTest(gate=gate):
                report_dir = context.report_dir(gate)
                ((_, argv, report),) = _generic_gate_commands(
                    context,
                    gate,
                    report_dir,
                    cpu_dry_run=False,
                )
                self.assertIn(f"/work/kernel-kit/{directory}/harness.py", argv)
                self.assertEqual(report, report_dir / _REPORT_NAMES[gate])

    def test_unregistered_gates_are_rejected_not_silently_accepted(self) -> None:
        lane = Path(__file__).resolve().parents[2] / "deploy" / K31_LANE_DIR
        context = load_context(lane, require_site_values=False)
        for gate in ("k13", "pp", "c01"):
            with self.subTest(gate=gate):
                with self.assertRaisesRegex(ValueError, "not a local standalone gate"):
                    _generic_gate_commands(
                        context,
                        gate,
                        context.report_dir(gate),
                        cpu_dry_run=True,
                    )

    def test_k11_k12_timeouts_follow_the_k09_precedent(self) -> None:
        lane = Path(__file__).resolve().parents[2] / "deploy" / K31_LANE_DIR
        context = load_context(lane, require_site_values=False)
        configured = int(context.env["K3_EXECUTE_MODEL_TIMEOUT_S"])

        for gate in ("k09", "k11", "k12"):
            with self.subTest(gate=gate):
                self.assertEqual(_gate_timeout(context, gate), max(configured, 7_200))
        with self.assertRaises(KeyError):
            _gate_timeout(context, "k13")

    def test_c01_runs_one_eight_rank_container_per_node(self) -> None:
        lane = (
            Path(__file__).resolve().parents[2]
            / "deploy"
            / "02-tp8-dp4-ep32-256k-safe"
        )
        context = load_context(lane, require_site_values=False)
        report_dir = context.report_dir("c01")

        for rank, profile in enumerate(("head", "worker1", "worker2", "worker3")):
            argv = _c01_command(context, profile, rank, report_dir)
            joined = " ".join(argv)
            self.assertIn("docker run", joined)
            self.assertIn("c01_ep32_a2a/launch.sh", joined)
            self.assertIn(f"NODE_RANK={rank}", joined)
            self.assertIn("NNODES=4", joined)

    def test_gpu_gate_holds_and_releases_global_cluster_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            context = load_context(
                root / "01-tp8-pp4-256k-safe",
                require_site_values=False,
            )
            with (
                patch("deploy.tooling.gates._acquire_cluster_lock") as acquire,
                patch("deploy.tooling.gates._release_cluster_lock") as release,
                patch(
                    "deploy.tooling.gates._run_gate_locked",
                    return_value=True,
                ) as locked,
            ):
                self.assertTrue(run_gate(context, "k08"))

            acquire.assert_called_once()
            release.assert_called_once_with(context.lane_path, context.env)
            locked.assert_called_once_with(context, "k08", cpu_dry_run=False)

    def test_cpu_dry_run_does_not_mutate_cluster_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            context = load_context(
                root / "01-tp8-pp4-256k-safe",
                require_site_values=False,
            )
            with (
                patch("deploy.tooling.gates._acquire_cluster_lock") as acquire,
                patch(
                    "deploy.tooling.gates._run_gate_locked",
                    return_value=False,
                ),
            ):
                self.assertFalse(run_gate(context, "k08", cpu_dry_run=True))

            acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
