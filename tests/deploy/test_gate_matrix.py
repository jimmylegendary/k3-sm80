from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.tooling.gates import (
    GATE_MATRIX,
    PREREQUISITE_GATES,
    _c01_command,
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
}


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
