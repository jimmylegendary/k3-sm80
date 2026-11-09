from __future__ import annotations

import unittest
from pathlib import Path

from deploy.tooling.command import build_vllm_args, validate_extra_args
from deploy.tooling.lane_schema import LaneEnv


REPO_ROOT = Path(__file__).resolve().parents[2]


def contains_in_order(values: list[str], expected: list[str]) -> bool:
    index = -1
    for value in expected:
        try:
            index = values.index(value, index + 1)
        except ValueError:
            return False
    return True


class VllmCommandTest(unittest.TestCase):
    def load_lane(self, lane_id: str) -> LaneEnv:
        return LaneEnv.load(REPO_ROOT / "deploy" / lane_id / ".env")

    def test_lane_01_mp_command(self) -> None:
        env = self.load_lane("01-tp8-pp4-256k-safe")

        args = build_vllm_args(env, node_rank=3)

        self.assertTrue(
            contains_in_order(
                args,
                [
                    "moonshotai/Kimi-K3",
                    "--distributed-executor-backend",
                    "mp",
                    "--data-parallel-backend",
                    "mp",
                    "--nnodes",
                    "4",
                    "--node-rank",
                    "3",
                    "--tensor-parallel-size",
                    "8",
                    "--pipeline-parallel-size",
                    "4",
                    "--data-parallel-size",
                    "1",
                    "--max-model-len",
                    "262144",
                    "--max-num-seqs",
                    "1",
                ],
            )
        )
        self.assertNotIn("--enable-expert-parallel", args)
        self.assertNotIn("--speculative-config", args)

    def test_lane_05_has_ep_fp8_and_dspark(self) -> None:
        env = self.load_lane("05-tp8-dp4-ep32-1m-fp8kv-dspark")

        args = build_vllm_args(env, node_rank=0)

        self.assertIn("--enable-expert-parallel", args)
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
        self.assertEqual(args[args.index("--max-model-len") + 1], "1048576")
        self.assertEqual(
            args[args.index("--speculative-config") + 1],
            env.values["VLLM_SPECULATIVE_CONFIG"],
        )
        self.assertEqual(
            args[args.index("--data-parallel-rpc-port") + 1],
            env.values["VLLM_DP_RPC_PORT"],
        )
        skip_index = args.index("--kv-cache-dtype-skip-layers")
        self.assertEqual(args[skip_index + 1 : skip_index + 6], ["93", "94", "95", "96", "97"])

    def test_lane_11_k31_command(self) -> None:
        env = self.load_lane("11-k31-tp8-pp4-256k-safe")

        for node_rank in range(4):
            with self.subTest(node_rank=node_rank):
                args = build_vllm_args(env, node_rank=node_rank)

                self.assertEqual(args[0], "moonshotai/Kimi-K3.1-Pro")
                self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "bfloat16")
                self.assertEqual(
                    args[args.index("--attention-backend") + 1], "TRITON_MLA_SPARSE"
                )
                self.assertEqual(
                    args[args.index("--sparse-indexer-topk-backend") + 1], "persistent"
                )
                self.assertEqual(
                    args[args.index("--served-model-name") + 1], "kimi-k31-pro"
                )
                self.assertEqual(
                    args[args.index("--pipeline-parallel-size") + 1], "4"
                )
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "8")
                self.assertEqual(args[args.index("--data-parallel-size") + 1], "1")
                self.assertEqual(args[args.index("--max-model-len") + 1], "262144")
                self.assertEqual(args[args.index("--max-num-seqs") + 1], "1")
                self.assertEqual(args[args.index("--tool-call-parser") + 1], "kimi_k3")
                self.assertEqual(args[args.index("--reasoning-parser") + 1], "kimi_k3")
                self.assertEqual(args[args.index("--quantization") + 1], "mxfp4")
                for flag in (
                    "--language-model-only",
                    "--trust-remote-code",
                    "--enforce-eager",
                ):
                    self.assertIn(flag, args)
                for flag in (
                    "--speculative-config",
                    "--kv-cache-dtype-skip-layers",
                    "--enable-expert-parallel",
                    "--enable-prefix-caching",
                ):
                    self.assertNotIn(flag, args)
                self.assertEqual("--headless" in args, node_rank > 0)

    def test_k3_lanes_never_emit_the_sparse_indexer_flag(self) -> None:
        for lane_id in (
            "01-tp8-pp4-256k-safe",
            "02-tp8-dp4-ep32-256k-safe",
            "03-tp8-dp4-ep32-256k-dspark",
            "04-tp8-dp4-ep32-1m-fp8kv",
            "05-tp8-dp4-ep32-1m-fp8kv-dspark",
        ):
            with self.subTest(lane=lane_id):
                args = build_vllm_args(self.load_lane(lane_id), node_rank=0)
                self.assertNotIn("--sparse-indexer-topk-backend", args)
                self.assertEqual(
                    args[args.index("--attention-backend") + 1], "TRITON_MLA"
                )

    def test_sparse_indexer_flag_follows_the_env_value(self) -> None:
        env = self.load_lane("11-k31-tp8-pp4-256k-safe")
        env.values["VLLM_SPARSE_INDEXER_TOPK_BACKEND"] = "auto"

        args = build_vllm_args(env, node_rank=0)

        self.assertEqual(args[args.index("--sparse-indexer-topk-backend") + 1], "auto")
        self.assertEqual(args.count("--sparse-indexer-topk-backend"), 1)

    def test_only_lane_05_keeps_dspark_rope_cache_in_bf16(self) -> None:
        for lane_id in (
            "01-tp8-pp4-256k-safe",
            "02-tp8-dp4-ep32-256k-safe",
            "03-tp8-dp4-ep32-256k-dspark",
            "04-tp8-dp4-ep32-1m-fp8kv",
        ):
            with self.subTest(lane=lane_id):
                args = build_vllm_args(self.load_lane(lane_id), node_rank=0)
                self.assertNotIn("--kv-cache-dtype-skip-layers", args)

    def test_all_explicit_serving_controls_are_emitted(self) -> None:
        env = self.load_lane("02-tp8-dp4-ep32-256k-safe")

        args = build_vllm_args(env, node_rank=1)

        expected_flags = {
            "--host",
            "--port",
            "--master-addr",
            "--master-port",
            "--data-parallel-rpc-port",
            "--data-parallel-backend",
            "--all2all-backend",
            "--moe-backend",
            "--attention-backend",
            "--quantization",
            "--dtype",
            "--kv-cache-dtype",
            "--max-num-batched-tokens",
            "--gpu-memory-utilization",
            "--compilation-config",
            "--served-model-name",
            "--tool-call-parser",
            "--reasoning-parser",
            "--enforce-eager",
            "--language-model-only",
        }
        self.assertTrue(expected_flags.issubset(args))
        self.assertNotIn("--enable-prefix-caching", args)

    def test_node_rank_must_be_one_of_four_nodes(self) -> None:
        env = self.load_lane("01-tp8-pp4-256k-safe")

        for invalid in (-1, 4):
            with self.subTest(node_rank=invalid):
                with self.assertRaisesRegex(ValueError, "node_rank"):
                    build_vllm_args(env, node_rank=invalid)

    def test_only_worker_nodes_are_headless(self) -> None:
        env = self.load_lane("02-tp8-dp4-ep32-256k-safe")

        self.assertNotIn("--headless", build_vllm_args(env, node_rank=0))
        for node_rank in (1, 2, 3):
            with self.subTest(node_rank=node_rank):
                self.assertIn("--headless", build_vllm_args(env, node_rank=node_rank))

    def test_extra_args_cannot_override_managed_flags(self) -> None:
        for args in (
            ["--max-model-len", "1024"],
            ["--max-model-len=1024"],
            ["--enable-expert-parallel"],
            ["--speculative-config", "{}"],
            ["--model", "other/model"],
            ["--nnodes", "1"],
            ["--headless"],
            ["--sparse-indexer-topk-backend", "torch"],
            ["--sparse-indexer-topk-backend=torch"],
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "managed"):
                    validate_extra_args(args)

    def test_extra_args_preserve_safe_tokens_without_eval(self) -> None:
        args = [
            "--trust-remote-code",
            "--disable-custom-all-reduce",
            "--chat-template",
            "/models/template with spaces.jinja",
        ]

        self.assertEqual(validate_extra_args(args), args)


if __name__ == "__main__":
    unittest.main()
