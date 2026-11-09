from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_schema import (
    ALLOWED_KEYS,
    K31_KEYS,
    SPARSE_INDEXER_TOPK_BACKENDS,
    LaneEnv,
)
from deploy.tooling.lane_specs import (
    FAMILY_K3,
    FAMILY_K31,
    LANES,
    LaneSpec,
    lane_by_id,
    lane_by_number,
    lanes_in_family,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
K31_LANE_ID = "11-k31-tp8-pp4-256k-safe"


class LaneSpecTest(unittest.TestCase):
    def test_exact_lane_contract(self) -> None:
        self.assertEqual(
            [
                (
                    lane.id,
                    lane.tp,
                    lane.pp,
                    lane.dp,
                    lane.ep,
                    lane.max_len,
                    lane.dspark,
                    lane.fp8_kv,
                )
                for lane in LANES
            ],
            [
                ("01-tp8-pp4-256k-safe", 8, 4, 1, False, 262144, False, False),
                (
                    "02-tp8-dp4-ep32-256k-safe",
                    8,
                    1,
                    4,
                    True,
                    262144,
                    False,
                    False,
                ),
                (
                    "03-tp8-dp4-ep32-256k-dspark",
                    8,
                    1,
                    4,
                    True,
                    262144,
                    True,
                    False,
                ),
                (
                    "04-tp8-dp4-ep32-1m-fp8kv",
                    8,
                    1,
                    4,
                    True,
                    1048576,
                    False,
                    True,
                ),
                (
                    "05-tp8-dp4-ep32-1m-fp8kv-dspark",
                    8,
                    1,
                    4,
                    True,
                    1048576,
                    True,
                    True,
                ),
                ("11-k31-tp8-pp4-256k-safe", 8, 4, 1, False, 262144, False, False),
            ],
        )

    def test_every_lane_uses_exactly_32_processes(self) -> None:
        self.assertEqual(
            [lane.tp * lane.pp * lane.dp for lane in LANES],
            [32, 32, 32, 32, 32, 32],
        )

    def test_k3_lanes_keep_k3_model_fields(self) -> None:
        for lane in lanes_in_family(FAMILY_K3):
            with self.subTest(lane=lane.id):
                self.assertFalse(lane.is_k31)
                self.assertEqual(lane.target_model, "moonshotai/Kimi-K3")
                self.assertEqual(lane.draft_model, "Inferact/Kimi-K3-DSpark")
                self.assertEqual(lane.image, "k3-sm80-openai:0498dc7e-sm80-v3")
                self.assertEqual(
                    lane.serving_root,
                    "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3",
                )
                self.assertEqual(lane.served_name, "kimi-k3")
                self.assertEqual(lane.attention_backend, "TRITON_MLA")
                self.assertEqual(lane.sparse_indexer_topk_backend, "")
                self.assertEqual(lane.image_verifier, "docker/verify_image.py")

    def test_k31_lane_model_contract(self) -> None:
        (lane,) = lanes_in_family(FAMILY_K31)

        self.assertEqual(lane.id, K31_LANE_ID)
        self.assertEqual(lane.number, 11)
        self.assertTrue(lane.is_k31)
        self.assertEqual(lane.target_model, "moonshotai/Kimi-K3.1-Pro")
        self.assertEqual(lane.draft_model, "")
        self.assertEqual(lane.image, "k31-sm80-openai:a3c14efb-sm80-v1")
        self.assertEqual(
            lane.serving_root,
            "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3.1-Pro",
        )
        self.assertEqual(lane.served_name, "kimi-k31-pro")
        self.assertEqual(lane.attention_backend, "TRITON_MLA_SPARSE")
        self.assertEqual(lane.sparse_indexer_topk_backend, "persistent")
        self.assertFalse(lane.dspark)
        self.assertFalse(lane.fp8_kv)
        self.assertFalse(lane.ep)

    def test_lane_lookup_never_confuses_01_with_11(self) -> None:
        for lane in LANES:
            with self.subTest(lane=lane.id):
                self.assertIs(lane_by_id(lane.id), lane)
                self.assertIs(lane_by_id(f"{lane.number:02d}"), lane)
                self.assertIs(lane_by_number(lane.number), lane)
        self.assertEqual(lane_by_id("01").id, "01-tp8-pp4-256k-safe")
        self.assertEqual(lane_by_id("11").id, K31_LANE_ID)
        for unknown in ("1", "10", "12", "0", "k31", "11-k3"):
            with self.subTest(unknown=unknown):
                with self.assertRaises(KeyError):
                    lane_by_id(unknown)

    def test_every_lane_image_verifier_exists_in_the_repo(self) -> None:
        # preflight는 각 노드에서 이 스크립트로 이미지를 검증한다. K3.1 이미지용
        # 검증기가 따로 생기면 lane_specs의 image_verifier 한 줄만 바꾸면 되고,
        # 존재하지 않는 경로로 바꾸면 이 테스트가 먼저 실패한다.
        for lane in LANES:
            with self.subTest(lane=lane.id):
                self.assertTrue((REPO_ROOT / lane.image_verifier).is_file())

    def test_lane_numbers_are_unique(self) -> None:
        numbers = [lane.number for lane in LANES]
        self.assertEqual(len(numbers), len(set(numbers)))

    def test_k31_rejects_unimplemented_dspark_and_fp8_kv(self) -> None:
        (lane,) = lanes_in_family(FAMILY_K31)

        for change in ({"dspark": True}, {"fp8_kv": True}):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "K3.1 lane supports neither"):
                    dataclasses.replace(lane, **change)

    def test_unknown_family_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown lane family"):
            LaneSpec("99-x", 8, 4, 1, False, 1, False, False, family="k4")


class LaneEnvParsingTest(unittest.TestCase):
    def test_rejects_unknown_and_duplicate_keys(self) -> None:
        env = LaneEnv.parse(
            "VLLM_MAX_MODEL_LEN=262144\n"
            "VLLM_MAX_MODEL_LEN=1048576\n"
            "BOGUS=1\n"
        )

        self.assertIn("duplicate key: VLLM_MAX_MODEL_LEN", env.errors)
        self.assertIn("unknown key: BOGUS", env.errors)

    def test_rejects_malformed_lines_without_executing_them(self) -> None:
        marker = Path(tempfile.gettempdir()) / "k3-sm80-lane-schema-must-not-exist"
        marker.unlink(missing_ok=True)

        env = LaneEnv.parse(f"touch {marker}\n")

        self.assertEqual(env.errors, ["line 1: expected KEY=VALUE"])
        self.assertFalse(marker.exists())

    def test_allowed_keys_cover_serving_controls(self) -> None:
        self.assertTrue(
            {
                "K3_HEAD_NODE_IP",
                "K3_WORKER3_SSH_HOST",
                "VLLM_TENSOR_PARALLEL_SIZE",
                "VLLM_DATA_PARALLEL_SIZE",
                "VLLM_DATA_PARALLEL_BACKEND",
                "VLLM_ENABLE_EXPERT_PARALLEL",
                "VLLM_MAX_MODEL_LEN",
                "VLLM_MAX_NUM_SEQS",
                "VLLM_SPECULATIVE_CONFIG",
                "VLLM_EXTRA_ARGS",
            }.issubset(ALLOWED_KEYS)
        )

    def test_k31_only_keys_are_parseable_but_not_part_of_the_k3_key_set(self) -> None:
        self.assertEqual(K31_KEYS, {"VLLM_SPARSE_INDEXER_TOPK_BACKEND"})
        self.assertFalse(K31_KEYS & ALLOWED_KEYS)
        env = LaneEnv.parse("VLLM_SPARSE_INDEXER_TOPK_BACKEND=persistent\n")
        self.assertEqual(env.errors, [])


class K31SchemaRulesTest(unittest.TestCase):
    """K3.1 lane 11 .env 규칙과 K3 lane 격리."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = lane_by_id(K31_LANE_ID)
        cls.text = (REPO_ROOT / "deploy" / K31_LANE_ID / ".env").read_text()

    def with_value(self, key: str, value: str) -> LaneEnv:
        lines = []
        replaced = 0
        for line in self.text.splitlines():
            if line.startswith(f"{key}="):
                line = f"{key}={value}"
                replaced += 1
            lines.append(line)
        self.assertEqual(replaced, 1, key)
        return LaneEnv.parse("\n".join(lines) + "\n")

    def without_key(self, key: str) -> LaneEnv:
        lines = [
            line for line in self.text.splitlines() if not line.startswith(f"{key}=")
        ]
        return LaneEnv.parse("\n".join(lines) + "\n")

    def errors(self, env: LaneEnv) -> list[str]:
        return env.validate(self.spec, require_site_values=False)

    def test_committed_lane_env_is_valid(self) -> None:
        self.assertEqual(self.errors(LaneEnv.parse(self.text)), [])

    def test_kv_cache_dtype_must_be_explicit_bfloat16(self) -> None:
        for value in ("auto", "fp8", "fp8_e4m3", "float16"):
            with self.subTest(value=value):
                errors = self.errors(self.with_value("VLLM_KV_CACHE_DTYPE", value))
                self.assertTrue(
                    any(
                        "VLLM_KV_CACHE_DTYPE" in error and "fp8_ds_mla" in error
                        for error in errors
                    ),
                    errors,
                )

    def test_attention_backend_must_be_the_sparse_backend(self) -> None:
        for value in ("TRITON_MLA", "FLASHMLA_SPARSE", "XPU_MLA_SPARSE"):
            with self.subTest(value=value):
                errors = self.errors(self.with_value("VLLM_ATTENTION_BACKEND", value))
                self.assertTrue(
                    any("TRITON_MLA_SPARSE" in error for error in errors), errors
                )

    def test_topk_backend_accepts_only_the_declared_choices(self) -> None:
        self.assertEqual(
            SPARSE_INDEXER_TOPK_BACKENDS,
            {"persistent", "auto", "cooperative", "torch"},
        )
        for value in sorted(SPARSE_INDEXER_TOPK_BACKENDS):
            with self.subTest(valid=value):
                env = self.with_value("VLLM_SPARSE_INDEXER_TOPK_BACKEND", value)
                self.assertEqual(self.errors(env), [])
        for value in ("", "deep_select", "per_row", "PERSISTENT"):
            with self.subTest(invalid=value):
                env = self.with_value("VLLM_SPARSE_INDEXER_TOPK_BACKEND", value)
                self.assertTrue(
                    any(
                        "VLLM_SPARSE_INDEXER_TOPK_BACKEND" in error
                        and "must be one of" in error
                        for error in self.errors(env)
                    )
                )

    def test_topk_backend_key_is_required(self) -> None:
        errors = self.errors(self.without_key("VLLM_SPARSE_INDEXER_TOPK_BACKEND"))

        self.assertIn("missing key: VLLM_SPARSE_INDEXER_TOPK_BACKEND", errors)

    def test_speculative_config_and_draft_model_must_be_empty(self) -> None:
        speculative = self.errors(
            self.with_value("VLLM_SPECULATIVE_CONFIG", "'{\"method\":\"dspark\"}'")
        )
        draft = self.errors(self.with_value("K3_DRAFT_MODEL", "Inferact/Kimi-K3-DSpark"))

        self.assertIn(
            "non-DSpark lane must leave VLLM_SPECULATIVE_CONFIG empty", speculative
        )
        self.assertTrue(any("K3_DRAFT_MODEL" in error for error in draft), draft)

    def test_empty_draft_model_key_is_present_and_valid(self) -> None:
        env = LaneEnv.parse(self.text)

        self.assertEqual(env.values["K3_DRAFT_MODEL"], "")
        self.assertEqual(self.errors(env), [])

    def test_target_model_must_be_k31(self) -> None:
        errors = self.errors(self.with_value("K3_TARGET_MODEL", "moonshotai/Kimi-K3"))

        self.assertTrue(any("K3_TARGET_MODEL" in error for error in errors), errors)

    def test_fp8_kv_and_skip_layers_stay_off(self) -> None:
        fp8 = self.errors(self.with_value("VLLM_K3_SM80_FP8_KV", "1"))
        skip = self.errors(self.with_value("VLLM_KV_CACHE_DTYPE_SKIP_LAYERS", "'93 94'"))

        self.assertTrue(any("VLLM_K3_SM80_FP8_KV" in error for error in fp8), fp8)
        self.assertTrue(
            any("VLLM_KV_CACHE_DTYPE_SKIP_LAYERS" in error for error in skip), skip
        )


class K3LaneIsolationTest(unittest.TestCase):
    def test_k3_lane_env_must_not_carry_k31_only_keys(self) -> None:
        spec = lane_by_id("01")
        text = (REPO_ROOT / "deploy" / spec.id / ".env").read_text()
        env = LaneEnv.parse(text + "VLLM_SPARSE_INDEXER_TOPK_BACKEND=persistent\n")

        errors = env.validate(spec, require_site_values=False)

        self.assertTrue(
            any(
                "K3.1-only key is not allowed" in error
                and "VLLM_SPARSE_INDEXER_TOPK_BACKEND" in error
                for error in errors
            ),
            errors,
        )

    def test_committed_k3_lane_envs_still_validate_cleanly(self) -> None:
        for spec in lanes_in_family(FAMILY_K3):
            with self.subTest(lane=spec.id):
                env = LaneEnv.load(REPO_ROOT / "deploy" / spec.id / ".env")
                self.assertEqual(env.validate(spec, require_site_values=False), [])

    def test_k3_attention_backend_is_not_newly_pinned(self) -> None:
        # K3 lane 검증 규칙은 K3.1 추가 전과 같다: backend 값은 spec 검증 대상이 아니다.
        spec = lane_by_id("01")
        text = (REPO_ROOT / "deploy" / spec.id / ".env").read_text()
        text = text.replace(
            "VLLM_ATTENTION_BACKEND=TRITON_MLA\n",
            "VLLM_ATTENTION_BACKEND=SOMETHING_ELSE\n",
        )

        self.assertEqual(
            LaneEnv.parse(text).validate(spec, require_site_values=False), []
        )


if __name__ == "__main__":
    unittest.main()
