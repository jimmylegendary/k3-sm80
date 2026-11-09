from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from deploy.tooling.lane_schema import K31_KEYS, LaneEnv
from deploy.tooling.lane_specs import FAMILY_K3, FAMILY_K31, LANES, lanes_in_family
from deploy.tooling.render_lanes import render_all, render_lane

REPO_ROOT = Path(__file__).resolve().parents[2]

# lane 01-05 생성물 35개의 파일별 SHA-256. K3.1 lane 11 추가 직전 커밋(HEAD 2df2bf6)의
# `git show HEAD:deploy/<path> | sha256sum` 값이며 sha256sum 출력 형식이다.
# 렌더러·템플릿·LaneSpec을 바꿔도 K3 lane 출력은 1바이트도 바뀌면 안 된다.
# 의도적으로 K3 lane을 바꿀 때만 이 값을 갱신한다.
K3_LANE_SNAPSHOT = """\
e3724ad56d720c914719875b8c43263c45869f8cf3146942813ef32c691fb26e  01-tp8-pp4-256k-safe/.env
0d835bf2a5ce515ca26b1de32961d710ecfffc7598b586b21a05107fce2e874a  01-tp8-pp4-256k-safe/Makefile
2d5108ba575525e8078adf8a247ad249e3a753a4a6d06613c6d0a108aac955cf  01-tp8-pp4-256k-safe/RUNBOOK.md
e19f1d45ee60ed0c6792d1c748b59bff5122dcf68800dd6d2da609d7fc2f9994  01-tp8-pp4-256k-safe/compose.yaml
59cc84cf30ecbf74536b1851edd0e3dcfc70be6c24d32c966fb6df9e3dcae1e6  01-tp8-pp4-256k-safe/scripts/cluster.py
5c32341ff2fb0a01d98d96fa1666031f80f13edc2900268279da0c0aca20aa58  01-tp8-pp4-256k-safe/scripts/preflight.py
0fc28ccbb760bc0705365777f399c9da7da145a5c24fbadaa9493d832d1669d4  01-tp8-pp4-256k-safe/scripts/smoke.py
a0710c8acf69b7e67cee3600bedbab4d1d93d3d92a9c10666ebdd2de850df1d6  02-tp8-dp4-ep32-256k-safe/.env
e143bf44f0e6f33a5a6143f484346bc351dcb83dd4c7c325fb340346231a347f  02-tp8-dp4-ep32-256k-safe/Makefile
991ea11954f4e98387c3fc135f70fe858e4d78ce026c50e5b9b27c38d973acef  02-tp8-dp4-ep32-256k-safe/RUNBOOK.md
20c60c7e347b7fc9316eec3373be8f7faf901d4f948cdcbd6ede386e6b9a46c1  02-tp8-dp4-ep32-256k-safe/compose.yaml
e55e2de126b657cc0a76433fbfe90e66ca8ad422b0e9a83617d83358950eb22e  02-tp8-dp4-ep32-256k-safe/scripts/cluster.py
5c4e629e77c4ef67a14e342aa89fcf76fd1e2578728afd917299878ac0853b65  02-tp8-dp4-ep32-256k-safe/scripts/preflight.py
a37c2fb8cb5b6fb67394b442497ce6ba1bd318c6b38719a35aa0f14f05ef6762  02-tp8-dp4-ep32-256k-safe/scripts/smoke.py
eefa11765495278f10604a40a12c5c509188435bc7f4d4fd4b45eefe9382f9cc  03-tp8-dp4-ep32-256k-dspark/.env
1e4f41b3bee7de01fdb03a3700e93a432c4db084b3f39b5841e98c12c670d135  03-tp8-dp4-ep32-256k-dspark/Makefile
827e86cad06998e64c4411874e0faedd3400dd222b2ef7ad6fbbd5ee1fa3ff06  03-tp8-dp4-ep32-256k-dspark/RUNBOOK.md
a78f927ae66759c956834a06202550ca540a97f0b01cae6897ff76b8a2d6757a  03-tp8-dp4-ep32-256k-dspark/compose.yaml
00aa87b5871a082254fe4414ee6b8638f77d17d611bd71302ff1948547638246  03-tp8-dp4-ep32-256k-dspark/scripts/cluster.py
a8bbe1de0fa86846fb013ea5d7f55bdd2e11814d6a82bdd3032b316d522c142e  03-tp8-dp4-ep32-256k-dspark/scripts/preflight.py
a7d190229f9e351a09e6bb6e633503e33552d35609eea05b5d659257c454bc40  03-tp8-dp4-ep32-256k-dspark/scripts/smoke.py
c59bb35238a169c8e9538b9ba24bcd6b56eea1996cc678b4fe76ff4a062b685d  04-tp8-dp4-ep32-1m-fp8kv/.env
44b142d4d88eb445f816b4f2bae5e8e9d5565aab638c9868b012c599d62ac696  04-tp8-dp4-ep32-1m-fp8kv/Makefile
fb1b39d84c16447dfac7be25140c06656de5c5f1104b71934f013caaa3098f83  04-tp8-dp4-ep32-1m-fp8kv/RUNBOOK.md
d2eac4e49b78b56baacd333c778337abd3103ac4a0e307910807680971cab87d  04-tp8-dp4-ep32-1m-fp8kv/compose.yaml
28a29ea5d1d2e4383d7f915fc3103fcea80b40dc42f09cd2448398ace91b9f32  04-tp8-dp4-ep32-1m-fp8kv/scripts/cluster.py
c3f11ca30d2d9e6024b5c109f36a93170fa583fd72b6dbf4579b73ea76418f0b  04-tp8-dp4-ep32-1m-fp8kv/scripts/preflight.py
4fd8a9ab199a0995ffcc62e6c12210e66f9a1adfccc60dd038140349a0d1f2e6  04-tp8-dp4-ep32-1m-fp8kv/scripts/smoke.py
094d88db7cd58911101b89af9c696a4176e73fd6be1a8b48f6bd63b5cb5ea7fa  05-tp8-dp4-ep32-1m-fp8kv-dspark/.env
272b11e8e15dd718b1c111b2328b586d7dc6f9f6783bdd1d6888ccc0dee470e0  05-tp8-dp4-ep32-1m-fp8kv-dspark/Makefile
00eed19a159f1a959656a03b9a4d99d637b0c6ac93c37371139d3d8bea10ce39  05-tp8-dp4-ep32-1m-fp8kv-dspark/RUNBOOK.md
adcb36148d5bfaad05a4c6fc481f88425d195ecdd3388fc11d8e18562e6c8389  05-tp8-dp4-ep32-1m-fp8kv-dspark/compose.yaml
d94d5c8d7c00093602cab8a14bc46a9b38f12598b03d89b379a4c01dcf1e3d5a  05-tp8-dp4-ep32-1m-fp8kv-dspark/scripts/cluster.py
516895bb735478c2702c3f421fc69299d568c155baba1a32dc4ac492828e7983  05-tp8-dp4-ep32-1m-fp8kv-dspark/scripts/preflight.py
06103ba5358a400bcf5cd2bb0623b6fe386441f634d88fef1c045a63f62e5966  05-tp8-dp4-ep32-1m-fp8kv-dspark/scripts/smoke.py"""


def parse_snapshot() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in K3_LANE_SNAPSHOT.splitlines():
        digest, path = line.split("  ", 1)
        entries[path] = digest
    return entries


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# 생성물이 아닌 런타임 부산물(.gitignore 대상)은 비교에서 뺀다.
RUNTIME_ARTIFACTS = frozenset({"__pycache__", ".k3-cluster.lock"})


def lane_files(root: Path, lane_id: str) -> set[str]:
    """Relative POSIX paths of every generated file under one lane directory."""

    return {
        path.relative_to(root).as_posix()
        for path in (root / lane_id).rglob("*")
        if path.is_file() and not RUNTIME_ARTIFACTS & set(path.parts)
    }


class RenderLanesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_rendered_lane_is_self_contained(self) -> None:
        render_lane(LANES[0], self.root)
        lane = self.root / LANES[0].id

        self.assertEqual(
            {path.name for path in lane.iterdir()},
            {".env", "compose.yaml", "Makefile", "RUNBOOK.md", "scripts"},
        )
        self.assertEqual(
            {path.name for path in (lane / "scripts").iterdir()},
            {"preflight.py", "cluster.py", "smoke.py"},
        )

    def test_render_all_produces_exactly_one_env_file_per_lane(self) -> None:
        render_all(self.root)

        env_files = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob(".env"))
        self.assertEqual(
            env_files,
            [f"{lane.id}/.env" for lane in sorted(LANES, key=lambda lane: lane.id)],
        )
        self.assertEqual(len(env_files), 6)

    def test_render_is_byte_deterministic(self) -> None:
        render_all(self.root)
        first = tree_hash(self.root)

        render_all(self.root)

        self.assertEqual(tree_hash(self.root), first)

    def test_lane_env_values_match_topology_and_features(self) -> None:
        render_all(self.root)

        lane_01 = (self.root / LANES[0].id / ".env").read_text()
        lane_05 = (self.root / LANES[4].id / ".env").read_text()
        self.assertIn("VLLM_PIPELINE_PARALLEL_SIZE=4\n", lane_01)
        self.assertIn("VLLM_DATA_PARALLEL_SIZE=1\n", lane_01)
        self.assertIn("VLLM_KV_CACHE_DTYPE=bfloat16\n", lane_01)
        self.assertIn("VLLM_PIPELINE_PARALLEL_SIZE=1\n", lane_05)
        self.assertIn("VLLM_DATA_PARALLEL_SIZE=4\n", lane_05)
        self.assertIn("VLLM_ENABLE_EXPERT_PARALLEL=1\n", lane_05)
        self.assertIn("VLLM_KV_CACHE_DTYPE=fp8\n", lane_05)
        self.assertIn('"method":"dspark"', lane_05)

    def test_compose_contains_four_distinct_profiles(self) -> None:
        render_lane(LANES[2], self.root)
        compose = (self.root / LANES[2].id / "compose.yaml").read_text()

        for profile in ("head", "worker1", "worker2", "worker3"):
            self.assertEqual(compose.count(f'profiles: ["{profile}"]'), 1)


class K3LaneByteIdentityTest(unittest.TestCase):
    """lane 01-05 생성물은 K3.1 lane 추가 전과 byte-identical이어야 한다."""

    def test_snapshot_covers_exactly_the_k3_lane_files(self) -> None:
        snapshot = parse_snapshot()

        self.assertEqual(len(snapshot), 35)
        self.assertEqual(
            {path.split("/", 1)[0] for path in snapshot},
            {lane.id for lane in lanes_in_family(FAMILY_K3)},
        )

    def test_fresh_render_of_k3_lanes_matches_snapshot(self) -> None:
        snapshot = parse_snapshot()
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for lane in lanes_in_family(FAMILY_K3):
                expected = {
                    path for path in snapshot if path.startswith(f"{lane.id}/")
                }
                self.assertEqual(lane_files(root, lane.id), expected, lane.id)
            for path, digest in snapshot.items():
                with self.subTest(path=path):
                    self.assertEqual(sha256_of(root / path), digest)

    def test_committed_k3_lanes_match_snapshot(self) -> None:
        snapshot = parse_snapshot()
        deploy = REPO_ROOT / "deploy"
        for lane in lanes_in_family(FAMILY_K3):
            expected = {path for path in snapshot if path.startswith(f"{lane.id}/")}
            self.assertEqual(lane_files(deploy, lane.id), expected, lane.id)
        for path, digest in snapshot.items():
            with self.subTest(path=path):
                self.assertEqual(sha256_of(deploy / path), digest)

    def test_k3_envs_never_contain_k31_only_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            render_all(root)
            for lane in lanes_in_family(FAMILY_K3):
                values = LaneEnv.load(root / lane.id / ".env").values
                self.assertFalse(K31_KEYS & values.keys(), lane.id)
                text = (root / lane.id / "scripts" / "cluster.py").read_text()
                self.assertNotIn("sparse-indexer-topk-backend", text, lane.id)
                makefile = (root / lane.id / "Makefile").read_text()
                self.assertNotIn("gate-k11", makefile, lane.id)
                self.assertNotIn("gate-k12", makefile, lane.id)


class K31LaneRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tempdir.name)
        render_all(cls.root)
        cls.spec = lanes_in_family(FAMILY_K31)[0]
        cls.lane = cls.root / cls.spec.id

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_lane_directory_layout(self) -> None:
        self.assertEqual(self.spec.id, "11-k31-tp8-pp4-256k-safe")
        self.assertEqual(
            lane_files(self.root, self.spec.id),
            {
                f"{self.spec.id}/{name}"
                for name in (
                    ".env",
                    "compose.yaml",
                    "Makefile",
                    "RUNBOOK.md",
                    "scripts/preflight.py",
                    "scripts/cluster.py",
                    "scripts/smoke.py",
                )
            },
        )

    def test_env_pins_k31_model_image_and_serving_contract(self) -> None:
        values = LaneEnv.load(self.lane / ".env").values

        expected = {
            "K3_IMAGE": "k31-sm80-openai:a3c14efb-sm80-v1",
            "K3_TARGET_MODEL": "moonshotai/Kimi-K3.1-Pro",
            "K3_DRAFT_MODEL": "",
            "K3_SERVING_ROOT": (
                "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3.1-Pro"
            ),
            "VLLM_API_PORT": "8111",
            "VLLM_MASTER_PORT": "29611",
            "VLLM_DP_RPC_PORT": "29711",
            "VLLM_TENSOR_PARALLEL_SIZE": "8",
            "VLLM_PIPELINE_PARALLEL_SIZE": "4",
            "VLLM_DATA_PARALLEL_SIZE": "1",
            "VLLM_ENABLE_EXPERT_PARALLEL": "0",
            "VLLM_MAX_MODEL_LEN": "262144",
            "VLLM_MAX_NUM_SEQS": "1",
            "VLLM_ENFORCE_EAGER": "1",
            "VLLM_LANGUAGE_MODEL_ONLY": "1",
            "VLLM_ATTENTION_BACKEND": "TRITON_MLA_SPARSE",
            "VLLM_SPARSE_INDEXER_TOPK_BACKEND": "persistent",
            "VLLM_KV_CACHE_DTYPE": "bfloat16",
            "VLLM_KV_CACHE_DTYPE_SKIP_LAYERS": "",
            "VLLM_K3_SM80_FP8_KV": "0",
            "VLLM_SPECULATIVE_CONFIG": "",
            "VLLM_SERVED_MODEL_NAME": "kimi-k31-pro",
            "VLLM_TOOL_CALL_PARSER": "kimi_k3",
            "VLLM_REASONING_PARSER": "kimi_k3",
        }
        for key, value in expected.items():
            self.assertEqual(values[key], value, key)
        self.assertIn("--trust-remote-code", values["VLLM_EXTRA_ARGS"])

    def test_env_has_exactly_the_k3_keys_plus_the_topk_key(self) -> None:
        k3_values = LaneEnv.load(
            self.root / lanes_in_family(FAMILY_K3)[0].id / ".env"
        ).values
        k31_values = LaneEnv.load(self.lane / ".env").values

        self.assertEqual(k31_values.keys() - k3_values.keys(), K31_KEYS)
        self.assertEqual(k3_values.keys() - k31_values.keys(), set())

    def test_ports_do_not_collide_with_k3_lanes(self) -> None:
        by_key: dict[str, list[str]] = {}
        for lane in LANES:
            values = LaneEnv.load(self.root / lane.id / ".env").values
            for key in ("VLLM_API_PORT", "VLLM_MASTER_PORT", "VLLM_DP_RPC_PORT"):
                by_key.setdefault(f"{key}={values[key]}", []).append(lane.id)
        self.assertEqual([k for k, v in by_key.items() if len(v) > 1], [])

    def test_env_passes_schema_with_site_placeholders_allowed(self) -> None:
        env = LaneEnv.load(self.lane / ".env")

        self.assertEqual(env.validate(self.spec, require_site_values=False), [])
        self.assertTrue(
            any(
                "unresolved site value" in error
                for error in env.validate(self.spec, require_site_values=True)
            )
        )

    def test_makefile_gates_and_phony_targets(self) -> None:
        makefile = (self.lane / "Makefile").read_text()

        self.assertIn("REQUIRED_GATES := k08 kda k11 k12 pp\n", makefile)
        self.assertIn("PREREQUISITE_GATES := k08 kda k11 k12\n", makefile)
        self.assertIn("gate-k11:\n\t@$(GATE) run --lane $(CURDIR) --gate k11\n", makefile)
        self.assertIn("gate-k12:\n\t@$(GATE) run --lane $(CURDIR) --gate k12\n", makefile)
        self.assertIn("\tgate-k11 gate-k12 \\\n", makefile)
        self.assertIn("--gate pp", makefile)
        self.assertNotIn("--gate 1m", makefile)

    def test_runbook_names_k31_model_root_and_omits_draft(self) -> None:
        runbook = (self.lane / "RUNBOOK.md").read_text()

        self.assertIn("`moonshotai/Kimi-K3.1-Pro` MXFP4 checkpoint", runbook)
        self.assertIn(
            "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3.1-Pro", runbook
        )
        self.assertIn("models--moonshotai--Kimi-K3.1-Pro/snapshots", runbook)
        self.assertNotIn("DSpark--", runbook)
        self.assertNotIn("Inferact", runbook)
        self.assertNotIn("moonshotai/Kimi-K3`", runbook)
        self.assertNotIn("kda-sm80-int64`, `mxfp4", runbook)
        self.assertIn("make gate-k11", runbook)
        self.assertIn("reasoning_effort", runbook)

    def test_smoke_script_has_k31_reasoning_and_repro_checks(self) -> None:
        smoke = (self.lane / "scripts" / "smoke.py").read_text()

        for needle in (
            '"reasoning_effort": REASONING_EFFORT',
            'REASONING_EFFORT = "high"',
            "/v1/chat/completions",
            "greedy completion was not reproducible",
            "reasoning_effort=high returned an empty reasoning channel",
        ):
            self.assertIn(needle, smoke)
        self.assertIn("seed", smoke)

    def test_k3_smoke_scripts_do_not_get_reasoning_checks(self) -> None:
        for lane in lanes_in_family(FAMILY_K3):
            smoke = (self.root / lane.id / "scripts" / "smoke.py").read_text()
            self.assertNotIn("reasoning_effort", smoke, lane.id)
            self.assertNotIn("_greedy_repro", smoke, lane.id)

    def test_preflight_skips_draft_cache_check_without_dspark(self) -> None:
        preflight = (self.lane / "scripts" / "preflight.py").read_text()

        self.assertIn("if 0:\n", preflight)
        self.assertIn('str(repo_root / "docker/verify_k31_image.py")', preflight)
        self.assertNotIn('str(repo_root / "docker/verify_image.py")', preflight)


if __name__ == "__main__":
    unittest.main()
