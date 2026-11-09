from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from deploy.tooling.lane_schema import LaneEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
K31_LANE = "11-k31-tp8-pp4-256k-safe"
K3_LANE = "01-tp8-pp4-256k-safe"


def load_smoke(lane_id: str) -> ModuleType:
    """Execute a rendered smoke.py without leaving __pycache__ in the lane directory."""

    path = REPO_ROOT / "deploy" / lane_id / "scripts" / "smoke.py"
    module = ModuleType(f"smoke_{lane_id[:2]}")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


class FakeServer:
    """In-process stand-in for the OpenAI endpoints used by smoke.py."""

    def __init__(
        self,
        *,
        reasoning_key: str | None = "reasoning_content",
        reasoning: str = "The user wants the word OK.",
        content: str | None = None,
        completion_texts: list[str] | None = None,
        chat_reasonings: list[str] | None = None,
    ) -> None:
        self.reasoning_key = reasoning_key
        self.reasoning = reasoning
        self.content = content
        self.completion_texts = completion_texts or [" OK"]
        self.chat_reasonings = chat_reasonings
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.completions = 0
        self.chats = 0

    def __call__(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        self.calls.append((method, url, payload))
        if url.endswith("/v1/models"):
            return {"data": [{"id": "kimi-k31-pro"}]}
        if url.endswith("/v1/completions"):
            text = self.completion_texts[self.completions % len(self.completion_texts)]
            self.completions += 1
            return {"choices": [{"text": text}]}
        if url.endswith("/v1/chat/completions"):
            reasoning = self.reasoning
            if self.chat_reasonings is not None:
                reasoning = self.chat_reasonings[self.chats % len(self.chat_reasonings)]
            self.chats += 1
            message: dict[str, Any] = {"content": self.content}
            if self.reasoning_key is not None:
                message[self.reasoning_key] = reasoning
            return {"choices": [{"message": message, "finish_reason": "length"}]}
        raise AssertionError(f"unexpected request: {method} {url}")

    def payloads(self, suffix: str) -> list[dict[str, Any]]:
        return [
            payload
            for _method, url, payload in self.calls
            if url.endswith(suffix) and payload is not None
        ]


class K31SmokeHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke = load_smoke(K31_LANE)

    def run_helper(self, name: str, server: FakeServer) -> Any:
        with patch.object(self.smoke, "_request", server):
            return getattr(self.smoke, name)("http://head:8111", "kimi-k31-pro", 30)

    def test_greedy_repro_sends_the_same_greedy_request_twice(self) -> None:
        server = FakeServer()

        result = self.run_helper("_greedy_repro", server)

        self.assertTrue(result["reproducible"])
        payloads = server.payloads("/v1/completions")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0]["temperature"], 0)
        self.assertEqual(payloads[0]["seed"], 1)
        self.assertEqual(payloads[0]["max_tokens"], 16)
        self.assertEqual(payloads[0]["model"], "kimi-k31-pro")

    def test_greedy_repro_fails_when_outputs_differ(self) -> None:
        server = FakeServer(completion_texts=[" OK", " Okay"])

        with self.assertRaisesRegex(RuntimeError, "not reproducible"):
            self.run_helper("_greedy_repro", server)

    def test_reasoning_case_sends_top_level_reasoning_effort_high(self) -> None:
        server = FakeServer()

        result = self.run_helper("_reasoning_case", server)

        payloads = server.payloads("/v1/chat/completions")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0]["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", payloads[0])
        self.assertEqual(payloads[0]["temperature"], 0)
        self.assertEqual(payloads[0]["seed"], 1)
        self.assertEqual(payloads[0]["messages"][0]["role"], "user")
        self.assertEqual(result["reasoning_effort"], "high")
        self.assertGreater(result["reasoning_chars"], 0)
        self.assertTrue(result["reproducible"])

    def test_reasoning_channel_may_be_named_reasoning(self) -> None:
        server = FakeServer(reasoning_key="reasoning")

        result = self.run_helper("_reasoning_case", server)

        self.assertGreater(result["reasoning_chars"], 0)

    def test_empty_reasoning_channel_fails(self) -> None:
        for kwargs in ({"reasoning_key": None}, {"reasoning": ""}):
            with self.subTest(kwargs=kwargs):
                server = FakeServer(content="OK", **kwargs)

                with self.assertRaisesRegex(RuntimeError, "empty reasoning channel"):
                    self.run_helper("_reasoning_case", server)

    def test_non_reproducible_reasoning_fails(self) -> None:
        server = FakeServer(chat_reasonings=["thought A", "thought B"])

        with self.assertRaisesRegex(RuntimeError, "not reproducible"):
            self.run_helper("_reasoning_case", server)

    def test_malformed_chat_response_is_a_value_error(self) -> None:
        with patch.object(self.smoke, "_request", return_value={"choices": []}):
            with self.assertRaises(ValueError):
                self.smoke._reasoning_case("http://head:8111", "kimi-k31-pro", 30)


class SmokeMainFlowTest(unittest.TestCase):
    """main()이 mode별로 K3.1 전용 검사를 호출하거나 건너뛰는지 확인한다."""

    def run_main(
        self,
        lane_id: str,
        mode: str,
        server: FakeServer,
    ) -> tuple[int, dict[str, Any], str]:
        smoke = load_smoke(lane_id)
        values = dict(LaneEnv.load(REPO_ROOT / "deploy" / lane_id / ".env").values)
        for key, value in list(values.items()):
            if value.startswith("CHANGE_ME_"):
                values[key] = "10.0.0.1" if key.endswith("NODE_IP") else "host"
        gpus = [
            {
                "node_rank": rank,
                "gpu_index": index,
                "name": "NVIDIA A100-SXM4-80GB",
                "total_mib": 81920,
                "used_mib": 1000,
            }
            for rank in range(4)
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            report = Path(tempdir) / f"{mode}.json"
            output = io.StringIO()
            with (
                patch.object(smoke, "_load_env", return_value=values),
                patch.object(smoke, "_request", server),
                patch.object(smoke, "_gpu_memory", return_value=(gpus, 98.0)),
                patch.object(
                    sys, "argv", ["smoke.py", "--mode", mode, "--report", str(report)]
                ),
                contextlib.redirect_stdout(output),
            ):
                code = smoke.main()
            payload = json.loads(report.read_text(encoding="utf-8"))
        return code, payload, output.getvalue()

    def test_k31_smoke_mode_runs_repro_and_reasoning_checks(self) -> None:
        code, payload, stdout = self.run_main(K31_LANE, "smoke", FakeServer())

        self.assertEqual(code, 0, stdout)
        self.assertEqual(payload["overall"], "PASS")
        evidence = payload["evidence"]
        self.assertEqual(evidence["completion"], " OK")
        self.assertTrue(evidence["greedy_repro"]["reproducible"])
        self.assertEqual(evidence["reasoning"]["reasoning_effort"], "high")
        self.assertEqual(payload["id"], f"{K31_LANE}:smoke")

    def test_k31_load_mode_stays_a_first_token_check(self) -> None:
        server = FakeServer()

        code, payload, _ = self.run_main(K31_LANE, "load", server)

        self.assertEqual(code, 0)
        self.assertEqual(set(payload["evidence"]) & {"greedy_repro", "reasoning"}, set())
        self.assertEqual(server.payloads("/v1/chat/completions"), [])

    def test_k31_smoke_fails_closed_without_a_reasoning_channel(self) -> None:
        code, payload, stdout = self.run_main(
            K31_LANE, "smoke", FakeServer(reasoning_key=None, content="OK")
        )

        self.assertEqual(code, 1)
        self.assertEqual(payload["overall"], "FAIL")
        self.assertIn("empty reasoning channel", payload["error"])
        self.assertIn("K3-SMOKE-FAIL", stdout)

    def test_k3_smoke_mode_is_unchanged(self) -> None:
        server = FakeServer()

        code, payload, _ = self.run_main(K3_LANE, "smoke", server)

        self.assertEqual(code, 0)
        self.assertEqual(set(payload["evidence"]) & {"greedy_repro", "reasoning"}, set())
        self.assertEqual(len(server.payloads("/v1/completions")), 1)
        self.assertEqual(server.payloads("/v1/chat/completions"), [])


if __name__ == "__main__":
    unittest.main()
