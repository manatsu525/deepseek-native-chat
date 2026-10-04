from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.db import Database
from app.main import _build_web_evidence_context
from app.mimo import _canonical_url
from app.mimo_local import stream_response


class WebEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_database_evidence_is_scoped_and_upserted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "chat.db")
            database.init()
            user_id = database.run(
                "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
                ("evidence-user", "hash", 1),
            )
            database.run(
                "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                ("evidence-conversation", user_id, "Evidence", 1, 1),
            )
            database.upsert_web_evidence(
                user_id,
                "evidence-conversation",
                "job-1",
                [
                    {
                        "canonical_url": "https://example.com/page",
                        "url": "https://example.com/page",
                        "title": "Page",
                        "content": "first body",
                    }
                ],
            )
            database.upsert_web_evidence(
                user_id,
                "evidence-conversation",
                "job-2",
                [
                    {
                        "canonical_url": "https://example.com/page",
                        "url": "https://example.com/page",
                        "title": "Page updated",
                        "content": "second body",
                    }
                ],
            )
            rows = database.web_evidence_for_conversation(user_id, "evidence-conversation")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["content"], "second body")
            self.assertEqual(database.web_evidence_for_conversation(user_id + 1, "evidence-conversation"), [])

    def test_canonical_url_normalizes_encoded_parentheses(self) -> None:
        literal = "https://example.com/wiki/Ghost_Bride_(Heroes)"
        encoded = "https://example.com/wiki/Ghost_Bride_%28Heroes%29"
        self.assertEqual(_canonical_url(literal), _canonical_url(encoded))

    def test_context_contains_previous_page_body(self) -> None:
        context = _build_web_evidence_context(
            [
                {
                    "canonical_url": "https://example.com/page",
                    "url": "https://example.com/page",
                    "title": "Rules",
                    "content": "The ring attack can stun the bride.",
                    "fetched_at": 1,
                }
            ],
            "鬼新娘能不能被指环攻击",
        )
        self.assertIn("https://example.com/page", context)
        self.assertIn("The ring attack can stun the bride.", context)
        self.assertIn("already read", context)

    async def test_cached_fetch_does_not_call_upstream_or_consume_fetch_quota(self) -> None:
        fetch_url = "https://example.com/wiki/Ghost_Bride_%28Heroes%29"
        canonical = _canonical_url(fetch_url)

        def event(lines: list[str]) -> list[str]:
            return lines + ["data: [DONE]"]

        responses = [
            event(
                [
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "fetch-1",
                                                "type": "function",
                                                "function": {
                                                    "name": "fetch_webpage",
                                                    "arguments": json.dumps({"url": fetch_url}),
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                ]
            ),
            event(["data: " + json.dumps({"choices": [{"delta": {"content": "完成"}}]})]),
        ]

        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in responses.pop(0):
                    yield line

            async def aread(self) -> bytes:
                return b""

        class FakeStreamContext:
            def __init__(self) -> None:
                self.response = FakeResponse()

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                return False

        class FakeHTTPClient:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **_kwargs):
                return FakeStreamContext()

        class FakeParallelClient:
            calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, *_args, **_kwargs):
                type(self).calls += 1
                raise AssertionError("cached page must not reach the upstream web tool")

        async def update(_state):
            return None

        with patch("app.mimo_local.httpx.AsyncClient", FakeHTTPClient), patch(
            "app.mimo_local.ParallelMCPClient", FakeParallelClient
        ):
            result = await stream_response(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                messages=[{"role": "user", "content": "检查鬼新娘规则"}],
                timeout=30,
                stopped=lambda: False,
                update=update,
                settings={"thinking": "disabled", "web_tool_backend": "parallel"},
                conversation_id="cached-page",
                workspace=None,
                cached_web_evidence={
                    canonical: {
                        "canonical_url": canonical,
                        "url": fetch_url,
                        "title": "Ghost Bride",
                        "content": "The ring attack can stun the bride.",
                        "summary": "ring attack",
                    }
                },
            )

        self.assertEqual(result["answer"], "完成")
        self.assertEqual(FakeParallelClient.calls, 0)
        self.assertTrue(result["tool_trace"][0].get("cached"))
        self.assertFalse(result["tool_trace"][0].get("quota_counted"))

    async def test_stale_exhausted_web_call_is_dropped_before_workspace_round(self) -> None:
        """A gateway must not turn a removed web tool into a visible failed step."""
        def event(lines: list[str]) -> list[str]:
            return lines + ["data: [DONE]"]

        responses = [
            event(
                [
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "stale-fetch",
                                                "type": "function",
                                                "function": {
                                                    "name": "fetch_webpage",
                                                    "arguments": json.dumps({"url": "https://example.com/page"}),
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                ]
            ),
            event(["data: " + json.dumps({"choices": [{"delta": {"content": "完成"}}]})]),
        ]

        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in responses.pop(0):
                    yield line

            async def aread(self) -> bytes:
                return b""

        class FakeStreamContext:
            def __init__(self) -> None:
                self.response = FakeResponse()

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                return False

        class FakeHTTPClient:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **_kwargs):
                return FakeStreamContext()

        class FakeParallelClient:
            calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, *_args, **_kwargs):
                type(self).calls += 1
                raise AssertionError("an exhausted fetch must be dropped before upstream execution")

        class FakeWorkspace:
            def tool_definitions(self, _access):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "read",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            def list_files(self):
                return []

        async def update(_state):
            return None

        with patch("app.mimo_local.httpx.AsyncClient", FakeHTTPClient), patch(
            "app.mimo_local.ParallelMCPClient", FakeParallelClient
        ):
            result = await stream_response(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
                messages=[{"role": "user", "content": "检查页面"}],
                timeout=30,
                stopped=lambda: False,
                update=update,
                settings={"thinking": "disabled", "web_tool_backend": "parallel"},
                workspace=FakeWorkspace(),
                workspace_access="edit",
                web_search_limit=0,
                web_fetch_limit=0,
            )

        self.assertEqual(result["answer"], "完成")
        self.assertEqual(result["searches"], [])
        self.assertEqual(result["tool_trace"], [])
        self.assertEqual(FakeParallelClient.calls, 0)


if __name__ == "__main__":
    unittest.main()
