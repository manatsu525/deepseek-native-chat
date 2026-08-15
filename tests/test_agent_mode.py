"""Regression tests for chat/coding mode selection and budgets."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import workspace as workspace_module
from app.db import Database
from app.mimo_local import stream_response
from app.mode import CHAT_PROFILE, CODING_PROFILE, looks_like_coding_request, resolve_profile
from app.workspace import ConversationWorkspace


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeApiClient:
    payloads: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    def stream(self, method: str, url: str, *, headers: dict, json: dict):
        self.payloads.append(json)
        if len(self.payloads) == 1:
            call = {
                "index": 0,
                "id": "call-write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":"hello.py","content":"print(\\"ok\\")\\n"}',
                },
            }
            event = {"choices": [{"delta": {"content": "I'll write the file now.", "reasoning_content": "act", "tool_calls": [call]}}]}
        else:
            event = {"choices": [{"delta": {"content": "已保存 hello.py。"}}]}
        return _FakeStreamResponse([f"data: {json_module.dumps(event)}", "data: [DONE]"])


# Avoid shadowing the imported json module in _FakeApiClient.stream's API-shaped
# keyword argument.
json_module = json


class AgentModeTests(unittest.TestCase):
    def test_auto_detects_obvious_chinese_and_english_coding_tasks(self) -> None:
        self.assertTrue(looks_like_coding_request("帮我写一个 HTML 贪吃蛇游戏"))
        self.assertTrue(looks_like_coding_request("用html写一个触屏人机对战国际象棋"))
        self.assertTrue(looks_like_coding_request("用 Python 帮我做个批量改名脚本"))
        self.assertTrue(looks_like_coding_request("Fix the Python script and save the file"))
        self.assertIs(resolve_profile("auto", "创建一个网页", False), CODING_PROFILE)

    def test_auto_keeps_normal_questions_in_chat_mode(self) -> None:
        self.assertIs(resolve_profile("auto", "今天有什么值得关注的新闻？", False), CHAT_PROFILE)

    def test_existing_workspace_continues_in_coding_mode(self) -> None:
        self.assertIs(resolve_profile("auto", "把按钮颜色换成蓝色", True), CODING_PROFILE)

    def test_explicit_mode_overrides_auto_detection(self) -> None:
        self.assertIs(resolve_profile("chat", "写一个完整网站", True), CHAT_PROFILE)
        self.assertIs(resolve_profile("coding", "你好", False), CODING_PROFILE)

    def test_coding_retains_web_tools_with_independent_workspace_budget(self) -> None:
        self.assertTrue(CODING_PROFILE.web_tools_enabled)
        self.assertTrue(CODING_PROFILE.workspace_tools_enabled)
        self.assertEqual(CODING_PROFILE.max_web_rounds, 4)
        self.assertEqual(CODING_PROFILE.max_workspace_rounds, 30)
        self.assertEqual(CODING_PROFILE.first_round_tool_choice, "required")
        self.assertEqual(CODING_PROFILE.first_round_effort, "low")
        self.assertEqual(CODING_PROFILE.first_round_max_tokens, 8192)

    def test_coding_loop_requires_an_early_tool_then_allows_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original_root = workspace_module.WORKSPACES_DIR
            workspace_module.WORKSPACES_DIR = Path(directory)
            _FakeApiClient.payloads = []

            async def update(state: dict) -> None:
                return None

            try:
                with patch("app.mimo_local.httpx.AsyncClient", _FakeApiClient):
                    result = asyncio.run(
                        stream_response(
                            base_url="https://provider.invalid/v1",
                            api_key="test-token",
                            model="coding-model",
                            messages=[{"role": "user", "content": "写一个 hello.py"}],
                            timeout=30,
                            stopped=lambda: False,
                            update=update,
                            settings={"web_tool_backend": "legacy", "max_completion_tokens": 65536},
                            conversation_id="mode-test",
                            workspace=ConversationWorkspace(1, "mode-test"),
                            mode="coding",
                        )
                    )
            finally:
                workspace_module.WORKSPACES_DIR = original_root

        first, second = _FakeApiClient.payloads
        self.assertEqual(first["tool_choice"], "required")
        self.assertEqual(first["max_tokens"], 8192)
        self.assertEqual(first["reasoning_effort"], "low")
        self.assertEqual(second["tool_choice"], "auto")
        self.assertEqual(second["reasoning_effort"], "medium")
        tool_turn = next(message for message in second["messages"] if message.get("tool_calls"))
        self.assertEqual(tool_turn["content"], "I'll write the file now.")
        self.assertEqual(tool_turn["tool_calls"][0]["function"]["arguments"], "{}")
        first_tools = {item["function"]["name"] for item in first["tools"]}
        self.assertIn("web_search", first_tools)
        self.assertIn("write_file", first_tools)
        self.assertEqual(result["mode"], "coding")
        self.assertEqual(result["reasoning_before_first_write"], 3)
        self.assertEqual(result["stall_nudges"], 0)
        self.assertEqual(result["answer"], "已保存 hello.py。")
        self.assertNotIn("I'll write", result["answer"])
        self.assertIn("I'll write", result["reasoning"])

    def test_database_migrates_old_jobs_with_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    provider_id INTEGER NOT NULL,
                    provider_type TEXT NOT NULL DEFAULT 'deepseek',
                    model TEXT NOT NULL,
                    effort TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    reasoning TEXT NOT NULL DEFAULT '',
                    searches_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            connection.commit()
            connection.close()
            database = Database(path)
            database.init()
            columns = {row["name"] for row in database.all("PRAGMA table_info(jobs)")}
            self.assertIn("mode", columns)


if __name__ == "__main__":
    unittest.main()
