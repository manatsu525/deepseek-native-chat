"""Offline checks for the dual coding-agent loop and shared web quota."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app import code_agents
from app.code_agents import (
    MAX_AGENT_ITERATIONS,
    SharedWebBudget,
    coding_job_active,
    harvest_files_from_text,
    looks_like_coding_request,
    run_verified_coding,
)
from app.mimo_local import _step_action
from app.config import Settings
from app.mimo import DEFAULT_SETTINGS, MIMO_MAX_SEARCHES


class CodeAgentTests(unittest.TestCase):
    def test_detector_keeps_non_coding_questions_on_the_normal_path(self) -> None:
        self.assertFalse(looks_like_coding_request("今天有什么新闻"))
        self.assertFalse(looks_like_coding_request("这段代码是什么意思"))
        self.assertTrue(looks_like_coding_request("写一个 python 脚本把 csv 转 json"))
        self.assertTrue(looks_like_coding_request("implement a function to merge intervals"))

    def test_write_code_tool_is_not_labeled_as_page_fetch(self) -> None:
        self.assertEqual(MAX_AGENT_ITERATIONS, 2)
        self.assertEqual(_step_action("write_and_verify_code"), "code_write")
        self.assertEqual(_step_action("fetch_webpage"), "open_page")
        self.assertEqual(_step_action("web_search"), "search")

    def test_harvests_fenced_html_instead_of_requiring_submit_code(self) -> None:
        files = harvest_files_from_text(
            "plan\n```html\n<!DOCTYPE html><html><body>象棋</body></html>\n```\n"
        )
        self.assertEqual(files[0]["path"], "index.html")
        self.assertIn("象棋", files[0]["content"])

    def test_prose_code_blocks_are_recovered_as_submitted_files(self) -> None:
        async def complete_round(**kwargs):
            if "submit_code" in {item["function"]["name"] for item in kwargs["tools"]}:
                return {
                    "content": "```python\ndef add(a, b):\n    return a + b\n```",
                    "reasoning": "",
                    "usage": {},
                    "tool_calls": [],
                }
            return {
                "content": "",
                "reasoning": "",
                "usage": {},
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {
                            "name": "submit_review",
                            "arguments": json.dumps(
                                {"passed": False, "issues": "need tests", "test_commands": []}
                            ),
                        },
                    }
                ],
            }

        async def runner():
            return await run_verified_coding(
                task="write add()",
                api_client=object(),
                base_url="https://invalid.example/v1",
                api_key="offline",
                model="dummy",
                config=dict(DEFAULT_SETTINGS),
                effort="high",
                timeout=30,
                stopped=lambda: False,
                update=AsyncMock(),
                budget=SharedWebBudget(),
                backend="parallel",
                clients={},
                parent_answer="",
                parent_reasoning="",
                parent_usage={},
                complete_round=complete_round,
            )

        import asyncio

        with tempfile.TemporaryDirectory() as temp, patch.object(code_agents, "settings", Settings(data_dir=Path(temp))):
            result = asyncio.run(runner())
        self.assertIn("main.py", result["files"])
        self.assertFalse(result["passed"])

    def test_html_project_ignores_illegal_python_inline_tests(self) -> None:
        async def complete_round(**kwargs):
            names = {item["function"]["name"] for item in kwargs["tools"]}
            if "submit_code" in names:
                return {
                    "content": "",
                    "reasoning": "",
                    "usage": {},
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "submit_code",
                                "arguments": json.dumps(
                                    {
                                        "plan": "html page",
                                        "files": [
                                            {
                                                "path": "index.html",
                                                "content": "<!DOCTYPE html><html><body><h1>象棋</h1></body></html>\n",
                                            }
                                        ],
                                    }
                                ),
                            },
                        }
                    ],
                }
            return {
                "content": "",
                "reasoning": "",
                "usage": {},
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {
                            "name": "submit_review",
                            "arguments": json.dumps(
                                {
                                    "passed": True,
                                    "issues": "looks complete",
                                    "test_commands": ["python3 -c 'print(1)'"],
                                }
                            ),
                        },
                    }
                ],
            }

        async def runner():
            return await run_verified_coding(
                task="html chess",
                api_client=object(),
                base_url="https://invalid.example/v1",
                api_key="offline",
                model="dummy",
                config=dict(DEFAULT_SETTINGS),
                effort="high",
                timeout=30,
                stopped=lambda: False,
                update=AsyncMock(),
                budget=SharedWebBudget(),
                backend="parallel",
                clients={},
                parent_answer="",
                parent_reasoning="",
                parent_usage={},
                complete_round=complete_round,
            )

        import asyncio

        with tempfile.TemporaryDirectory() as temp, patch.object(code_agents, "settings", Settings(data_dir=Path(temp))):
            result = asyncio.run(runner())
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["tests"][0]["command"], "static_verify")
        self.assertNotIn("已达到 2 轮编码上限", result["review"])

    def test_nested_coding_job_is_rejected(self) -> None:
        token = coding_job_active.set(True)
        try:
            import asyncio

            async def runner():
                return await run_verified_coding(
                    task="nested",
                    api_client=object(),
                    base_url="https://invalid.example/v1",
                    api_key="offline",
                    model="dummy",
                    config=dict(DEFAULT_SETTINGS),
                    effort="high",
                    timeout=30,
                    stopped=lambda: False,
                    update=AsyncMock(),
                    budget=SharedWebBudget(),
                    backend="parallel",
                    clients={},
                    parent_answer="",
                    parent_reasoning="",
                    parent_usage={},
                )

            with self.assertRaisesRegex(RuntimeError, "不能再次套用"):
                asyncio.run(runner())
        finally:
            coding_job_active.reset(token)

    def test_loop_uses_review_and_real_tests_then_passes(self) -> None:
        calls: list[str] = []

        async def complete_round(**kwargs):
            tools = {item["function"]["name"] for item in kwargs["tools"]}
            if "submit_code" in tools:
                calls.append("code")
                return {
                    "content": "",
                    "reasoning": "",
                    "usage": {},
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "submit_code",
                                "arguments": json.dumps(
                                    {
                                        "plan": "add then test",
                                        "files": [
                                            {"path": "mod.py", "content": "def add(a,b):\n    return a+b\n"},
                                            {
                                                "path": "test_mod.py",
                                                "content": (
                                                    "import unittest\nfrom mod import add\n"
                                                    "class T(unittest.TestCase):\n"
                                                    "    def test_add(self):\n"
                                                    "        self.assertEqual(add(2, 3), 5)\n"
                                                ),
                                            },
                                        ],
                                    }
                                ),
                            },
                        }
                    ],
                }
            calls.append("review")
            return {
                "content": "",
                "reasoning": "",
                "usage": {},
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {
                            "name": "submit_review",
                            "arguments": json.dumps(
                                {
                                    "passed": True,
                                    "issues": "looks good",
                                    "test_commands": ["python3 -m unittest test_mod.py"],
                                }
                            ),
                        },
                    }
                ],
            }

        async def runner():
            budget = SharedWebBudget()
            return await run_verified_coding(
                task="write add()",
                api_client=object(),
                base_url="https://invalid.example/v1",
                api_key="offline",
                model="dummy",
                config=dict(DEFAULT_SETTINGS),
                effort="high",
                timeout=30,
                stopped=lambda: False,
                update=AsyncMock(),
                budget=budget,
                backend="parallel",
                clients={},
                parent_answer="",
                parent_reasoning="",
                parent_usage={},
                complete_round=complete_round,
            )

        import asyncio

        with tempfile.TemporaryDirectory() as temp, patch.object(code_agents, "settings", Settings(data_dir=Path(temp))):
            result = asyncio.run(runner())
        self.assertEqual(calls, ["code", "review"])
        self.assertTrue(result["passed"], result)
        self.assertIn("mod.py", result["files"])
        self.assertTrue(all(item["ok"] for item in result["tests"]))

    def test_exhausted_search_quota_is_not_expanded_for_coding_agents(self) -> None:
        seen_tools: list[set[str]] = []

        async def complete_round(**kwargs):
            names = {item["function"]["name"] for item in kwargs["tools"]}
            seen_tools.append(names)
            return {
                "content": "",
                "reasoning": "",
                "usage": {},
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {
                            "name": "submit_code" if "submit_code" in names else "submit_review",
                            "arguments": json.dumps(
                                {
                                    "plan": "x",
                                    "files": [{"path": "a.py", "content": "X=1\n"}],
                                    "passed": False,
                                    "issues": "no search allowed",
                                    "test_commands": ["python3 -m unittest"],
                                }
                            ),
                        },
                    }
                ],
            }

        def fake_tests(commands, workspace: Path):
            return [{"command": commands[0], "ok": False, "exit_code": 1, "stdout": "", "stderr": "fail"}]

        async def runner():
            budget = SharedWebBudget(search_count=MIMO_MAX_SEARCHES, fetch_count=3)
            return await run_verified_coding(
                task="write something",
                api_client=object(),
                base_url="https://invalid.example/v1",
                api_key="offline",
                model="dummy",
                config=dict(DEFAULT_SETTINGS),
                effort="high",
                timeout=30,
                stopped=lambda: False,
                update=AsyncMock(),
                budget=budget,
                backend="parallel",
                clients={},
                parent_answer="",
                parent_reasoning="",
                parent_usage={},
                complete_round=complete_round,
                run_tests=fake_tests,
            )

        import asyncio

        with tempfile.TemporaryDirectory() as temp, patch.object(code_agents, "settings", Settings(data_dir=Path(temp))):
            asyncio.run(runner())
        for names in seen_tools:
            self.assertNotIn("web_search", names)
            self.assertNotIn("fetch_webpage", names)


if __name__ == "__main__":
    unittest.main()
