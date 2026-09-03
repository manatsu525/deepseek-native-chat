from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import workspace as workspace_module
from app.mimo import custom_auth_headers
from app import mimo_local
from app.mimo_local import (
    AGENT_CONTEXT_COMPACT_THRESHOLD,
    _compact_workspace_call_arguments,
    _final_answer_prompt,
    _maybe_compact_agent_context,
    _select_round_tool_calls,
    stream_response,
)
from app.workspace import ConversationWorkspace


class ContextEfficiencyTests(unittest.TestCase):
    def test_final_answer_prompt_describes_enabled_tool_families(self) -> None:
        all_tools = _final_answer_prompt(web_enabled=True, workspace_enabled=True)
        self.assertIn("工具调用额度已经全部耗尽；搜索、网页读取和工作区文件操作均已不可用", all_tools)
        self.assertIn(
            "Unavailable tools now: search, webpage-reading, workspace file operations.",
            all_tools,
        )

        workspace_only = _final_answer_prompt(web_enabled=False, workspace_enabled=True)
        self.assertIn("工作区文件操作均已不可用", workspace_only)
        self.assertNotIn("搜索、网页读取和工作区文件操作", workspace_only)

    def test_round_selector_keeps_workspace_calls_and_one_web_call(self) -> None:
        def call(call_id: str, name: str) -> dict:
            return {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }

        calls = [
            call("web-1", "web_search"),
            call("read-1", "read_file"),
            call("fetch-1", "fetch_webpage"),
            call("write-1", "write_file"),
            call("web-2", "web_search"),
            call("patch-1", "apply_line_edits"),
        ]
        selected = _select_round_tool_calls(calls, {})
        self.assertEqual(
            [item["function"]["name"] for item in selected],
            ["web_search", "read_file", "write_file", "apply_line_edits"],
        )


    def test_xai_chat_uses_stable_conversation_routing(self) -> None:
        headers = custom_auth_headers(
            "secret",
            base_url="https://api.x.ai/v1",
            stream=True,
            conversation_id="conversation-123",
        )
        self.assertEqual(headers["x-grok-conv-id"], "conversation-123")
        other = custom_auth_headers(
            "secret",
            base_url="https://example.com/v1",
            conversation_id="conversation-123",
        )
        self.assertNotIn("x-grok-conv-id", other)

    def test_ordinary_successful_write_stays_available_for_self_review(self) -> None:
        raw = json.dumps({"path": "app.js", "content": "x" * 10_000})
        function = {"name": "write_file", "arguments": raw}
        changed = _compact_workspace_call_arguments(
            function,
            name="write_file",
            path="app.js",
            succeeded=True,
        )
        self.assertFalse(changed)
        self.assertEqual(function["arguments"], raw)

    def test_very_large_successful_write_is_compacted_before_replay(self) -> None:
        function = {
            "name": "write_file",
            "arguments": json.dumps({"path": "app.js", "content": "x" * 100_000}),
        }
        changed = _compact_workspace_call_arguments(
            function,
            name="write_file",
            path="app.js",
            succeeded=True,
        )
        self.assertTrue(changed)
        compact = json.loads(function["arguments"])
        self.assertEqual(compact["path"], "app.js")
        self.assertLess(len(function["arguments"]), 200)

    def test_small_patch_stays_byte_identical_for_cache(self) -> None:
        raw = json.dumps({"path": "a.py", "old_text": "x", "new_text": "y"})
        function = {"name": "apply_patch", "arguments": raw}
        self.assertFalse(
            _compact_workspace_call_arguments(
                function,
                name="apply_patch",
                path="a.py",
                succeeded=True,
            )
        )
        self.assertEqual(function["arguments"], raw)

    def test_large_line_edit_is_compacted_after_success(self) -> None:
        function = {
            "name": "apply_line_edits",
            "arguments": json.dumps(
                {
                    "path": "app.js",
                    "revision": "a" * 64,
                    "edits": [{"start_line": 1, "end_line": 20, "new_text": "x" * 10_000}],
                }
            ),
        }
        self.assertTrue(
            _compact_workspace_call_arguments(
                function,
                name="apply_line_edits",
                path="app.js",
                succeeded=True,
            )
        )
        compact = json.loads(function["arguments"])
        self.assertEqual(compact["path"], "app.js")
        self.assertIn("edits", compact)
        self.assertLess(len(function["arguments"]), 250)

    def test_high_water_mark_keeps_base_and_latest_tool_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = workspace_module.WORKSPACES_DIR
            workspace_module.WORKSPACES_DIR = Path(directory)
            try:
                workspace = ConversationWorkspace(1, "conv")
                workspace.write_file("app.py", "print('ok')\n")
                base = [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "fix it"},
                ]
                old_pair = [
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "old"}]},
                    {"role": "tool", "tool_call_id": "old", "content": "x" * (AGENT_CONTEXT_COMPACT_THRESHOLD + 1)},
                ]
                latest_pair = [
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "new"}]},
                    {"role": "tool", "tool_call_id": "new", "content": "latest"},
                ]
                conversation = [*base, *old_pair, *latest_pair]
                changed = _maybe_compact_agent_context(
                    conversation,
                    base_message_count=len(base),
                    workspace=workspace,
                    sources={},
                    tool_trace=[{"name": "read_file", "path": "app.py", "status": "completed"}],
                )
                self.assertTrue(changed)
                self.assertTrue(conversation[0]["content"].startswith("system\n\nCONTEXT CHECKPOINT:"))
                self.assertEqual(conversation[1], base[1])
                self.assertEqual(conversation[-2:], latest_pair)
                checkpoint = json.loads(conversation[0]["content"].split("CONTEXT CHECKPOINT:\n", 1)[1])
                self.assertEqual(checkpoint["workspace_files"][0]["path"], "app.py")
            finally:
                workspace_module.WORKSPACES_DIR = original


class WorkspaceLoopGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_validation_is_skipped_until_workspace_changes(self) -> None:
        class CountingWorkspace(ConversationWorkspace):
            validation_runs = 0

            def execute(self, name: str, arguments: dict) -> str:
                if name == "run_python":
                    self.validation_runs += 1
                    return json.dumps({"ok": True, "stdout": "ok\n"})
                return super().execute(name, arguments)

        def tool_response(call_id: str, name: str, arguments: dict) -> list[str]:
            event = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }]
                    }
                }]
            }
            return ["data: " + json.dumps(event), "data: [DONE]"]

        class FakeResponse:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def aread(self) -> bytes:
                return b""

        class FakeStreamContext:
            def __init__(self, lines: list[str]) -> None:
                self.response = FakeResponse(lines)

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                return False

        responses = [
            tool_response("read-1", "read_file", {"path": "app.py", "start_line": 1, "end_line": 1}),
            tool_response("run-1", "run_python", {"path": "app.py"}),
            tool_response("run-2", "run_python", {"path": "app.py"}),
            tool_response("write-1", "write_file", {"path": "app.py", "content": "print('changed')\n"}),
            tool_response("run-3", "run_python", {"path": "app.py"}),
            ["data: " + json.dumps({"choices": [{"delta": {"content": "完成"}}]}), "data: [DONE]"],
        ]

        class FakeAsyncClient:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **_kwargs):
                return FakeStreamContext(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            workspace = CountingWorkspace(1, "loop-guard")
            workspace.root = Path(directory)
            workspace.write_file("app.py", "print('ok')\n")

            async def update(_):
                return None

            with patch.object(mimo_local.httpx, "AsyncClient", FakeAsyncClient):
                result = await stream_response(
                    base_url="https://example.test/v1",
                    api_key="test-key",
                    model="test-model",
                    messages=[{"role": "user", "content": "修改并检查代码"}],
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                    settings={"thinking": "disabled", "max_completion_tokens": 1024},
                    conversation_id="loop-guard",
                    workspace=workspace,
                    web_enabled=False,
                )

        self.assertEqual(result["answer"], "完成")
        self.assertEqual(workspace.validation_runs, 2)
        self.assertEqual(
            [(item["name"], item["status"]) for item in result["tool_trace"]],
            [
                ("read_file", "completed"),
                ("run_python", "completed"),
                ("run_python", "skipped"),
                ("write_file", "completed"),
                ("run_python", "completed"),
            ],
        )
        read_trace = result["tool_trace"][0]
        self.assertEqual(read_trace["requested_start_line"], 1)
        self.assertEqual(read_trace["requested_end_line"], 1)
        self.assertEqual(read_trace["line_count"], 1)
        self.assertEqual(read_trace["returned_from_line"], 1)
        self.assertEqual(read_trace["returned_through_line"], 1)
        self.assertFalse(read_trace["truncated"])


    async def test_multiple_workspace_calls_run_in_one_model_round(self) -> None:
        def tool_response(calls: list[tuple[str, str, dict]]) -> list[str]:
            event_calls = []
            for index, (call_id, name, arguments) in enumerate(calls):
                event_calls.append(
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                )
            event = {"choices": [{"delta": {"tool_calls": event_calls}}]}
            return ["data: " + json.dumps(event), "data: [DONE]"]

        class FakeResponse:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def aread(self) -> bytes:
                return b""

        class FakeStreamContext:
            def __init__(self, lines: list[str]) -> None:
                self.response = FakeResponse(lines)

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                return False

        responses = [
            tool_response(
                [
                    ("write-1", "write_file", {"path": "one.txt", "content": "one"}),
                    ("write-2", "write_file", {"path": "two.txt", "content": "two"}),
                ]
            ),
            ["data: " + json.dumps({"choices": [{"delta": {"content": "完成"}}]}), "data: [DONE]"],
        ]

        class FakeAsyncClient:
            calls = 0

            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **_kwargs):
                type(self).calls += 1
                return FakeStreamContext(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "multi-call")
            workspace.root = Path(directory)

            async def update(_):
                return None

            with patch.object(mimo_local.httpx, "AsyncClient", FakeAsyncClient):
                result = await stream_response(
                    base_url="https://example.test/v1",
                    api_key="test-key",
                    model="test-model",
                    messages=[{"role": "user", "content": "创建两个文件"}],
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                    settings={"thinking": "disabled", "max_completion_tokens": 1024},
                    conversation_id="multi-call",
                    workspace=workspace,
                    web_enabled=False,
                )

            self.assertEqual((workspace.root / "one.txt").read_text(), "one")
            self.assertEqual((workspace.root / "two.txt").read_text(), "two")

        self.assertEqual(FakeAsyncClient.calls, 2)
        self.assertEqual(result["answer"], "完成")
        self.assertEqual(
            [item["name"] for item in result["tool_trace"]],
            ["write_file", "write_file"],
        )

    async def test_tool_round_narration_is_not_accumulated_in_final_answer(self) -> None:
        def response(content: str, call: tuple[str, str, dict] | None = None) -> list[str]:
            delta: dict = {"content": content}
            if call is not None:
                call_id, name, arguments = call
                delta["tool_calls"] = [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            return ["data: " + json.dumps({"choices": [{"delta": delta}]}), "data: [DONE]"]

        class FakeResponse:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def aread(self) -> bytes:
                return b""

        class FakeStreamContext:
            def __init__(self, lines: list[str]) -> None:
                self.response = FakeResponse(lines)

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                return False

        responses = [
            response(
                "收到，我现在修改。",
                ("read-1", "read_file", {"path": "app.py"}),
            ),
            response(
                "找到问题，继续修改。",
                ("write-1", "write_file", {"path": "app.py", "content": "print('fixed')\n"}),
            ),
            response("已经修复。"),
        ]
        request_payloads: list[dict] = []

        class FakeAsyncClient:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **kwargs):
                request_payloads.append(json.loads(json.dumps(kwargs["json"])))
                return FakeStreamContext(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "tool-narration")
            workspace.root = Path(directory)
            workspace.write_file("app.py", "print('old')\n")

            async def update(_):
                return None

            with patch.object(mimo_local.httpx, "AsyncClient", FakeAsyncClient):
                result = await stream_response(
                    base_url="https://example.test/v1",
                    api_key="test-key",
                    model="test-model",
                    messages=[{"role": "user", "content": "修复代码"}],
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                    settings={"thinking": "disabled", "max_completion_tokens": 1024},
                    conversation_id="tool-narration",
                    workspace=workspace,
                    web_enabled=False,
                )

            self.assertEqual((workspace.root / "app.py").read_text(), "print('fixed')\n")

        self.assertEqual(result["answer"], "已经修复。")
        self.assertNotIn("我现在修改", result["answer"])
        self.assertNotIn("继续修改", result["answer"])
        self.assertEqual(request_payloads[1]["messages"][-2]["content"], "收到，我现在修改。")
        self.assertEqual(request_payloads[2]["messages"][-2]["content"], "找到问题，继续修改。")


if __name__ == "__main__":
    unittest.main()
