from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import workspace as workspace_module
from app.mimo import custom_auth_headers
from app.mimo_local import (
    AGENT_CONTEXT_COMPACT_THRESHOLD,
    _compact_workspace_call_arguments,
    _maybe_compact_agent_context,
)
from app.workspace import ConversationWorkspace


class ContextEfficiencyTests(unittest.TestCase):
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

    def test_large_successful_write_is_compacted_before_replay(self) -> None:
        function = {
            "name": "write_file",
            "arguments": json.dumps({"path": "app.js", "content": "x" * 10_000}),
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


if __name__ == "__main__":
    unittest.main()
