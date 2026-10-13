"""Streaming tool calls are shown while their arguments arrive; offline only."""
import copy
import json
import unittest
from unittest.mock import patch

from app import mimo_local


class ToolCallPreviewTests(unittest.TestCase):
    def test_preview_names_tool_path_and_size(self):
        partial = '{"path": "games/snake.html", "content": "<html>'
        found = {
            0: {"id": "call_1", "function": {"name": "write_file", "arguments": partial}},
            1: {"function": {"name": "", "arguments": ""}},
            2: {"function": {"name": "web_search", "arguments": '{"obj'}},
        }
        self.assertEqual(mimo_local._tool_call_preview(found), [
            {"index": 0, "id": "call_1", "tool": "write_file", "path": "games/snake.html", "chars": len(partial)},
            {"index": 2, "id": "", "tool": "web_search", "path": "", "chars": 5},
        ])

    def test_preview_decodes_escaped_path(self):
        found = {0: {"function": {"name": "write_file", "arguments": '{"path": "a\\"b.txt"'}}}
        self.assertEqual(mimo_local._tool_call_preview(found)[0]["path"], 'a"b.txt')


class ToolCallPreviewStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_write_is_previewed_then_cleared(self):
        arguments = json.dumps({"path": "site/index.html", "content": "<html>" + "x" * 4000 + "</html>"})
        chunks = [arguments[i:i + 500] for i in range(0, len(arguments), 500)]
        rounds = [
            [{"index": 0, "id": "w", "type": "function", "function": {"name": "host_write_file", "arguments": ""}}]
            + [{"index": 0, "function": {"arguments": chunk}} for chunk in chunks],
            None,
        ]
        updates, written = [], []

        def event(obj):
            return "data: " + json.dumps(obj)

        class Response:
            status_code = 200

            async def aiter_lines(self):
                deltas = rounds.pop(0)
                if deltas is None:
                    yield event({"choices": [{"delta": {"content": "done"}}]})
                else:
                    for delta in deltas:
                        yield event({"choices": [{"delta": {"tool_calls": [delta]}}]})
                yield "data: [DONE]"

            async def aread(self):
                return b""

        class Context:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_):
                return False

        class Client:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_, **__):
                return Context()

        async def update(state):
            updates.append(copy.deepcopy(state))

        async def execute(name, args):
            written.append(args["path"])
            return json.dumps({"ok": True})

        tool = {"type": "function", "function": {"name": "host_write_file", "description": "write", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}}
        with patch.object(mimo_local.httpx, "AsyncClient", Client):
            result = await mimo_local.stream_response(
                base_url="https://example.invalid/v1", api_key="test", model="test",
                messages=[{"role": "user", "content": "write the page"}], timeout=5, stopped=lambda: False,
                update=update, web_enabled=False, agent_mode=True, max_tool_rounds=5,
                extra_tools=[tool], extra_tool_handler=execute)

        self.assertEqual(written, ["site/index.html"])
        previews = [u["tool_preview"] for u in updates if u.get("tool_preview")]
        self.assertTrue(previews)
        self.assertEqual(previews[0][0]["tool"], "host_write_file")
        self.assertEqual(previews[0][0]["id"], "w")
        self.assertEqual(previews[-1][0]["path"], "site/index.html")
        self.assertEqual(previews[-1][0]["chars"], len(arguments))
        self.assertLess(previews[0][0]["chars"], previews[-1][0]["chars"])
        # Once the call has run, later updates no longer carry a preview.
        last_preview = max(i for i, u in enumerate(updates) if u.get("tool_preview"))
        self.assertTrue(any(not u.get("tool_preview") for u in updates[last_preview + 1:]))
        self.assertNotIn("tool_preview", result)


if __name__ == "__main__":
    unittest.main()
