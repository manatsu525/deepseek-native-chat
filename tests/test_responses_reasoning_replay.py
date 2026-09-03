from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import mimo_local
from app.mimo_local import stream_response
from app.workspace import ConversationWorkspace


class ResponsesReasoningReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_reasoning_is_replayed_before_tool_output(self) -> None:
        reasoning_item = {
            "id": "rs_hidden",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-plan-state",
            "status": "completed",
        }
        message_item = {
            "id": "msg_progress",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "我来修改。", "annotations": []}],
        }
        function_item = {
            "id": "fc_read",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_read",
            "name": "read_file",
            "arguments": '{"path":"app.py"}',
        }

        def event(data: dict) -> str:
            return "data: " + json.dumps(data, ensure_ascii=False)

        responses = [
            [
                event({"type": "response.output_text.delta", "delta": "我来修改。"}),
                event({"type": "response.output_item.done", "output_index": 0, "item": reasoning_item}),
                event({"type": "response.output_item.done", "output_index": 1, "item": message_item}),
                event({"type": "response.output_item.done", "output_index": 2, "item": function_item}),
                event(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [reasoning_item, message_item, function_item],
                            "usage": {},
                        },
                    }
                ),
                "data: [DONE]",
            ],
            [
                event({"type": "response.output_text.delta", "delta": "完成。"}),
                "data: [DONE]",
            ],
        ]
        payloads: list[dict] = []

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

        class FakeAsyncClient:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, *_args, **kwargs):
                payloads.append(json.loads(json.dumps(kwargs["json"])))
                return FakeStreamContext(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "responses-reasoning")
            workspace.root = Path(directory)
            workspace.write_file("app.py", "print('old')\n")

            async def update(_):
                return None

            with patch.object(mimo_local.httpx, "AsyncClient", FakeAsyncClient):
                result = await stream_response(
                    base_url="https://opencode.ai/zen/v1",
                    api_key="test-key",
                    model="muse-spark-1.3-contributor-free",
                    messages=[{"role": "user", "content": "检查代码"}],
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                    settings={
                        "thinking": "enabled",
                        "reasoning_effort_enabled": True,
                        "max_completion_tokens": 1024,
                    },
                    api_protocol="responses",
                    conversation_id="responses-reasoning",
                    workspace=workspace,
                    web_enabled=False,
                )

        self.assertEqual(result["answer"], "我来修改。完成。")
        self.assertEqual(payloads[0]["include"], ["reasoning.encrypted_content"])
        second_input = payloads[1]["input"]
        self.assertIn(reasoning_item, second_input)
        self.assertIn(message_item, second_input)
        self.assertIn(function_item, second_input)
        tool_output = next(item for item in second_input if item.get("type") == "function_call_output")
        self.assertEqual(tool_output["call_id"], "call_read")
        self.assertLess(second_input.index(reasoning_item), second_input.index(tool_output))
        self.assertEqual(sum(item.get("type") == "function_call" for item in second_input), 1)


if __name__ == "__main__":
    unittest.main()
