import asyncio
import json
import unittest
from unittest.mock import patch

from app.mimo_local import _responses_input, stream_response


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeAsyncClient:
    response_batches: list[list[str]] = []
    payloads: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs):
        self.__class__.payloads.append(kwargs["json"])
        return _FakeStreamResponse(self.__class__.response_batches.pop(0))


class ResponsesReasoningReplayTests(unittest.TestCase):
    def test_native_output_items_are_replayed_without_reconstruction(self) -> None:
        reasoning_item = {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-state",
            "status": "completed",
        }
        message_item = {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will inspect it.", "annotations": []}],
        }
        function_item = {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "host_read_file",
            "arguments": '{"path":"/home/share/index.html"}',
        }
        converted = _responses_input(
            [
                {
                    "role": "assistant",
                    "content": "I will inspect it.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "host_read_file",
                                "arguments": '{"path":"/home/share/index.html"}',
                            },
                        }
                    ],
                    "responses_output_items": [reasoning_item, message_item, function_item],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
            ]
        )

        self.assertEqual(converted[:3], [reasoning_item, message_item, function_item])
        self.assertEqual(
            converted[3],
            {"type": "function_call_output", "call_id": "call_1", "output": "file contents"},
        )
        self.assertEqual(sum(item.get("type") == "function_call" for item in converted), 1)

    def test_hidden_reasoning_item_does_not_require_visible_reasoning_text(self) -> None:
        reasoning_item = {
            "id": "rs_hidden",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-state",
        }
        converted = _responses_input(
            [{"role": "assistant", "content": "", "responses_output_items": [reasoning_item]}]
        )

        self.assertEqual(converted, [reasoning_item])

    def test_stream_replays_hidden_reasoning_before_tool_output(self) -> None:
        reasoning_item = {
            "id": "rs_hidden",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-state",
            "status": "completed",
        }
        function_item = {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "host_read_file",
            "arguments": "{}",
        }
        final_message = {
            "id": "msg_2",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "完成", "annotations": []}],
        }

        def event(data: dict) -> str:
            return "data: " + json.dumps(data, ensure_ascii=False)

        _FakeAsyncClient.payloads = []
        _FakeAsyncClient.response_batches = [
            [
                event({"type": "response.output_item.done", "output_index": 0, "item": reasoning_item}),
                event({"type": "response.output_item.done", "output_index": 1, "item": function_item}),
                event(
                    {
                        "type": "response.completed",
                        "response": {"output": [reasoning_item, function_item], "usage": {}},
                    }
                ),
                "data: [DONE]",
            ],
            [
                event({"type": "response.output_text.delta", "delta": "完成"}),
                event({"type": "response.output_item.done", "output_index": 0, "item": final_message}),
                "data: [DONE]",
            ],
        ]

        async def update(payload: dict) -> None:
            return None

        tool = {
            "type": "function",
            "function": {
                "name": "host_read_file",
                "description": "Read a host file.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        with patch("app.mimo_local.httpx.AsyncClient", _FakeAsyncClient):
            result = asyncio.run(
                stream_response(
                    base_url="https://provider.example/v1",
                    api_key="test-key",
                    model="muse-spark-1.2-contributor-free",
                    messages=[{"role": "user", "content": "修复页面"}],
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                    settings={"reasoning_effort_enabled": True},
                    web_enabled=False,
                    api_protocol="responses",
                    agent_mode=True,
                    extra_tools=[tool],
                    extra_tool_handler=lambda name, arguments: "file contents",
                )
            )

        self.assertEqual(result["answer"], "完成")
        self.assertEqual(len(_FakeAsyncClient.payloads), 2)
        self.assertEqual(
            _FakeAsyncClient.payloads[0]["include"],
            ["reasoning.encrypted_content"],
        )
        second_input = _FakeAsyncClient.payloads[1]["input"]
        self.assertIn(reasoning_item, second_input)
        self.assertIn(function_item, second_input)
        tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "file contents"}
        self.assertIn(tool_output, second_input)
        self.assertLess(second_input.index(reasoning_item), second_input.index(tool_output))


if __name__ == "__main__":
    unittest.main()
