import json
import unittest
from unittest.mock import patch

from app import main
from app import mimo_local
from app.mimo_local import _anthropic_messages, _anthropic_tools, _normalize_anthropic_usage


class CustomMessagesProtocolTests(unittest.TestCase):
    def test_system_images_and_tool_history_are_translated(self) -> None:
        system, messages = _anthropic_messages(
            [
                {"role": "system", "content": "system prompt"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "anthropic_thinking_blocks": [{"type": "thinking", "thinking": "check", "signature": "signed"}],
                    "tool_calls": [{"id": "tool-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}],
                },
                {"role": "tool", "tool_call_id": "tool-1", "content": "file body"},
            ]
        )
        self.assertEqual(system, "system prompt")
        self.assertEqual(messages[0]["content"][1]["source"]["type"], "base64")
        self.assertEqual(messages[1]["content"][0]["signature"], "signed")
        self.assertEqual(messages[1]["content"][1]["type"], "tool_use")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "tool-1")

    def test_tool_schema_uses_input_schema(self) -> None:
        tools = _anthropic_tools(
            [{"type": "function", "function": {"name": "web_search", "description": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
        )
        self.assertEqual(tools[0]["name"], "web_search")
        self.assertIn("input_schema", tools[0])
        self.assertNotIn("function", tools[0])

    def test_usage_and_provider_kind(self) -> None:
        usage = _normalize_anthropic_usage(
            {
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_read_input_tokens": 9,
                "output_tokens_details": {"thinking_tokens": 3},
            }
        )
        self.assertEqual(usage["total_tokens"], 17)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 9)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 3)
        main.validate_provider_selection("custom_messages", "claude-sonnet-4-6")


class AnthropicAdaptivePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_effort_is_sent_as_adaptive_output_config(self) -> None:
        captured: dict = {}

        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                yield "data: " + json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    }
                )
                yield "data: [DONE]"

            async def aread(self) -> bytes:
                return b""

        class FakeStream:
            def __init__(self) -> None:
                self.response = FakeResponse()

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *args):
                return False

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url, *, headers, json):
                captured.update({"method": method, "url": url, "headers": headers, "json": json})
                return FakeStream()

        async def update(_: dict) -> None:
            return None

        with patch.object(mimo_local.httpx, "AsyncClient", FakeClient):
            result = await mimo_local.stream_response(
                base_url="https://provider.invalid/v1",
                api_key="anthropic-test-key",
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                timeout=30,
                stopped=lambda: False,
                update=update,
                settings={
                    "thinking": "enabled",
                    "reasoning_effort": "high",
                    "reasoning_effort_enabled": True,
                    "max_completion_tokens": 8192,
                },
                conversation_id="anthropic-adaptive",
                effort="high",
                web_enabled=False,
                api_protocol="messages",
            )

        self.assertEqual(result["answer"], "ok")
        self.assertEqual(captured["url"], "https://provider.invalid/v1/messages")
        self.assertEqual(captured["json"]["max_tokens"], 8192)
        self.assertEqual(captured["json"]["thinking"], {"type": "adaptive"})
        self.assertEqual(captured["json"]["output_config"], {"effort": "high"})
        self.assertNotIn("budget_tokens", captured["json"]["thinking"])
        self.assertNotIn("reasoning_effort", captured["json"])
        self.assertNotIn("max_completion_tokens", captured["json"])


if __name__ == "__main__":
    unittest.main()
