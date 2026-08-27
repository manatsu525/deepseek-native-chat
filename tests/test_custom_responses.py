import unittest
from unittest.mock import patch

from app import main
from app.mimo import custom_output_token_field
from app.mimo_local import _normalize_responses_usage, _responses_input, _responses_tools


class CustomResponsesProtocolTests(unittest.TestCase):
    def test_output_token_field_matches_each_custom_protocol(self) -> None:
        self.assertEqual(custom_output_token_field("chat_completions"), "max_completion_tokens")
        self.assertEqual(custom_output_token_field("responses"), "max_output_tokens")
        self.assertEqual(custom_output_token_field("messages"), "max_tokens")

    def test_provider_type_is_independent_custom_protocol(self) -> None:
        self.assertTrue(main.is_custom_provider("custom"))
        self.assertTrue(main.is_custom_provider("custom_response"))
        self.assertTrue(main.is_custom_provider("custom_messages"))
        self.assertFalse(main.is_custom_provider("deepseek"))
        main.validate_provider_selection("custom_response", "muse-spark-1.2")

    def test_chat_tool_history_becomes_responses_items(self) -> None:
        converted = _responses_input(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"test"}'}}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            ]
        )
        self.assertEqual(converted[2]["type"], "function_call")
        self.assertEqual(converted[2]["call_id"], "call-1")
        self.assertEqual(converted[3], {"type": "function_call_output", "call_id": "call-1", "output": "result"})

    def test_image_and_function_schema_are_translated(self) -> None:
        converted = _responses_input(
            [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}]}]
        )
        self.assertEqual(converted[0]["content"][0], {"type": "input_text", "text": "look"})
        self.assertEqual(converted[0]["content"][1]["type"], "input_image")
        tools = _responses_tools(
            [{"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        )
        self.assertEqual(tools[0]["name"], "read_file")
        self.assertNotIn("function", tools[0])

    def test_responses_usage_keeps_cache_and_reasoning_details(self) -> None:
        usage = _normalize_responses_usage(
            {"input_tokens": 10, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 7}, "output_tokens_details": {"reasoning_tokens": 3}}
        )
        self.assertEqual(usage["total_tokens"], 14)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 7)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 3)


class CustomResponsesConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_model_probe_respects_common_sixteen_token_minimum(self) -> None:
        captured: dict = {}

        class Response:
            status_code = 200

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, *, headers, json):
                captured.update({"url": url, "headers": headers, "json": json})
                return Response()

        with patch.object(main.httpx, "AsyncClient", return_value=Client()):
            await main.test_custom_model(
                "https://provider.invalid/v1",
                "response-test-key",
                "tencent/hy3",
                api_protocol="responses",
            )

        self.assertTrue(captured["url"].endswith("/responses"))
        self.assertEqual(captured["json"]["max_output_tokens"], 16)
        self.assertNotIn("max_tokens", captured["json"])
        self.assertEqual(captured["json"]["input"], "Reply only OK")

    async def test_manual_chat_and_anthropic_probes_use_their_protocol_fields(self) -> None:
        captured: list[dict] = []

        class Response:
            status_code = 200

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, url, *, headers, json):
                captured.append({"url": url, "headers": headers, "json": json})
                return Response()

        with patch.object(main.httpx, "AsyncClient", return_value=Client()):
            await main.test_custom_model(
                "https://provider.invalid/v1",
                "chat-test-key",
                "xiaomi/mimo-v2.5:thinking",
                api_protocol="chat_completions",
            )
            await main.test_custom_model(
                "https://provider.invalid/v1",
                "anthropic-test-key",
                "claude-sonnet-4-6",
                api_protocol="messages",
            )

        self.assertTrue(captured[0]["url"].endswith("/chat/completions"))
        self.assertEqual(captured[0]["json"]["max_completion_tokens"], 1)
        self.assertNotIn("max_tokens", captured[0]["json"])
        self.assertTrue(captured[1]["url"].endswith("/messages"))
        self.assertEqual(captured[1]["json"]["max_tokens"], 1)
        self.assertNotIn("max_completion_tokens", captured[1]["json"])


if __name__ == "__main__":
    unittest.main()
