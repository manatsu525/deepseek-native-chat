import unittest

from app import main
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


if __name__ == "__main__":
    unittest.main()
