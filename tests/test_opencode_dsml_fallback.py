import json
import os
import unittest
from unittest.mock import patch

from app.opencode_dsml_fallback import (
    DsmlStreamBuffer,
    applies_to,
    parse_dsml,
    recover_tool_calls,
)


DSML = """先查一下。
<｜DSML｜tool_calls>
<｜DSML｜invoke name="web_search">
<｜DSML｜parameter name="objective" string="true">核实天气</｜DSML｜parameter>
<｜DSML｜parameter name="search_queries" string="false">["深圳 天气", "深圳 气温"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""


class OpenCodeDsmlFallbackTests(unittest.TestCase):
    def test_scope_is_exact_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENCODE_DSML_FALLBACK", None)
            self.assertTrue(applies_to("https://opencode.ai/zen/v1", "deepseek-v4-flash-free"))
            self.assertFalse(applies_to("https://opencode.ai/zen/v1", "deepseek-v4-flash"))
            self.assertFalse(applies_to("https://integrate.api.nvidia.com/v1", "deepseek-v4-flash"))
            self.assertFalse(applies_to("https://evil-opencode.ai/v1", "deepseek-v4-flash-free"))
            self.assertFalse(applies_to("https://opencode.ai/zen/v1", "mimo-v2.5-free"))
            self.assertFalse(applies_to("https://opencode.ai/zen/v1", "deepseek-v4-flash-free", False))

        with patch.dict(os.environ, {"OPENCODE_DSML_FALLBACK": "off"}):
            self.assertFalse(applies_to("https://opencode.ai/zen/v1", "deepseek-v4-flash-free"))

    def test_parse_full_width_dsml_and_json_parameters(self):
        clean, calls = parse_dsml(DSML, id_prefix="round-1")
        self.assertEqual(clean, "先查一下。")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "round-1_0_web_search")
        self.assertEqual(calls[0]["function"]["name"], "web_search")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"objective": "核实天气", "search_queries": ["深圳 天气", "深圳 气温"]},
        )

    def test_ascii_pipe_variant_is_supported(self):
        clean, calls = parse_dsml(DSML.replace("｜", "|"))
        self.assertEqual(clean, "先查一下。")
        self.assertEqual(calls[0]["function"]["name"], "web_search")

    def test_native_calls_always_win(self):
        native = [{"id": "native", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]
        clean, calls = recover_tool_calls(DSML, native, id_prefix="ignored", tools_available=True)
        self.assertEqual(clean, "先查一下。")
        self.assertIs(calls, native)

    def test_native_search_query_string_is_also_normalized(self):
        native = [
            {
                "id": "call_from_opencode",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps(
                        {"objective": "查资料", "search_queries": '["玄凤鹦鹉 夜惊", "虎皮鹦鹉 夜惊"]'},
                        ensure_ascii=False,
                    ),
                },
            }
        ]
        content, calls = recover_tool_calls("", native, id_prefix="unused", tools_available=True)
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["id"], "call_from_opencode")
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["search_queries"], ["玄凤鹦鹉 夜惊", "虎皮鹦鹉 夜惊"])

    def test_fallback_recovers_only_while_tools_are_available(self):
        clean, calls = recover_tool_calls(DSML, [], id_prefix="recovered", tools_available=True)
        self.assertEqual(clean, "先查一下。")
        self.assertEqual(len(calls), 1)

        _, final_calls = recover_tool_calls(DSML, [], id_prefix="final", tools_available=False)
        self.assertEqual(final_calls, [])

    def test_recovered_search_queries_string_is_normalized_to_array(self):
        leaked = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="web_search">
<｜DSML｜parameter name="objective" string="true">查行情</｜DSML｜parameter>
<｜DSML｜parameter name="search_queries" string="true">["兆易创新 今日走势", "603986 行情"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        _, calls = recover_tool_calls(leaked, [], id_prefix="first", tools_available=True)
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["search_queries"], ["兆易创新 今日走势", "603986 行情"])

    def test_plain_search_query_string_becomes_single_item_array(self):
        leaked = """<|DSML|tool_calls>
<|DSML|invoke name="web_search">
<|DSML|parameter name="objective" string="true">查行情</|DSML|parameter>
<|DSML|parameter name="search_queries" string="true">兆易创新 今日走势</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>"""
        _, calls = recover_tool_calls(leaked, [], id_prefix="first", tools_available=True)
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["search_queries"], ["兆易创新 今日走势"])

    def test_orphan_block_is_removed_without_inventing_a_call(self):
        clean, calls = parse_dsml("正文<｜DSML｜tool_calls><｜DSML｜invoke")
        self.assertEqual(clean, "正文")
        self.assertEqual(calls, [])

    def test_stream_buffer_hides_split_marker_and_preserves_plain_text(self):
        buffer = DsmlStreamBuffer()
        chunks = ["这是准备文字 " * 5, "<｜DS", "ML｜tool_calls>", DSML.split("tool_calls>", 1)[1]]
        preview = "".join(buffer.feed(chunk) for chunk in chunks) + buffer.flush()
        self.assertNotIn("DSML", preview)
        self.assertTrue(preview.startswith("这是准备文字"))

        plain = "一段超过缓冲长度的普通回答，前后空格和词语都应当原样保留。" * 3
        plain_buffer = DsmlStreamBuffer()
        rebuilt = plain_buffer.feed(plain) + plain_buffer.flush()
        self.assertEqual(rebuilt, plain)


if __name__ == "__main__":
    unittest.main()
