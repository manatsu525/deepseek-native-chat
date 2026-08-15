from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from app.inkling_tool_compat import (
    InklingStreamBuffer,
    applies_to,
    bind_patch_tools,
    parse_tool_markup,
    recover_tool_calls,
)
from app.workspace import WORKSPACE_TOOLS


MARKUP = (
    '<|message_model|>apply_patch<|content_invoke_tool_json|>'
    '{"name":"apply_patch","args":{"path":"app.js","old_text":"OLD","new_text":"NEW"}}'
    '<|end_message|><|content_model_end_sampling|>'
)


class InklingToolCompatTests(unittest.TestCase):
    def test_scope_and_disable_switch(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INKLING_TOOL_COMPAT", None)
            self.assertTrue(applies_to("lihuahuili--ep-inkling-nvfp4"))
            self.assertFalse(applies_to("deepseek-v4-flash"))
        with patch.dict(os.environ, {"INKLING_TOOL_COMPAT": "off"}):
            self.assertFalse(applies_to("inkling-nvfp4"))

    def test_parses_official_typed_tool_block(self) -> None:
        clean, calls = parse_tool_markup(MARKUP, "round-1")
        self.assertEqual(clean, "")
        self.assertEqual(calls[0]["function"]["name"], "apply_patch")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "app.js", "old_text": "OLD", "new_text": "NEW"},
        )

    def test_native_calls_win_and_visible_text_is_preserved(self) -> None:
        native = [{"id": "native", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        content = "prefix" + MARKUP
        clean, calls = recover_tool_calls(content, native, id_prefix="ignored", tools_available=True)
        self.assertEqual(clean, "prefix")
        self.assertIs(calls, native)

    def test_patch_schemas_are_explicitly_bound_without_path_argument(self) -> None:
        tools, bindings = bind_patch_tools(WORKSPACE_TOOLS, ["chinese-chess.html"])
        bound = [item for item in tools if item["function"]["name"].startswith("inkling_apply_patch_")]
        self.assertEqual(len(bound), 2)
        for tool in bound:
            function = tool["function"]
            self.assertNotIn("path", function["parameters"]["properties"])
            self.assertNotIn("path", function["parameters"]["required"])
            self.assertEqual(bindings[function["name"]][1], "chinese-chess.html")

    def test_stream_buffer_hides_split_private_marker(self) -> None:
        buffer = InklingStreamBuffer()
        preview = "".join(buffer.feed(item) for item in ["准备修改\n<|mess", "age_model|>", "private"])
        preview += buffer.flush()
        self.assertEqual(preview, "准备修改\n")


if __name__ == "__main__":
    unittest.main()
