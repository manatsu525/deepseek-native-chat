from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from app.minimax_tool_fallback import MiniMaxStreamBuffer, applies_to, parse_minimax_markup, recover_tool_calls


LEAK = """让我修复：
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="apply_patch">]<]minimax[>[<path>xiangqi.html]<]minimax[>[</path>]<]minimax[>[
<old_text><canvas id="board</canvas></old_text>]<]minimax[>[
<new_text><canvas id="board"></canvas></new_text>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>"""


class MiniMaxToolFallbackTests(unittest.TestCase):
    def test_scope_is_model_name_and_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_TOOL_FALLBACK", None)
            self.assertTrue(applies_to("minimaxai/minimax-m3"))
            self.assertTrue(applies_to("MiniMax-M2.7"))
            self.assertFalse(applies_to("deepseek-v4-flash"))
        with patch.dict(os.environ, {"MINIMAX_TOOL_FALLBACK": "off"}):
            self.assertFalse(applies_to("minimax-m3"))

    def test_parse_apply_patch_with_html_values(self) -> None:
        clean, calls = parse_minimax_markup(LEAK, "round-1")
        self.assertEqual(clean, "让我修复：")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "apply_patch")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {
                "path": "xiangqi.html",
                "old_text": '<canvas id="board</canvas>',
                "new_text": '<canvas id="board"></canvas>',
            },
        )

    def test_native_calls_win_and_final_leak_forces_real_answer(self) -> None:
        native = [{"id": "native", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        _, calls = recover_tool_calls(LEAK, native, id_prefix="ignored", tools_available=True)
        self.assertIs(calls, native)
        content, calls = recover_tool_calls(LEAK, [], id_prefix="final", tools_available=False)
        self.assertEqual(content, "")
        self.assertEqual(calls, [])

    def test_stream_buffer_hides_split_marker(self) -> None:
        buffer = MiniMaxStreamBuffer()
        chunks = ["准备修改。\n]", "<]mini", "max[>[<tool_call>", "private markup"]
        preview = "".join(buffer.feed(chunk) for chunk in chunks) + buffer.flush()
        self.assertEqual(preview, "准备修改。\n")
        plain = "普通回答" * 20
        plain_buffer = MiniMaxStreamBuffer()
        self.assertEqual(plain_buffer.feed(plain) + plain_buffer.flush(), plain)


if __name__ == "__main__":
    unittest.main()
