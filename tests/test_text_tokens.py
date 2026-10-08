import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.text_tokens import count_tokens, truncate_tokens
from app.keyless_web import KeylessWebProvider
from app.workspace import numbered_window
from app.agent import bounded_output
from app.attachments import _extract_plain_text, MAX_FILE_TEXT_TOKENS


class ContentTokenLimitsTests(unittest.IsolatedAsyncioTestCase):
    def test_unicode_and_literal_special_tokens_have_lossless_boundaries(self):
        for text in ['hello world ' * 1000, '中文内容🙂测试' * 1000, '<|endoftext|>' * 1000]:
            for limit in [0, 1, 7, 500]:
                head = truncate_tokens(text, limit)
                tail = truncate_tokens(text, limit, tail=True)
                self.assertLessEqual(count_tokens(head), limit)
                self.assertLessEqual(count_tokens(tail), limit)
                self.assertTrue(text.startswith(head))
                self.assertTrue(text.endswith(tail))
                self.assertNotIn('\ufffd', head + tail)
        self.check_files_and_command_output_keep_more_text_without_changing_numbers()
        self.check_plain_attachment_is_not_preclipped_to_old_character_limit()

    async def test_web_limits_use_tokens_and_do_not_preshrink_upstream(self):
        snippet = 'hello world ' * 1000
        raw = f'Title: source\nURL: https://example.com\nSnippets:\n{snippet}'
        provider = KeylessWebProvider('keenable')
        provider._call_tool = AsyncMock(return_value=({}, raw))
        result = await provider.search('hello')
        self.assertGreater(len(result[0]['snippet']), 500)
        self.assertLessEqual(count_tokens(result[0]['snippet']), 500)
        self.assertEqual(provider._call_tool.call_args.args[1]['snippet_max_length'], 10000)
        content = 'hello world ' * 10000
        provider._call_tool = AsyncMock(return_value=({}, content))
        page = await provider.fetch('https://example.com')
        body = page.split('\n\n[网页内容已截断')[0]
        self.assertGreater(len(body), 8000)
        self.assertLessEqual(count_tokens(body), 8000)
        self.assertGreater(provider._call_tool.call_args.args[1]['max_chars'], 8000)

    def check_files_and_command_output_keep_more_text_without_changing_numbers(self):
        text = 'hello world ' * 1000
        view = numbered_window(text, max_tokens=5000)
        self.assertFalse(view['truncated'])
        self.assertGreater(len(view['content']), 5000)
        long_lines = '\n'.join('x y z' for _ in range(1000))
        first = numbered_window(long_lines, max_tokens=500)
        self.assertTrue(first['truncated'])
        self.assertLessEqual(count_tokens(first['content']), 500)
        rest = numbered_window(long_lines, first['next_start_line'], max_tokens=500)
        self.assertEqual(rest['from_line'], first['through_line'] + 1)
        self.assertEqual(bounded_output(text, limit=5000), text)
        bounded = bounded_output(text, limit=500)
        self.assertIn('token', bounded)
        self.assertTrue(bounded.startswith(text[:30]))
        self.assertTrue(bounded.endswith(text[-30:]))

    def check_plain_attachment_is_not_preclipped_to_old_character_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_text('hello world ' * 40000)
            text = _extract_plain_text(path)
            self.assertGreater(len(text), MAX_FILE_TEXT_TOKENS)
            self.assertLessEqual(count_tokens(text), MAX_FILE_TEXT_TOKENS)
