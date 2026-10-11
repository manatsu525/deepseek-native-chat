from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.keyless_web import KeylessWebError, KeylessWebProvider
from app.page_focus import focus_page
from app.text_tokens import count_tokens
from tests import test_responses_state as responses_state
from tests.test_responses_state import Transport, call, final


def rule_book() -> str:
    """A long PDF-like page whose wanted section sits far past the first 8000 tokens."""
    sections = []
    for number in range(51, 101):
        body = f"Haunt {number} setup. " + " ".join(f"An explorer takes step {i} through the house." for i in range(60))
        if number == 96:
            body += "\n\nYou've been pulled into an alternate dimension. House of Leavings: heroes must rescue the trapped ghosts."
        sections.append(f"{number}\n\nHAUNT\n\n{body}")
    return "\n\n".join(sections)


class FocusPageTests(unittest.TestCase):
    def test_short_page_is_unchanged(self):
        self.assertEqual(focus_page("short page", 8000, "anything"), "short page")

    def test_long_page_keeps_head_and_the_section_matching_the_objective(self):
        page = rule_book()
        self.assertGreater(count_tokens(page), 30000)
        for objective in ("Find haunt 96 House of Leavings rules", "山屋惊魂 作祟96 玩法"):
            focused = focus_page(page, 8000, objective)
            self.assertLessEqual(count_tokens(focused), 8000)
            self.assertIn("alternate dimension", focused)
            self.assertTrue(focused.startswith("51\n\nHAUNT"))
            self.assertIn("[……中间省略……]", focused)
            self.assertIn("最相关的片段", focused)

    def test_numbers_match_whole_numbers_only(self):
        page = rule_book().replace("Haunt 96 setup", "Haunt ninety-six setup").replace("96\n\nHAUNT", "XX\n\nHAUNT")
        page += "\n\nFootnote: printed in 1996."
        focused = focus_page(page, 8000, "haunt 96")
        self.assertNotIn("alternate dimension", focused)

    def test_without_objective_the_head_is_kept_within_the_limit(self):
        focused = focus_page(rule_book(), 8000)
        self.assertLessEqual(count_tokens(focused), 8000)
        self.assertNotIn("alternate dimension", focused)
        self.assertIn("仅保留前面部分", focused)


class KeenableFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_refused_live_crawl_retries_from_keenable_cache_and_focuses(self):
        provider = KeylessWebProvider("keenable")
        provider._call_tool = AsyncMock(side_effect=[
            KeylessWebError("Error fetching https://wiki.example/Haunts: The target server denied access to this URL"),
            ({}, rule_book()),
        ])
        page = await provider.fetch("https://wiki.example/Haunts", "haunt 96 House of Leavings")
        live_flags = [c.args[1]["live"] for c in provider._call_tool.call_args_list]
        self.assertEqual(live_flags, [True, False])
        self.assertIn("alternate dimension", page)
        self.assertLessEqual(count_tokens(page), 8000)

    async def test_other_keyless_errors_are_not_retried(self):
        provider = KeylessWebProvider("tavily")
        provider._call_tool = AsyncMock(side_effect=KeylessWebError("denied"))
        with self.assertRaises(KeylessWebError):
            await provider.fetch("https://wiki.example/Haunts")
        self.assertEqual(provider._call_tool.call_count, 1)


def web_call(rid, name, args):
    events = call(rid, (name,))
    events[0]["item"]["arguments"] = json.dumps(args)
    return events


class FetchLoopTests(unittest.IsolatedAsyncioTestCase):
    """The reader result handed to the model, for each search backend."""

    run_stream = responses_state.ResponsesStateTests.run_stream

    def fetch_output(self, transport):
        return transport.payloads[-1]["input"][-1]["output"]

    async def test_parallel_requests_full_text_falls_back_to_cache_and_focuses(self):
        transport = Transport([
            web_call("fetch", "fetch_webpage", {"url": "https://rules.example/ww.pdf", "objective": "haunt 96 House of Leavings"}),
            final(),
        ])
        requests = []

        class Web:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, name, args):
                requests.append(dict(args))
                if args.get("allow_live_fetch") is False:
                    return {"results": [{"url": args["urls"][0], "title": "WW", "full_content": rule_book()}]}
                return {"results": [], "errors": [{"error_type": "fetch_failed"}]}

        with patch("app.mimo_local.ParallelMCPClient", Web):
            await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                  settings={"web_tool_backend": "parallel"})
        self.assertEqual([r.get("allow_live_fetch") for r in requests], [None, False])
        self.assertTrue(all(r["full_content"] for r in requests))
        output = self.fetch_output(transport)
        self.assertIn("alternate dimension", output)
        self.assertIn("最相关的片段", output)
        self.assertLessEqual(count_tokens(output), 8300)

    async def test_keenable_reader_receives_objective_and_search_terms(self):
        transport = Transport([
            web_call("search", "web_search", {"query": "House of Leavings haunt"}),
            web_call("fetch", "fetch_webpage", {"url": "https://rules.example/ww.pdf"}),
            final(),
        ])

        class Keyless:
            def __init__(self, provider):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def search(self, query, limit):
                return [{"url": "https://rules.example/ww.pdf", "title": "WW", "snippet": "", "publish_date": ""}]

            async def fetch(self, url, objective="", queries=()):
                return focus_page(rule_book(), 8000, objective, queries)

        with patch("app.mimo_local.KeylessWebProvider", Keyless):
            await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                  settings={"web_tool_backend": "keenable"})
        # No objective argument: the earlier search query still steers the page fit.
        self.assertIn("alternate dimension", self.fetch_output(transport))

    async def test_you_backend_fits_jina_page_around_the_objective(self):
        transport = Transport([
            web_call("fetch", "fetch_webpage", {"url": "https://rules.example/ww.pdf", "objective": "haunt 96 House of Leavings"}),
            final(),
        ])
        page = rule_book().encode()

        class Stream:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def aiter_content(self):
                yield page

        class Jina:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def stream(self, method, url, headers):
                return Stream()

        class YouSearch:
            def __init__(self, provider):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        with patch("app.mimo_local.curl_requests.AsyncSession", Jina), patch("app.mimo_local.KeylessWebProvider", YouSearch):
            await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                  settings={"web_tool_backend": "you"})
        output = self.fetch_output(transport)
        self.assertIn("Jina Reader", output)
        self.assertIn("alternate dimension", output)
        self.assertIn("最相关的片段", output)

