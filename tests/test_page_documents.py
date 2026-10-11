import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.keyless_web import KeylessWebError, KeylessWebProvider
from app.page_documents import PageDocuments, TTL_SECONDS, cleanup_page_documents, read_document
from tests import test_responses_state as responses_state
from tests.test_responses_state import Transport, call, final


def web_call(rid, args):
    events = call(rid, ("fetch_webpage",))
    events[0]["item"]["arguments"] = json.dumps(args)
    return events


class PageDocumentTests(unittest.TestCase):
    def test_locate_complete_chapter_and_paginate_without_losing_the_tail(self):
        body = "\n".join("A hero explores the house and rescues a ghost. " * 10 for _ in range(200))
        page = "# Haunt 93\nOld rule\n# Haunt 96\n" + body + "\nVICTORY_RULE_MARKER\n# Haunt 97\nOther rule"
        data = {"content": page, "source_complete": True}
        index = json.loads(read_document(data, {"objective": "Haunt 96"}))
        self.assertEqual(index["mode"], "index")
        self.assertNotIn("content", index)
        self.assertTrue(any(item["line"] == 3 for item in index["related_headings"]))
        found = json.loads(read_document(data, {"mode": "find", "query": "Haunt 96"}))
        self.assertEqual(found["matches"][0]["line"], 3)
        following = json.loads(read_document(data, {"mode": "find", "query": "Haunt 97"}))["matches"][0]["line"]
        start, pieces = 3, []
        while True:
            result = json.loads(read_document(data, {"start_line": start, "end_line": following - 1, "max_tokens": 1000}))
            pieces.append(result["content"])
            if result["complete"]:
                break
            self.assertGreater(result["next_line"], start)
            start = result["next_line"]
        self.assertIn("VICTORY_RULE_MARKER", "\n".join(pieces))
        self.assertNotIn("Other rule", "\n".join(pieces))
        # A huge single line is wrapped losslessly, rather than repeatedly
        # re-encoding the remaining source for each chunk.
        raw = "中文🙂" * 30000
        start, pieces = 1, []
        while True:
            result = json.loads(read_document({"content": raw}, {"start_line": start}))
            pieces.extend(line.split(": ", 1)[1] for line in result["content"].split("\n"))
            if result["complete"]:
                break
            start = result["next_line"]
        self.assertEqual("".join(pieces), raw)
        wrapped = {"content": "x" * 198 + "VICTORY_RULE_MARKER"}
        self.assertEqual(json.loads(read_document(wrapped, {"mode": "find", "query": "VICTORY_RULE_MARKER"}))["matches"][0]["line"], 1)

    def test_cache_survives_restart_is_isolated_and_cleans_only_owned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PageDocuments("chat1", root)
            store.save("https://example.com", "full source with late victory")
            self.assertIn("victory", PageDocuments("chat1", root).load("https://example.com")["content"])
            self.assertIsNone(PageDocuments("chat2", root).load("https://example.com"))
            keep = root / "user-file.txt"
            keep.write_text("keep")
            old = time.time() - TTL_SECONDS - 1
            os.utime(store.path("https://example.com"), (old, old))
            cleanup_page_documents(root)
            self.assertFalse(store.has("https://example.com"))
            self.assertTrue(keep.exists())
            store.save("https://example.com/new", "cache")
            with patch("app.page_documents.MAX_CACHE_BYTES", 1):
                cleanup_page_documents(root)
            self.assertFalse(store.has("https://example.com/new"))
            self.assertTrue(keep.exists())
            self.assertFalse(json.loads(read_document({"content": "old excerpt", "source_complete": False}, {}))["source_complete"])


class PageLoopTests(unittest.IsolatedAsyncioTestCase):
    run_stream = responses_state.ResponsesStateTests.run_stream

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = PageDocuments("test", Path(self.directory.name))
        self.patcher = patch("app.mimo_local.PageDocuments", return_value=self.store)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def test_fetch_once_then_find_and_read_different_chapters_locally(self):
        url = "https://rules.example/ww.pdf"
        raw = "# Haunt 93\nOld rule\n" + "Filler.\n" * 10000 + "# Haunt 96\nVICTORY_RULE_MARKER\n# Haunt 97\nOther rule"
        self.store.save(url, raw)
        transport = Transport([web_call("find", {"url": url, "mode": "find", "query": "Haunt 96"}),
                               web_call("read", {"url": url, "start_line": 10003, "end_line": 10004}), final()])
        upstream = AsyncMock()
        class Web:
            call_tool = upstream
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
        with patch("app.mimo_local.ParallelMCPClient", Web):
            result = await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                          web_fetch_limit=0, web_search_limit=0, web_tool_round_limit=0,
                                          messages=[{"role": "user", "content": url}],
                                          settings={"web_tool_backend": "parallel"})
        upstream.assert_not_called()
        self.assertIn("VICTORY_RULE_MARKER", transport.payloads[-1]["input"][-1]["output"])
        self.assertEqual([step["status"] for step in result["tool_trace"]], ["completed", "completed"])
        self.assertTrue(all(not step["quota_counted"] for step in result["tool_trace"]))

    async def test_parallel_full_source_retained_and_same_url_can_continue(self):
        raw = "# Book\n" + "Filler.\n" * 10000 + "# Haunt 96\nVICTORY_RULE_MARKER"
        url = "https://rules.example/book.pdf"
        transport = Transport([web_call("fetch", {"url": url}),
                               web_call("read", {"url": url, "mode": "find", "query": "VICTORY_RULE_MARKER"}), final()])
        requests = []

        class Web:
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
            async def call_tool(self, name, args):
                requests.append(args)
                if args.get("allow_live_fetch") is False:
                    return {"results": [{"full_content": raw, "title": "Book"}]}
                return {"results": []}

        with patch("app.mimo_local.ParallelMCPClient", Web):
            await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                  settings={"web_tool_backend": "parallel"})
        self.assertEqual(len(requests), 2)  # live refusal, then provider cache; no third fetch
        self.assertEqual(self.store.load(url)["content"], raw)
        self.assertIn("VICTORY_RULE_MARKER", transport.payloads[-1]["input"][-1]["output"])

    async def test_keenable_returns_full_raw_source_after_cached_crawl_fallback(self):
        provider = KeylessWebProvider("keenable")
        raw = "Content. " * 50000 + "VICTORY_RULE_MARKER"
        provider._call_tool = AsyncMock(side_effect=[KeylessWebError("denied"), ({}, raw)])
        self.assertEqual(await provider.fetch("https://example.com"), raw.strip())
        self.assertEqual([c.args[1]["live"] for c in provider._call_tool.call_args_list], [True, False])

    async def test_old_excerpt_is_marked_partial_and_explicit_refresh_gets_raw_source(self):
        url = "https://rules.example/legacy.pdf"
        transport = Transport([web_call("legacy", {"url": url}),
                               web_call("refresh", {"url": url, "refresh": True}), final()])
        upstream = AsyncMock(return_value={"results": [{"full_content": "# Haunt 96\nVICTORY_RULE_MARKER"}]})
        class Web:
            call_tool = upstream
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
        with patch("app.mimo_local.ParallelMCPClient", Web):
            await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                  settings={"web_tool_backend": "parallel"},
                                  cached_web_evidence={url: {"content": "Haunt 93 old excerpt"}})
        self.assertIn('"source_complete": false', transport.payloads[1]["input"][-1]["output"])
        self.assertIn("VICTORY_RULE_MARKER", transport.payloads[-1]["input"][-1]["output"])
        self.assertEqual(upstream.call_count, 1)

    async def test_jina_keeps_content_past_the_old_window(self):
        from app.mimo import _read_with_jina
        raw = "Filler.\n" * 20000 + "VICTORY_RULE_MARKER"
        class Stream:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
            async def aiter_content(self): yield raw.encode()
        class Client:
            def stream(self, *_args, **_kwargs): return Stream()
        with patch("app.mimo._acquire_jina_slot", new_callable=AsyncMock):
            self.assertEqual(await _read_with_jina(Client(), "https://example.com", lambda: False), raw)
