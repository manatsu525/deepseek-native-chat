from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import mimo_local
from app.agent import AgentRuntime
from app.db import Database
from app.work_log import WORK_LOG_HEADER, build_work_log, with_work_log
from app.workspace import ConversationWorkspace, WorkspaceError, replace_text_in_content


def fake_client(rounds, requests):
    class Response:
        status_code = 200

        def __init__(self, events):
            self.events = events

        async def aiter_lines(self):
            for event in self.events:
                yield "data: " + json.dumps(event)
            yield "data: [DONE]"

        async def aread(self):
            return b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def stream(self, *args, **kwargs):
            requests.append(copy.deepcopy(kwargs.get("json")))
            return Response(rounds.pop(0))

    return Client


def tool_round(call_id, name, arguments, usage=None):
    events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                                        "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}]
    if usage:
        events.append({"choices": [], "usage": usage})
    return events


def answer_round(text, usage=None):
    events = [{"choices": [{"delta": {"content": text}}]}]
    if usage:
        events.append({"choices": [], "usage": usage})
    return events


async def run_stream(workspace, rounds, **kwargs):
    requests = []

    async def update(_):
        pass

    options = dict(
        base_url="https://example.test/v1", api_key="k", model="m",
        messages=[{"role": "user", "content": "fix"}], timeout=30, stopped=lambda: False,
        update=update, settings={"thinking": "disabled"}, web_enabled=False, workspace=workspace,
    )
    options.update(kwargs)
    with patch.object(mimo_local.httpx, "AsyncClient", fake_client(rounds, requests)):
        result = await mimo_local.stream_response(**options)
    return result, requests


class ReplaceTextTests(unittest.TestCase):
    def test_exact_replacement_reports_new_line_numbers(self):
        content = "a\nb\nc\nd\n"
        updated, count, regions, mode = replace_text_in_content(content, "b\n", "b1\nb2\n", False)
        self.assertEqual((updated, count, mode), ("a\nb1\nb2\nc\nd\n", 1, "exact"))
        workspace_excerpt = __import__("app.workspace", fromlist=["edited_excerpt"]).edited_excerpt(updated, regions)
        self.assertIn("2|b1", workspace_excerpt["updated_excerpt"])
        self.assertIn("3|b2", workspace_excerpt["updated_excerpt"])

    def test_copied_line_number_prefixes_are_tolerated(self):
        content = "def f():\n    return 1\n"
        updated, _, _, mode = replace_text_in_content(content, "2|    return 1", "2|    return 2", False)
        self.assertEqual(mode, "line_numbers_removed")
        self.assertEqual(updated, "def f():\n    return 2\n")

    def test_trailing_whitespace_is_tolerated(self):
        content = "x = 1   \r\ny = 2\r\n"
        updated, _, _, mode = replace_text_in_content(content, "x = 1\ny = 2", "x = 3\ny = 4", False)
        self.assertEqual(mode, "whitespace_insensitive")
        self.assertEqual(updated, "x = 3\ny = 4\r\n")

    def test_ambiguous_and_missing_text_fail_without_changes(self):
        with self.assertRaisesRegex(WorkspaceError, "出现 2 次.*1、3"):
            replace_text_in_content("x\ny\nx\n", "x", "z", False)
        updated, count, _, _ = replace_text_in_content("x\ny\nx\n", "x", "z", True)
        self.assertEqual((updated, count), ("z\ny\nz\n", 2))
        with self.assertRaisesRegex(WorkspaceError, "最接近的当前内容：\n2\\|    total = price \\* qty"):
            replace_text_in_content("def f():\n    total = price * qty\n", "  total = price*qty", "x", False)

    def test_workspace_tool_is_opt_in_and_returns_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "replace")
            workspace.root = Path(directory)
            names = {item["function"]["name"] for item in workspace.tool_definitions()}
            self.assertNotIn("replace_text", names)
            names = {item["function"]["name"] for item in workspace.tool_definitions("edit", text_replace=True)}
            self.assertIn("replace_text", names)
            workspace.write_file("app.js", "".join(f"line{n}\n" for n in range(1, 11)))
            result = json.loads(workspace.execute("replace_text", {"path": "app.js", "old_text": "line5\n", "new_text": "five\n"}))
            self.assertEqual(result["replacements"], 1)
            self.assertIn("5|five", result["updated_excerpt"])
            self.assertIn("3|line3", result["updated_excerpt"])
            self.assertTrue(result["revision"])

    def test_line_edits_return_shifted_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "edits")
            workspace.root = Path(directory)
            workspace.write_file("a.txt", "".join(f"l{n}\n" for n in range(1, 21)))
            revision = json.loads(workspace.execute("read_file", {"path": "a.txt"}))["revision"]
            result = workspace.apply_line_edits("a.txt", revision, [
                {"start_line": 2, "end_line": 1, "new_text": "new-a\nnew-b"},
                {"start_line": 15, "end_line": 15, "new_text": "fifteen"},
            ])
            self.assertIn("2|new-a", result["updated_excerpt"])
            self.assertIn("3|new-b", result["updated_excerpt"])
            self.assertIn("17|fifteen", result["updated_excerpt"])
            self.assertIn("...", result["updated_excerpt"])

    def test_host_patch_uses_tolerant_matching_and_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mod.json"
            path.write_text('{\n  "spell": "old"  \n}\n')
            result = json.loads(AgentRuntime(None, 1, "t").execute("host_apply_patch", {
                "path": str(path), "old_text": '2|  "spell": "old"', "new_text": '2|  "spell": "new"'}))
            self.assertTrue(result["ok"], result)
            self.assertIn('2|  "spell": "new"', result["updated_excerpt"])
            self.assertEqual(result["match"], "line_numbers_removed")
            self.assertEqual(path.read_text(), '{\n  "spell": "new"  \n}\n')
            failed = json.loads(AgentRuntime(None, 1, "t").execute("host_apply_patch", {
                "path": str(path), "old_text": "missing", "new_text": "x"}))
            self.assertFalse(failed["ok"])


class StreamCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_replace_setting_controls_tool_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "stream")
            workspace.root = Path(directory)
            _, off = await run_stream(workspace, [answer_round("ok")])
            _, on = await run_stream(workspace, [answer_round("ok")],
                                     settings={"thinking": "disabled", "text_replace_tool": True})
        names_off = {tool["function"]["name"] for tool in off[0]["tools"]}
        names_on = {tool["function"]["name"] for tool in on[0]["tools"]}
        self.assertNotIn("replace_text", names_off)
        self.assertIn("replace_text", names_on)
        self.assertIn("replace_text", on[0]["messages"][0]["content"])
        self.assertNotIn("replace_text", off[0]["messages"][0]["content"])

    async def test_user_context_addendum_keeps_system_prompt_stable(self):
        _, plain = await run_stream(None, [answer_round("ok")])
        _, extra = await run_stream(None, [answer_round("ok")], user_context_addendum="WEB EVIDENCE: abc")
        self.assertEqual(plain[0]["messages"][0], extra[0]["messages"][0])
        self.assertTrue(extra[0]["messages"][-1]["content"].startswith("fix\n\n---\n"))
        self.assertIn("WEB EVIDENCE: abc", extra[0]["messages"][-1]["content"])
        _, multimodal = await run_stream(
            None, [answer_round("ok")], user_context_addendum="WEB EVIDENCE: abc",
            messages=[{"role": "user", "content": [{"type": "text", "text": "look"}]}])
        parts = multimodal[0]["messages"][-1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "look"})
        self.assertIn("WEB EVIDENCE: abc", parts[-1]["text"])

    async def test_round_stats_record_usage_and_prefix_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "stats")
            workspace.root = Path(directory)
            usage = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                     "prompt_tokens_details": {"cached_tokens": 80}}
            result, _ = await run_stream(workspace, [
                tool_round("c1", "list_files", {}, usage),
                answer_round("done", usage),
            ])
        stats = result["round_stats"]
        self.assertEqual([item["round"] for item in stats], [1, 2])
        self.assertEqual(stats[0]["cached_tokens"], 80)
        self.assertEqual(stats[0]["system_hash"], stats[1]["system_hash"])
        self.assertEqual(stats[0]["tools_hash"], stats[1]["tools_hash"])
        self.assertFalse(stats[0]["chained"])


class HistoryAndLogTests(unittest.TestCase):
    def test_history_window_moves_in_steps(self):
        from app.main import history_window_size

        starts = []
        for total in range(1, 61):
            size = history_window_size(total)
            self.assertGreaterEqual(size, min(total, 20))
            self.assertLess(size, 30)
            starts.append(total - size)
        self.assertEqual(sorted(set(starts)), [0, 10, 20, 30, 40])
        self.assertEqual(history_window_size(1), 1)

    def test_work_log_is_deterministic_and_compact(self):
        trace = [
            {"name": "read_file", "path": "index.html", "status": "completed"},
            {"name": "read_file", "path": "index.html", "status": "skipped"},
            {"name": "write_file", "path": "app.js", "status": "completed"},
            {"name": "replace_text", "path": "index.html", "status": "completed"},
            {"name": "apply_line_edits", "path": "style.css", "status": "failed"},
            {"name": "check_web_syntax", "path": "index.html", "status": "completed"},
            {"name": "host_run_command", "path": "npm   test", "status": "failed"},
            {"name": "web_search", "path": "", "status": "completed"},
        ]
        log = build_work_log(trace)
        self.assertEqual(log, build_work_log(copy.deepcopy(trace)))
        self.assertTrue(log.startswith(WORK_LOG_HEADER))
        self.assertIn("新建或覆盖：app.js", log)
        self.assertIn("修改：index.html", log)
        self.assertIn("查看：index.html", log)
        self.assertIn("`npm test` 失败", log)
        self.assertIn("未成功的文件操作：1 次", log)
        self.assertNotIn("style.css", log)
        self.assertEqual(build_work_log([{"name": "web_search", "status": "completed"}]), "")
        self.assertEqual(with_work_log("answer", log), f"answer\n\n{log}")
        self.assertEqual(with_work_log("answer", ""), "answer")

    def test_responses_capability_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "chat.db")
            database.init()
            self.assertIsNone(database.responses_capability("k", max_age_seconds=60))
            database.set_responses_capability("k", "upstream_rejected_response_state")
            self.assertEqual(database.responses_capability("k", max_age_seconds=60), "upstream_rejected_response_state")
            database.run("UPDATE responses_capabilities SET updated_at=?", (int(time.time()) - 120,))
            self.assertIsNone(database.responses_capability("k", max_age_seconds=60))
            from app import main

            with patch.object(main, "db", database):
                main.record_responses_capability("k", {"disabled": True, "fallback_reason": "store_false"})
                self.assertIsNone(database.responses_capability("k", max_age_seconds=60))
                main.record_responses_capability("k", {"disabled": True, "fallback_reason": "upstream_missing_stored_response_id"})
                self.assertIsNotNone(database.responses_capability("k", max_age_seconds=60))
                main.record_responses_capability("k", {"response_id": "resp_1", "disabled": False})
                self.assertIsNone(database.responses_capability("k", max_age_seconds=60))
            key_a = main.responses_capability_key({"base_url": "https://x.test/v1/"}, "m")
            self.assertEqual(key_a, main.responses_capability_key({"base_url": "https://x.test/v1"}, "m"))
            self.assertNotEqual(key_a, main.responses_capability_key({"base_url": "https://x.test/v1"}, "n"))


if __name__ == "__main__":
    unittest.main()
