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
from app.workspace import ConversationWorkspace, WorkspaceError, apply_text_edits, edited_excerpt


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


def one_edit(content, old, new, replace_all=False):
    return apply_text_edits(content, [{"old_text": old, "new_text": new, "replace_all": replace_all}])


class EditEngineTests(unittest.TestCase):
    def test_exact_replacement_reports_new_line_numbers(self):
        updated, regions, details = one_edit("a\nb\nc\nd\n", "b\n", "b1\nb2\n")
        self.assertEqual((updated, details), ("a\nb1\nb2\nc\nd\n", [{"replacements": 1, "match": "exact"}]))
        excerpt = edited_excerpt(updated, regions)["updated_excerpt"]
        self.assertIn("2|b1", excerpt)
        self.assertIn("3|b2", excerpt)

    def test_copied_line_number_prefixes_are_tolerated(self):
        updated, _, details = one_edit("def f():\n    return 1\n", "2|    return 1", "2|    return 2")
        self.assertEqual(details[0]["match"], "line_numbers_removed")
        self.assertEqual(updated, "def f():\n    return 2\n")

    def test_trailing_whitespace_is_tolerated(self):
        updated, _, details = one_edit("x = 1   \r\ny = 2\r\n", "x = 1\ny = 2", "x = 3\ny = 4")
        self.assertEqual(details[0]["match"], "whitespace_insensitive")
        self.assertEqual(updated, "x = 3\ny = 4\r\n")

    def test_ambiguous_and_missing_text_fail_without_changes(self):
        with self.assertRaisesRegex(WorkspaceError, "出现 2 次.*1、3"):
            one_edit("x\ny\nx\n", "x", "z")
        updated, _, details = one_edit("x\ny\nx\n", "x", "z", True)
        self.assertEqual((updated, details[0]["replacements"]), ("z\ny\nz\n", 2))
        with self.assertRaisesRegex(WorkspaceError, "最接近的当前内容：\n2\\|    total = price \\* qty"):
            one_edit("def f():\n    total = price * qty\n", "  total = price*qty", "x")

    def test_insert_and_delete_with_anchors(self):
        content = "import a\n\ndef main():\n    run()\n    debug()\n"
        updated, _, _ = apply_text_edits(content, [
            {"old_text": "import a\n", "new_text": "import a\nimport b\n"},
            {"old_text": "    debug()\n", "new_text": ""},
        ])
        self.assertEqual(updated, "import a\nimport b\n\ndef main():\n    run()\n")

    def test_host_edit_uses_tolerant_matching_and_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mod.json"
            path.write_text('{\n  "spell": "old"  \n}\n')
            result = json.loads(AgentRuntime(None, 1, "t").execute("host_edit_file", {
                "path": str(path), "edits": [{"old_text": '2|  "spell": "old"', "new_text": '2|  "spell": "new"'}]}))
            self.assertTrue(result["ok"], result)
            self.assertIn('2|  "spell": "new"', result["updated_excerpt"])
            self.assertEqual(result["match"], ["line_numbers_removed"])
            self.assertEqual(path.read_text(), '{\n  "spell": "new"  \n}\n')


class StreamCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_edit_read_does_not_resend_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "edit-flow")
            workspace.root = Path(directory)
            workspace.write_file("game.js", "".join(f"step{n}();\n" for n in range(1, 301)))
            result, requests = await run_stream(workspace, [
                tool_round("r1", "read_file", {"path": "game.js"}),
                tool_round("e1", "edit_file", {"path": "game.js", "edits": [
                    {"old_text": "step10();\n", "new_text": "step10();\nextra();\n"}]}),
                tool_round("r2", "read_file", {"path": "game.js", "start_line": 200}),
                tool_round("r3", "read_file", {"path": "game.js"}),
                answer_round("done"),
            ])
        self.assertEqual([(item["name"], item["status"]) for item in result["tool_trace"]], [
            ("read_file", "completed"), ("edit_file", "completed"),
            ("read_file", "skipped"), ("read_file", "completed"),
        ])
        tool_results = [message["content"] for message in requests[-1]["messages"] if message["role"] == "tool"]
        self.assertIn("1|step1();", tool_results[0])
        edit = json.loads(tool_results[1])
        self.assertIn("11|extra();", edit["updated_excerpt"])
        second = json.loads(tool_results[2])
        self.assertTrue(second["unchanged"])
        self.assertNotIn("content", second)
        self.assertIn("updated_excerpt", second["message"])
        # A second consecutive request is answered with the whole current file.
        third = json.loads(tool_results[3])
        self.assertEqual((third["from_line"], third["through_line"]), (1, 301))
        self.assertIn("11|extra();", third["content"])
        tool_names = {tool["function"]["name"] for tool in requests[0]["tools"]}
        self.assertIn("edit_file", tool_names)
        self.assertNotIn("apply_line_edits", tool_names)

    async def test_other_argument_conventions_and_empty_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "aliases")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "let x = 1;\n")
            result, requests = await run_stream(workspace, [
                tool_round("e0", "edit_file", {}),
                tool_round("e1", "edit_file", {"file_path": "a.js", "old_string": "x = 1", "new_string": "x = 2"}),
                tool_round("e2", "edit_file", {"arguments": {"path": "a.js", "changes": [
                    {"old": "x = 2", "new": "x = 3"}]}}),
                answer_round("done"),
            ])
            self.assertEqual(workspace.read_file("a.js"), "let x = 3;\n")
        statuses = [(item["name"], item["status"]) for item in result["tool_trace"]]
        self.assertEqual(statuses, [("edit_file", "failed"), ("edit_file", "completed"), ("edit_file", "completed")])
        failed = result["tool_trace"][0]
        self.assertEqual((failed["received_argument_keys"], failed["received_argument_chars"]), ([], 2))
        error = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("参数为空", error)
        self.assertIn("拆成更小的 edit_file", error)

    async def test_web_tools_stay_listed_after_web_rounds_in_coding(self):
        class NoWeb:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, *_):
                raise AssertionError("refused web calls must not reach the network")

        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "web-rounds")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "x();\n")
            rounds = [tool_round(f"l{n}", "list_files", {}) for n in range(6)]
            rounds += [
                tool_round("s1", "web_search", {"objective": "o", "search_queries": ["q"]}),
                tool_round("s2", "web_search", {"objective": "o", "search_queries": ["q2"]}),
                tool_round("l7", "list_files", {}),
                answer_round("done"),
            ]
            with patch.object(mimo_local, "ParallelMCPClient", NoWeb):
                result, requests = await run_stream(workspace, rounds, web_enabled=True)
        stats = result["round_stats"]
        # Same tool schema through the refused calls, then web tools are
        # dropped after two refusals instead of looping on them.
        self.assertEqual(len({item["tools_hash"] for item in stats[:8]}), 1)
        self.assertNotEqual(stats[8]["tools_hash"], stats[0]["tools_hash"])
        names = lambda request: {tool["function"]["name"] for tool in request["tools"]}  # noqa: E731
        self.assertIn("web_search", names(requests[7]))
        self.assertNotIn("web_search", names(requests[8]))
        self.assertIn("edit_file", names(requests[8]))
        refused = [item for item in result["tool_trace"] if item["name"] == "web_search"]
        self.assertEqual([item["status"] for item in refused], ["failed", "failed"])
        self.assertIn("轮次额度已用完", refused[0]["error"])
        self.assertIn("web_search=0, fetch_webpage=0", json.dumps(requests[7], ensure_ascii=False))

    async def test_file_written_by_the_model_is_not_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "write-flow")
            workspace.root = Path(directory)
            result, requests = await run_stream(workspace, [
                tool_round("w1", "write_file", {"path": "a.html", "content": "<h1>x</h1>\n"}),
                tool_round("r1", "read_file", {"path": "a.html"}),
                answer_round("done"),
            ])
        self.assertEqual(result["tool_trace"][1]["status"], "skipped")
        self.assertTrue(json.loads(requests[-1]["messages"][-1]["content"])["unchanged"])

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
