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
                result, requests = await run_stream(workspace, rounds, web_enabled=True, web_tool_round_limit=6)
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

    async def test_spent_tool_budget_ends_as_incomplete_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "budget")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "x();\n")
            narrated = [
                [{"choices": [{"delta": {"content": f"第{n}步。", "tool_calls": [{"index": 0, "id": f"c{n}", "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"}}]}}]}]
                for n in range(mimo_local.MAX_AGENT_TOOL_ROUNDS)
            ]
            still_calling = [tool_round(f"f{n}", "list_files", {}) for n in range(mimo_local.FINAL_ANSWER_ATTEMPTS)]
            result, requests = await run_stream(workspace, narrated + still_calling)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["tool_round_limit"], mimo_local.MAX_AGENT_TOOL_ROUNDS)
        self.assertTrue(result["answer"].startswith("第0步。"))
        self.assertEqual(len(result["round_stats"]), mimo_local.MAX_AGENT_TOOL_ROUNDS + mimo_local.FINAL_ANSWER_ATTEMPTS)
        self.assertNotIn("tools", requests[-1])

    async def test_empty_answer_without_tools_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, "空正文"):
            await run_stream(None, [answer_round(""), answer_round("")])

    async def test_reworded_search_is_answered_from_earlier_results(self):
        upstream = []

        class Web:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, name, args):
                upstream.append(args["search_queries"])
                return {"results": [{"url": "https://vcmi.eu/x", "title": "x", "excerpts": ["CSTORM.DEF"]}]}

        def search(call_id, objective, queries):
            return tool_round(call_id, "web_search", {"objective": objective, "search_queries": queries})

        with patch.object(mimo_local, "ParallelMCPClient", Web):
            result, requests = await run_stream(None, [
                search("s1", "storm elemental def file", ["vcmi storm elemental DEF name"]),
                search("s2", "storm elemental def file name", ["storm elemental DEF name vcmi sprites"]),
                search("s3", "unrelated", ["python asyncio cancel task"]),
                answer_round("done"),
            ], web_enabled=True, settings={"thinking": "disabled", "web_tool_backend": "parallel"})
        self.assertEqual(len(upstream), 2)
        statuses = [(item["status"], item.get("similar_to_search")) for item in result["tool_trace"]]
        self.assertEqual(statuses, [("completed", None), ("skipped", 1), ("completed", None)])
        skipped = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"][1]
        self.assertIn("高度相似", skipped)
        self.assertIn("第 1 次搜索", skipped)

    def test_query_similarity_handles_chinese_and_distinct_topics(self):
        a = mimo_local._query_terms("VCMI 风暴元素 DEF 文件名")
        b = mimo_local._query_terms("风暴元素的 DEF 文件名 VCMI")
        self.assertIsNotNone(mimo_local._similar_search(b, [(1, a, "first")]))
        c = mimo_local._query_terms("比亚迪 元UP 电机功率")
        self.assertIsNone(mimo_local._similar_search(c, [(1, a, "first")]))

    async def test_web_calls_without_progress_are_warned_then_refused(self):
        class Web:
            calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, name, args):
                type(self).calls += 1
                return {"results": [{"url": f"https://example.test/{type(self).calls}", "title": "t", "excerpts": ["e"]}]}

        runtime = AgentRuntime(None, 1, "stall")
        topics = ["python asyncio", "rust borrow checker", "kubernetes ingress", "postgres vacuum", "redis streams",
                  "nginx caching", "sqlite wal mode", "docker networking", "git rebase", "bash arrays"]
        rounds = [tool_round(f"s{n}", "web_search", {"objective": t, "search_queries": [t]}) for n, t in enumerate(topics)]
        rounds.append(answer_round("done"))
        with patch.object(mimo_local, "ParallelMCPClient", Web):
            result, requests = await run_stream(
                None, rounds, web_enabled=True, agent_mode=True,
                settings={"thinking": "disabled", "web_tool_backend": "parallel"},
                extra_tools=runtime.tool_definitions, extra_tool_handler=runtime.execute_async,
                max_tool_rounds=96, web_search_limit=96, web_fetch_limit=96, web_tool_round_limit=96,
            )
        trace = result["tool_trace"]
        self.assertEqual(Web.calls, mimo_local.WEB_STALL_REFUSE_CALLS)
        self.assertEqual([item["status"] for item in trace[:8]], ["completed"] * 8)
        self.assertEqual([item["status"] for item in trace[8:]], ["failed", "failed"])
        self.assertIn("联网查询已暂停", trace[8]["error"])
        self.assertIn("git clone", trace[8]["error"])
        self.assertEqual(trace[3]["web_stall_warning"], mimo_local.WEB_STALL_WARN_CALLS)
        self.assertNotIn("web_stall_warning", trace[2])
        tool_results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("[Runtime note] 已经连续 4 次联网", tool_results[3])
        # After two refusals the web tools are dropped so the answer finalizes.
        self.assertNotIn("web_search", {t["function"]["name"] for t in requests[-1].get("tools", [])})

    async def test_progress_resets_the_web_stall_counter(self):
        class Web:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def call_tool(self, name, args):
                return {"results": [{"url": "https://example.test/a", "title": "t", "excerpts": ["e"]}]}

        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "progress")
            workspace.root = Path(directory)
            # Standard mode allows 3 searches per answer; the counter itself is
            # exercised in agent mode above. Here two searches, progress, one more.
            rounds = [tool_round(f"s{n}", "web_search", {"objective": f"topic {n} {'ab'[n % 2]}", "search_queries": [f"query {n} {'xyz'[n % 3]} distinct{n}"]}) for n in range(2)]
            rounds.append(tool_round("w", "write_file", {"path": "notes.md", "content": "plan\n"}))
            rounds.append(tool_round("s9", "web_search", {"objective": "another thing", "search_queries": ["something else entirely"]}))
            rounds.append(answer_round("done"))
            with patch.object(mimo_local, "ParallelMCPClient", Web):
                result, _ = await run_stream(workspace, rounds, web_enabled=True,
                                             settings={"thinking": "disabled", "web_tool_backend": "parallel"},
                                             web_search_limit=10, web_tool_round_limit=10)
        self.assertTrue(all(item["status"] == "completed" for item in result["tool_trace"]))
        self.assertNotIn("web_stall_warning", result["tool_trace"][-1])

    async def test_calls_without_any_mutation_get_a_periodic_nudge(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "nudge")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "x();\n")
            # Standard mode allows 12 tool rounds; the nudge starts at the 10th
            # call without a file change and repeats every 4 calls.
            rounds = [tool_round(f"l{n}", "search_files", {"query": f"needle{n}"}) for n in range(12)]
            rounds.append(answer_round("done"))
            result, requests = await run_stream(workspace, rounds)
        trace = result["tool_trace"]
        warned = [item.get("mutation_stall_warning") for item in trace]
        self.assertEqual(warned, [None] * 9 + [10, None, None])
        tool_results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("没有修改任何文件", tool_results[9])
        self.assertNotIn("没有修改任何文件", tool_results[8])

    async def test_plan_tool_is_listed_and_survives_compaction(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "plan")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "x();\n")
            rounds = [
                tool_round("p1", "update_plan", {"steps": [{"step": "read a.js", "status": "in_progress"}, {"step": "edit", "status": "pending"}]}),
                tool_round("r1", "read_file", {"path": "a.js"}),
                tool_round("s1", "search_files", {"query": "x"}),
                tool_round("s2", "search_files", {"query": "y"}),
                tool_round("s3", "search_files", {"query": "z"}),
                answer_round("done"),
            ]
            result, requests = await run_stream(workspace, rounds, settings={"thinking": "disabled", "context_budget_chars": 40_000})
        self.assertIn("update_plan", {t["function"]["name"] for t in requests[0]["tools"]})
        self.assertEqual(result["tool_trace"][0]["name"], "update_plan")
        self.assertEqual(result["plan"]["steps"][0]["step"], "read a.js")
        plan_result = json.loads([m["content"] for m in requests[1]["messages"] if m["role"] == "tool"][0])
        self.assertIn("[~] 1. read a.js", plan_result["plan"])
        compactions = [s for s in result["round_stats"] if s.get("compacted_after")]
        if compactions:
            checkpoint = mimo_local.checkpoint_payload(requests[-1]["messages"])
            self.assertEqual(checkpoint["plan"]["steps"][1]["step"], "edit")

    def test_long_command_output_keeps_head_and_tail(self):
        from app.agent import bounded_output

        text = "H" * 10_000 + "M" * 50_000 + "T" * 10_000
        out = bounded_output(text, 12_000)
        self.assertTrue(out.startswith("H" * 100))
        self.assertTrue(out.endswith("T" * 100))
        self.assertIn("中间省略", out)
        self.assertLess(len(out), 12_400)
        self.assertEqual(bounded_output("short", 12_000), "short")
        runtime = AgentRuntime(None, 1, "t")
        result = json.loads(runtime.execute("host_run_command", {"command": "seq 1 20000", "cwd": "/tmp"}))
        self.assertTrue(result["ok"])
        self.assertIn("\n20000", result["stdout"])
        self.assertIn("1\n2\n", result["stdout"])
        self.assertIn("中间省略", result["stdout"])

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
    def test_incomplete_answer_notice_lists_completed_work(self):
        from app.main import incomplete_answer

        trace = [{"name": "edit_file", "path": "car.html", "status": "completed"},
                 {"name": "search_files", "path": "", "status": "completed"}]
        text = incomplete_answer("价格已改好。\n", trace, 12)
        self.assertTrue(text.startswith("价格已改好。\n\n---\n"))
        self.assertIn("最多 12 轮", text)
        self.assertIn("发送“继续”", text)
        self.assertIn("- 修改：car.html", text)
        self.assertNotIn(WORK_LOG_HEADER, text)
        self.assertTrue(incomplete_answer("", [], 12).startswith("**本轮工具调用次数已用完"))

    def test_search_accepts_slash_as_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = ConversationWorkspace(1, "slash")
            workspace.root = Path(directory)
            workspace.write_file("a.js", "needle();\n")
            self.assertEqual(workspace.search_files("needle", "/")["matches"][0]["path"], "a.js")
            with self.assertRaises(WorkspaceError):
                workspace.read_snapshot("/")

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
