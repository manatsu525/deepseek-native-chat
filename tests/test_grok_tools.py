"""Offline source-contract tests: no model, search or extraction API calls."""
import hashlib
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from app.grok_tools import read_window, replace_string, web_preview, grep_content, BashTasks, clean_web_content
from app.plan import GrokTodo
from app.prompts import build_system_prompt
from app.workspace import ConversationWorkspace
from app.skills import SkillRegistry
import test_plan_execution as plan_tests


class GrokToolsTests(unittest.TestCase):
    def test_read_offset_is_honored_even_for_small_file(self):
        result = read_window("\n".join(str(n) for n in range(1, 41)), 10, 3)
        self.assertEqual(result["content"].splitlines()[:3], ["10→10", "11", "12"])
        self.assertEqual(result["next_start_line"], 13)

    def test_default_read_limit_and_line_anchors(self):
        result = read_window("x\n" * 1100)
        self.assertEqual(result["through_line"], 1000)
        self.assertEqual(result["next_start_line"], 1001)
        self.assertTrue(result["content"].startswith("1→x\nx\n"))
        self.assertIn("10→x", result["content"])

    def test_utf8_window_is_byte_bounded(self):
        result = read_window(("中文" * 100 + "\n") * 1000)
        self.assertLessEqual(len(result["content"].encode()), 100_100)
        self.assertNotIn("�", result["content"])
        self.assertTrue(result["truncated"])

    def test_replace_matches_exactly_and_creates_files(self):
        self.assertEqual(replace_string("old", "", "new"), "new")
        self.assertEqual(replace_string("a a", "a", "b", True), "b b")
        self.assertEqual(replace_string("a\r\nb\r\n", "a\nb", "a\nc"), "a\r\nc\r\n")
        for old in ("a", "A", "a\u00a0a"):
            with self.assertRaises(ValueError):
                replace_string("a a", old, "b")

    def test_preview_uses_context_percentage_and_preserves_full_unicode_tail(self):
        body = "段落中文\n" * 20_000 + "IMPORTANT TAIL"
        with tempfile.TemporaryDirectory() as root:
            rendered = web_preview(body, root=Path(root), conversation_id="c", context_window=128_000)
            shown = int(rendered.split("showing first ")[-1].split(" ")[0])
            self.assertLessEqual(shown, 15360)
            self.assertGreaterEqual(shown, 15357)
            self.assertLessEqual(len(rendered.encode()), 100_000)
            path = next(Path(root).rglob("*.md"))
            self.assertEqual(path.read_text(), body)
            self.assertNotIn("IMPORTANT TAIL", rendered)

    def test_full_inline_content_is_not_truncated(self):
        self.assertEqual(clean_web_content("![image](data:image/png;base64,abcd)"), "![image]([base64 image/png data removed])")
        self.assertEqual(clean_web_content("metadata:image/png;base64,abcd"), "metadata:image/png;base64,abcd")
        with tempfile.TemporaryDirectory() as root:
            body = "完整正文" * 100
            self.assertEqual(web_preview(body, root=Path(root), conversation_id="c"), body)
            self.assertFalse(list(Path(root).rglob("*.md")))

    def test_artifact_is_private_reusable_and_recoverable_in_ordinary_mode(self):
        with tempfile.TemporaryDirectory() as root, patch("app.workspace.settings", SimpleNamespace(data_dir=Path(root))), patch("app.workspace.WORKSPACES_DIR", Path(root) / "workspaces"):
            convo = "preview-test"
            body = "first\n" * 12_000 + "TAIL EVIDENCE"
            archive = Path(root) / "agent_sessions"
            preview = web_preview(body, root=archive, conversation_id=convo, ordinary=True)
            self.assertIn(".context/web_fetch/", preview)
            workspace = ConversationWorkspace(1, convo)
            workspace.write_file("visible.txt", "visible")
            artifact = next(archive.rglob("*.md"))
            path = ".context/web_fetch/" + artifact.name
            result = json.loads(workspace.execute("read_file", {"target_file": path, "offset": 12001}))
            self.assertIn("TAIL EVIDENCE", result["content"])
            command = "python3 -c 'from pathlib import Path; print(Path(\"" + path + "\").read_text()[-13:])'"
            queried = json.loads(workspace.execute("bash", {"command": command, "description": "query saved content"}))
            self.assertTrue(queried["ok"], queried)
            self.assertIn("TAIL EVIDENCE", queried["stdout"])
            self.assertNotIn(".context", json.loads(workspace.execute("list_dir", {"target_directory": "."}))["content"])
            with self.assertRaises(ValueError):
                workspace.execute("search_replace", {"file_path": path, "old_string": "", "new_string": "bad"})
            web_preview(body, root=archive, conversation_id=convo, ordinary=True)
            self.assertEqual(len(list(archive.rglob("*.md"))), 2)

    def test_grep_is_regex_and_obeys_context_flags(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "example.py").write_text("first\nvalue=12\nlast\n")
            result = grep_content(path, {"pattern": r"value=\d+", "-A": 1}, cwd=path)
            self.assertIn("value=12", result["content"])
            self.assertIn("last", result["content"])

    def test_todo_merge_ids_partial_updates_and_atomic_validation(self):
        todo = GrokTodo()
        todo.apply({"todos": [{"id": "a", "content": "edit", "status": "in_progress"},
                              {"id": "b", "content": "verify", "status": "in_progress"}]})
        todo.apply({"todos": [{"id": "a", "status": "completed"}]})
        self.assertEqual(todo.steps[0]["step"], "edit")
        self.assertEqual(len(todo.steps), 2)
        before = todo.export()
        with self.assertRaises(ValueError):
            todo.apply({"todos": [{"id": "a"}, {"id": "a"}]})
        self.assertEqual(todo.export(), before)
        todo.apply({"merge": False, "todos": [{"id": "c"}]})
        self.assertEqual(todo.steps, [{"id": "c", "step": "c", "status": "pending"}])
        self.assertEqual(GrokTodo(todo.export()).export(), todo.export())

    def test_background_wait_does_not_kill_and_is_conversation_owned(self):
        with tempfile.TemporaryDirectory() as root:
            tasks = BashTasks((1, "a"), Path(root), threading.Event())
            result = tasks.run({"command": "sleep .1; printf finished", "block_until_ms": 0}, Path(root))
            self.assertEqual(result["status"], "running")
            other = BashTasks((2, "a"), Path(root), threading.Event())
            with self.assertRaises(ValueError):
                other.output({"task_id": result["task_id"]})
            done = tasks.output({"task_id": result["task_id"], "timeout_ms": 1000})
            self.assertEqual(done["exit_code"], 0)
            self.assertEqual(done["stdout"], "finished")
            self.assertEqual(tasks.get_output({"task_ids": [result["task_id"]]})["exit_code"], 0)

    def test_background_full_output_is_recoverable(self):
        with tempfile.TemporaryDirectory() as root:
            tasks = BashTasks((1, "logs"), Path(root), threading.Event())
            result = tasks.run({"command": "python3 -c 'print(\"x\" * 50000)'", "block_until_ms": 2000}, Path(root))
            self.assertIn("output truncated", result["stdout"])
            self.assertEqual(len(Path(result["output_file"]).read_text()), 50_001)

    def test_prompt_is_source_rendered_without_old_execution_rules(self):
        for agent in (True, False):
            prompt = build_system_prompt(agent_mode=agent, web_enabled=True, web_backend="parallel",
                                         workspace_access="full", user_timezone="UTC")
            self.assertIn("<work_policy>", prompt)
            self.assertIn("Claim that something is done", prompt)
            for removed in ("never with cat", "never re-read", "exactly one in_progress", "do not read an engine", "${"):
                self.assertNotIn(removed, prompt)
            self.assertNotIn("released by xAI", prompt)

    def test_skill_loader_uses_metadata_then_zero_based_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "test").mkdir()
            (directory / "test/SKILL.md").write_text("---\nname: test\ndescription: demo\n---\n$0 / $ARGUMENTS[1] / $ARGUMENTS / ${SKILL_DIR}")
            with patch("app.skills.SKILL_CONFIG_PATH", directory / "config.json"):
                registry = SkillRegistry(builtin_root=directory / "empty", user_root=directory)
                prompt = registry.prompt()
                self.assertNotIn("$ARGUMENTS", prompt)
                body = registry.invoke("test", "alpha beta")
                self.assertIn("alpha / beta / alpha beta", body)
                self.assertIn('<skill name="test"', body)


class GrokLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_full_body_cached_and_search_results_unabridged_all_protocols(self):
        from app import mimo_local
        from app.grok_tools import READ_TOOL
        from app.db import Database
        body = "web paragraph\n" * 12000 + "TAIL FACT"
        calls = []
        class Provider:
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
            async def call_tool(self, name, arguments):
                calls.append((name, arguments))
                if name == "web_search":
                    return {"results": [{"url": f"https://example.test/{n}", "title": str(n),
                                         "excerpts": ["SOURCE-" + str(n) + "-" + "x" * 2000 + "-END"]} for n in range(25)]}
                return {"results": [{"url": "https://example.test/page", "full_content": body}]}
        for protocol in ("chat_completions", "responses", "messages"):
            for agent in (True, False):
                calls.clear()
                with tempfile.TemporaryDirectory() as root, patch("app.workspace.WORKSPACES_DIR", Path(root) / "workspaces"), \
                     patch("app.workspace.settings", SimpleNamespace(data_dir=Path(root))), \
                     patch.object(mimo_local, "app_settings", SimpleNamespace(data_dir=Path(root))), \
                     patch.object(mimo_local, "ParallelMCPClient", Provider):
                    workspace = ConversationWorkspace(1, "web")
                    artifact = Path(root) / "agent_sessions" / hashlib.sha256(b"web").hexdigest() / "web_fetch" / "1.md"
                    target = str(artifact) if agent else ".context/web_fetch/" + artifact.name
                    events = []
                    def read(name, args):
                        return json.dumps(read_window(Path(args["target_file"]).read_text(), args.get("offset"), args.get("limit")))
                    helper = plan_tests.PlanLoopTests()
                    result, payloads, _, _ = await helper.run_loop([
                        [("search", "web_search", {"objective": "find sources", "search_queries": ["facts"]}),
                         ("fetch", "fetch_webpage", {"url": "https://example.test/page"}),
                         ("cached", "fetch_webpage", {"url": "https://example.test/page"})],
                        [("tail", "read_file", {"target_file": target, "offset": 12001, "limit": 1})], ["done"]],
                        protocol=protocol, agent_mode=agent, workspace=workspace,
                        web_enabled=True, settings={"thinking": "disabled", "web_tool_backend": "parallel"},
                        custom_tools=[READ_TOOL] if agent else None, custom_handler=read if agent else None,
                        record_event=lambda kind, value: events.append((kind, value)))
                    self.assertEqual(result["answer"], "done")
                    self.assertEqual([name for name, _ in calls], ["web_search", "web_fetch"])
                    self.assertTrue(calls[-1][1]["full_content"])
                    self.assertEqual(artifact.read_text(), body)
                    results = {value["message"].get("tool_call_id"): value["message"]["content"]
                               for kind, value in events if kind == "tool/result"}
                    search = json.loads(results["search"])
                    self.assertEqual(len(search["citations"]), 25)
                    self.assertIn("SOURCE-24-" + "x" * 2000 + "-END", search["content"])
                    self.assertEqual(results["fetch"].split("Full content saved to:")[0], results["cached"].split("Full content saved to:")[0])
                    self.assertIn("TAIL FACT", results["tail"])
                    self.assertEqual(result["web_evidence"][0]["content"], body)
                    database = Database(Path(root) / "cache.db")
                    database.init()
                    uid = database.run("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)", ("test", "hash", 1))
                    database.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", ("web", uid, "test", 1, 1))
                    database.upsert_web_evidence(uid, "web", "job", result["web_evidence"])
                    saved = database.web_evidence_for_conversation(uid, "web")[0]
                    self.assertEqual(saved["content"], body)
                    self.assertEqual(saved["content_complete"], 1)

    async def test_many_reads_then_write_no_stall_gate_or_read_suppression_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with tempfile.TemporaryDirectory() as root, patch("app.workspace.WORKSPACES_DIR", Path(root)):
                workspace = ConversationWorkspace(1, "read-loop")
                workspace.write_file("code.py", "before\nsecond\n")
                calls = [(f"read{n}", "read_file", {"target_file": "code.py", "offset": 2, "limit": 1}) for n in range(20)]
                calls.append(("edit", "search_replace", {"file_path": "code.py", "old_string": "before", "new_string": "after"}))
                helper = plan_tests.PlanLoopTests()
                result, payloads, _, _ = await helper.run_loop([calls, ["done"]], protocol=protocol, workspace=workspace, agent_mode=False)
                self.assertEqual(workspace.read_file("code.py"), "after\nsecond\n")
                self.assertEqual(result["answer"], "done")
                self.assertTrue(all(item["status"] == "completed" for item in result["tool_trace"]))
                self.assertNotIn("只读操作已暂停", json.dumps(payloads, ensure_ascii=False))
                self.assertIn("2→second", json.dumps(payloads, ensure_ascii=False))

    async def test_native_todo_does_not_require_execution_evidence_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            helper = plan_tests.PlanLoopTests()
            result, _, commands, _ = await helper.run_loop([
                [("todo", "todo_write", {"todos": [{"id": "edit", "status": "in_progress"}, {"id": "check", "status": "in_progress"}]})],
                [("exec", "run_command", {"command": "write"}), ("done", "todo_write", {"todos": [{"id": "edit", "status": "completed"}]})],
                ["finished"]], protocol=protocol)
            self.assertEqual(commands, ["write"])
            self.assertEqual(result["plan"]["steps"][0]["status"], "completed")
            self.assertEqual(result["plan"]["steps"][1]["status"], "in_progress")
            self.assertFalse(result["incomplete"])
