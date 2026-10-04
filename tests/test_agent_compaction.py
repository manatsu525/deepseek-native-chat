"""Offline source-parity vectors and long-context protocol regressions."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import agent_compaction as compact
from app import mimo_local
from app.agent import AgentRuntime, HOST_TOOLS
import test_plan_execution as loops
from app import workspace as workspaces


def history(size=160_000):
    return [{"role": "system", "content": "stable system"}, {"role": "user", "content": "edit /home/share/app.py"},
            {"role": "assistant", "content": "read", "tool_calls": [{"id": "r", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/home/share/app.py"}'}}]},
            {"role": "tool", "tool_call_id": "r", "content": "old code " + "x" * size}]


SUMMARY = "<summary>\n1. Primary Request and Intent: edit /home/share/app.py\n" + "saved state " * 90 + "\n</summary>"


class CompactionParityTests(unittest.TestCase):
    def test_meter_utf8_native_mirrors_and_reseed(self):
        self.assertEqual(compact.estimate_item({"role": "user", "content": "中文" * 100}), 150)
        item = {"role": "assistant", "content": "abcd", "tool_calls": [{"function": {"arguments": "1234"}}],
                "reasoning_content": "z" * 40, "responses_output_items": [
                    {"type": "message", "content": [{"type": "output_text", "text": "abcd"}]},
                    {"type": "function_call", "arguments": "1234"},
                    {"type": "reasoning", "encrypted_content": "y" * 80}]}
        self.assertEqual(compact.estimate_item(item), 17)  # 8/4 + max(40,80*3/4)/4
        meter = compact.TokenMeter()
        messages = [{"role": "user", "content": "x" * 400}]
        meter.observe(messages, {"input_tokens": 180, "output_tokens": 20})
        messages.append({"role": "tool", "content": "y" * 400})
        self.assertEqual(meter.used(messages), 300)
        meter.reseed([{"role": "user", "content": "a" * 100}])
        self.assertEqual(meter.total, 50)
        restored = compact.TokenMeter(meter.export())
        self.assertEqual(restored.used([{"role": "user", "content": "a" * 100}]), 50)

    def test_summary_cleaning_upstream_vectors(self):
        self.assertEqual(compact.clean_summary("<analysis>draft</analysis><summary>1. Request: fix</summary>"),
                         "Summary:\n1. Request: fix")
        self.assertEqual(compact.clean_summary("<analysis>unfinished"), "")
        self.assertEqual(compact.clean_summary("<analysis>draft<summary>1. Request: fix</summary>"),
                         "Summary:\n1. Request: fix")
        echoed = "<summary>1. Request: fix\n6. Messages: 'ONLY <summary>, </analysis>'\n9. Next: test</summary>"
        cleaned = compact.clean_summary(echoed)
        self.assertIn("1. Request: fix", cleaned)
        self.assertIn("9. Next: test", cleaned)
        self.assertNotIn("<summary>", cleaned)
        self.assertEqual(compact.clean_summary("<summary>**Analysis** draft</analysis>1. Request: fix</summary>"),
                         "Summary:\n1. Request: fix")
        self.assertTrue(compact.SUMMARY_PROMPT.endswith("conversation text only.\n"))
        self.assertEqual(len([line for line in compact.SUMMARY_PROMPT.splitlines() if line[:1].isdigit()]), 9)

    def test_fit_and_segments_preserve_tool_units(self):
        messages = history()
        fitted = compact.fit_history(messages, 300)
        self.assertEqual(fitted[0], messages[0])
        self.assertEqual(fitted[-2]["tool_calls"][0]["id"], fitted[-1]["tool_call_id"])
        self.assertIn("truncated", fitted[-1]["content"])
        self.assertEqual(messages[-1]["content"], "old code " + "x" * 160_000)
        with tempfile.TemporaryDirectory() as directory:
            store = compact.SegmentStore(Path(directory), "../../unsafe")
            first = store.save(messages, SUMMARY)
            second = store.save(messages, SUMMARY)
            self.assertEqual(first["index"], 0)
            self.assertEqual(second["index"], 1)
            self.assertTrue(Path(first["path"]).is_relative_to(Path(directory)))
            self.assertIn("x" * 160_000, Path(first["path"]).read_text())
            self.assertIn("[tool_request: read_file]", Path(first["path"]).read_text())
            self.assertIn("segment_001.md", (store.directory / "INDEX.md").read_text())
            projected = compact.rebuilt_history(messages, SUMMARY, first["directory"], {"todos": [{"step": "edit"}]})
            self.assertIn("edit /home/share/app.py", json.dumps(projected))
            self.assertFalse(any(m.get("tool_calls") or m.get("role") == "tool" for m in projected))
            self.assertEqual(projected[-1]["content"], compact.AUTO_CONTINUE)

    def test_two_pass_split_note_keywords_and_size_classification(self):
        items = [{"role": "user", "content": "x" * 40} for _ in range(40)]
        self.assertEqual(compact.split_two_pass(items), 38)
        items = history(1000)
        split = compact.split_two_pass(items)
        self.assertNotEqual(items[split]["role"], "tool")
        note = "<summary>" + "n" * 1100 + "</summary>"
        self.assertEqual(compact.note_for_pass2(note), "n" * 1100)
        self.assertIn("NOTE₁ truncated", compact.note_for_pass2("n" * 60_001))
        self.assertEqual(compact.keywords("8. Current Work: Authentication render_widget\n9. Next: unrelated"),
                         ["Authentication", "render_widget"])
        self.assertTrue(compact.overflow_error(413, "generic proxy error"))
        self.assertTrue(compact.overflow_error(400, "API error: context_length_exceeded"))
        self.assertFalse(compact.overflow_error(429, "maximum context length rate limit"))
        self.assertFalse(compact.overflow_error(400, "metadata exceeds maximum allowed length"))

    def test_pruning_age_and_image_defaults(self):
        items = []
        for i in range(12):
            items += [{"role": "user", "content": str(i)}, {"role": "tool", "content": str(i) + "x" * 5000}]
        projected = compact.prune_history(items)
        self.assertEqual(projected[1]["content"], compact.HARD_CLEAR)
        self.assertIn("[…trimmed…]", projected[9]["content"])
        self.assertEqual(projected[-1], items[-1])
        self.assertNotIn(compact.HARD_CLEAR, json.dumps(items))
        synthetic = [*items, {"role": "user", "content": "note", "agent_synthetic": "reminder"}]
        retained = compact.prune_history(synthetic, retained=True)
        self.assertEqual(retained[-2]["content"], items[-1]["content"])
        images = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        self.assertIs(compact.image_budget(images, "messages"), images)
        for status, body, reason in ((401, "denied", "auth"), (403, "denied", "turn"),
                                      (400, "invalid_request_error", "schema"), (402, "out of credits", "credit")):
            self.assertEqual(compact.suppress_reason(compact.CompactError(body, status)), reason)


class CompactionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_write_then_compact_then_read_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, \
                 patch.object(mimo_local, "app_settings") as config, \
                 patch.object(workspaces, "settings") as workspace_config, \
                 patch.object(workspaces, "WORKSPACES_DIR", Path(directory) / "workspaces"):
                config.data_dir = workspace_config.data_dir = Path(directory)
                workspace = workspaces.ConversationWorkspace(1, "coding")
                content = "BEFORE\n" + "# payload line\n" * 5000
                events = []
                result, _, _, _ = await loops.PlanLoopTests.run_loop(self, [
                    [("load", "load_tools", {"groups": ["files"]})],
                    [("write", "write_file", {"path": "real.py", "content": content})], [SUMMARY],
                    [("read", "read_file", {"path": "real.py"})], [SUMMARY], ["done"]],
                    protocol=protocol, agent_mode=False, workspace=workspace, context_window_tokens=20_000,
                    record_event=lambda k, p: events.append((k, copy.deepcopy(p))))
                self.assertEqual(result["answer"], "done")
                self.assertEqual((workspace.root / "real.py").read_text(), content)
                read = next(p["message"]["content"] for k, p in events if k == "tool/result" and p["message"].get("tool_call_id") == "read")
                self.assertIn("BEFORE", read)
                self.assertNotIn('"skipped": true', read)
                self.assertEqual(len(list(Path(directory).rglob("segment_*.md"))), 2)

    async def test_ordinary_chat_compacts_and_recovers_private_archive_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, \
                 patch.object(mimo_local, "app_settings") as config, \
                 patch.object(workspaces, "settings") as workspace_config, \
                 patch.object(workspaces, "WORKSPACES_DIR", Path(directory) / "workspaces"):
                config.data_dir = workspace_config.data_dir = Path(directory)
                workspace = workspaces.ConversationWorkspace(1, "ordinary")
                events = []
                messages = history(160_000)[1:]
                messages[-1]["content"] = "old code\n" + ("x" * 100 + "\n") * 1600
                result, payloads, _, _ = await loops.PlanLoopTests.run_loop(self, [
                    [SUMMARY], [("load", "load_tools", {"groups": ["files"]})],
                    [("recover", "read_file", {"path": ".context/compaction/segment_000.md"})], ["done"]],
                    protocol=protocol, messages=messages, context_window_tokens=40_000,
                    agent_mode=False, workspace=workspace, settings={"context_budget_chars": 40_000},
                    record_event=lambda k, p: events.append((k, copy.deepcopy(p))))
                self.assertEqual(result["answer"], "done")
                self.assertEqual(result["round_stats"][0]["context_policy"], "grok-build")
                self.assertIn(".context/compaction/INDEX.md", json.dumps(payloads[1]))
                recovered = next(p["message"]["content"] for k, p in events if k == "tool/result" and p["message"].get("tool_call_id") == "recover")
                self.assertIn("old code", recovered)
                self.assertNotIn('"agent_synthetic":', json.dumps(payloads))
                names = {t["function"]["name"] if "function" in t else t["name"] for t in payloads[2]["tools"]}
                self.assertIn("read_file", names)
                self.assertFalse(any(n.startswith("host_") for n in names))
                self.assertEqual(workspace.list_files(), [])
                matches = workspace.search_files("old code", ".context/compaction")
                self.assertTrue(matches["matches"])
                self.assertTrue(matches["matches"][0]["path"].startswith(".context/"))
                for action in (lambda: workspace.write_file(".context/compaction/INDEX.md", "overwrite"),
                               lambda: workspace.delete_file(".context/compaction/segment_000.md"),
                               lambda: workspace.read_file(".context/../other/secret"),
                               lambda: workspaces.ConversationWorkspace(1, "other").read_file(".context/compaction/segment_000.md")):
                    with self.assertRaises(workspaces.WorkspaceError):
                        action()

    async def test_failure_does_not_mutate_history_or_execute_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            messages, events = history(), []
            original = copy.deepcopy(messages)
            engine = compact.AgentCompactor(compact.SegmentStore(Path(directory), "chat"), 40_000,
                "chat_completions", compact.TokenMeter(), lambda k, p: events.append((k, p)), lambda: False)
            attempts = []
            async def sample(messages, tools, stage):
                attempts.append(stage)
                raise compact.CompactError("invalid_request_error: unknown parameter", 400)
            self.assertIsNone(await engine.compact(messages, [], sample, {}))
            self.assertEqual(attempts, ["verbatim_fitted"])
            self.assertEqual(messages, original)
            self.assertEqual(engine.suppressed, "schema")
            self.assertFalse(list(Path(directory).rglob("segment_*.md")))

    async def test_transient_degenerate_retry_and_overflow_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = compact.AgentCompactor(compact.SegmentStore(Path(directory), "chat"), 40_000,
                "responses", compact.TokenMeter(), lambda *_: None, lambda: False)
            calls = []
            async def sample(messages, tools, stage):
                calls.append(stage)
                if len(calls) == 1:
                    raise compact.CompactError("context_length_exceeded", 413)
                if len(calls) == 2:
                    return "tiny"
                return SUMMARY
            with patch.object(compact.asyncio, "sleep") as sleep:
                projected = await engine.compact(history(), [], sample, {})
            self.assertIsNotNone(projected)
            self.assertEqual(calls, ["verbatim_fitted", "lossy", "lossy"])
            sleep.assert_awaited_once_with(3)

    async def test_prefire_pass2_and_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = compact.TokenMeter()
            engine = compact.AgentCompactor(compact.SegmentStore(Path(directory), "chat"), 50_000,
                "messages", meter, lambda *_: None, lambda: False)
            messages = history(153_000)
            phases = []
            async def sample(messages, tools, stage):
                phases.append(stage)
                return SUMMARY
            engine.maybe_prefire(messages, [], sample)
            await engine.prefire
            messages.append({"role": "user", "content": "new results " + "y" * 20_000})
            projected = await engine.compact(messages, [], sample, {})
            self.assertIsNotNone(projected)
            self.assertEqual(phases, ["prefire", "pass2"])
            engine.cache = {"length": 1, "fingerprint": "stale", "note": SUMMARY}
            engine.maybe_prefire(messages, [], sample)
            self.assertIsNone(engine.cache)
            await engine.close()

    async def test_archive_failure_cancellation_and_non_degenerate_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            stopped = False
            engine = compact.AgentCompactor(compact.SegmentStore(Path(directory), "chat"), 40_000,
                "chat_completions", compact.TokenMeter(), lambda *_: None, lambda: stopped)
            original = history()
            async def sample(*_):
                return SUMMARY
            with patch.object(engine.store, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    await engine.compact(original, [], sample, {})
            self.assertEqual(original, history())
            stopped = True
            with self.assertRaises(asyncio.CancelledError):
                await engine.compact(original, [], sample, {})

    async def test_two_successive_compactions_tool_receipts_and_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, \
                 patch.object(mimo_local, "app_settings") as config:
                config.data_dir = Path(directory)
                messages = history(160_000)[1:]
                # Large real arguments/results force compaction twice. Scripted
                # upstream sees exactly the three wire protocols; no real API.
                events = []
                result, payloads, executed, updates = await loops.PlanLoopTests.run_loop(self, [
                    [SUMMARY], [("write", "run_command", {"command": "x" * 160_000})], [SUMMARY], ["done"]],
                    protocol=protocol, messages=messages, context_window_tokens=40_000,
                    record_event=lambda k, p: events.append((k, copy.deepcopy(p))))
                self.assertEqual(len(executed), 1)
                self.assertEqual(result["answer"], "done")
                self.assertEqual(len(list(Path(directory).rglob("segment_*.md"))), 2)
                self.assertEqual(len([e for e in events if e[0] == "context/compaction_end"]), 2)
                self.assertNotIn("previous_response_id", payloads[-1])
                self.assertIn("Summary:", json.dumps(payloads[-1]))
                self.assertTrue(any(s.get("retry_status", {}).get("status") == "compacting" for s in updates))

    async def test_real_file_write_compact_read_twice_edit_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, \
                 patch.object(mimo_local, "app_settings") as config:
                config.data_dir = Path(directory)
                target = str(Path(directory) / "real.py")
                runtime = AgentRuntime(None, 1, "fixture", is_admin=True)
                calls, outputs = [], []
                async def execute(name, args):
                    calls.append(name)
                    output = runtime.execute(name, args)
                    outputs.append(output)
                    return output
                content = "BEFORE\n" + "# payload line\n" * 5000
                result, _, _, _ = await loops.PlanLoopTests.run_loop(self, [
                    [("w", "write_file", {"path": target, "content": content})], [SUMMARY],
                    [("r1", "read_file", {"path": target})], [SUMMARY],
                    [("r2", "read_file", {"path": target})], [SUMMARY],
                    [("edit", "edit_file", {"path": target, "edits": [{"old_text": "BEFORE", "new_text": "AFTER"}]})],
                    ["done"]], protocol=protocol, context_window_tokens=20_000,
                    custom_tools=HOST_TOOLS, custom_handler=execute)
                self.assertEqual(result["answer"], "done")
                self.assertEqual(calls, ["write_file", "read_file", "read_file", "edit_file"])
                self.assertIn("BEFORE", outputs[1])
                self.assertIn("BEFORE", outputs[2])  # no FileKnowledge "already in context" stub
                self.assertTrue(Path(target).read_text().startswith("AFTER\n"))
                self.assertEqual(len(list(Path(directory).rglob("segment_*.md"))), 3)
                archives = "\n".join(p.read_text() for p in Path(directory).rglob("segment_*.md"))
                self.assertIn("# payload line", archives)
                # Existing host read_file really can recover an archive; no new
                # special memory tool or network operation is required.
                archive = next(Path(directory).rglob("segment_000.md"))
                self.assertIn("BEFORE", runtime.execute("read_file", {"path": str(archive)}))

    async def test_long_search_fetch_transcripts_compact_without_network(self):
        messages = [{"role": "user", "content": "research the fixture"}]
        for i, name in enumerate(("web_search", "fetch_webpage", "fetch_webpage")):
            messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                "function": {"name": name, "arguments": json.dumps({"url": "https://example.invalid/" + str(i)})}}]},
                {"role": "tool", "tool_call_id": str(i), "content": ("SEARCH_EVIDENCE " if i == 0 else "PAGE_BODY ") * 9000}]
        for protocol, agent_mode in ((p, a) for p in ("chat_completions", "responses", "messages") for a in (True, False)):
            with self.subTest(protocol=protocol, agent_mode=agent_mode), tempfile.TemporaryDirectory() as directory, \
                 patch.object(mimo_local, "app_settings") as config:
                config.data_dir = Path(directory)
                result, payloads, executed, _ = await loops.PlanLoopTests.run_loop(self, [[SUMMARY], ["done"]],
                    protocol=protocol, agent_mode=agent_mode, context_window_tokens=80_000, messages=messages)
                self.assertEqual(executed, [])
                self.assertEqual(result["answer"], "done")
                self.assertIn("PAGE_BODY " * 100, next(Path(directory).rglob("segment_000.md")).read_text())
                self.assertNotIn("PAGE_BODY " * 100, json.dumps(payloads[-1]))

    async def test_auth_aborts_without_sampling_oversized_history(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = compact.AgentCompactor(compact.SegmentStore(Path(directory), "chat"), 40_000,
                "responses", compact.TokenMeter(), lambda *_: None, lambda: False)
            async def denied(*_):
                raise compact.CompactError("unauthorized", 401)
            with self.assertRaises(compact.CompactError):
                await engine.compact(history(), [], denied, {})
            self.assertEqual(engine.suppressed, "auth")

    async def test_http_overflow_resubmits_same_tool_round_after_compaction(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mimo_local, "app_settings") as config:
            config.data_dir = Path(directory)
            events = []
            result, payloads, executed, _ = await loops.PlanLoopTests.run_loop(self, [[SUMMARY], ["done"]],
                status_sequence=[413, 200, 200], context_window_tokens=40_000,
                record_event=lambda k, p: events.append((k, p)))
            self.assertEqual(result["answer"], "done")
            self.assertEqual(executed, [])
            self.assertEqual(len(payloads), 3)
            self.assertEqual(len(result["round_stats"]), 1)
            self.assertTrue(any(k == "context/overflow_resubmit" for k, _ in events))
