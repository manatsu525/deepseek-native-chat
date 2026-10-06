"""Offline durable-history and DSH-style checklist regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent_session import AgentJournal, project_history, remove_legacy_web_evidence, LEGACY_WEB_EVIDENCE_PREFIX
from app.db import Database
from app.plan import ChecklistPlan
from app import main, mimo_local
import test_plan_execution as loops
import test_responses_state as responses_tests


def seed(db):
    db.init()
    uid = db.run("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)", ("test", "hash", 1))
    pid = db.run("INSERT INTO providers(user_id,name,api_key,base_url,model,created_at) VALUES(?,?,?,?,?,?)",
                 (uid, "test", "secret", "https://test.invalid", "test", 1))
    db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", ("chat", uid, "test", 1, 1))
    return uid, pid


def job(db, uid, pid, ident):
    db.run("INSERT INTO jobs(id,user_id,conversation_id,provider_id,model,effort,chat_mode,status,created_at,updated_at) "
           "VALUES(?,?,?,?,?,?,?,?,?,?)", (ident, uid, "chat", pid, "test", "high", "agent", "queued", 1, 1))


def assistant(call_id="write"):
    return {"role": "assistant", "content": "save file", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "run_command", "arguments": '{"command":"write"}'}}]}


class AgentSessionTests(unittest.TestCase):
    def test_checklist_replacement_validation_and_legacy_restore(self):
        plan = ChecklistPlan()
        self.assertIsNone(plan.export())
        self.assertFalse(plan.needs_plan)
        plan.apply({"todos": [{"content": "build", "status": "in_progress"},
                              {"content": "test", "status": "in_progress"}]})
        before = plan.export()
        for invalid in ([{"content": "", "status": "pending"}],
                        [{"content": "build", "status": "pending"}] * 2,
                        [{"content": "test", "status": "completed", "evidence": ["id"]}]):
            with self.assertRaises(ValueError):
                plan.apply({"todos": invalid})
            self.assertEqual(plan.export(), before)
        plan.apply({"todos": [{"content": "different approach", "status": "completed"}]})
        self.assertFalse(plan.unfinished)
        self.assertFalse(plan.needs_plan)
        plan.apply({"todos": []})
        self.assertEqual(plan.export()["steps"], [])
        legacy = ChecklistPlan({"version": 1, "steps": [{"id": "s1", "step": "old", "status": "done",
                                                         "evidence": ["write"], "outcome": "ok"}]})
        self.assertEqual(legacy.steps, [{"step": "old", "status": "completed"}])

    def test_durable_projection_request_immutability_and_unknown_outcome(self):
        quoted = "quoted" + LEGACY_WEB_EVIDENCE_PREFIX + "user's own quotation"
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,original"}}
        sample = [{"role": "user", "content": "question" + LEGACY_WEB_EVIDENCE_PREFIX + "duplicate page"},
                  {"role": "tool", "content": "original webpage result"},
                  {"role": "user", "content": quoted},
                  {"role": "user", "content": [{"type": "text", "text": "question"}, image,
                    {"type": "text", "text": LEGACY_WEB_EVIDENCE_PREFIX.lstrip("\n") + "duplicate page"}]}]
        remove_legacy_web_evidence(sample, {"question", quoted})
        self.assertEqual(sample[0]["content"], "question")
        self.assertEqual(sample[1]["content"], "original webpage result")
        self.assertEqual(sample[2]["content"], quoted)
        self.assertEqual(sample[3]["content"], [{"type": "text", "text": "question"}, image])
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "test.db")
            uid, pid = seed(db)
            job(db, uid, pid, "one")
            journal = AgentJournal(db, "one", "chat")
            journal.append("turn/start", {"scope": "same"})
            journal.append("history/start", {"messages": [{"role": "user", "content": "build"}]})
            body = {"tools": [{"name": "write"}], "input": "original"}
            journal.append("model/request", {"body": body})
            body["input"] = "mutated"
            journal.append("model/attempt_failed", {"content": "failed draft"})
            journal.append("model/preview", {"content": "interrupted stream"})
            journal.flush_preview()
            message = assistant()
            message["reasoning_content"] = "opaque reasoning"
            journal.append("assistant/message", {"message": message})
            journal.append("tool/start", {"call_id": "write"})
            reopened = AgentJournal(Database(db.path), "one", "chat")
            history = reopened.history(scope="same")
            self.assertIn("结果未知", history[-1]["content"])
            self.assertNotIn("failed draft", json.dumps(history))
            self.assertNotIn("interrupted stream", json.dumps(history))
            self.assertEqual(history[-2]["reasoning_content"], "opaque reasoning")
            self.assertNotIn("reasoning_content", reopened.history(scope="different")[-2])
            self.assertEqual(reopened.events()[2]["payload"]["body"]["input"], "original")
            self.assertIsNone(AgentJournal(db, "one", "other-conversation").history())
            journal.append("tool/result", {"message": {"role": "tool", "tool_call_id": "write", "content": "saved"}})
            self.assertEqual(reopened.history(scope="same")[-1]["content"], "saved")
            journal.append("context/checkpoint", {"messages": [{"role": "user", "content": "bounded projection"}]})
            self.assertEqual(reopened.history(scope="same"), [{"role": "user", "content": "bounded projection"}])
            journal.append("context/state", {"meter": {"total": 1000, "baseline": 500}, "window": 256_000,
                                             "loaded_groups": ["skills"], "plan": {"steps": []},
                                             "prefire_cache": {"note": "saved note"}})
            self.assertEqual(reopened.context_state(scope="same")["meter"]["total"], 1000)
            self.assertNotIn("meter", reopened.context_state(scope="changed"))
            self.assertEqual(reopened.context_state(scope="changed")["loaded_groups"], ["skills"])
            raw = next(Path(directory).rglob("updates.jsonl")).read_text()
            self.assertIn('"content": "saved"', raw)
            self.assertNotIn("failed draft", raw)
            with patch.object(main, "db", db):
                self.assertTrue(main.get_execution("one", user={"id": uid})["events"])
                with self.assertRaises(main.HTTPException):
                    main.get_execution("one", user={"id": uid + 1})
            db.run("DELETE FROM conversations WHERE id=?", ("chat",))
            self.assertFalse(reopened.events())


class AgentSessionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_tools_require_explicit_loading_and_survive_restoration(self):
        from app.workspace import ConversationWorkspace
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory, patch.object(mimo_local, "app_settings") as config:
                config.data_dir = Path(directory)
                workspace = ConversationWorkspace(1, "deferred")
                workspace.root = Path(directory) / "workspace"
                workspace.context_archive = Path(directory) / "archive"
                workspace.context_archive.mkdir()
                workspace.write_file("uploaded.txt", "An uploaded file does not load schemas.")
                events = []
                def record(kind, payload):
                    events.append({"kind": kind, "payload": copy.deepcopy(payload)})
                def names(payload):
                    return [t.get("name", (t.get("function") or {}).get("name")) for t in payload.get("tools", [])]
                async def run(rounds, messages=None, state=None):
                    return await loops.PlanLoopTests.run_loop(self, rounds, protocol=protocol, agent_mode=False,
                        workspace=workspace, custom_tools=[], messages=messages, record_event=record,
                        agent_context_state=state, context_window_tokens=128_000)
                _, first, _, _ = await run([["hello"]])
                self.assertIn("load_tools", names(first[0]))
                self.assertNotIn("write_file", names(first[0]))
                history = project_history(events)
                state = next(e["payload"] for e in reversed(events) if e["kind"] == "context/state")
                events.clear()
                _, second, _, _ = await run([["another ordinary answer"]], [*history, {"role": "user", "content": "hello again"}], state)
                self.assertEqual(first[0]["tools"], second[0]["tools"])
                self.assertEqual(first[0].get("instructions", first[0].get("system", first[0].get("messages", [{}])[0])),
                                 second[0].get("instructions", second[0].get("system", second[0].get("messages", [{}])[0])))
                history = project_history(events)
                events.clear()
                result, loaded, _, _ = await run([
                    [("load-files", "load_tools", {"groups": ["files"]})],
                    [("write-file", "write_file", {"path": "new.txt", "content": "saved"})], ["done"]],
                    [*history, {"role": "user", "content": "write a file"}], state)
                self.assertNotIn("write_file", names(loaded[0]))
                self.assertIn("write_file", names(loaded[1]))
                self.assertEqual(workspace.read_file("new.txt"), "saved")
                self.assertFalse(result["incomplete"])
                history = project_history(events)
                state = next(e["payload"] for e in reversed(events) if e["kind"] == "context/state")
                self.assertIn("files", state["loaded_groups"])
                events.clear()
                # A new loop/process reconstructs schemas from persisted state
                # even when compaction has removed the old load_tools receipt.
                _, resumed, _, _ = await run([["still loaded"]], [{"role": "user", "content": "continue"}], state)
                self.assertEqual(loaded[-1]["tools"], resumed[0]["tools"])
                self.assertEqual(names(resumed[0]).count("write_file"), 1)
                events.clear()
                # Older histories with successful file work but no load receipt
                # also retain their loaded state on the next request.
                old = [{"role": "user", "content": "old task"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "old-read", "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"new.txt"}'}}]},
                    {"role": "tool", "tool_call_id": "old-read", "content": "saved"},
                    {"role": "user", "content": "continue"}]
                _, recovered, _, _ = await run([["restored"]], old)
                self.assertIn("read_file", names(recovered[0]))

    async def test_many_turns_keep_web_results_once_and_clean_legacy_history(self):
        from app.agent_compaction import estimate_history
        page = "WEB_FACT_unique_document_" + "网页事实，仅保留原始工具结果。" * 1500 + "_TAIL_FACT"
        old = "LEGACY_DOCUMENT_unique_" + "历史正文。" * 1000 + "_LEGACY_TAIL"
        for mode in ("standard", "agent"):
            for protocol, kind in (("chat_completions", "custom"), ("responses", "custom_response"), ("messages", "custom_messages")):
                with self.subTest(mode=mode, protocol=protocol), tempfile.TemporaryDirectory() as directory:
                    db = Database(Path(directory) / "test.db")
                    uid, pid = seed(db)
                    db.run("UPDATE providers SET provider_type=?,settings_json=? WHERE id=?", (kind, json.dumps({
                        "model_settings": {"test": {"context_window_tokens": 64_000,
                            "advanced_enabled": True, "advanced_request": {"model": "test", "store": False}}}}), pid))
                    job(db, uid, pid, "legacy")
                    db.run("UPDATE jobs SET status='completed',chat_mode=? WHERE id='legacy'", (mode,))
                    legacy_meta = {"job_id": "legacy"}
                    if kind == "custom_response":
                        provider = db.one("SELECT * FROM providers WHERE id=?", (pid,))
                        legacy_meta["responses_state"] = {"response_id": "obsolete-server-history", "disabled": False,
                            "scope": main.state_scope(provider, db.one("SELECT * FROM jobs WHERE id='legacy'"), main.custom_settings_for_model(provider, "test"))}
                    db.run("INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)", ("chat", "user", "legacy question", "{}", 1))
                    db.run("INSERT INTO messages(conversation_id,role,content,meta_json,created_at) VALUES(?,?,?,?,?)", ("chat", "assistant", "legacy done", json.dumps(legacy_meta), 1))
                    legacy = AgentJournal(db, "legacy", "chat")
                    legacy.append("history/start", {"messages": [
                        {"role": "user", "content": "legacy question" + LEGACY_WEB_EVIDENCE_PREFIX + old},
                        assistant("old-fetch"), {"role": "tool", "tool_call_id": "old-fetch", "content": old},
                        {"role": "assistant", "content": "legacy done"}]})
                    db.upsert_web_evidence(uid, "chat", "legacy", [{"url": "https://example.test/old", "canonical_url": "https://example.test/old", "content": old}])
                    sizes = []
                    request_sizes = []
                    async def execute(name, args):
                        return page
                    async def stream(**kwargs):
                        # Exercise the actual serializers and loop with mocked
                        # model output; the acquisition result is a local fixture.
                        self.assertFalse(kwargs.get("user_context_addendum"))
                        self.assertEqual(kwargs["context_window_tokens"], 64_000)
                        if kind == "custom_response" and not sizes:
                            self.assertFalse(kwargs.get("responses_state", {}).get("response_id"))
                        messages = kwargs["messages"]
                        sent = json.dumps(messages, ensure_ascii=False)
                        self.assertNotIn(LEGACY_WEB_EVIDENCE_PREFIX, sent)
                        for prior in range(len(sizes) + 1):
                            self.assertIn(f'"follow-up {prior}"', sent)
                        self.assertLessEqual(sent.count("LEGACY_DOCUMENT_unique_"), 1)
                        self.assertLessEqual(sent.count("WEB_FACT_unique_document_"), 1)
                        if len(sizes) < 8:
                            self.assertEqual(sent.count("LEGACY_DOCUMENT_unique_"), 1)
                            self.assertEqual(sent.count("WEB_FACT_unique_document_"), 0 if not sizes else 1)
                        sizes.append(estimate_history(messages))
                        rounds = [[("new-fetch", "run_command", {"command": "mock acquisition"})], ["done"]] if len(sizes) == 1 else [["done"]]
                        result, payloads, _, _ = await loops.PlanLoopTests.run_loop(
                            self, rounds, protocol=protocol, record_event=kwargs["record_event"], messages=messages,
                            context_window_tokens=kwargs["context_window_tokens"], agent_mode=mode == "agent", custom_handler=execute,
                            settings={"advanced_enabled": True, "advanced_request": {"model": "test", "store": False}})
                        for payload in payloads:
                            serialized = json.dumps(payload, ensure_ascii=False)
                            self.assertLessEqual(serialized.count("LEGACY_DOCUMENT_unique_"), 1)
                            self.assertLessEqual(serialized.count("WEB_FACT_unique_document_"), 1)
                            self.assertNotIn("WEB EVIDENCE FROM THIS CONVERSATION:", serialized)
                        request_sizes.append(len(json.dumps(payloads[0], ensure_ascii=False).encode()))
                        if len(sizes) == 1:
                            result["web_evidence"] = [{"url": "https://example.test/new", "canonical_url": "https://example.test/new", "content": page}]
                        return result
                    with patch.object(main, "db", db), patch.object(main, "custom_streamer", return_value=stream), \
                         patch.object(main, "AgentRuntime"), patch.object(main, "AgentSharedWorkspace") as workspace, \
                         patch.object(main, "context_window_tokens", return_value=1_048_576), \
                         patch.object(main, "build_agent_skills_prompt", return_value=""), \
                         patch.object(mimo_local, "app_settings") as config:
                        config.data_dir = Path(directory)
                        workspace.return_value.list_files.return_value = []
                        for index in range(24):
                            ident = f"turn-{index}"
                            job(db, uid, pid, ident)
                            db.run("UPDATE jobs SET chat_mode=? WHERE id=?", (mode, ident))
                            db.run("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", ("chat", "user", f"follow-up {index}", 1))
                            await main.run_job(ident)
                            state = db.one("SELECT status,error FROM jobs WHERE id=?", (ident,))
                            self.assertEqual(state["status"], "completed", f"{ident}: {state['error']}")
                    # Only tiny user/assistant exchanges are added after the
                    # document was first acquired; no per-turn page-sized growth.
                    self.assertLess(sizes[-1] - sizes[1], 3000)
                    self.assertLess(max(sizes[1:]) - sizes[1], 3000)
                    self.assertTrue(all(b - a < 750 for a, b in zip(sizes[1:], sizes[2:])))
                    self.assertLess(request_sizes[-1] - request_sizes[1], 8000)
                    # The existing Grok ten-turn pruning can remove old bodies
                    # from requests; their full receipts remain recoverable.
                    transcript = next(Path(directory).rglob("updates.jsonl")).read_text()
                    self.assertIn("_TAIL_FACT", transcript)
                    self.assertIn("_LEGACY_TAIL", transcript)
                    print(f"web-history regression {mode}/{protocol}: 24 turns, estimated tokens first={sizes[1]} peak={max(sizes[1:])} final={sizes[-1]}, request bytes {request_sizes[1]} -> {request_sizes[-1]}")

    async def test_committed_tools_replay_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol):
                events = []
                def record(kind, payload):
                    events.append({"kind": kind, "payload": copy.deepcopy(payload)})
                result, payloads, executed, _ = await loops.PlanLoopTests.run_loop(self, [
                    [("first", "run_command", {"command": "write"})], ["saved"]],
                    protocol=protocol, record_event=record)
                history = project_history(events)
                self.assertEqual(executed, ["write"])
                self.assertEqual(history[-1]["content"], "saved")
                self.assertEqual(history[-2]["role"], "tool")
                self.assertIn('"exit_code": 0', history[-2]["content"])
                requests = [event["payload"]["body"] for event in events if event["kind"] == "model/request"]
                self.assertEqual(requests, payloads)
                self.assertNotIn("Authorization", json.dumps(events))
                self.assertFalse(result["incomplete"])
                _, next_payloads, _, _ = await loops.PlanLoopTests.run_loop(self, [["already saved"]],
                    protocol=protocol, messages=[*history, {"role": "user", "content": "what changed?"}])
                self.assertIn("first", json.dumps(next_payloads[0]))
                self.assertIn("exit_code", json.dumps(next_payloads[0]))
        # Record both the chained request and actual fallback body, while
        # proving that the completed tool is not executed twice.
        events = []
        def record(kind, payload):
            events.append({"kind": kind, "payload": copy.deepcopy(payload)})
        transport = responses_tests.Transport([
            responses_tests.call("resp_tool"), (400, "previous_response_id not found"), responses_tests.final()])
        result = await responses_tests.ResponsesStateTests.run_stream(
            self, transport, agent_mode=True, record_event=record)
        requests = [event["payload"]["body"] for event in events if event["kind"] == "model/request"]
        self.assertEqual(requests, transport.payloads)
        self.assertEqual(len([event for event in events if event["kind"] == "tool/start"]), 1)
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertIn("previous_response_id", requests[1])
        self.assertNotIn("previous_response_id", requests[2])

    async def test_restored_history_compacts_without_losing_user_requests(self):
        history = [{"role": "user", "content": "original requirements"}]
        for index in range(8):
            history.extend([assistant(str(index)), {"role": "tool", "tool_call_id": str(index), "content": "x" * 15_000},
                            {"role": "user", "content": f"requirement {index}"}])
        history.append({"role": "user", "content": "latest request"})
        events = []
        def record(kind, payload):
            events.append({"kind": kind, "payload": copy.deepcopy(payload)})
        # The old 40k character switch no longer applies to Agent.
        _, payloads, _, _ = await loops.PlanLoopTests.run_loop(self, [["done"]], messages=history,
                    settings={"context_budget_chars": 40_000}, record_event=record)
        self.assertIn("x" * 15_000, json.dumps(payloads[0]))
        summary = "<summary>1. Primary Request: original requirements; " + "; ".join(
            f"requirement {index}" for index in range(8)) + "; latest request. " + "state " * 120 + "</summary>"
        for protocol in ("chat_completions", "responses", "messages"):
            with tempfile.TemporaryDirectory() as directory, patch.object(mimo_local, "app_settings") as config:
                config.data_dir = Path(directory)
                events.clear()
                _, payloads, executed, _ = await loops.PlanLoopTests.run_loop(self, [[summary], ["done"]],
                    protocol=protocol, messages=history, context_window_tokens=32_000, record_event=record)
                self.assertEqual(executed, [])
                sent = json.dumps(payloads[1])
                self.assertLess(len(sent), 20_000)
                for index in range(8):
                    self.assertIn(f"requirement {index}", sent)
                self.assertIn("latest request", sent)
                self.assertNotIn("previous_response_id", payloads[1])
                projected = project_history(events)
                self.assertEqual(projected[-1]["content"], "done")
                self.assertFalse(any(m.get("tool_calls") or m.get("role") == "tool" for m in projected))
                archive = next(Path(directory).rglob("segment_000.md"))
                self.assertIn("x" * 15_000, archive.read_text())

    async def test_main_cross_turn_and_restart_use_committed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "test.db")
            uid, pid = seed(db)
            captured = []
            async def stream(**kwargs):
                captured.append(copy.deepcopy(kwargs["messages"]))
                record = kwargs["record_event"]
                record("history/start", {"messages": kwargs["messages"]})
                if len(captured) == 1:
                    record("assistant/message", {"message": assistant()})
                    record("tool/start", {"call_id": "write"})
                    record("tool/result", {"message": {"role": "tool", "tool_call_id": "write", "content": "real result"}})
                record("assistant/message", {"message": {"role": "assistant", "content": "done"}})
                return {"answer": "done", "reasoning": "", "searches": [], "sources": [], "usage": {}}
            with patch.object(main, "db", db), patch.object(main, "custom_stream_response", stream), \
                 patch.object(main, "AgentRuntime"), patch.object(main, "AgentSharedWorkspace") as workspace, \
                 patch.object(main, "context_window_tokens", return_value=120_000), \
                 patch.object(main, "build_agent_skills_prompt", return_value=""):
                workspace.return_value.list_files.return_value = []
                for ident in ("first", "second"):
                    job(db, uid, pid, ident)
                    db.run("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", ("chat", "user", ident, 1))
                    await main.run_job(ident)
                    self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (ident,))["status"], "completed")
                self.assertEqual(captured[1][-1]["content"], "second")
                self.assertIn("real result", json.dumps(captured[1]))
                # A restarted job resumes its own committed operation even before
                # an assistant message has been added to the UI history.
                job(db, uid, pid, "restart")
                journal = AgentJournal(db, "restart", "chat")
                journal.append("history/start", {"messages": [{"role": "user", "content": "restart"}]})
                journal.append("assistant/message", {"message": assistant("unknown")})
                journal.append("tool/start", {"call_id": "unknown"})
                await main.run_job("restart")
                self.assertIn("结果未知", captured[2][-1]["content"])
                self.assertEqual(captured[2][0]["content"], "restart")
                self.assertEqual(AgentJournal(db, "restart", "chat").events()[-1]["kind"], "turn/end")

    async def test_ordinary_main_cross_turn_and_restart_use_committed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "test.db")
            uid, pid = seed(db)
            captured = []
            async def stream(**kwargs):
                self.assertFalse(kwargs.get("agent_mode", False))
                self.assertIsNotNone(kwargs.get("workspace"))
                captured.append(copy.deepcopy(kwargs["messages"]))
                record = kwargs["record_event"]
                record("history/start", {"messages": kwargs["messages"]})
                if len(captured) == 1:
                    record("assistant/message", {"message": assistant()})
                    record("tool/start", {"call_id": "write"})
                    record("tool/result", {"message": {"role": "tool", "tool_call_id": "write", "content": "real result"}})
                record("assistant/message", {"message": {"role": "assistant", "content": "done"}})
                return {"answer": "done", "reasoning": "", "searches": [], "sources": [], "usage": {}}
            with patch.object(main, "db", db), patch.object(main, "custom_stream_response", stream), \
                 patch.object(main, "AgentRuntime"), patch.object(main, "AgentSharedWorkspace") as workspace, \
                 patch.object(main, "context_window_tokens", return_value=120_000), \
                 patch.object(main, "build_agent_skills_prompt", return_value=""):
                workspace.return_value.list_files.return_value = []
                for ident in ("first", "second"):
                    job(db, uid, pid, ident)
                    db.run("UPDATE jobs SET chat_mode='standard' WHERE id=?", (ident,))
                    db.run("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", ("chat", "user", ident, 1))
                    await main.run_job(ident)
                    self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (ident,))["status"], "completed")
                self.assertEqual(captured[1][-1]["content"], "second")
                self.assertIn("real result", json.dumps(captured[1]))
                # A restarted job resumes its own committed operation even before
                # an assistant message has been added to the UI history.
                job(db, uid, pid, "restart")
                db.run("UPDATE jobs SET chat_mode='standard' WHERE id='restart'")
                journal = AgentJournal(db, "restart", "chat")
                journal.append("history/start", {"messages": [{"role": "user", "content": "restart"}]})
                journal.append("assistant/message", {"message": assistant("unknown")})
                journal.append("tool/start", {"call_id": "unknown"})
                await main.run_job("restart")
                self.assertIn("结果未知", captured[2][-1]["content"])
                self.assertEqual(captured[2][0]["content"], "restart")
                self.assertEqual(AgentJournal(db, "restart", "chat").events()[-1]["kind"], "turn/end")
