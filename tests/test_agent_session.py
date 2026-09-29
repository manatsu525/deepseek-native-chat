"""Offline durable-history and DSH-style checklist regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent_session import AgentJournal, project_history
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
            with patch.object(main, "db", db):
                self.assertTrue(main.get_execution("one", user={"id": uid})["events"])
                with self.assertRaises(main.HTTPException):
                    main.get_execution("one", user={"id": uid + 1})
            db.run("DELETE FROM conversations WHERE id=?", ("chat",))
            self.assertFalse(reopened.events())


class AgentSessionLoopTests(unittest.IsolatedAsyncioTestCase):
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
        _, payloads, _, _ = await loops.PlanLoopTests.run_loop(self, [["done"]], messages=history,
                    settings={"context_budget_chars": 40_000}, record_event=record)
        sent = payloads[0]["messages"]
        self.assertLess(len(json.dumps(sent)), 70_000)
        for index in range(8):
            self.assertIn(f"requirement {index}", json.dumps(sent))
        self.assertEqual(sent[-1]["content"], "latest request")
        self.assertEqual(project_history(events)[:-1], sent[1:])
        call_ids = {call["id"] for message in sent for call in message.get("tool_calls") or []}
        result_ids = {message["tool_call_id"] for message in sent if message.get("role") == "tool"}
        self.assertEqual(call_ids, result_ids)

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
