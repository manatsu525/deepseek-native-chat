"""Offline execution-contract tests; no provider credentials or API calls."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.plan import ExecutionPlan
from app import mimo_local


def steps():
    return [{"step": "implement", "status": "in_progress"}, {"step": "verify", "status": "pending"}]


class ExecutionPlanTests(unittest.TestCase):
    def test_evidence_transition_and_atomic_rejection(self):
        p = ExecutionPlan()
        p.apply({"steps": steps(), "note": "keep decision"})
        proposed = copy.deepcopy(p.steps)
        proposed[0].update(status="done", outcome="wrote file", evidence=["write1"])
        proposed[1]["status"] = "in_progress"
        before = p.export()
        with self.assertLogs("app.plan.debug", level="WARNING") as logs:
            with self.assertRaises(ValueError):
                p.apply({"steps": proposed})
        self.assertIn('"validation": "done_transition"', "\n".join(logs.output))
        self.assertIn('"available_evidence"', "\n".join(logs.output))
        self.assertEqual(p.export(), before)
        p.record("write1", "write_file", "failed", "a.py", "error")
        with self.assertRaises(ValueError):
            p.apply({"steps": proposed})
        p.record("write2", "write_file", "completed", "a.py", "saved")
        proposed[0]["evidence"] = ["write2"]
        p.apply({"steps": proposed})
        self.assertEqual(p.active["id"], "s2")
        self.assertEqual(p.note, "keep decision")
        proposed[1].update(status="done", outcome="tested", evidence=["write2"])
        with self.assertRaises(ValueError):
            p.apply({"steps": proposed})
        p.record("check", "run_command", "completed", "", "passed")
        proposed[1]["evidence"] = ["check"]
        p.apply({"steps": proposed})
        self.assertFalse(p.unfinished)
        self.assertTrue(p.needs_plan)
        reopened = copy.deepcopy(p.steps)
        reopened[0]["status"] = "in_progress"
        with self.assertRaises(ValueError):
            p.apply({"steps": reopened})
        p.apply({"steps": reopened, "replan_reason": "regression found"})
        self.assertEqual(p.receipts["s1"], [])
        proposed = copy.deepcopy(p.steps)
        proposed[0].update(status="done",outcome="fixed",evidence=["write2"])
        with self.assertRaises(ValueError):
            p.apply({"steps": proposed})

    def test_replan_and_single_active_contract(self):
        p = ExecutionPlan()
        invalid = steps()
        invalid[1]["status"] = "in_progress"
        with self.assertRaises(ValueError):
            p.apply({"steps": invalid})
        p.apply({"steps": steps()})
        modified = copy.deepcopy(p.steps)
        modified[0]["step"] = "implement new format"
        with self.assertRaises(ValueError):
            p.apply({"steps": modified})
        p.apply({"steps": modified, "replan_reason": "format discovered"})
        self.assertEqual(p.revisions[-1]["reason"], "format discovered")
        self.assertEqual(p.active["id"], "s1")

    def test_gate_blocked_and_checkpoint_restore(self):
        p = ExecutionPlan()
        for i in range(2):
            p.require_execution()
            p.record(str(i), "read_file", "completed", "a", "read")
        with self.assertRaises(ValueError):
            p.require_execution()
        p.apply({"steps": [{"step": "inspect", "status": "in_progress"}]})
        p.record("read", "read_file", "completed", "a", "read")
        saved = p.export()
        restored = ExecutionPlan(saved)
        self.assertFalse(restored.receipts)
        self.assertIn("恢复", restored.runtime_note())
        p.apply({"steps": [{"step": "inspect", "status": "blocked", "outcome": "missing archive"}]})
        self.assertTrue(p.unfinished)
        self.assertIsNone(p.active)
        self.assertEqual(saved["steps"][0]["status"], "in_progress")
        from app.db import Database
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "test.db")
            db.init()
            db.init()
            self.assertIn("plan_json", {r["name"] for r in db.all("PRAGMA table_info(jobs)")})


class PlanLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, rounds, protocol="chat_completions", status_sequence=None):
        payloads, executed, updates = [], [], []
        statuses = list(status_sequence or [])
        def event(obj):
            return "data: " + json.dumps(obj)
        class Response:
            def __init__(self, status_code=200):
                self.status_code = status_code
            async def aiter_lines(self):
                actions = rounds.pop(0)
                if protocol == "responses":
                    output = []
                    for i, action in enumerate(actions):
                        if isinstance(action, str):
                            yield event({"type": "response.output_text.delta", "delta": action})
                        else:
                            ident, name, args = action
                            item = {"type": "function_call", "id": "fc_"+ident, "call_id": ident, "name": name, "arguments": json.dumps(args)}
                            output.append(item)
                            yield event({"type": "response.output_item.done", "output_index": i, "item": item})
                    yield event({"type": "response.completed", "response": {"id": "resp_"+str(len(payloads)), "output": output, "usage": {}}})
                elif protocol == "messages":
                    for i, action in enumerate(actions):
                        if isinstance(action, str):
                            yield event({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": action}})
                        else:
                            ident, name, args = action
                            yield event({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": ident, "name": name, "input": {}}})
                            yield event({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}})
                    yield event({"type": "message_stop"})
                else:
                    for i, action in enumerate(actions):
                        delta = {"content": action} if isinstance(action, str) else {"tool_calls": [{"index": i, "id": action[0], "type": "function", "function": {"name": action[1], "arguments": json.dumps(action[2])}}]}
                        yield event({"choices": [{"delta": delta}]})
                yield "data: [DONE]"
            async def aread(self):
                return b""
        class Context:
            def __init__(self, status_code=200):
                self.status_code = status_code
            async def __aenter__(self): return Response(self.status_code)
            async def __aexit__(self, *_): return False
        class Client:
            def __init__(self, **_): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
            def stream(self, *_, **kwargs):
                payloads.append(copy.deepcopy(kwargs["json"]))
                return Context(statuses.pop(0) if statuses else 200)
        async def update(state): updates.append(copy.deepcopy(state))
        async def execute(name, args):
            executed.append(args["command"])
            return json.dumps({"exit_code": 0, "stdout": "ok"})
        with patch.object(mimo_local.httpx, "AsyncClient", Client):
            result = await mimo_local.stream_response(
                base_url="https://example.invalid/v1", api_key="test", model="test", messages=[{"role":"user","content":"edit files"}],
                timeout=5, stopped=lambda:False, update=update, web_enabled=False, agent_mode=True,
                max_tool_rounds=15, api_protocol=protocol,
                extra_tools=[{"type":"function","function":{"name":"run_command","description":"run","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
                extra_tool_handler=execute)
        self.assertFalse(rounds)
        return result, payloads, executed, updates

    async def test_gate_execute_verify_and_finish_all_protocols(self):
        for protocol in ("chat_completions", "responses", "messages"):
            with self.subTest(protocol=protocol):
                first = steps()
                second = [{"id":"s1","step":"implement","status":"done","outcome":"saved","evidence":["write"]},
                          {"id":"s2","step":"verify","status":"in_progress"}]
                final = copy.deepcopy(second)
                final[1].update(status="done",outcome="passed",evidence=["check"])
                result, payloads, executed, updates = await self.run_loop([
                    [("a","run_command",{"command":"inspect a"}),("b","run_command",{"command":"inspect b"}), ("denied","run_command",{"command":"should not run"})],
                    [("p","update_plan",{"steps":first})],
                    [("write","run_command",{"command":"implement"})],
                    [("p2","update_plan",{"steps":second}), ("check","run_command",{"command":"verify"})],
                    [("p3","update_plan",{"steps":final})], ["done"]], protocol)
                self.assertEqual(executed, ["inspect a","inspect b","implement","verify"])
                names = [t.get("name", t.get("function",{}).get("name")) for t in payloads[1]["tools"]]
                self.assertEqual(names,["update_plan"])
                self.assertFalse(result["incomplete"])
                self.assertEqual(result["plan"]["steps"][1]["evidence"],["check"])
                self.assertTrue(any(u.get("plan",{}).get("steps") for u in updates))
                self.assertIn("write", json.dumps(payloads[3]))
                if protocol == "responses":
                    self.assertIn("previous_response_id",payloads[2])
                    self.assertIn("服务端执行状态",json.dumps(payloads[3],ensure_ascii=False))

    async def test_unfinished_final_is_reconciled_and_bounded(self):
        result, payloads, executed, _ = await self.run_loop([
            [("p","update_plan",{"steps":steps()})], ["done"], ["done"], ["done"]])
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["incomplete_reason"], "plan_unfinished")
        self.assertEqual(len(payloads),4)
        self.assertFalse(executed)

    async def test_simple_answer_has_no_planning_overhead(self):
        result, payloads, _, _ = await self.run_loop([["hello"]])
        self.assertEqual(len(payloads),1)
        self.assertFalse(result["incomplete"])
        self.assertIsNone(result["plan"])

    async def test_503_waits_five_seconds_and_reports_recovery(self):
        delays = []
        async def fake_sleep(seconds):
            delays.append(seconds)
        with patch.object(mimo_local.asyncio, "sleep", fake_sleep):
            result, payloads, _, updates = await self.run_loop(
                [["hello"]], status_sequence=[503, 503]
            )
        self.assertEqual(len(payloads), 3)
        self.assertEqual(delays, [5, 5])
        self.assertEqual(result["retry_status"]["status"], "recovered")
        self.assertEqual(result["retry_status"]["attempt"], 2)
        self.assertTrue(any(item.get("retry_status", {}).get("active") for item in updates))
