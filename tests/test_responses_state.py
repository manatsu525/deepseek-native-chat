from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.mimo_local import stream_response
from app.responses_state import ResponsesState, resume_state, state_scope
from app import main
from app.db import Database


def final(rid="resp_final", text="完成"):
    return [{"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed", "response": {"id": rid, "output": []}}]


def call(rid, names=("host_check",)):
    items = [{"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}",
              "name": name, "arguments": "{}"} for i, name in enumerate(names)]
    return [*({"type": "response.output_item.done", "output_index": i, "item": item}
              for i, item in enumerate(items)),
            {"type": "response.completed", "response": {"id": rid, "output": items}}]


class Transport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def stream(self, method, url, *, headers, json):
        self.payloads.append(copy.deepcopy(json))
        reply = self.replies.pop(0)

        class Response:
            status_code = reply[0] if isinstance(reply, tuple) else 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def aread(self):
                return reply[1].encode()

            async def aiter_lines(self):
                for event in reply:
                    yield "data: " + __import__("json").dumps(event)
                yield "data: [DONE]"

        return Response()


class ResponsesStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_searches_still_allow_fetch_after_stale_search(self):
        def web_call(rid, name, args):
            events = call(rid, (name,))
            events[0]["item"]["arguments"] = json.dumps(args)
            return events
        queries = [web_call(f"s{i}", "web_search", {"objective": "verify age", "search_queries": [f"query{i}"]}) for i in range(3)]
        transport = Transport([*queries,
            web_call("stale", "web_search", {"objective": "verify age", "search_queries": ["query4"]}),
            web_call("fetch", "fetch_webpage", {"url": "https://example.com/age"}), final()])
        requests = []
        class Web:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                return False
            async def call_tool(self, name, args):
                requests.append(name)
                return {"results": [{"url": "https://example.com/age", "title": "age", "excerpts": ["Born in 2000."]}]}
        with patch("app.mimo_local.ParallelMCPClient", Web):
            result = await self.run_stream(transport, web_enabled=True, extra_tools=[], extra_tool_handler=None,
                                          settings={"web_tool_backend": "parallel"})
        self.assertEqual(requests, ["web_search"] * 3 + ["web_fetch"])
        self.assertEqual(result["answer"], "完成")
        for payload in transport.payloads[3:5]:
            self.assertEqual([item["name"] for item in payload["tools"]], ["fetch_webpage"])
            self.assertIn("web_search=0, fetch_webpage=3", payload["instructions"])
        self.assertTrue(all(item["status"] == "completed" for item in result["tool_trace"]))

    async def test_completed_snapshot_supplies_calls_and_final_text(self):
        events = call("resp_tool")[-1:]
        end = [{"type": "response.completed", "response": {"id": "resp_final", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "完成"}]}]}}]
        transport = Transport([events, end])
        result = await self.run_stream(transport)
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertEqual(result["answer"], "完成")
        self.assertEqual(transport.payloads[1]["previous_response_id"], "resp_tool")

    async def test_arguments_done_survives_empty_item_snapshot_and_replay(self):
        item = {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "host_check", "arguments": ""}
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"value":42}'},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {"id": "resp_tool", "output": [item]}},
        ]
        executed = []
        async def execute(name, args):
            executed.append(args)
            return "checked"
        transport = Transport([events, final()])
        await self.run_stream(transport, extra_tool_handler=execute)
        self.assertEqual(executed, [{"value": 42}])
        second = transport.payloads[1]
        self.assertNotIn("previous_response_id", second)
        replay = next(item for item in second["input"] if item.get("type") == "function_call")
        self.assertEqual(json.loads(replay["arguments"]), {"value": 42})

    async def test_invalid_arguments_are_not_executed_or_replayed(self):
        events = call("resp_bad")
        for event in events:
            if "item" in event:
                event["item"]["arguments"] = '{"broken":'
        transport = Transport([events, call("resp_good"), final()])
        result = await self.run_stream(transport)
        self.assertEqual(len(result["tool_trace"]), 1)
        retry = transport.payloads[1]
        self.assertNotIn("previous_response_id", retry)
        self.assertFalse(any(item.get("type") == "function_call" for item in retry["input"]))
        self.assertIn("not valid JSON", retry["instructions"])

    async def test_unavailable_web_call_does_not_force_remaining_tools_off(self):
        transport = Transport([call("resp_stale", ("web_search",)), call("resp_valid"), final()])
        result = await self.run_stream(transport)
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertEqual(transport.payloads[1]["tool_choice"], "auto")
        self.assertIn("host_check", [item["name"] for item in transport.payloads[1]["tools"]])
        self.assertIn("Available tools", transport.payloads[1]["instructions"])

    async def run_stream(self, transport, **kwargs):
        async def update(state):
            pass

        async def execute(name, args):
            return "checked"

        options = dict(base_url="https://test.invalid/v1", api_key="test", model="test",
                       messages=[{"role": "user", "content": "检查"}], timeout=30,
                       stopped=lambda: False, update=update, web_enabled=False,
                       api_protocol="responses", settings={"thinking": "disabled"},
                       extra_tools=[{"type": "function", "function": {"name": "host_check",
                                     "parameters": {"type": "object", "properties": {}}}}],
                       extra_tool_handler=execute)
        options.update(kwargs)
        with patch("app.mimo_local.httpx.AsyncClient", return_value=transport):
            return await stream_response(**options)

    async def test_tools_and_next_chat_send_only_new_items(self):
        transport = Transport([call("resp_tool"), final()])
        result = await self.run_stream(transport)
        first, second = transport.payloads
        self.assertTrue(first["store"])
        self.assertNotIn("previous_response_id", first)
        self.assertEqual(second["previous_response_id"], "resp_tool")
        self.assertEqual(second["input"], [{"type": "function_call_output", "call_id": "call_0", "output": "checked"}])
        self.assertEqual(first["instructions"], second["instructions"])
        self.assertEqual(first["tools"], second["tools"])
        self.assertEqual(result["responses_state"]["response_id"], "resp_final")
        followup = Transport([final("resp_next")])
        await self.run_stream(followup, responses_state=result["responses_state"], messages=[
            {"role": "user", "content": "检查"}, {"role": "assistant", "content": "完成"},
            {"role": "user", "content": [{"type": "text", "text": "再检查图片"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])
        request = followup.payloads[0]
        self.assertEqual(request["previous_response_id"], "resp_final")
        self.assertEqual(len(request["input"]), 1)
        self.assertEqual(request["input"][0]["content"][1]["type"], "input_image")

    async def test_expired_state_rebases_once_without_reexecuting_tools(self):
        transport = Transport([call("resp_tool"), (400, 'previous_response_id not found'), final()])
        result = await self.run_stream(transport)
        self.assertEqual(len(result["tool_trace"]), 1)
        retry = transport.payloads[2]
        self.assertNotIn("previous_response_id", retry)
        self.assertNotIn("store", retry)
        self.assertTrue(any(item.get("type") == "function_call_output" for item in retry["input"]))
        self.assertTrue(result["responses_state"]["disabled"])

    async def test_store_rejection_and_missing_id_use_manual_history(self):
        for replies in ([(400, "store is not supported"), call("resp_tool"), final()],
                        [call(""), final()]):
            with self.subTest(replies=replies):
                transport = Transport(replies)
                result = await self.run_stream(transport)
                self.assertTrue(result["responses_state"]["disabled"])
                self.assertNotIn("previous_response_id", transport.payloads[-1])
                self.assertTrue(any(item.get("type") == "function_call_output" for item in transport.payloads[-1]["input"]))

    async def test_unrelated_errors_are_not_retried(self):
        for error in [(503, "temporarily unavailable"), (401, "invalid key"), (400, "invalid temperature")]:
            with self.subTest(error=error):
                transport = Transport([error])
                with self.assertRaises(RuntimeError):
                    await self.run_stream(transport)
                self.assertEqual(len(transport.payloads), 1)

    async def test_unexecuted_calls_are_not_left_in_stored_chain(self):
        transport = Transport([call("resp_bad", ("fetch_webpage", "host_check")), final()])
        result = await self.run_stream(transport)
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertNotIn("previous_response_id", transport.payloads[1])
        self.assertFalse(any(item.get("name") == "fetch_webpage" for item in transport.payloads[1]["input"]))

    async def test_store_false_disables_chaining(self):
        transport = Transport([call("resp_tool"), final()])
        await self.run_stream(transport, settings={"request_overrides": {"store": False}})
        self.assertTrue(all(p["store"] is False and "previous_response_id" not in p for p in transport.payloads))

    def test_scope_and_retry_parent_selection(self):
        provider = {"id": 1, "api_key": "secret", "base_url": "https://test.invalid"}
        job = {"user_id": 1, "conversation_id": "chat", "model": "model", "chat_mode": "standard"}
        scope = state_scope(provider, job, {})
        saved = {"response_id": "parent", "scope": scope}
        rows = [{"role": "user"}, {"role": "assistant", "meta_json": json.dumps({"responses_state": saved})}]
        self.assertEqual(resume_state(rows, scope), saved)
        for field, value in [("user_id", 2), ("conversation_id", "other"), ("model", "other"), ("chat_mode", "agent")]:
            self.assertEqual(resume_state(rows, state_scope(provider, {**job, field: value}, {})), {})
        self.assertNotEqual(scope, state_scope({**provider, "api_key": "new"}, job, {}))
        self.assertNotEqual(scope, state_scope(provider, job, {"request_overrides": {"provider": "other"}}))
        rows[1]["meta_json"] = json.dumps({"failed": True, "responses_state": saved})
        self.assertEqual(resume_state(rows, scope), {})

    def test_created_only_and_reset_cannot_persist_incomplete_state(self):
        state = ResponsesState()
        state.observe({"type": "response.created", "response": {"id": "incomplete"}})
        state.accept()
        self.assertEqual(state.export()["response_id"], "")
        state = ResponsesState({"response_id": "old"})
        state.reset()
        payload = {}
        state.prepare(payload, [{"role": "user", "content": "full"}])
        self.assertNotIn("previous_response_id", payload)

    async def test_completed_job_persists_state_and_next_job_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "chat.db")
            db.init()
            uid = db.run("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)", ("test", "hash", 1))
            pid = db.run("INSERT INTO providers(user_id,name,api_key,base_url,model,provider_type,created_at) VALUES(?,?,?,?,?,?,?)",
                         (uid, "test", "key", "https://test.invalid", "test", "custom_response", 1))
            db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", ("chat", uid, "test", 1, 1))
            captured = []

            async def fake_stream(**kwargs):
                captured.append(kwargs)
                return {"answer": "done", "reasoning": "", "searches": [], "sources": [], "usage": {},
                        "responses_state": {"response_id": "resp_saved", "disabled": False}}

            with patch.object(main, "db", db), patch.object(main, "custom_responses_stream_response", fake_stream), \
                 patch.object(main, "ConversationWorkspace") as workspace, patch.object(main, "AgentSharedWorkspace"):
                workspace.return_value.list_files.return_value = []
                for index in range(2):
                    db.run("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", ("chat", "user", "question", index))
                    db.run("INSERT INTO jobs(id,user_id,conversation_id,provider_id,provider_type,model,effort,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (str(index), uid, "chat", pid, "custom_response", "test", "high", "queued", index, index))
                    await main._execute_job(str(index))
                    self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (str(index),))["status"], "completed")
                self.assertEqual(captured[0]["responses_state"], {})
                self.assertEqual(captured[1]["responses_state"]["response_id"], "resp_saved")
                meta = json.loads(db.one("SELECT meta_json FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")["meta_json"])
                self.assertIn("scope", meta["responses_state"])
