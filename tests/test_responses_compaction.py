"""Offline regression coverage for intact tool arguments and fixed budgets."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import mimo_local
from app.context import effective_context_budget, serialized_chars, compact_request
from app.workspace import ConversationWorkspace, WorkspaceError
import test_responses_state as responses_tests


class ResponsesCompactionTests(unittest.IsolatedAsyncioTestCase):
    def test_wire_and_meter_preserve_original_arguments_and_native_items(self):
        reasoning = {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque-signature", "summary": []}
        for name, succeeded, arguments in (
            ("write_file", True, {"path": "game.html", "content": "x" * 65_000}),
            ("edit_file", True, {"path": "game.html", "edits": [{"old_text": "old", "new_text": "x" * 10_000}]}),
            ("write_file", False, {"path": "game.html", "content": "x" * 65_000}),
        ):
            with self.subTest(name=name, succeeded=succeeded):
                raw = json.dumps(arguments)
                function = {"name": name, "arguments": raw}
                native = {"id": "fc_1", "call_id": "call_1", "type": "function_call", "status": "completed",
                          "name": name, "arguments": raw}
                message = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": function}],
                           "responses_output_items": [reasoning, native]}
                before = serialized_chars([message])
                self.assertFalse(mimo_local._compact_workspace_call_arguments(function, name=name, path="game.html", succeeded=succeeded))
                wire = mimo_local._responses_input([message])
                self.assertEqual(wire[0], reasoning)
                self.assertEqual(wire[1]["arguments"], function["arguments"])
                self.assertEqual(wire[1]["arguments"], raw)
                self.assertEqual(wire[1]["id"], "fc_1")
                self.assertEqual(wire[1]["call_id"], "call_1")
                self.assertEqual(serialized_chars([message]), before)
                self.assertEqual(native["arguments"], raw)  # original diagnostic item stays intact
                self.assertEqual(mimo_local._responses_input([{"role": "assistant", "responses_output_items": [reasoning, native]}]),
                                 [reasoning, native])
        small_raw = '{"path":"game.html","content":"small"}'
        small = {"name": "write_file", "arguments": small_raw}
        self.assertFalse(mimo_local._compact_workspace_call_arguments(small, name="write_file", path="game.html", succeeded=True))
        self.assertEqual(small["arguments"], small_raw)

    def test_default_budget_stays_240000_and_triggers_compaction(self):
        for window, chars, tokens in ((1_048_576, 219099, 37906), (32_000, 0, 0), (None, 664536, 128812)):
            self.assertEqual(effective_context_budget(240_000, window_tokens=window, request_chars=chars, input_tokens=tokens), 240_000)
        self.assertEqual(effective_context_budget(80_000, window_tokens=1_000_000, request_chars=219099, input_tokens=37906), 80_000)
        history = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        for index in range(6):
            history.extend([{"role": "assistant", "content": "", "tool_calls": [
                {"id": str(index), "function": {"name": "run_command", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": str(index), "content": "x" * 50_000}])
        self.assertGreater(serialized_chars(history), 240_000)
        compacted = compact_request(history, base_message_count=2, budget=240_000)
        self.assertTrue(compacted)
        self.assertLess(serialized_chars(history), 240_000)
        self.assertEqual(history[-1]["content"], "x" * 50_000)

    async def test_next_actual_request_keeps_arguments_without_reexecuting(self):
        content = "UNIQUE_LARGE_BODY_" * 4000
        reasoning = {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []}
        for agent_mode, store, failed in ((False, False, False), (False, True, False),
                                         (True, True, False), (False, True, True)):
            with self.subTest(agent_mode=agent_mode, store=store, failed=failed), tempfile.TemporaryDirectory() as directory:
                workspace = ConversationWorkspace(1, "regression")
                workspace.root = Path(directory)
                workspace.write_file("existing.txt", "initial")
                function = {"id": "fc_1", "call_id": "write_1", "type": "function_call", "name": "write_file",
                            "arguments": json.dumps({"path": "game.html", "content": content})}
                items = [reasoning, function]
                first = [{"type": "response.output_item.done", "output_index": i, "item": item} for i, item in enumerate(items)]
                first.append({"type": "response.completed", "response": {"id": "resp_write", "output": items}})
                transport = responses_tests.Transport([first, responses_tests.final()])
                executed = []
                original_execute = workspace.execute
                def execute(name, arguments):
                    executed.append((name, copy.deepcopy(arguments)))
                    if failed:
                        raise WorkspaceError("deliberate write failure")
                    return original_execute(name, arguments)
                workspace.execute = execute
                async def update(_):
                    pass
                with patch.object(mimo_local.httpx, "AsyncClient", return_value=transport), \
                     patch.object(mimo_local, "_host_read_snapshot", return_value=None):
                    result = await mimo_local.stream_response(
                        base_url="https://test.invalid/v1", api_key="test", model="test", messages=[{"role": "user", "content": "write file"}],
                        timeout=5, stopped=lambda: False, update=update, api_protocol="responses", web_enabled=False,
                        agent_mode=agent_mode, workspace=None if agent_mode else workspace,
                        extra_tools=workspace.tool_definitions() if agent_mode else None,
                        extra_tool_handler=execute if agent_mode else None,
                        settings={"advanced_enabled": True, "advanced_request": {"model": "test", "store": store}})
                self.assertEqual(len(transport.payloads), 2)
                writes = [call for call in executed if call[0] == "write_file"]
                self.assertEqual(len(writes), 1)
                self.assertEqual(len(result["tool_trace"]), 1)
                self.assertEqual(writes[0][1]["content"], content)
                sent = transport.payloads[1]
                if store:
                    self.assertEqual(sent.get("previous_response_id"), "resp_write")
                else:
                    self.assertNotIn("previous_response_id", sent)
                    calls = [item for item in sent["input"] if item.get("type") == "function_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(json.loads(calls[0]["arguments"])["content"], content)
                    self.assertIn(reasoning, sent["input"])
                self.assertEqual(len([item for item in sent["input"] if item.get("type") == "function_call_output"]), 1)
                if agent_mode:
                    self.assertEqual(result["round_stats"][0]["context_policy"], "grok-build")
                else:
                    self.assertEqual(result["round_stats"][0]["context_budget"], 240_000)
                self.assertFalse(result["round_stats"][0].get("arguments_compacted", False))
                if not failed:
                    self.assertEqual((workspace.root / "game.html").read_text(), content)
