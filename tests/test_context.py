from __future__ import annotations

import json
import unittest

from app.context import (
    DEFAULT_CONTEXT_BUDGET_CHARS,
    LOW_WATER_RATIO,
    STUB_PREFIX,
    checkpoint_payload,
    compact_request,
    normalize_budget,
    serialized_chars,
)
from app.file_knowledge import FileKnowledge
from app.plan import TaskPlan


def exchange(call_id: str, name: str, arguments: dict, result: str, text: str = "") -> list[dict]:
    return [
        {"role": "assistant", "content": text, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


BASE = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "do the task"}]


class CompactRequestTests(unittest.TestCase):
    def test_under_budget_is_untouched(self):
        conversation = BASE + exchange("a", "list_files", {}, "files")
        before = json.dumps(conversation)
        self.assertIsNone(compact_request(conversation, base_message_count=2, budget=100_000))
        self.assertEqual(json.dumps(conversation), before)

    def test_old_results_become_stubs_and_recent_ones_survive(self):
        conversation = list(BASE)
        for n in range(6):
            conversation += exchange(f"c{n}", "host_run_command", {"command": f"grep thing{n} -r ."}, f"line {n}\n" * 2000, f"第{n}步")
        budget = 60_000
        info = compact_request(conversation, base_message_count=2, budget=budget)
        self.assertIsNotNone(info)
        self.assertLessEqual(serialized_chars(conversation), int(budget * LOW_WATER_RATIO) + 2_000)
        self.assertEqual(info["dropped_rounds"], 0)
        # Every assistant message and tool call is still there, in order.
        texts = [m["content"] for m in conversation if m["role"] == "assistant"]
        self.assertEqual(texts, [f"第{n}步" for n in range(6)])
        results = [m["content"] for m in conversation if m["role"] == "tool"]
        self.assertTrue(all(r.startswith(STUB_PREFIX) for r in results[:4]))
        self.assertTrue(all(not r.startswith(STUB_PREFIX) for r in results[4:]))
        self.assertIn("grep thing0", results[0])
        self.assertEqual([name for name, _ in info["stubbed"]], ["host_run_command"] * 4)
        checkpoint = checkpoint_payload(conversation)
        self.assertTrue(checkpoint["context_checkpoint"])
        self.assertEqual(conversation[0], BASE[0])

    def test_whole_rounds_are_dropped_only_as_a_last_resort(self):
        conversation = list(BASE)
        for n in range(4):
            conversation += exchange(f"c{n}", "search_files", {"query": f"q{n}"}, "r" * 20_000, "决定：新建独立兵种" if n == 0 else f"步骤{n}")
        info = compact_request(conversation, base_message_count=2, budget=45_000)
        self.assertGreater(info["dropped_rounds"], 0)
        notes = checkpoint_payload(conversation)["progress_notes"]
        self.assertEqual(notes[0], "决定：新建独立兵种")
        # The newest exchange is always kept.
        self.assertEqual(conversation[-1]["tool_call_id"], "c3")
        # A later compaction keeps accumulating notes rather than restarting.
        conversation += exchange("c9", "search_files", {"query": "q9"}, "z" * 40_000, "再一步")
        compact_request(conversation, base_message_count=2, budget=45_000)
        self.assertEqual(checkpoint_payload(conversation)["progress_notes"][0], "决定：新建独立兵种")

    def test_checkpoint_carries_files_plan_and_sources(self):
        knowledge = FileKnowledge()
        knowledge.record_read({"path": "a.js", "revision": "r1", "line_count": 1, "from_line": 1,
                               "through_line": 1, "truncated": False, "content": "1|x();"})
        plan = TaskPlan()
        plan.apply({"steps": [{"step": "read", "status": "done"}, {"step": "edit", "status": "in_progress"}]})
        conversation = BASE + exchange("c0", "read_file", {"path": "a.js"}, "1|x();" + " " * 5000) + exchange("c1", "list_files", {}, "f" * 5000) + exchange("c2", "list_files", {}, "g" * 5000)
        info = compact_request(conversation, base_message_count=2, budget=12_000, knowledge=knowledge,
                               workspace_files=lambda: [{"path": "a.js", "size": 5}],
                               sources={"u": {"url": "https://x", "title": "t", "summary": "s"}}, plan=plan.export())
        self.assertIsNotNone(info)
        checkpoint = checkpoint_payload(conversation)
        self.assertEqual(checkpoint["file_snapshots"][0]["path"], "a.js")
        self.assertEqual(checkpoint["plan"]["steps"][1]["status"], "in_progress")
        self.assertEqual(checkpoint["workspace_files"], [{"path": "a.js", "size": 5}])
        self.assertEqual(checkpoint["sources"][0]["url"], "https://x")
        self.assertIn("file_snapshots", checkpoint_payload(conversation)["instruction"])

    def test_snapshots_that_do_not_fit_are_forgotten(self):
        knowledge = FileKnowledge()
        knowledge.record_read({"path": "huge.py", "revision": "r", "line_count": 1, "from_line": 1,
                               "through_line": 1, "truncated": False, "content": "1|" + "x" * 60_000})
        conversation = BASE + exchange("c0", "list_files", {}, "f" * 30_000) + exchange("c1", "list_files", {}, "g" * 30_000) + exchange("c2", "list_files", {}, "h" * 100)
        compact_request(conversation, base_message_count=2, budget=50_000, knowledge=knowledge)
        self.assertFalse(knowledge.known("huge.py"))
        self.assertEqual(checkpoint_payload(conversation)["file_snapshots"], [])

    def test_encrypted_reasoning_is_discounted_in_the_measure(self):
        blob = "A" * 80_000
        with_reasoning = [{"role": "assistant", "content": "x", "responses_output_items": [
            {"type": "reasoning", "id": "r", "encrypted_content": blob}]}]
        self.assertLess(serialized_chars(with_reasoning), 12_000)
        self.assertGreater(serialized_chars(with_reasoning), 10_000)
        conversation = BASE + with_reasoning + [{"role": "tool", "tool_call_id": "c", "content": "ok"}]
        self.assertIsNone(compact_request(conversation, base_message_count=2, budget=40_000))

    def test_budget_normalization(self):
        self.assertEqual(normalize_budget(None), DEFAULT_CONTEXT_BUDGET_CHARS)
        self.assertEqual(normalize_budget("abc"), DEFAULT_CONTEXT_BUDGET_CHARS)
        self.assertEqual(normalize_budget(1), 40_000)
        self.assertEqual(normalize_budget(10**9), 2_000_000)


class TaskPlanTests(unittest.TestCase):
    def test_plan_round_trips_and_renders(self):
        plan = TaskPlan()
        result = json.loads(plan.apply({"steps": ["read files", {"step": "edit", "status": "in_progress"}], "note": "assume X"}))
        self.assertEqual((result["steps"], result["done"]), (2, 0))
        self.assertIn("[ ] 1. read files", result["plan"])
        self.assertIn("[~] 2. edit", result["plan"])
        self.assertIn("note: assume X", result["plan"])
        self.assertEqual(plan.export()["steps"][0], {"step": "read files", "status": "pending"})
        with self.assertRaises(ValueError):
            plan.apply({"steps": []})
        self.assertEqual(plan.updates, 1)
        # Other todo-tool conventions are accepted.
        result = json.loads(plan.apply({"todos": [{"title": "解压", "status": "completed"}, {"description": "改 mod.json", "state": "todo"}]}))
        self.assertEqual(result["done"], 1)
        self.assertEqual(plan.export()["steps"], [{"step": "解压", "status": "done"}, {"step": "改 mod.json", "status": "pending"}])

    def test_plan_accepts_loose_shapes_instead_of_failing(self):
        plan = TaskPlan()
        # Free text, one step per line, with checkbox marks.
        result = json.loads(plan.apply({"plan": "1. [x] unzip mods\n2. [~] read vanilla stats\n- write creature file\n"}))
        self.assertEqual(result["done"], 1)
        self.assertEqual([s["status"] for s in plan.export()["steps"]], ["done", "in_progress", "pending"])
        self.assertEqual(plan.export()["steps"][2]["step"], "write creature file")
        # Unknown field names: the longest text wins; numeric step ids are not text.
        result = json.loads(plan.apply({"steps": [{"id": 1, "action": "inspect both mods", "done": True},
                                                  {"step": 2, "summary": "add new creature", "completed": False}]}))
        self.assertEqual(plan.export()["steps"], [{"step": "inspect both mods", "status": "done"},
                                                  {"step": "add new creature", "status": "pending"}])
        # A dict keyed by step number, and a single step given at the top level.
        plan.apply({"steps": {"1": "a", "2": {"task": "b", "status": "in_progress"}}})
        self.assertEqual([s["step"] for s in plan.export()["steps"]], ["a", "b"])
        plan.apply({"step": "only one", "status": "done"})
        self.assertEqual(plan.export()["steps"], [{"step": "only one", "status": "done"}])
        with self.assertRaises(ValueError):
            plan.apply({"steps": [{"id": 1}, ""]})

    def test_budget_follows_the_model_window_and_measured_token_ratio(self):
        from app.context import CONTEXT_TOKEN_CAP, effective_context_budget

        # Explicit setting wins; default without usage stays as is.
        self.assertEqual(effective_context_budget(100_000, window_tokens=1_000_000, request_chars=50_000, input_tokens=10_000), 100_000)
        self.assertEqual(effective_context_budget(DEFAULT_CONTEXT_BUDGET_CHARS, window_tokens=1_000_000, request_chars=0, input_tokens=0),
                         DEFAULT_CONTEXT_BUDGET_CHARS)
        # 1M window: 60% share is capped at 512K tokens, times 4.6 chars/token.
        budget = effective_context_budget(DEFAULT_CONTEXT_BUDGET_CHARS, window_tokens=1_048_576, request_chars=239_000, input_tokens=52_000)
        self.assertEqual(budget, int(CONTEXT_TOKEN_CAP * 239_000 / 52_000))
        # Unknown window: assume 128K tokens.
        budget = effective_context_budget(DEFAULT_CONTEXT_BUDGET_CHARS, window_tokens=None, request_chars=40_000, input_tokens=10_000)
        self.assertEqual(budget, int(128_000 * 0.6 * 4))
        # A small window with CJK-heavy text (few chars per token) shrinks the budget.
        budget = effective_context_budget(DEFAULT_CONTEXT_BUDGET_CHARS, window_tokens=32_000, request_chars=15_000, input_tokens=10_000)
        self.assertEqual(budget, 40_000)

    def test_round_text_is_joined_as_paragraphs(self):
        from app.mimo_local import _join_round_text

        self.assertEqual(_join_round_text("", "first"), "first")
        self.assertEqual(_join_round_text("first", ""), "first")
        self.assertEqual(_join_round_text("first", "second"), "first\n\nsecond")
        self.assertEqual(_join_round_text("first\n", "second"), "first\nsecond")


if __name__ == "__main__":
    unittest.main()
