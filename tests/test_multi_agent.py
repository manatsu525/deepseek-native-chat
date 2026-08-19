from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.multi_agent import (
    INSPECTOR_PROMPT,
    LEADER_DECISION_PROMPT,
    LEADER_FINAL_PROMPT,
    PROGRAMMER_PROMPT,
    RESEARCHER_PROMPT,
    _parse_decision,
    run_collaboration,
)
from app.workspace import ConversationWorkspace


def result(answer: str, *, searches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "answer": answer,
        "reasoning": "role reasoning",
        "searches": searches or [],
        "sources": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "tool_trace": [],
    }


class MultiAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_leader_drives_real_repair_loop_and_on_demand_research(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = ConversationWorkspace(1, "conversation")
        workspace.root = Path(temporary.name)
        decisions = iter(
            [
                '{"action":"program","task":"创建程序","reason":"需要实际文件"}',
                '{"action":"inspect","task":"测试程序","reason":"程序改动必须检查"}',
                # Research is deliberately requested after a failed inspection,
                # proving it is available on demand rather than a fixed first stage.
                '{"action":"research","task":"核对正确算法","reason":"修复前需要资料"}',
                '{"action":"program","task":"按检查意见修复","reason":"存在阻断问题"}',
                '{"action":"inspect","task":"复检修复结果","reason":"确认问题已解决"}',
                '{"action":"finish","task":"","reason":"复检通过"}',
            ]
        )
        calls: list[dict[str, Any]] = []
        program_packets: list[str] = []
        program_count = 0
        inspection_count = 0

        async def fake_streamer(**kwargs: Any) -> dict[str, Any]:
            nonlocal program_count, inspection_count
            calls.append(kwargs)
            await kwargs["update"]({"answer": "partial", "reasoning": "", "searches": [], "sources": [], "usage": {}})
            prompt = kwargs["system_addendum"]
            packet = kwargs["messages"][-1]["content"]
            if prompt == LEADER_DECISION_PROMPT:
                return result(next(decisions))
            if prompt == PROGRAMMER_PROMPT:
                program_count += 1
                program_packets.append(packet)
                if program_count == 1:
                    workspace.write_file("app.py", "print('broken')\n")
                    return result("已创建 app.py，但尚未解决输出问题。")
                workspace.write_file("app.py", "print('fixed')\n")
                return result("已按检查反馈修复 app.py，并完成本地验证。")
            if prompt == INSPECTOR_PROMPT:
                inspection_count += 1
                names = {item["function"]["name"] for item in workspace.tool_definitions("read_only")}
                self.assertNotIn("write_file", names)
                self.assertNotIn("apply_line_edits", names)
                if inspection_count == 1:
                    return result("app.py 输出不符合任务，需要程序员修复。\nVERDICT: REVISE")
                self.assertIn("fixed", workspace.read_file("app.py"))
                return result("实际运行和内容检查均通过。\nVERDICT: PASS")
            if prompt == RESEARCHER_PROMPT:
                return result(
                    "资料确认应输出 fixed。",
                    searches=[{"id": "s1", "action": "search", "status": "completed", "query": "algorithm"}],
                )
            if prompt == LEADER_FINAL_PROMPT:
                return result("程序已经修复并通过检查，可下载 app.py。")
            raise AssertionError("unexpected role prompt")

        updates: list[dict[str, Any]] = []

        async def update(state: dict[str, Any]) -> None:
            updates.append(state)

        final = await run_collaboration(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            messages=[{"role": "user", "content": "请创建并修好程序"}],
            timeout=30,
            stopped=lambda: False,
            update=update,
            settings={"max_completion_tokens": 65_536},
            conversation_id="conversation",
            user_timezone="UTC",
            effort="high",
            workspace=workspace,
            streamer=fake_streamer,
        )

        self.assertEqual(final["answer"], "程序已经修复并通过检查，可下载 app.py。")
        self.assertEqual(program_count, 2)
        self.assertEqual(inspection_count, 2)
        self.assertIn("必须处理的检查员反馈", program_packets[1])
        self.assertIn("输出不符合任务", program_packets[1])
        roles = [item["role"] for item in final["agents"]]
        self.assertEqual(
            [role for role in roles if role != "leader"],
            ["programmer", "inspector", "researcher", "programmer", "inspector"],
        )
        self.assertEqual([item["verdict"] for item in final["agents"] if item["role"] == "inspector"], ["REVISE", "PASS"])
        self.assertTrue(updates)
        self.assertEqual(updates[-1]["answer"], final["answer"])

        researcher_call = next(item for item in calls if item["system_addendum"] == RESEARCHER_PROMPT)
        self.assertTrue(researcher_call["web_enabled"])
        self.assertIsNone(researcher_call["workspace"])
        inspector_calls = [item for item in calls if item["system_addendum"] == INSPECTOR_PROMPT]
        self.assertTrue(all(not item["web_enabled"] and item["workspace_access"] == "read_only" for item in inspector_calls))
        programmer_calls = [item for item in calls if item["system_addendum"] == PROGRAMMER_PROMPT]
        self.assertTrue(all(item["web_enabled"] and item["workspace_access"] == "full" for item in programmer_calls))

    async def test_finish_is_rejected_until_program_changes_are_inspected(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = ConversationWorkspace(1, "guarded")
        workspace.root = Path(temporary.name)
        decisions = iter(
            [
                '{"action":"program","task":"写文件","reason":"执行"}',
                '{"action":"finish","task":"","reason":"想直接结束"}',
                '{"action":"inspect","task":"检查文件","reason":"需要复检"}',
                '{"action":"finish","task":"","reason":"已经通过"}',
            ]
        )
        prompts: list[str] = []

        async def fake_streamer(**kwargs: Any) -> dict[str, Any]:
            prompt = kwargs["system_addendum"]
            prompts.append(prompt)
            if prompt == LEADER_DECISION_PROMPT:
                return result(next(decisions))
            if prompt == PROGRAMMER_PROMPT:
                workspace.write_file("ok.py", "print('ok')\n")
                return result("完成")
            if prompt == INSPECTOR_PROMPT:
                return result("检查通过。\nVERDICT: PASS")
            if prompt == LEADER_FINAL_PROMPT:
                return result("最终完成")
            raise AssertionError("unexpected role")

        async def update(_: dict[str, Any]) -> None:
            return None

        final = await run_collaboration(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            messages=[{"role": "user", "content": "写代码"}],
            timeout=30,
            stopped=lambda: False,
            update=update,
            settings={"max_completion_tokens": 65_536},
            conversation_id="guarded",
            user_timezone="UTC",
            effort="high",
            workspace=workspace,
            streamer=fake_streamer,
        )
        self.assertEqual(final["answer"], "最终完成")
        self.assertEqual(prompts.count(LEADER_FINAL_PROMPT), 1)
        self.assertLess(prompts.index(INSPECTOR_PROMPT), prompts.index(LEADER_FINAL_PROMPT))

    def test_decision_parser_accepts_fenced_json(self) -> None:
        decision = _parse_decision('```json\n{"action":"research","task":"查文档","reason":"需要依据"}\n```')
        self.assertEqual(decision["action"], "research")
        self.assertEqual(decision["task"], "查文档")


if __name__ == "__main__":
    unittest.main()
