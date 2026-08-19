from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .mimo import _merge_usage
from .mimo_local import stream_response as custom_stream_response
from .workspace import ConversationWorkspace


MAX_WORKER_ACTIONS = 10
MAX_LEADER_DECISIONS = 14
MAX_ROLE_ACTIONS = 5
MAX_VISIBLE_ANSWER_CHARS = 20_000
MAX_VISIBLE_REASONING_CHARS = 12_000
MAX_STATE_OUTPUT_CHARS = 5_000

ROLE_LABELS = {
    "leader": "Leader",
    "researcher": "调研员",
    "programmer": "程序员",
    "inspector": "检查员",
}

LEADER_DECISION_PROMPT = """你是四智能体协作组的 Leader，负责理解用户目标、决定下一步由谁工作，并最终整合答案。你自己没有工具，也不要假装执行过搜索、文件修改或测试。

每次只决定一个下一步动作，并且只输出一个 JSON 对象，不要 Markdown 代码块或额外文字：
{"action":"research|program|inspect|finish","task":"交给该角色的具体任务，finish 时为空","reason":"简短决策依据","final_answer":"仅 finish 时填写的完整最终回答，其他动作为空"}

规则：
- research：需要外部资料、文档、事实核查时交给调研员；可以在工作中的任何阶段再次调用。
- program：需要创建/修改文件、执行代码或修复检查员发现的问题时交给程序员。
- inspect：程序员产出后交给检查员独立读取、测试和审查。
- finish：证据和产出已经足够且没有待修复或待复检的问题时结束；必须同时在 final_answer 中直接写好给用户的完整回答，以免再调用一次模型。
- 不要为了凑齐角色而调用不需要的智能体，也不要机械串联；根据共享状态作真实决策。
- 检查员报告 REVISE 后，必须让程序员实际修复，并再次让检查员复检通过，才能 finish。
"""

LEADER_FINAL_PROMPT = """你是协作组 Leader。根据用户原始对话、调研结论、程序员实际保存的文件和检查员结果，给出最终回答。只回答用户，不要再输出调度 JSON，不要声称做过记录中没有发生的工作。清楚说明完成内容、关键结论、文件和仍存在的限制。使用与用户提问相同的语言。"""

RESEARCHER_PROMPT = """你是协作组的调研员，只负责外部信息调研和事实核查。你可以使用搜索和网页抓取工具，但没有工作区文件权限。围绕 Leader 指派的任务寻找可靠资料，交叉核对关键结论，并给程序员或 Leader 提供简洁、可执行且带来源的报告。不要假装修改或测试文件。"""

PROGRAMMER_PROMPT = """你是协作组的程序员，负责具体执行、创建和修改共享工作区文件，并运行适用的检查。你拥有联网和完整工作区工具。先查看当前工作区真实状态，再完成 Leader 的任务；收到检查员反馈时必须针对反馈实际修改文件，不要只描述建议。尽量局部修改，避免无意义地重读或重写文件。完成后简洁列出改动文件、执行的验证及结果，不要在回答中重复粘贴已保存的完整代码。"""

INSPECTOR_PROMPT = """你是协作组的检查员，负责独立审查程序员产出。你只有只读文件、搜索文件和本地测试/语法检查工具，绝不能创建、修改或删除文件。必须读取相关文件并尽可能运行实际检查，指出可复现的问题、文件路径和证据。整个检查最多有 6 次工具调用：先选最相关的文件，再运行覆盖面最大的验证；工作区未发生变化时，绝对不要重复读取同一范围或重复运行相同测试。一次综合测试成功且代码审查无阻断问题后，应立即给出结论。报告最后单独输出且仅输出以下两种结论之一：
VERDICT: PASS
VERDICT: REVISE
只在产出满足任务且检查未发现阻断问题时 PASS；需要程序员修复时必须 REVISE。"""


def _trim(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[内容过长，协作记录已截断]"


def _extract_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            decoded, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("Leader 没有返回有效的调度 JSON")


def _parse_decision(value: str) -> dict[str, str]:
    raw = _extract_json_object(value)
    action = str(raw.get("action") or "").strip().casefold()
    if action not in {"research", "program", "inspect", "finish"}:
        raise ValueError(f"Leader 返回了无效动作：{action or '空'}")
    task = " ".join(str(raw.get("task") or "").split())[:2_000]
    reason = " ".join(str(raw.get("reason") or "").split())[:1_000]
    final_answer = str(raw.get("final_answer") or "").strip()
    if action != "finish" and not task:
        raise ValueError("Leader 调度任务为空")
    return {"action": action, "task": task, "reason": reason, "final_answer": final_answer}


def _inspection_verdict(value: str) -> str:
    matches = re.findall(r"VERDICT\s*[:：]\s*(PASS|REVISE)\b", str(value or ""), re.IGNORECASE)
    return matches[-1].upper() if matches else "REVISE"


def _role_settings(settings: dict[str, Any], role: str, phase: str = "work") -> dict[str, Any]:
    result = dict(settings)
    configured = int(result.get("max_completion_tokens") or 65_536)
    limits = {
        ("leader", "decision"): 4_096,
        ("leader", "final"): 16_384,
        ("researcher", "work"): 16_384,
        ("programmer", "work"): 32_768,
        ("inspector", "work"): 16_384,
    }
    result["max_completion_tokens"] = min(configured, limits.get((role, phase), configured))
    return result


def _usage_for(agents: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for agent in agents:
        usage = _merge_usage(usage, agent.get("usage") or {})
    return usage


def _sources_for(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for agent in agents:
        for source in agent.get("sources") or []:
            url = str(source.get("url") or "") if isinstance(source, dict) else ""
            key = url or json.dumps(source, ensure_ascii=False, sort_keys=True)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(source)
    return result


def _searches_for(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for agent in agents:
        for step in agent.get("searches") or []:
            item = dict(step)
            item["agent_id"] = agent["id"]
            item["agent_role"] = agent["role"]
            item["agent_label"] = agent["label"]
            result.append(item)
    return result


def _public_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "id", "role", "label", "status", "task", "answer", "reasoning",
        "searches", "sources", "usage", "verdict", "error", "decision_action",
        "decision_reason",
    }
    return [{key: value for key, value in agent.items() if key in fields} for agent in agents]


def _state_packet(
    agents: list[dict[str, Any]],
    workspace: ConversationWorkspace,
    *,
    pending_inspection: bool,
    blocker: str,
    guard_message: str = "",
) -> str:
    completed = []
    for agent in agents:
        if agent.get("role") == "leader" or agent.get("status") != "completed":
            continue
        completed.append(
            {
                "role": agent.get("label"),
                "task": agent.get("task"),
                "result": _trim(agent.get("answer"), MAX_STATE_OUTPUT_CHARS),
                "verdict": agent.get("verdict", ""),
            }
        )
    packet = {
        "shared_workspace_files": workspace.list_files(),
        "completed_work": completed[-8:],
        "program_changes_need_inspection": pending_inspection,
        "unresolved_inspector_feedback": _trim(blocker, MAX_STATE_OUTPUT_CHARS),
        "orchestrator_guard": guard_message,
    }
    return (
        "以下是协作组的当前共享状态。请结合此前的用户原始对话作决定：\n"
        + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    )


async def run_collaboration(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    stopped: Callable[[], bool],
    update: Callable[[dict[str, Any]], Awaitable[None]],
    settings: dict[str, Any],
    conversation_id: str,
    user_timezone: str,
    effort: str,
    workspace: ConversationWorkspace,
    streamer: Callable[..., Awaitable[dict[str, Any]]] = custom_stream_response,
) -> dict[str, Any]:
    """Run a Leader-directed, shared-workspace collaboration for Custom models."""
    agents: list[dict[str, Any]] = []
    final_answer = ""
    final_reasoning = ""

    async def emit() -> None:
        await update(
            {
                "answer": final_answer,
                "reasoning": final_reasoning,
                "searches": _searches_for(agents),
                "sources": _sources_for(agents),
                "usage": _usage_for(agents),
                "agents": _public_agents(agents),
            }
        )

    async def run_role(
        role: str,
        task: str,
        packet: str,
        *,
        phase: str = "work",
    ) -> dict[str, Any]:
        capabilities = {
            "leader": (False, "none", LEADER_FINAL_PROMPT if phase == "final" else LEADER_DECISION_PROMPT),
            "researcher": (True, "none", RESEARCHER_PROMPT),
            "programmer": (True, "full", PROGRAMMER_PROMPT),
            "inspector": (False, "read_only", INSPECTOR_PROMPT),
        }
        web_enabled, workspace_access, system_prompt = capabilities[role]
        tool_round_limits = {"leader": 0, "researcher": 6, "programmer": 12, "inspector": 6}
        record: dict[str, Any] = {
            "id": f"agent-{len(agents) + 1}",
            "role": role,
            "label": ROLE_LABELS[role],
            "status": "running",
            "task": task,
            "answer": "",
            "reasoning": "",
            "searches": [],
            "sources": [],
            "usage": {},
            "verdict": "",
            "error": "",
        }
        agents.append(record)
        await emit()

        async def role_update(state: dict[str, Any]) -> None:
            record["answer"] = _trim(state.get("answer"), MAX_VISIBLE_ANSWER_CHARS)
            record["reasoning"] = _trim(state.get("reasoning"), MAX_VISIBLE_REASONING_CHARS)
            record["searches"] = state.get("searches") or []
            record["sources"] = state.get("sources") or []
            record["usage"] = state.get("usage") or {}
            await emit()

        try:
            result = await streamer(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[*messages, {"role": "user", "content": packet}],
                timeout=timeout,
                stopped=stopped,
                update=role_update,
                settings=_role_settings(settings, role, phase),
                conversation_id=conversation_id,
                user_timezone=user_timezone,
                effort=effort,
                workspace=workspace if workspace_access != "none" else None,
                web_enabled=web_enabled,
                workspace_access=workspace_access,
                system_addendum=system_prompt,
                max_tool_rounds=tool_round_limits[role],
            )
        except asyncio.CancelledError:
            record["status"] = "stopped"
            await emit()
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)[:3_000]
            await emit()
            raise
        record["status"] = "completed"
        record["answer"] = _trim(result.get("answer"), MAX_VISIBLE_ANSWER_CHARS)
        record["reasoning"] = _trim(result.get("reasoning"), MAX_VISIBLE_REASONING_CHARS)
        record["searches"] = result.get("searches") or []
        record["sources"] = result.get("sources") or []
        record["usage"] = result.get("usage") or {}
        if result.get("tool_trace"):
            record["tool_trace"] = result["tool_trace"]
        await emit()
        return result

    pending_inspection = False
    blocker = ""
    guard_message = ""
    worker_actions = 0
    decisions = 0
    role_counts = {"research": 0, "program": 0, "inspect": 0}

    while decisions < MAX_LEADER_DECISIONS and worker_actions < MAX_WORKER_ACTIONS:
        if stopped():
            raise asyncio.CancelledError
        state_packet = _state_packet(
            agents,
            workspace,
            pending_inspection=pending_inspection,
            blocker=blocker,
            guard_message=guard_message,
        )
        decision_result = await run_role("leader", "决定下一步工作", state_packet, phase="decision")
        decisions += 1
        leader_record = agents[-1]
        try:
            decision = _parse_decision(str(decision_result.get("answer") or ""))
        except ValueError as exc:
            leader_record["status"] = "failed"
            leader_record["error"] = str(exc)
            guard_message = f"上一轮 Leader 调度格式无效：{exc}。请严格返回指定 JSON。"
            await emit()
            continue
        leader_record["decision_action"] = decision["action"]
        leader_record["decision_reason"] = decision["reason"]
        leader_record["task"] = decision["task"] or "汇总最终回答"
        await emit()

        action = decision["action"]
        if action == "finish":
            if pending_inspection or blocker:
                guard_message = "存在尚未通过检查的程序改动或待修复问题，不能结束。请安排程序员修复或检查员复检。"
                continue
            final_answer = decision["final_answer"]
            final_reasoning = str(leader_record.get("reasoning") or "")
            if not final_answer:
                guard_message = "finish 缺少必填的 final_answer，不能结束。请重新返回 finish JSON，并直接写好给用户的完整最终回答。"
                continue
            await emit()
            break

        if role_counts[action] >= MAX_ROLE_ACTIONS:
            guard_message = f"{action} 已达到单角色调用上限，请基于现有结果选择其他动作或在条件允许时结束。"
            continue

        role_counts[action] += 1
        worker_actions += 1
        guard_message = ""
        if action == "research":
            packet = (
                f"Leader 指派的调研任务：{decision['task']}\n\n"
                + _state_packet(agents, workspace, pending_inspection=pending_inspection, blocker=blocker)
            )
            await run_role("researcher", decision["task"], packet)
        elif action == "program":
            feedback = f"\n\n必须处理的检查员反馈：\n{blocker}" if blocker else ""
            packet = (
                f"Leader 指派的程序任务：{decision['task']}{feedback}\n\n"
                + _state_packet(agents, workspace, pending_inspection=pending_inspection, blocker=blocker)
            )
            await run_role("programmer", decision["task"], packet)
            blocker = ""
            pending_inspection = True
        else:
            packet = (
                f"Leader 指派的独立检查任务：{decision['task']}\n\n"
                + _state_packet(agents, workspace, pending_inspection=pending_inspection, blocker=blocker)
            )
            inspection = await run_role("inspector", decision["task"], packet)
            verdict = _inspection_verdict(str(inspection.get("answer") or ""))
            agents[-1]["verdict"] = verdict
            if verdict == "PASS":
                pending_inspection = False
                blocker = ""
            else:
                pending_inspection = True
                blocker = str(inspection.get("answer") or "检查员要求修改，但没有提供明确说明。")
            await emit()

    if not final_answer:
        unresolved = (
            "协作轮次已达到上限，且仍有未通过的程序检查。必须明确告诉用户未完成及剩余问题。"
            if pending_inspection or blocker
            else "协作轮次已达到上限。请基于已有产出给出最佳最终回答并说明限制。"
        )
        final_packet = _state_packet(
            agents,
            workspace,
            pending_inspection=pending_inspection,
            blocker=blocker,
            guard_message=unresolved,
        ) + "\n\n现在停止调度并向用户给出最终回答。"
        final_result = await run_role("leader", "在轮次上限内整合回答", final_packet, phase="final")
        final_answer = str(final_result.get("answer") or "").strip()
        final_reasoning = str(final_result.get("reasoning") or "")
        if not final_answer:
            raise RuntimeError("Leader 未生成最终回答")
        await emit()

    searches = _searches_for(agents)
    sources = _sources_for(agents)
    usage = _usage_for(agents)
    tool_trace = []
    for agent in agents:
        for item in agent.get("tool_trace") or []:
            tool_trace.append({**item, "agent_id": agent["id"], "agent_role": agent["role"]})
    return {
        "answer": final_answer,
        "reasoning": final_reasoning,
        "searches": searches,
        "sources": sources,
        "usage": usage,
        "tool_calls": [],
        "tool_trace": tool_trace,
        "agents": _public_agents(agents),
        "response": {"tool_trace": tool_trace},
    }
