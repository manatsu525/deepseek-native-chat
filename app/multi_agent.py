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
MAX_MODEL_CALLS = 20
MAX_VISIBLE_ANSWER_CHARS = 20_000
MAX_VISIBLE_REASONING_CHARS = 12_000
MAX_STATE_OUTPUT_CHARS = 5_000
MAX_RESEARCH_SEARCHES_PER_ENTRY = 1
MAX_RESEARCH_FETCHES_PER_ENTRY = 2
MAX_RESEARCH_TOOLS_PER_ENTRY = 3
MAX_FINAL_QUALITY_RETRIES = 1

ROLE_LABELS = {
    "leader": "Nexus",
    "researcher": "Atlas",
    "programmer": "Forge",
    "inspector": "Sentinel",
}


class ModelCallLimitExceeded(RuntimeError):
    pass


class ModelCallBudget:
    def __init__(self, limit: int = MAX_MODEL_CALLS) -> None:
        self.limit = int(limit)
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def consume(self) -> None:
        if self.used >= self.limit:
            raise ModelCallLimitExceeded(f"多智能体协作每个问题最多调用模型 {self.limit} 次")
        self.used += 1

LEADER_DECISION_PROMPT = """你是四智能体协作组的 Nexus，负责决定下一步由哪个角色工作，并最终整合答案。你自己没有工具，也不要假装执行过搜索、文件修改或测试。

每次只决定一个下一步动作，并且只输出一个 JSON 对象，不要 Markdown 代码块或额外文字：
{"action":"research|program|inspect|finish","reason":"简短决策依据","final_answer":"仅 finish 时填写的完整最终回答，其他动作为空"}

规则：
- 用户原始对话是唯一任务来源，优先级高于你的理解、共享状态和其他智能体结论。你只能选择角色，不能改写用户任务后再分派；编排器会直接把用户原始请求交给所选角色。
- 必须原样尊重用户表达的对象、名称、版本、人物、时间、数量、否定关系、条件、范围和输出要求。你认为用户可能写错、概念可疑或含义不明确时，不得静默纠正、替换成相似概念或自行补全；需要外部确认就选 research，否则在最终回答中明确保留不确定性或请用户澄清。
- “现有知识里没有”“没有搜到”和“资料不足”都不等于事实不存在。不得把猜测、近似对象或未验证结论升级成确定事实。
- research：需要外部资料、文档、事实核查时交给 Atlas；可以在工作中的任何阶段再次调用。
- program：需要创建/修改文件、执行代码或修复 Sentinel 发现的问题时交给 Forge。
- inspect：Forge 产出后交给 Sentinel 独立读取、测试和审查。
- finish：证据和产出已经足够且没有待修复或待复检的问题时结束；必须同时在 final_answer 中直接写好给用户的完整回答，以免再调用一次模型。
- 多智能体模式的最终回答默认必须全面、详细，因为用户选择该模式就是希望得到完整的协作成果。只要调用过 Atlas、Forge 或 Sentinel，就要充分整合各角色的有效产出；除非用户明确要求一句话、简短或只给结论，否则不能只给一段压缩总结。调研类要保留关键证据、重要数据或对比、正反观点、局限与不确定性、实用建议及可用来源；工程类要说明实际改动、关键实现、验证结果、文件和遗留限制。不要机械复制各角色全文，但也不能丢掉核心信息。没有调用其他角色的简单问题可以直接简洁回答，禁止为了凑长度灌水。
- 不要为了凑齐角色而调用不需要的智能体，也不要机械串联；根据共享状态作真实决策。
- Sentinel 报告 REVISE 后，必须让 Forge 实际修复，并再次让 Sentinel 复检通过，才能 finish。
"""

LEADER_FINAL_PROMPT = """你是协作组 Nexus。根据用户原始对话、Atlas 调研结论、Forge 实际保存的文件和 Sentinel 检查结果，给出最终回答。用户原始对话是唯一任务来源；其他智能体的任务理解和结论只能作为证据，不能改变用户原意。必须保留用户表达的对象、名称、版本、人物、时间、数量、否定关系、条件、范围和输出要求。资料不足、没有搜到或无法确认时必须如实保留不确定性，绝不能改成相似对象继续回答，也不能宣称其不存在。多智能体模式默认输出全面、详细的综合成果：只要调用过其他角色，除非用户明确要求一句话、简短或只给结论，否则必须充分整合其有效产出。调研类应包含结论、关键证据与数据、重要对比、正反观点、局限与不确定性、实用建议和可用来源；工程类应包含实际改动、关键实现、验证结果、文件和遗留限制。不要机械复制角色全文，但禁止把丰富成果压成一小段泛泛结论。没有调用其他角色的简单问题可以直接简洁回答，禁止为了长度灌水。只回答用户，不要再输出调度 JSON，不要声称做过记录中没有发生的工作。使用与用户提问相同的语言。"""

RESEARCHER_PROMPT = """你是协作组的 Atlas，只负责外部信息调研和事实核查。你可以使用搜索和网页抓取工具，但没有工作区文件权限。用户原始对话是唯一任务来源；Nexus 只选择了调研角色，无权改写调研对象。开始前必须对照用户原文，原样保留其中的对象、名称、版本、人物、时间、数量、否定关系、条件和范围。遇到陌生、可疑或歧义表达，先按用户原文核实身份，禁止擅自替换为你熟悉的相似概念；没有搜到只能报告“暂未验证”，不能推断“不存在”。采用渐进式调研：先用一次搜索同时提出少量互补查询，再只抓取最有价值的页面；证据足以回答当前问题就立即停止，绝不能为了用满额度而继续搜索或抓取。每次出场最多搜索 1 次、抓取 2 次、合计 3 次，未使用的共享额度留给后续再次调研。整个用户问题内，Atlas 所有出场合计仍最多搜索 3 次、抓取 3 次、合计 6 次。寻找可靠资料、交叉核对关键结论，并给 Forge 或 Nexus 提供简洁、可执行且带来源的报告。不要假装修改或测试文件，禁止绕过或无意义消耗额度。"""

PROGRAMMER_PROMPT = """你是协作组的 Forge，负责具体执行、创建和修改共享工作区文件，并运行适用的检查。你拥有联网和完整工作区工具。用户原始对话是唯一任务来源；Nexus 只选择了编程角色，无权重写需求。必须以用户原文及工作区真实状态为准，保留对象、技术栈、版本、文件、条件、范围、禁止项和输出要求；其他角色的结论与用户原意冲突时，以用户原意为准并明确报告冲突，绝不能按被篡改的理解执行。先查看当前工作区真实状态，再落实用户请求；收到 Sentinel 反馈时必须针对反馈实际修改文件，不要只描述建议。尽量局部修改，避免无意义地重读或重写文件。完成后简洁列出改动文件、执行的验证及结果，不要在回答中重复粘贴已保存的完整代码。"""

INSPECTOR_PROMPT = """你是协作组的 Sentinel，负责独立审查 Forge 产出。你只有只读文件、搜索文件和本地测试/语法检查工具，绝不能创建、修改或删除文件。用户原始对话是唯一验收标准；Nexus 只选择了检查角色，无权重写验收目标。必须逐项对照用户原文中的对象、技术栈、版本、文件、条件、范围、禁止项和输出要求；如果 Forge 或 Atlas 改变了原意，即使代码本身能运行也必须判定 REVISE，并指出偏离之处。必须读取相关文件并尽可能运行实际检查，指出可复现的问题、文件路径和证据。整个检查最多有 6 次工具调用：先选最相关的文件，再运行覆盖面最大的验证；工作区未发生变化时，绝对不要重复读取同一范围或重复运行相同测试。一次综合测试成功且代码审查无阻断问题后，应立即给出结论。报告最后单独输出且仅输出以下两种结论之一：
VERDICT: PASS
VERDICT: REVISE
只在产出满足任务且检查未发现阻断问题时 PASS；需要 Forge 修复时必须 REVISE。"""


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
    raise ValueError("Nexus 没有返回有效的调度 JSON")


def _parse_decision(value: str) -> dict[str, str]:
    raw = _extract_json_object(value)
    action = str(raw.get("action") or "").strip().casefold()
    if action not in {"research", "program", "inspect", "finish"}:
        raise ValueError(f"Nexus 返回了无效动作：{action or '空'}")
    task = " ".join(str(raw.get("task") or "").split())[:2_000]
    reason = " ".join(str(raw.get("reason") or "").split())[:1_000]
    final_answer = str(raw.get("final_answer") or "").strip()
    return {"action": action, "task": task, "reason": reason, "final_answer": final_answer}


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _latest_user_request(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").casefold() != "user":
            continue
        text = _message_content_text(message.get("content")).strip()
        if text:
            return text
    return ""


def _orchestrated_role_task(action: str, *, has_blocker: bool = False) -> str:
    """Create a role objective without letting Nexus paraphrase user intent."""
    if action == "research":
        return "依据用户原始请求和当前共享状态，核实完成任务仍缺少的外部事实与资料"
    if action == "program":
        if has_blocker:
            return "依据用户原始请求落实工作，并修复当前 Sentinel 指出的未通过项"
        return "依据用户原始请求、已验证资料和当前工作区状态完成具体实现"
    if action == "inspect":
        return "以用户原始请求为唯一验收标准，独立检查当前产出并运行必要测试"
    return "依据用户原始请求和已有真实产出整合最终回答"


def _research_entry_limits(remaining: tuple[int, int, int]) -> tuple[int, int, int]:
    searches, fetches, total = remaining
    return (
        min(searches, MAX_RESEARCH_SEARCHES_PER_ENTRY),
        min(fetches, MAX_RESEARCH_FETCHES_PER_ENTRY),
        min(total, MAX_RESEARCH_TOOLS_PER_ENTRY),
    )


def _explicitly_requests_concise(user_request: str) -> bool:
    text = str(user_request or "").casefold()
    return bool(re.search(
        r"(?:一句(?:话|结论)|只(?:需|要)?回答|只给(?:一句|结论|答案|要点)|简短|简要|精简|简单说|不要展开|无需展开|直接回答|"
        r"brief|concise|short answer|one sentence|just (?:answer|the answer)|tldr only)",
        text,
    ))


def _required_final_answer_chars(user_request: str, agents: list[dict[str, Any]]) -> int:
    if _explicitly_requests_concise(user_request):
        return 0
    research_chars = sum(
        len(str(agent.get("answer") or "").strip())
        for agent in agents
        if agent.get("role") == "researcher" and agent.get("status") == "completed"
    )
    if research_chars:
        return min(1_800, max(800, research_chars // 5))
    worker_chars = sum(
        len(str(agent.get("answer") or "").strip())
        for agent in agents
        if agent.get("role") in {"programmer", "inspector"} and agent.get("status") == "completed"
    )
    if worker_chars:
        return min(1_200, max(600, worker_chars // 6))
    return 0


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
    usage["model_calls"] = sum(int((agent.get("usage") or {}).get("model_calls") or 0) for agent in agents)
    return usage


def _research_tool_usage(agents: list[dict[str, Any]]) -> dict[str, int]:
    searches = 0
    fetches = 0
    for agent in agents:
        if agent.get("role") != "researcher":
            continue
        for step in agent.get("searches") or []:
            if not step.get("quota_counted"):
                continue
            if step.get("action") == "search":
                searches += 1
            elif step.get("action") == "open_page":
                fetches += 1
    return {"searches": searches, "fetches": fetches, "total": searches + fetches}


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
    model_calls_used: int = 0,
    model_call_limit: int = MAX_MODEL_CALLS,
    current_user_request: str = "",
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
        "immutable_current_user_request": current_user_request,
        "intent_policy": "当前请求必须结合此前用户原始对话理解；二者共同构成唯一任务来源。共享状态和任何角色结论只能补充证据，不得改写、纠正、替换或缩小用户原意。",
        "shared_workspace_files": workspace.list_files(),
        "completed_work": completed[-8:],
        "program_changes_need_inspection": pending_inspection,
        "unresolved_inspector_feedback": _trim(blocker, MAX_STATE_OUTPUT_CHARS),
        "orchestrator_guard": guard_message,
        "model_calls": {"used": model_calls_used, "limit": model_call_limit, "remaining": max(0, model_call_limit - model_calls_used)},
        "research_tool_usage": _research_tool_usage(agents),
        "research_tool_limits": {"searches": 3, "fetches": 3, "total": 6},
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
    """Run a Nexus-directed, shared-workspace collaboration for Custom models."""
    agents: list[dict[str, Any]] = []
    final_answer = ""
    final_reasoning = ""
    budget = ModelCallBudget()
    current_user_request = _latest_user_request(messages)

    def shared_state(
        *,
        pending_inspection: bool,
        blocker: str,
        guard_message: str = "",
    ) -> str:
        return _state_packet(
            agents,
            workspace,
            pending_inspection=pending_inspection,
            blocker=blocker,
            guard_message=guard_message,
            model_calls_used=budget.used,
            model_call_limit=budget.limit,
            current_user_request=current_user_request,
        )

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
        reserve_calls: int = 0,
        web_limits: tuple[int, int, int] = (3, 3, 6),
    ) -> dict[str, Any]:
        capabilities = {
            "leader": (False, "none", LEADER_FINAL_PROMPT if phase == "final" else LEADER_DECISION_PROMPT),
            "researcher": (True, "none", RESEARCHER_PROMPT),
            "programmer": (True, "full", PROGRAMMER_PROMPT),
            "inspector": (False, "read_only", INSPECTOR_PROMPT),
        }
        web_enabled, workspace_access, system_prompt = capabilities[role]
        tool_round_limits = {"leader": 0, "researcher": 6, "programmer": 12, "inspector": 6}
        if budget.remaining <= reserve_calls:
            raise ModelCallLimitExceeded(f"模型调用剩余 {budget.remaining} 次，必须预留 {reserve_calls} 次完成协作")
        role_call_allowance = budget.remaining - reserve_calls
        effective_tool_rounds = min(tool_round_limits[role], max(0, role_call_allowance - 1))
        if role == "researcher":
            effective_tool_rounds = min(effective_tool_rounds, max(0, web_limits[2]))
        start_model_calls = budget.used
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
            record["usage"] = dict(state.get("usage") or {})
            record["usage"]["model_calls"] = budget.used - start_model_calls
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
                max_tool_rounds=effective_tool_rounds,
                web_search_limit=web_limits[0],
                web_fetch_limit=web_limits[1],
                web_tool_round_limit=web_limits[2],
                before_model_call=budget.consume,
            )
        except asyncio.CancelledError:
            record["status"] = "stopped"
            await emit()
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)[:3_000]
            record["usage"] = dict(record.get("usage") or {})
            record["usage"]["model_calls"] = budget.used - start_model_calls
            await emit()
            raise
        record["status"] = "completed"
        record["answer"] = _trim(result.get("answer"), MAX_VISIBLE_ANSWER_CHARS)
        record["reasoning"] = _trim(result.get("reasoning"), MAX_VISIBLE_REASONING_CHARS)
        record["searches"] = result.get("searches") or []
        record["sources"] = result.get("sources") or []
        record["usage"] = dict(result.get("usage") or {})
        record["usage"]["model_calls"] = budget.used - start_model_calls
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
    budget_exhausted = False
    final_quality_retries = 0

    while decisions < MAX_LEADER_DECISIONS and worker_actions < MAX_WORKER_ACTIONS and budget.remaining > 0:
        if stopped():
            raise asyncio.CancelledError
        forced_final = budget.remaining == 1
        if forced_final:
            guard_message = (
                f"只剩最后 1 次模型调用，必须立即选择 finish 并在 final_answer 中如实汇总；"
                f"如仍有未修复或未复检问题，必须明确告知用户。总调用绝不能超过 {budget.limit} 次。"
            )
        state_packet = shared_state(
            pending_inspection=pending_inspection,
            blocker=blocker,
            guard_message=guard_message,
        )
        try:
            decision_result = await run_role("leader", "决定下一步工作", state_packet, phase="decision")
        except ModelCallLimitExceeded:
            budget_exhausted = True
            break
        decisions += 1
        leader_record = agents[-1]
        try:
            decision = _parse_decision(str(decision_result.get("answer") or ""))
        except ValueError as exc:
            leader_record["status"] = "failed"
            leader_record["error"] = str(exc)
            guard_message = f"上一轮 Nexus 调度格式无效：{exc}。请严格返回指定 JSON。"
            await emit()
            continue
        leader_record["decision_action"] = decision["action"]
        leader_record["decision_reason"] = decision["reason"]
        safe_task = _orchestrated_role_task(decision["action"], has_blocker=bool(blocker))
        leader_record["task"] = "选择下一步角色" if decision["action"] != "finish" else safe_task
        await emit()

        action = decision["action"]
        if action == "finish":
            if (pending_inspection or blocker) and not forced_final:
                guard_message = "存在尚未通过 Sentinel 检查的 Forge 改动或待修复问题，不能结束。请安排 Forge 修复或 Sentinel 复检。"
                continue
            candidate_answer = decision["final_answer"]
            if not candidate_answer:
                guard_message = "finish 缺少必填的 final_answer，不能结束。请重新返回 finish JSON，并直接写好给用户的完整最终回答。"
                continue
            required_chars = _required_final_answer_chars(current_user_request, agents)
            if (
                required_chars
                and len(candidate_answer.strip()) < required_chars
                and final_quality_retries < MAX_FINAL_QUALITY_RETRIES
                and not forced_final
            ):
                final_quality_retries += 1
                guard_message = (
                    f"最终回答过度压缩：当前只有 {len(candidate_answer.strip())} 个字符，未充分呈现已有协作成果。"
                    f"不要重新调研；请立即重新选择 finish，把最终回答扩展到至少约 {required_chars} 个字符，"
                    "全面整合相关角色的有效产出；调研类保留关键数据或对比、正反证据、局限与不确定性、实用建议和来源，"
                    "工程类保留实际改动、关键实现、验证、文件和遗留限制。"
                )
                continue
            final_answer = candidate_answer
            final_reasoning = str(leader_record.get("reasoning") or "")
            if forced_final and (pending_inspection or blocker):
                final_answer = (
                    f"⚠️ 多智能体协作已达到每个问题最多 {budget.limit} 次模型调用的硬上限；"
                    "以下结果仍有未修复或未复检项目。\n\n" + final_answer
                )
            await emit()
            break

        if role_counts[action] >= MAX_ROLE_ACTIONS:
            guard_message = f"{action} 已达到单角色调用上限，请基于现有结果选择其他动作或在条件允许时结束。"
            continue

        if action == "research":
            research_used = _research_tool_usage(agents)
            research_limits = (
                max(0, 3 - research_used["searches"]),
                max(0, 3 - research_used["fetches"]),
                max(0, 6 - research_used["total"]),
            )
            if research_limits[2] <= 0 or (research_limits[0] <= 0 and research_limits[1] <= 0):
                guard_message = "Atlas 已用完本问题的 3 次搜索、3 次抓取或合计 6 次工具额度，不能继续调研。"
                continue
            if budget.remaining < 3:
                guard_message = "模型调用预算不足以完成一次调研并让 Nexus 汇总，请立即 finish。"
                continue
            role_counts[action] += 1
            worker_actions += 1
            guard_message = ""
            entry_limits = _research_entry_limits(research_limits)
            packet = (
                f"编排器分配的角色目标：{safe_task}\n"
                "注意：这不是对用户请求的改写；必须以共享状态中的 immutable_current_user_request 和此前用户原始对话为准。\n\n"
                + shared_state(pending_inspection=pending_inspection, blocker=blocker)
            )
            try:
                await run_role("researcher", safe_task, packet, reserve_calls=1, web_limits=entry_limits)
            except ModelCallLimitExceeded:
                budget_exhausted = True
                break
        elif action == "program":
            if budget.remaining < 5:
                guard_message = "模型调用预算不足以执行编程、检查和最终汇总，请立即 finish 并说明未完成项。"
                continue
            role_counts[action] += 1
            worker_actions += 1
            guard_message = ""
            feedback = f"\n\n必须处理的 Sentinel 反馈：\n{blocker}" if blocker else ""
            packet = (
                f"编排器分配的角色目标：{safe_task}{feedback}\n"
                "注意：这不是对用户请求的改写；必须以共享状态中的 immutable_current_user_request 和此前用户原始对话为准。\n\n"
                + shared_state(pending_inspection=pending_inspection, blocker=blocker)
            )
            try:
                await run_role("programmer", safe_task, packet, reserve_calls=3)
            except ModelCallLimitExceeded:
                budget_exhausted = True
                break
            blocker = ""
            pending_inspection = True
        else:
            if budget.remaining < 3:
                guard_message = "模型调用预算不足以执行检查并让 Nexus 汇总，请立即 finish 并说明未复检。"
                continue
            role_counts[action] += 1
            worker_actions += 1
            guard_message = ""
            packet = (
                f"编排器分配的角色目标：{safe_task}\n"
                "注意：这不是对用户请求的改写；必须以共享状态中的 immutable_current_user_request 和此前用户原始对话为准。\n\n"
                + shared_state(pending_inspection=pending_inspection, blocker=blocker)
            )
            try:
                inspection = await run_role("inspector", safe_task, packet, reserve_calls=1)
            except ModelCallLimitExceeded:
                budget_exhausted = True
                break
            verdict = _inspection_verdict(str(inspection.get("answer") or ""))
            agents[-1]["verdict"] = verdict
            if verdict == "PASS":
                pending_inspection = False
                blocker = ""
            else:
                pending_inspection = True
                blocker = str(inspection.get("answer") or "Sentinel 要求修改，但没有提供明确说明。")
            await emit()

    if not final_answer:
        unresolved = (
            "模型调用或协作轮次即将达到上限，且仍有未通过的程序检查。必须明确告诉用户未完成及剩余问题。"
            if pending_inspection or blocker
            else "模型调用或协作轮次即将达到上限。请基于已有产出给出最佳最终回答并说明限制。"
        )
        if budget.remaining > 0:
            final_packet = shared_state(
                pending_inspection=pending_inspection,
                blocker=blocker,
                guard_message=unresolved,
            ) + "\n\n现在停止调度并向用户给出最终回答。"
            try:
                final_result = await run_role("leader", "在硬限制内整合回答", final_packet, phase="final")
            except ModelCallLimitExceeded:
                budget_exhausted = True
            else:
                final_answer = str(final_result.get("answer") or "").strip()
                final_reasoning = str(final_result.get("reasoning") or "")
        if not final_answer:
            completed = [
                f"- {agent.get('label')}：{_trim(agent.get('answer'), 1_500)}"
                for agent in agents
                if agent.get("role") != "leader" and agent.get("status") == "completed" and agent.get("answer")
            ][-3:]
            files = "、".join(item["path"] for item in workspace.list_files()) or "无"
            reason = "模型调用已达到硬上限" if budget_exhausted or budget.remaining <= 0 else "协作轮次已达到上限"
            final_answer = (
                f"⚠️ {reason}（每个问题最多 {budget.limit} 次），已停止继续请求以避免触发 RPM 限流。\n\n"
                + ("当前已完成：\n" + "\n".join(completed) + "\n\n" if completed else "")
                + f"当前工作区文件：{files}。"
                + ("\n\n仍有未修复或未复检的问题，请在下一条消息中继续。" if pending_inspection or blocker else "")
            )
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
