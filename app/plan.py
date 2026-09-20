"""The model's task plan for one answer.

A plan written with update_plan is server-side state: it is echoed back in
the tool result, carried in every context checkpoint, and stored with the
answer. That gives a small model a place to keep its intent that survives
compaction, instead of re-deriving the approach every round.
"""

from __future__ import annotations

import json
import re
import copy
from typing import Any

MAX_PLAN_ITEMS = 20
MAX_ITEM_CHARS = 200
STATUSES = ("pending", "in_progress", "done", "blocked")
STEP_ALIASES = ("step", "title", "description", "task", "content", "text", "name", "item")
STATUS_ALIASES = {"todo": "pending", "not_started": "pending", "open": "pending", "doing": "in_progress",
                  "active": "in_progress", "wip": "in_progress", "completed": "done", "complete": "done",
                  "finished": "done", "closed": "done"}

UPDATE_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": (
            "Manage the execution plan. For multi-step work, create concrete deliverables and a verification step; "
            "keep exactly one step in_progress while work remains. Execute that step, then mark it done with "
            "outcome and evidence (successful tool call IDs from its results), activating the next step. "
            "Always send the full list; retain returned step IDs. Use blocked with an outcome explaining the blocker. "
            "Changing steps or revisiting completed work requires replan_reason. State survives compaction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PLAN_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "Short description of the step"},
                            "status": {"type": "string", "enum": list(STATUSES)},
                            "id": {"type": "string", "description": "Keep the server-assigned step ID on updates"},
                            "outcome": {"type": "string", "description": "Concrete result or blocker"},
                            "evidence": {"type": "array", "items": {"type": "string"}, "description": "Successful tool call IDs for this step"},
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
                "note": {"type": "string", "description": "Optional: decisions or assumptions made so far"},
                "replan_reason": {"type": "string", "description": "New evidence justifying a changed plan"},
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
    },
}

PLAN_PROMPT = (
    "For any task that needs more than two or three tool calls, call update_plan first with the concrete steps, "
    "then update it as steps complete; it is the one place your plan survives context compaction."
)


_STATUS_KEYS = ("status", "state", "done", "completed", "complete", "finished")
_NON_TEXT_KEYS = set(_STATUS_KEYS) | {"id", "index", "order", "priority", "number", "no", "n"}
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\)|#+)?\s*(\[(?P<mark>[ xX~])\])?\s*")


def _split_plan_text(text: str) -> list[dict[str, str]]:
    """Turn a plan written as free text (one step per line) into items."""
    items: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _LINE_PREFIX_RE.match(line)
        body = line[match.end():].strip() if match else line.strip()
        if not body:
            continue
        mark = (match.group("mark") if match else None) or " "
        status = {"x": "done", "X": "done", "~": "in_progress"}.get(mark, "pending")
        items.append({"step": body, "status": status})
    return items


def _parse_step(item: Any) -> tuple[str, str]:
    """Extract (text, status) from one step in any common convention."""
    if isinstance(item, (str, int, float)):
        parsed = _split_plan_text(str(item))
        return (parsed[0]["step"][:MAX_ITEM_CHARS], parsed[0]["status"]) if parsed else ("", "pending")
    if not isinstance(item, dict):
        return "", "pending"
    # Models trained on other todo tools send title/description/task/content.
    text = next((str(item[key]) for key in STEP_ALIASES if isinstance(item.get(key), str) and item[key].strip()), "")
    if not text:
        candidates = [
            str(value) for key, value in item.items()
            if str(key).lower() not in _NON_TEXT_KEYS and isinstance(value, (str, int, float)) and str(value).strip()
        ]
        text = max(candidates, key=len) if candidates else ""
    status_value: Any = next((item[key] for key in _STATUS_KEYS if key in item), "pending")
    if isinstance(status_value, bool):
        status = "done" if status_value else "pending"
    else:
        status = str(status_value or "pending").strip().lower()
        status = STATUS_ALIASES.get(status, status)
    if status not in STATUSES:
        status = "pending"
    return " ".join(text.split())[:MAX_ITEM_CHARS], status


class TaskPlan:
    def __init__(self) -> None:
        self.steps: list[dict[str, str]] = []
        self.note = ""
        self.updates = 0

    def apply(self, arguments: dict[str, Any]) -> str:
        """Accept whatever shape the model used for its plan.

        A rejected plan call is worse than a loosely parsed one: the model
        rarely retries, and then works the whole task without the anchor.
        """
        raw = next((arguments[key] for key in ("steps", "plan", "items", "todos", "tasks", "checklist") if key in arguments), None)
        if raw is None and any(key in arguments for key in STEP_ALIASES):
            raw = [arguments]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            raw = parsed if isinstance(parsed, (list, dict)) else _split_plan_text(raw)
        if isinstance(raw, dict):
            raw = [
                value if isinstance(value, (dict, str)) else str(value)
                for value in raw.values()
            ]
        if not isinstance(raw, list) or not raw:
            raise ValueError("steps 必须是非空数组，每项包含 step 和 status")
        steps: list[dict[str, str]] = []
        for item in raw[:MAX_PLAN_ITEMS]:
            text, status = _parse_step(item)
            if text:
                steps.append({"step": text, "status": status})
        if not steps:
            raise ValueError("每个步骤需要非空的 step 文本")
        self.steps = steps
        self.note = " ".join(str(arguments.get("note") or "").split())[:1000]
        self.updates += 1
        done = sum(1 for item in steps if item["status"] == "done")
        return json.dumps(
            {"ok": True, "steps": len(steps), "done": done, "plan": self.render()},
            ensure_ascii=False,
        )

    def render(self) -> str:
        marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}
        lines = [f"{marks[item['status']]} {index}. {item['step']}" for index, item in enumerate(self.steps, 1)]
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines)

    def export(self) -> dict[str, Any] | None:
        if not self.steps:
            return None
        return {"steps": self.steps, "note": self.note}


class ExecutionPlan(TaskPlan):
    """Validated execution state, independent of model protocol and reasoning visibility.

    Evidence proves that an operation succeeded, not that the user's goal was
    semantically met. The model must state the outcome and plan verification.
    """

    def __init__(self, saved: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.operations = 0
        self.serial = 0
        self.revisions: list[dict[str, Any]] = []
        self.receipts: dict[str, list[dict[str, Any]]] = {}
        if saved and saved.get("version") == 1:
            self.steps = copy.deepcopy(saved.get("steps") or [])
            self.note = str(saved.get("note") or "")
            self.operations = int(saved.get("operations") or 0)
            self.serial = int(saved.get("serial") or len(self.steps))
            self.revisions = copy.deepcopy(saved.get("revisions") or [])
            # A process restart does not restore tool history or guarantee that
            # files are unchanged. Retain completed outcomes, but require fresh
            # evidence before completing the interrupted active step.
            self.note += "\n恢复的计划：先核对当前步骤涉及的文件，再继续；中断前的操作可能已经生效。"

    @property
    def active(self) -> dict[str, Any] | None:
        return next((s for s in self.steps if s["status"] == "in_progress"), None)

    @property
    def needs_plan(self) -> bool:
        return (not self.steps and self.operations >= 2) or bool(self.steps and not self.active)

    @property
    def unfinished(self) -> bool:
        return bool(self.steps and any(s["status"] != "done" for s in self.steps))

    def require_execution(self) -> None:
        if self.needs_plan:
            raise ValueError("当前处于规划阶段：先调用 update_plan 建立或调整计划并指定一个 in_progress 步骤，再执行工具。")

    def apply(self, arguments: dict[str, Any]) -> str:
        raw = arguments.get("steps")
        if not isinstance(raw, list) or not raw or len(raw) > MAX_PLAN_ITEMS:
            raise ValueError("steps 必须包含 1–20 个完整步骤")
        reason = str(arguments.get("replan_reason") or "").strip()[:1000]
        previous = {s["id"]: s for s in self.steps}
        titles = {s["step"]: s for s in self.steps}
        candidate = []
        serial = self.serial
        for item in raw:
            if not isinstance(item, dict) or item.get("status") not in STATUSES:
                raise ValueError("每个步骤需要 step 和有效的 status")
            title = " ".join(str(item.get("step") or "").split())[:MAX_ITEM_CHARS]
            if not title:
                raise ValueError("步骤描述不能为空")
            old = previous.get(str(item.get("id") or "")) if item.get("id") else titles.get(title)
            if item.get("id") and not old:
                raise ValueError("未知步骤 ID；新增步骤请省略 id")
            if not old:
                serial += 1
            step = {"id": old["id"] if old else f"s{serial}", "step": title, "status": item["status"],
                    "outcome": str(item.get("outcome", (old or {}).get("outcome", ""))).strip()[:1000],
                    "evidence": list((old or {}).get("evidence") or [])}
            status_changed = not old or old["status"] != step["status"]
            if step["status"] == "done" and status_changed:
                available = {r["id"] for r in self.receipts.get(step["id"], []) if r["status"] == "completed"}
                evidence = item.get("evidence") or []
                if not old or old["status"] != "in_progress" or old["step"] != title or not step["outcome"] or not isinstance(evidence, list) or not evidence or any(not isinstance(e, str) or e not in available for e in evidence):
                    raise ValueError("完成步骤需要先执行该当前步骤，再提供 outcome 和该步骤成功工具调用的 evidence ID")
                step["evidence"] = evidence[:8]
            if step["status"] == "blocked" and not step["outcome"]:
                raise ValueError("blocked 步骤必须用 outcome 说明缺失条件")
            if old and old["status"] == "done" and step != old and not reason:
                raise ValueError("修改已完成步骤需要 replan_reason")
            if old and old["status"] == "done" and step != old and step["status"] != "in_progress":
                raise ValueError("返工已完成步骤时，先用相同 ID 设为 in_progress，再执行并提交新证据")
            if old and old["status"] != "in_progress" and step["status"] == "in_progress":
                step["evidence"] = []
                step["outcome"] = ""
            candidate.append(step)
        ids = [s["id"] for s in candidate]
        if len(set(ids)) != len(ids) or len({s["step"] for s in candidate}) != len(candidate):
            raise ValueError("步骤不能重复")
        active_count = sum(s["status"] == "in_progress" for s in candidate)
        if active_count > 1 or (any(s["status"] == "pending" for s in candidate) and active_count != 1):
            raise ValueError("有待处理步骤时必须且只能有一个 in_progress 步骤")
        structural_change = [(s["id"], s["step"]) for s in candidate] != [(s["id"], s["step"]) for s in self.steps]
        deferred = self.active and next((s["status"] for s in candidate if s["id"] == self.active["id"]), "removed") == "pending"
        if self.steps and (structural_change or deferred) and not reason:
            raise ValueError("调整步骤或推迟当前步骤需要 replan_reason，说明新发现")
        if any(s["status"] == "done" and s["id"] not in ids for s in self.steps):
            raise ValueError("保留已完成步骤作为执行记录；需要返工时用相同 ID 重新激活")
        if reason:
            self.revisions = (self.revisions + [{"reason": reason, "previous": copy.deepcopy(self.steps)}])[-5:]
        for s in candidate:
            old = previous.get(s["id"])
            if s["status"] == "in_progress" and (not old or old["status"] != "in_progress" or old["step"] != s["step"]):
                self.receipts[s["id"]] = []
        self.steps, self.serial = candidate, serial
        if "note" in arguments:
            self.note = str(arguments.get("note") or "")[:1000]
        self.updates += 1
        return json.dumps({"ok": True, "plan": self.export()}, ensure_ascii=False)

    def record(self, call_id: str, name: str, status: str, path: str, result: str) -> None:
        self.operations += 1
        if self.active:
            key = self.active["id"]
            receipts = self.receipts.setdefault(key, [])
            receipts.append({"id": call_id, "tool": name, "status": status, "path": path[:300],
                             "result": result[:200]})

    def runtime_note(self) -> str:
        if not self.steps:
            return ("规划阶段：已完成初步探索。若任务仍需工具，先用 update_plan 建立具体交付步骤和验证步骤，"
                    "其中一个为 in_progress；若两次操作已足够，可直接回答。") if self.needs_plan else ""
        receipts = self.receipts.get((self.active or {}).get("id", ""), [])
        state = {"steps": self.steps, "note": self.note, "recent_results": receipts[-3:],
                 "eligible_evidence_ids": [r["id"] for r in receipts if r["status"] == "completed"]}
        return ("服务端执行状态（权威）：" + json.dumps(state, ensure_ascii=False) +
                "\n只推进当前步骤；达到该步骤结果后，用 update_plan 提交 outcome 和成功调用的 evidence ID，激活下一步。"
                "遇到新事实可用 replan_reason 调整；无法继续则标记 blocked 并说明原因。所有步骤完成后再给最终交付。")

    def export(self) -> dict[str, Any] | None:
        if not self.steps and not self.operations:
            return None
        return copy.deepcopy({"version": 1, "steps": self.steps, "note": self.note, "operations": self.operations,
                              "serial": self.serial, "revisions": self.revisions, "receipts": self.receipts})
