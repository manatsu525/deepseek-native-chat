"""The model's task plan for one answer.

A plan written with update_plan is server-side state: it is echoed back in
the tool result, carried in every context checkpoint, and stored with the
answer. That gives a small model a place to keep its intent that survives
compaction, instead of re-deriving the approach every round.
"""

from __future__ import annotations

import json
import re
from typing import Any

MAX_PLAN_ITEMS = 20
MAX_ITEM_CHARS = 200
STATUSES = ("pending", "in_progress", "done")
STEP_ALIASES = ("step", "title", "description", "task", "content", "text", "name", "item")
STATUS_ALIASES = {"todo": "pending", "not_started": "pending", "open": "pending", "doing": "in_progress",
                  "active": "in_progress", "wip": "in_progress", "completed": "done", "complete": "done",
                  "finished": "done", "closed": "done"}

UPDATE_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": (
            "Record or update your step-by-step plan for the current task. Call it once with all steps before "
            "starting a task that needs more than two or three tool calls, then again whenever a step's status "
            "changes (mark the finished step done and the next one in_progress). Always send the full list. "
            "The plan is kept for you across context compaction."
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
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
                "note": {"type": "string", "description": "Optional: decisions or assumptions made so far"},
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
        marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
        lines = [f"{marks[item['status']]} {index}. {item['step']}" for index, item in enumerate(self.steps, 1)]
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines)

    def export(self) -> dict[str, Any] | None:
        if not self.steps:
            return None
        return {"steps": self.steps, "note": self.note}
