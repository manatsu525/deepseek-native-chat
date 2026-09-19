"""Tool groups that are loaded on demand.

Every request used to carry the full schema of every tool and the file-work
rules, about 3,000 tokens before the user's question, although most chat
questions never touch a file. As in mainstream agents (deferred tools that
are fetched by a search/load tool), a group that is not needed yet is listed
only by name and one line in the ``load_tools`` tool. When the model loads a
group, its schemas are added to the next request and its working rules come
back in the tool result, which keeps the system prompt byte-stable.
"""

from __future__ import annotations

from typing import Any

GROUP_SUMMARIES = {
    "files": (
        "read, write and edit files in this conversation's workspace, run Python or shell commands, check web "
        "pages, and keep a task plan. Load it for any coding, file, data-processing or calculation task."
    ),
    "conversations": "list, read, create, rename and delete this user's conversations.",
    "skills": "list, read, install, enable and remove Skills (working instructions).",
}

CONVERSATION_PREFIX = "conversation_"
SKILL_PREFIX = "skill_"


def group_of_extra_tool(name: str) -> str | None:
    """The deferred group an Agent-mode tool belongs to, or None when always loaded."""
    if name.startswith(CONVERSATION_PREFIX):
        return "conversations"
    if name.startswith(SKILL_PREFIX):
        return "skills"
    return None


def load_tools_definition(groups: list[str]) -> dict[str, Any]:
    lines = "; ".join(f"{group}: {GROUP_SUMMARIES[group]}" for group in groups)
    return {
        "type": "function",
        "function": {
            "name": "load_tools",
            "description": (
                "Load additional tools before you need them. They are available from your next step on. "
                f"Groups — {lines}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {"type": "array", "items": {"type": "string", "enum": list(groups)}, "minItems": 1},
                },
                "required": ["groups"],
                "additionalProperties": False,
            },
        },
    }


def requested_groups(arguments: dict[str, Any], available: list[str]) -> list[str]:
    """Accept a list, a single string or a comma-separated string."""
    raw = arguments.get("groups", arguments.get("group", arguments.get("names")))
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, list):
        raw = []
    wanted = [str(item).strip().lower() for item in raw if str(item).strip()]
    unknown = [item for item in wanted if item not in available]
    if unknown or not wanted:
        raise ValueError(f"可加载的工具组：{', '.join(available)}")
    return list(dict.fromkeys(wanted))
