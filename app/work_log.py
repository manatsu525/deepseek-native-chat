"""Compact cross-turn record of what an answer's tool calls actually did.

Only the final assistant text is replayed to later turns, so without this the
model has to rediscover which files it created or changed. The log is built
deterministically from the stored tool trace and appended to that historical
assistant message when history is assembled, keeping the prompt prefix stable
across turns. The stored message and the UI are not changed.
"""

from __future__ import annotations

from typing import Any

WORK_LOG_HEADER = "[工具操作记录（应用自动生成，仅供后续对话参考，不是回答内容）]"
WORK_LOG_MAX_CHARS = 1_500
MAX_ITEMS_PER_LINE = 12

WRITE_TOOLS = {"write_file", "host_write_file", "frontend_write_page"}
EDIT_TOOLS = {"edit_file", "host_edit_file", "apply_line_edits", "apply_patch", "apply_patch_batch", "replace_text", "host_apply_patch"}
DELETE_TOOLS = {"delete_file", "host_delete_path"}
READ_TOOLS = {"read_file", "host_read_file", "frontend_read_page"}
VALIDATION_TOOLS = {"run_python", "check_web_syntax", "frontend_validate_page"}
COMMAND_TOOLS = {"host_run_command"}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _joined(values: list[str]) -> str:
    items = _unique(values)
    text = "、".join(items[:MAX_ITEMS_PER_LINE])
    if len(items) > MAX_ITEMS_PER_LINE:
        text += f" 等 {len(items)} 项"
    return text


def build_work_log(tool_trace: list[dict[str, Any]] | None) -> str:
    written: list[str] = []
    edited: list[str] = []
    deleted: list[str] = []
    viewed: list[str] = []
    checks: list[str] = []
    commands: list[str] = []
    failed = 0
    for item in tool_trace or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        path = str(item.get("path") or "")
        status = str(item.get("status") or "")
        ok = status in {"completed", "skipped"}
        if name in VALIDATION_TOOLS:
            # Workspace validators report ok=false inside a completed call, so
            # only "failed" is a verdict; "executed" makes no claim either way.
            checks.append(f"{name} {path} {'已执行' if ok else '未通过或出错'}".strip())
            continue
        if name in COMMAND_TOOLS:
            command = " ".join(path.split())[:120]
            commands.append(f"`{command}` {'成功' if ok else '失败'}")
            continue
        if not ok:
            if name in WRITE_TOOLS | EDIT_TOOLS | DELETE_TOOLS:
                failed += 1
            continue
        if name in WRITE_TOOLS:
            written.append(path)
        elif name in EDIT_TOOLS:
            edited.append(path)
        elif name in DELETE_TOOLS:
            deleted.append(path)
        elif name in READ_TOOLS:
            viewed.append(path)
    lines: list[str] = []
    for label, values in (("新建或覆盖", written), ("修改", edited), ("删除", deleted), ("查看", viewed)):
        if values:
            lines.append(f"- {label}：{_joined(values)}")
    if checks:
        lines.append(f"- 验证：{'；'.join(_unique(checks)[-6:])}")
    if commands:
        lines.append(f"- 命令：{'；'.join(commands[-6:])}")
    if failed:
        lines.append(f"- 未成功的文件操作：{failed} 次")
    if not lines:
        return ""
    text = WORK_LOG_HEADER + "\n" + "\n".join(lines)
    return text[:WORK_LOG_MAX_CHARS]


def with_work_log(content: Any, work_log: str) -> Any:
    if not work_log:
        return content
    if isinstance(content, list):
        return [*content, {"type": "text", "text": work_log}]
    return f"{content or ''}\n\n{work_log}"
