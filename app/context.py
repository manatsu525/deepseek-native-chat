"""Context budgeting for one answer's tool loop.

The request is measured as a whole (system prompt, history, current user
message, tool traffic). When it exceeds the model's budget, old tool results
are replaced by one-line stubs, oldest first, until the request is back under
the low-water mark. The assistant's own messages, its tool calls and any
provider reasoning items are never touched, so the model keeps its plan and
the sequence of what it did; only bulky outputs it has already acted on are
dropped. Whole exchanges are removed only as a last resort.

A checkpoint attached to the current user message carries what stubs cannot:
the current content of files the model has read or written, its task plan,
and the sources it found. It is rebuilt only when compaction runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .file_knowledge import FileKnowledge

# The settings value. Left at the default it means "size the budget from the
# model's context window"; any other value is an explicit character budget.
DEFAULT_CONTEXT_BUDGET_CHARS = 240_000
MIN_CONTEXT_BUDGET_CHARS = 40_000
MAX_CONTEXT_BUDGET_CHARS = 2_000_000
# Automatic sizing: this share of the model's window, capped in tokens.
# Compaction throws away work the model has done, so on large-window models
# it starts only past 512K tokens; smaller windows stay bounded by the share.
CONTEXT_WINDOW_SHARE = 0.6
CONTEXT_TOKEN_CAP = 512_000
ASSUMED_WINDOW_TOKENS = 128_000
# Snapshots in the checkpoint never take more than this many characters.
SNAPSHOT_MAX_CHARS = 200_000
LOW_WATER_RATIO = 0.6
# The newest exchanges are never stubbed: the model is acting on them now.
PROTECTED_RECENT_EXCHANGES = 2
# Results shorter than this are kept; a stub would not save anything.
STUB_MIN_CHARS = 400
# Share of the budget the checkpoint's file snapshots may use.
SNAPSHOT_BUDGET_RATIO = 0.35
PROGRESS_NOTE_COUNT = 12
PROGRESS_NOTE_CHARS = 400
CONTEXT_CHECKPOINT_MARKER = "\n\nCONTEXT CHECKPOINT:\n"
STUB_PREFIX = "[已省略的工具结果] "
READ_TOOLS = {"read_file", "host_read_file", "frontend_read_page"}
COMMAND_TOOLS = {"run_command", "host_run_command"}


ENCRYPTED_REASONING_WEIGHT = 8


def serialized_chars(messages: list[dict[str, Any]]) -> int:
    """Request size in characters, with opaque reasoning blobs discounted.

    Providers replay ``encrypted_content`` without charging it as input, and
    it is base64 (about 8 chars per token), so counting it at face value made
    a 25K-token request look like 150K characters and forced needless
    compaction.
    """
    total = 0
    for message in messages:
        total += len(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
        for item in message.get("responses_output_items") or []:
            blob = item.get("encrypted_content") if isinstance(item, dict) else None
            if isinstance(blob, str):
                total -= len(blob) - len(blob) // ENCRYPTED_REASONING_WEIGHT
    return total


def normalize_budget(value: Any) -> int:
    try:
        budget = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_BUDGET_CHARS
    return max(MIN_CONTEXT_BUDGET_CHARS, min(MAX_CONTEXT_BUDGET_CHARS, budget))


def effective_context_budget(
    setting: int,
    *,
    window_tokens: int | None,
    request_chars: int,
    input_tokens: int,
) -> int:
    """The character budget for the next request.

    A character count is only a proxy for tokens, so the budget is derived from
    the model's window using the ratio the provider itself reported for the
    last request (characters sent versus input tokens billed). Before any
    round has reported usage, or when the user set an explicit budget, the
    settings value is used as is. A 50K-token budget on a 1M-token model was
    dropping whole rounds of finished work mid-task.
    """
    if setting != DEFAULT_CONTEXT_BUDGET_CHARS:
        return setting
    if input_tokens <= 0 or request_chars <= 0:
        return setting
    ratio = request_chars / input_tokens
    tokens = min(int((window_tokens or ASSUMED_WINDOW_TOKENS) * CONTEXT_WINDOW_SHARE), CONTEXT_TOKEN_CAP)
    return int(max(MIN_CONTEXT_BUDGET_CHARS, min(MAX_CONTEXT_BUDGET_CHARS, tokens * ratio)))


def with_message_block(message: dict[str, Any], marker: str, text: str) -> dict[str, Any]:
    """Return a copy of ``message`` with its ``marker`` block replaced by ``text``."""
    result = dict(message)
    content = result.get("content")
    if isinstance(content, list):
        label = marker.lstrip("\n")
        parts = [
            part for part in content
            if not (isinstance(part, dict) and part.get("type") == "text" and str(part.get("text") or "").startswith(label))
        ]
        if text:
            parts.append({"type": "text", "text": f"{label}{text}"})
        result["content"] = parts
    else:
        original = str(content or "").split(marker, 1)[0]
        result["content"] = f"{original}{marker}{text}" if text else original
    return result


def checkpoint_present(messages: list[dict[str, Any]]) -> bool:
    label = CONTEXT_CHECKPOINT_MARKER.lstrip("\n")
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(isinstance(part, dict) and str(part.get("text") or "").startswith(label) for part in content):
                return True
        elif CONTEXT_CHECKPOINT_MARKER in str(content or "") or str(content or "").startswith(label):
            return True
    return False


def checkpoint_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Decode the checkpoint JSON from wherever it was attached."""
    label = CONTEXT_CHECKPOINT_MARKER.lstrip("\n")
    for message in messages:
        content = message.get("content")
        texts = (
            [str(part.get("text") or "") for part in content if isinstance(part, dict)]
            if isinstance(content, list) else [str(content or "")]
        )
        for text in texts:
            if label in text:
                try:
                    return json.loads(text.split(label, 1)[1])
                except ValueError:
                    return {}
    return {}


def _exchanges(internal: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(start, end) index pairs: an assistant message plus its tool results."""
    starts = [index for index, message in enumerate(internal) if message.get("role") == "assistant"]
    return [(start, starts[position + 1] if position + 1 < len(starts) else len(internal)) for position, start in enumerate(starts)]


def _call_index(assistant: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for call in assistant.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(str(function.get("arguments") or "{}"))
        except ValueError:
            arguments = {}
        calls[str(call.get("id") or "")] = (str(function.get("name") or ""), arguments if isinstance(arguments, dict) else {})
    return calls


def stub_for(name: str, arguments: dict[str, Any], result: str) -> str:
    """One line that says what the dropped result was and how to get it back."""
    size = len(result)
    data: dict[str, Any] = {}
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            data = parsed
    except ValueError:
        pass
    path = str(data.get("path") or arguments.get("path") or "")
    if name in READ_TOOLS:
        return (
            f"{STUB_PREFIX}{name} {path}：此前读取的 {size} 字符已省略。"
            "当前内容若在 CONTEXT CHECKPOINT 的 file_snapshots 中则直接使用；否则需要时重新读取。"
        )
    if name in COMMAND_TOOLS:
        command = " ".join(str(arguments.get("command") or "").split())[:160]
        exit_code = data.get("exit_code", "?")
        return f"{STUB_PREFIX}{name} `{command}`（exit {exit_code}）：{size} 字符的输出已省略。需要时重新执行更精确的命令。"
    if name == "web_search":
        return f"{STUB_PREFIX}web_search 的结果（{size} 字符）已省略；来源见 CONTEXT CHECKPOINT 的 sources。同一问题不要重复搜索。"
    if name == "fetch_webpage":
        url = str(arguments.get("url") or "")[:200]
        return f"{STUB_PREFIX}fetch_webpage {url} 的正文（{size} 字符）已省略；确实需要时可以再次读取，本对话内已缓存。"
    return f"{STUB_PREFIX}{name}{(' ' + path) if path else ''} 的结果（{size} 字符）已省略。"


def compact_request(
    conversation: list[dict[str, Any]],
    *,
    base_message_count: int,
    budget: int,
    knowledge: FileKnowledge | None = None,
    workspace_files: Callable[[], list[dict[str, Any]]] | None = None,
    sources: dict[str, dict[str, str]] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bring the request under ``budget`` chars; return what was done, or None.

    The returned ``stubbed`` list holds (tool name, arguments) of every result
    replaced this time, so the caller can lift dedupe marks that pointed at
    content which is no longer in the request.
    """
    if serialized_chars(conversation) <= budget:
        return None
    target = int(budget * LOW_WATER_RATIO)
    base = [dict(message) for message in conversation[:base_message_count]]
    internal = [dict(message) for message in conversation[base_message_count:]]
    previous = checkpoint_payload(base)

    snapshots = knowledge.snapshots(min(int(budget * SNAPSHOT_BUDGET_RATIO), SNAPSHOT_MAX_CHARS)) if knowledge is not None else []
    checkpoint: dict[str, Any] = {
        "context_checkpoint": True,
        "instruction": (
            "Older tool results were replaced by one-line stubs to fit the context budget; your own messages and "
            "tool calls are intact. file_snapshots hold the exact current content of files you read or wrote "
            "(numbered lines, including your own edits): use them instead of reading those files again. "
            "plan is your task plan; continue from it and keep it updated. Do not repeat completed operations."
        ),
        "file_snapshots": snapshots,
        "sources": [
            {"url": item.get("url", ""), "title": item.get("title", ""), "summary": str(item.get("summary") or "")[:200]}
            for item in list((sources or {}).values())[:30]
        ],
    }
    if workspace_files is not None:
        checkpoint["workspace_files"] = workspace_files()
    if plan:
        checkpoint["plan"] = plan
    notes: list[str] = list(previous.get("progress_notes") or [])

    def attach(notes_now: list[str]) -> None:
        if notes_now:
            checkpoint["progress_notes"] = notes_now[-PROGRESS_NOTE_COUNT:]
            checkpoint["instruction"] += (
                " progress_notes are your own earlier statements from rounds that were removed entirely, oldest first."
            )
        else:
            checkpoint.pop("progress_notes", None)
        text = json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":"))
        if base and base[-1].get("role") == "user":
            base[-1] = with_message_block(conversation[base_message_count - 1], CONTEXT_CHECKPOINT_MARKER, text)
        elif base and base[0].get("role") == "system":
            original = str(conversation[0].get("content") or "").split(CONTEXT_CHECKPOINT_MARKER, 1)[0]
            base[0] = {**base[0], "content": f"{original}{CONTEXT_CHECKPOINT_MARKER}{text}"}

    attach(notes)
    room = target - serialized_chars(base)
    stubbed: list[tuple[str, dict[str, Any]]] = []
    exchanges = _exchanges(internal)

    # Pass 1: stub old tool results, oldest exchange first.
    for start, end in exchanges[: max(0, len(exchanges) - PROTECTED_RECENT_EXCHANGES)]:
        if serialized_chars(internal) <= room:
            break
        calls = _call_index(internal[start])
        for index in range(start + 1, end):
            message = internal[index]
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            if content.startswith(STUB_PREFIX) or len(content) < STUB_MIN_CHARS:
                continue
            name, arguments = calls.get(str(message.get("tool_call_id") or ""), ("", {}))
            internal[index] = {**message, "content": stub_for(name, arguments, content)}
            stubbed.append((name, arguments))

    # Pass 2: drop whole old exchanges, keeping the model's words as notes.
    dropped_rounds = 0
    while serialized_chars(internal) > room and len(_exchanges(internal)) > 1:
        start, end = _exchanges(internal)[0]
        for message in internal[start:end]:
            if message.get("role") == "assistant":
                text = " ".join(str(message.get("content") or "").split())
                if text:
                    notes.append(text[:PROGRESS_NOTE_CHARS])
        del internal[start:end]
        dropped_rounds += 1
    if dropped_rounds:
        attach(notes)

    conversation[:] = [*base, *internal]
    return {"stubbed": stubbed, "dropped_rounds": dropped_rounds, "request_chars": serialized_chars(conversation)}
