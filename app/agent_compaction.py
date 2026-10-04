"""Grok Build session compaction, adapted to our conversation/journal types.

Upstream: xai-org/grok-build @ 2bdd1d6 (Apache-2.0).
See third_party/grok-build/NOTICE and docs/agent-context.md for source mappings.
This module is Agent-only; the ordinary chat character-budget policy is separate.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

DEFAULT_WINDOW = 256_000
THRESHOLD_PERCENT = 85
PREFIRE_PERCENT = 75
SUMMARY_RESERVE = 32_768
MAX_ATTEMPTS = 3
RETRY_DELAY = 3
WALL_CLOCK_SECONDS = 300
MIN_SUMMARY_CHARS = 500
SEGMENT_MAX_BYTES = 5 * 1024 * 1024
HARD_CLEAR = "[Tool result omitted — too old]"
CONTINUATION = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation."
)
AUTO_CONTINUE = (
    'Continue the conversation from where it left off without asking the user any further questions. '
    'Resume directly - do not acknowledge the summary, do not recap what was happening, do not preface '
    'with "I\'ll continue" or similar.\n'
    'Pick up the last task as if the break never happened.'
)
INDEX_HEADER = ("# Compaction Segment Index\n\n"
                "| Segment | File | Turns | Approx bytes | Keywords |\n"
                "|---|---|---|---|---|\n")
SUMMARY_PROMPT = (Path(__file__).parent / "templates/grok_compaction_prompt.txt").read_text(encoding="utf-8").replace(
    "{user_context_section}", "")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def text_content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text") or "") for part in content
                     if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"})


def estimate_item(message: dict[str, Any]) -> int:
    """Upstream bytes/4, 765/image; native item mirrors never counted twice."""
    content = message.get("content") or ""
    if isinstance(content, list):
        raw_bytes = sum(len(str(p.get("text") or "").encode()) for p in content if isinstance(p, dict))
        images = sum(1 for p in content if isinstance(p, dict) and p.get("type") in {"image_url", "image", "input_image"})
    else:
        raw_bytes, images = len(str(content).encode()), 0
    if message.get("role") == "assistant":
        raw_bytes += sum(len(str((c.get("function") or {}).get("arguments") or "").encode())
                         for c in message.get("tool_calls") or [])
    tokens = raw_bytes // 4 + images * 765
    reasoning = str(message.get("reasoning_content") or "")
    encrypted_bytes = 0
    for item in message.get("responses_output_items") or []:
        if item.get("type") == "reasoning":
            encrypted_bytes += len(str(item.get("encrypted_content") or "").encode())
            if not reasoning:
                reasoning = "\n".join(str(p.get("text") or "") for p in item.get("summary") or [])
    if not reasoning:
        reasoning = "\n".join(str(b.get("thinking") or "") for b in message.get("anthropic_thinking_blocks") or [])
    return tokens + max(len(reasoning.encode()), encrypted_bytes * 3 // 4) // 4


def estimate_history(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_item(m) for m in messages)


def estimate_tools(tools: list[dict[str, Any]]) -> int:
    result = 0
    for item in tools:
        f = item.get("function") or item
        result += (len(str(f.get("name") or "").encode()) + len(str(f.get("description") or "").encode())
                   + len(json_text(f.get("parameters") or f.get("input_schema") or {}).encode())) // 4
    return result


class TokenMeter:
    """Last provider total + appended item estimates, not lifetime billed tokens."""
    def __init__(self, state: dict[str, Any] | None = None):
        state = state or {}
        self.total = int(state.get("total") or 0)
        self.baseline = int(state.get("baseline") or 0)

    def used(self, messages: list[dict[str, Any]]) -> int:
        current = estimate_history(messages)
        return self.total + max(0, current - self.baseline) if self.total else current

    def observe(self, messages: list[dict[str, Any]], usage: dict[str, Any]) -> None:
        total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        self.baseline = estimate_history(messages)
        self.total = total or self.baseline

    def reseed(self, messages: list[dict[str, Any]]) -> None:
        baseline = estimate_history(messages)
        scaled = math.floor(baseline * self.total / self.baseline + .5) if self.total and self.baseline else baseline
        self.total = min(scaled, self.total) if self.total else scaled
        self.baseline = baseline

    def export(self) -> dict[str, int]:
        return {"total": self.total, "baseline": self.baseline}


def clean_summary(raw: str) -> str:
    result = raw
    while "<analysis>" in result:
        start = result.index("<analysis>")
        sp = result.find("<summary>")
        leading = (start < sp or not result[sp + 9:start].strip()) if sp >= 0 else not result[:start].strip()
        if not leading:
            break
        end = result.find("</analysis>", start)
        if end >= 0:
            result = result[:start] + result[end + 11:]
        else:
            next_summary = result.find("<summary>", start)
            result = result[:start] + (result[next_summary:] if next_summary >= 0 else "")
            break
    start, end = result.find("<summary>"), result.rfind("</summary>")
    if start >= 0 and end > start:
        inner = result[start + 9:end].strip()
        lead = inner.lstrip("#*-> \t")
        if not (lead and lead[0].isascii() and lead[0].isdigit()) and "</analysis>" in inner:
            inner = inner[inner.rfind("</analysis>") + 11:].lstrip()
        if inner.startswith("<summary>"):
            inner = inner[9:].lstrip()
        result = result[:start] + "Summary:\n" + inner + result[end + 10:]
    for tag in ("</summary>", "<summary>", "</analysis>", "<analysis>", "</summary_request>", "<summary_request>"):
        result = result.replace(tag, "<\u200b" + tag[1:])
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def prepared(messages: list[dict[str, Any]], *, lossy: bool = False, strip_reasoning: bool = False,
             strip_images: bool = False) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    if lossy:
        result = [m for m in result if m.get("role") != "tool"]
    for message in result:
        if lossy and message.get("tool_calls"):
            names = [str(c["function"]["name"]) for c in message.pop("tool_calls")]
            message["content"] = text_content(message) + "\n[Called tools: " + ", ".join(names) + "]"
        if lossy or strip_reasoning or strip_images:
            # Native output mirrors contain reasoning and unstripped images. Use
            # canonical text/tool messages for lossy and archived projections.
            message.pop("responses_output_items", None)
        if lossy or strip_reasoning or strip_images:
            message.pop("reasoning_content", None)
            message.pop("anthropic_thinking_blocks", None)
        if (lossy or strip_images) and isinstance(message.get("content"), list):
            message["content"] = [({"type": "text", "text": "[image]"} if p.get("type") in
                                   {"image", "image_url", "input_image"} else p) for p in message["content"]]
    while result and result[-1].get("role") == "assistant" and result[-1].get("tool_calls"):
        result.pop()
    return result


def prune_history(messages: list[dict[str, Any]], *, retained: bool = False) -> list[dict[str, Any]]:
    """Upstream 3 recent turns / 4000→1500+1500 / 10-turn hard clear.

    Retained pruning raises the threshold by the number of synthetic user
    items, just as Grok's prompt_index accounting does. Request pruning counts
    all User items and never changes the canonical/durable transcript.
    """
    result = copy.deepcopy(messages)
    synthetic = sum(1 for m in result if m.get("role") == "user" and m.get("agent_synthetic"))
    hard_age = 10 + synthetic if retained else 10
    age, seen = 0, False
    for m in reversed(result):
        if m.get("role") == "user":
            age += int(seen)
            seen = True
        elif m.get("role") == "tool" and age >= 3:
            content = text_content(m)
            if age >= hard_age:
                m["content"] = HARD_CLEAR
            elif not retained and len(content) > 4000:
                m["content"] = content[:1500] + "\n\n[…trimmed…]\n\n" + content[-1500:]
    return result


def image_budget(messages: list[dict[str, Any]], protocol: str, tool_tokens: int = 0) -> list[dict[str, Any]]:
    """Upstream high-water request size / reclaim-to-half image eviction."""
    cap = 30_000_000 if protocol == "messages" else 50 * 1024 * 1024
    body_bytes = len(json_text(messages).encode())
    if body_bytes < max(0, cap - 3 * 1024 * 1024 - tool_tokens * 4):
        return messages
    target = max(0, cap // 2 - tool_tokens * 4)
    result = copy.deepcopy(messages)
    placeholder = ("[An earlier image was removed to keep the request within its size limit and is no longer visible. "
                   "Do not describe or reason about its contents from memory; ask the user to re-share it if you need to see it again.]")
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for i, part in enumerate(message["content"]):
            if body_bytes <= target:
                return result
            if part.get("type") in {"image", "image_url", "input_image"}:
                replacement = {"type": "text", "text": placeholder}
                body_bytes -= len(json_text(part).encode()) - len(json_text(replacement).encode())
                message["content"][i] = replacement
    return result


def truncate_text(text: str, tokens: int) -> str:
    raw = text.encode()
    if len(raw) <= tokens * 4:
        return text
    prefix = raw[:max(0, tokens * 4 - 64)].decode(errors="ignore")
    return prefix + f"\n[... truncated {len(raw) - len(prefix.encode())} bytes to fit the compaction window ...]"


def truncate_item(message: dict[str, Any], tokens: int) -> dict[str, Any]:
    message = copy.deepcopy(message)
    if isinstance(message.get("content"), str):
        message["content"] = truncate_text(message["content"], tokens)
    elif isinstance(message.get("content"), list):
        for part in message["content"]:
            if isinstance(part.get("text"), str):
                part["text"] = truncate_text(part["text"], tokens)
    # Do not replay a full native text block after truncating its canonical copy.
    message.pop("responses_output_items", None)
    return message


def fit_history(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Newest suffix; owning assistant and contiguous tool results stay together."""
    if estimate_history(messages) <= budget:
        return copy.deepcopy(messages)
    body = copy.deepcopy(messages)
    head = [body.pop(0)] if body and body[0].get("role") == "system" else []
    remaining = max(0, budget - estimate_history(head))
    body_budget = remaining
    start = len(body)
    for i in range(len(body) - 1, -1, -1):
        cost = estimate_item(body[i])
        if cost > remaining:
            break
        remaining -= cost
        start = i
    while start < len(body) and body[start].get("role") == "tool":
        start += 1
    if start < len(body):
        return head + body[start:]
    results = []
    while body and body[-1].get("role") == "tool":
        results.insert(0, body.pop())
    if not results:
        return head + ([truncate_item(body[-1], body_budget)] if body else [])
    owner = body[-1] if body and body[-1].get("role") == "assistant" and body[-1].get("tool_calls") else None
    per = max(1, max(0, body_budget - (estimate_item(owner) if owner else 0)) // len(results))
    return head + ([owner] if owner else []) + [truncate_item(r, per) for r in results]


def split_two_pass(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    target = max(1, estimate_history(messages)) * .95
    accumulated, split = 0, max(1, len(messages) - 1)
    for i, message in enumerate(messages):
        accumulated += estimate_item(message)
        if accumulated >= target:
            split = max(1, i + 1)
            break
    split = min(split, len(messages) - 1) if len(messages) > 1 else split
    while split < len(messages) and messages[split].get("role") == "tool":
        split += 1
    if split < len(messages) and messages[split].get("tool_calls"):
        split += 1
        while split < len(messages) and messages[split].get("role") == "tool":
            split += 1
    if split >= len(messages) and len(messages) > 1:
        split = len(messages) - 1
        while split > 1 and messages[split].get("role") == "tool":
            split -= 1
    return min(split, len(messages))


def note_for_pass2(raw: str) -> str:
    blocks = re.findall(r"<summary>(.*?)</summary>", raw, re.I | re.S)
    note = next((b.strip() for b in reversed(blocks) if len(b.strip()) > 1000), raw.strip())
    return note[:60_000] + "\n\n[… NOTE₁ truncated for pass2 input budget …]" if len(note) > 60_000 else note


def fingerprint(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json_text(messages).encode()).hexdigest()


def keywords(summary: str) -> list[str]:
    start = re.search(r"(?m)^#{0,6}\s*8\.\s+Current Work", summary)
    section = summary[start.start():] if start else summary
    if start:
        end = re.search(r"(?m)^#{0,6}\s*\d+\.\s+[A-Z]", section[start.end() - start.start():])
        if end:
            section = section[:start.end() - start.start() + end.start()]
    stop = set("section summary current work errors analysis primary request intent technical concepts pending problem solving include outline describe specific messages feedback snippet snippets session explicit thorough language important convention".split())
    return list(dict.fromkeys(w for w in re.findall(r"[A-Z][A-Za-z0-9_]{3,}|[a-z][a-z0-9_]{5,}", section)
                             if w.lower() not in stop))[:8]


def render_segment(messages: list[dict[str, Any]], summary: str, index: int, timestamp: str) -> str:
    labels = {"system": "System", "user": "Human", "assistant": "Assistant", "tool": "Function"}
    roles, tools, files = Counter(), Counter(), set()
    errors, approx, last, blocks = 0, 0, "", []
    for i, message in enumerate(prepared(messages, strip_images=True)):
        label = labels.get(message.get("role"), "Human")
        roles[label] += 1
        content = text_content(message)
        approx += 64 + len(content.encode())
        parts = [f"### Turn {i} ({label})"]
        if message.get("role") == "tool":
            parts.append("[tool_response]")
            errors += int(content.startswith("Error") or "Failed tool validation" in content)
        if content:
            parts.append(content)
            if label == "Assistant":
                last = content
        for call in message.get("tool_calls") or []:
            f = call.get("function") or {}
            name = str(f.get("name") or "")
            tools[name] += 1
            parts.append(f"[tool_request: {name}]")
            try:
                args = json.loads(f.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            for key in ("target_file", "file_path", "path", "target_directory"):
                if isinstance(args.get(key), str) and args[key]:
                    files.add(args[key])
                    break
            for key, value in sorted(args.items()):
                value_text = value if isinstance(value, str) else json_text(value)
                parts.append(f"- {key}: {value_text}")
                approx += 32 + len(key.encode()) + len(value_text.encode())
        blocks.append("\n".join(parts) + "\n")
    file_list = sorted(files)
    file_text = (", ".join(file_list) if len(file_list) <= 8 else
                 ", ".join(file_list[:5]) + f", ... and {len(file_list) - 5} more") or "(none)"
    tool_text = ", ".join(f"{n} ({c})" for n, c in sorted(tools.items(), key=lambda p: (-p[1], p[0]))) or "(none)"
    stats = (f"## Turn statistics\n\n- Turns: {sum(roles.values())} (" +
             ", ".join(f"{k}={v}" for k, v in sorted(roles.items())) + ")\n" +
             f"- Tools used: {tool_text}\n- Unique target files ({len(files)}): {file_text}\n" +
             f"- Tool errors: {errors}\n- Verbose-render size estimate: {approx:,} B\n")
    if last.strip():
        excerpt = last[-500:].strip().replace("\n", " ")[:300]
        stats += f'- Last assistant response excerpt: "{excerpt}"\n'
    preamble = (f"# HISTORICAL -- DO NOT EDIT\n# Record of compaction segment {index:03} (detail=verbose) from this same task.\n"
                "# Use read_file or grep to look up details, but do not modify.\n\n"
                f"## Segment metadata\n- Index: {index:03}\n- Turn count: {sum(roles.values())}\n- Timestamp: {timestamp}\n\n"
                + stats + "\n\n## Summary (curated by compaction step)\n\n" + (summary.strip() or "(empty)")
                + "\n\n## Verbatim turns\n\n")
    notice = "\n\n[... TRUNCATED at {limit} bytes, {omitted} turns omitted ...]\n"
    budget = max(0, SEGMENT_MAX_BYTES - len(preamble.encode()) - len(notice.encode()) - 64)
    kept, used = [], 0
    for block in blocks:
        if used + len(block.encode()) > budget:
            break
        used += len(block.encode())
        kept.append(block)
    body = "\n".join(kept)
    if len(kept) < len(blocks):
        body += notice.format(limit=SEGMENT_MAX_BYTES, omitted=len(blocks) - len(kept))
    return preamble + body


class SegmentStore:
    def __init__(self, root: Path, conversation_id: str):
        # User-controlled IDs never become filesystem paths. No shared /home/share.
        self.directory = root / "agent_sessions" / hashlib.sha256(conversation_id.encode()).hexdigest() / "compaction"

    def save(self, messages: list[dict[str, Any]], summary: str) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory.parent, 0o700)
        os.chmod(self.directory, 0o700)
        index = max((int(p.stem.split("_")[1]) for p in self.directory.glob("segment_*.md")
                     if p.stem.split("_")[1].isdigit()), default=-1) + 1
        rendered = render_segment(messages, summary, index, datetime.now(timezone.utc).isoformat())
        segment = self.directory / f"segment_{index:03}.md"
        # Exclusive creation prevents silently overwriting an existing archive.
        fd = os.open(segment, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        index_file = self.directory / "INDEX.md"
        empty = not index_file.exists() or index_file.stat().st_size == 0
        fd = os.open(index_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if empty:
                stream.write(INDEX_HEADER)
            kw = ", ".join('"' + k + '"' for k in keywords(summary))
            stream.write(f"| {index:03} | {segment.name} | {len(messages)} | {len(rendered.encode())} | {kw} |\n")
            stream.flush()
            os.fsync(stream.fileno())
        return {"index": index, "path": str(segment), "directory": str(self.directory)}


def delete_session_archive(root: Path, conversation_id: str) -> None:
    directory = root / "agent_sessions" / hashlib.sha256(conversation_id.encode()).hexdigest()
    if directory.exists():
        shutil.rmtree(directory)


def rebuilt_history(messages: list[dict[str, Any]], summary: str, directory: str,
                    state: dict[str, Any]) -> list[dict[str, Any]]:
    result = [copy.deepcopy(messages[0])]
    latest = next((m for m in reversed(messages) if m.get("role") == "user" and not m.get("agent_synthetic")
                   and not text_content(m).startswith(CONTINUATION) and text_content(m) != AUTO_CONTINUE), None)
    if latest:
        query = copy.deepcopy(latest)
        original = text_content(query)
        for tag in ("user_info", "project_layout", "git_status", "fork-context", "system-reminder", "agent-memory",
                    "system_reminder", "background_context", "command-name", "command-message", "command-args", "rules"):
            original = re.sub("<" + tag + r">.*?</" + tag + ">", "", original, flags=re.S)
        wrapped = re.search(r"<user_query>(.*?)</user_query>", original, re.S)
        original = wrapped.group(1).strip() if wrapped else original.strip()
        parts = query.get("content")
        query["content"] = "<user_query>\n" + original + "\n</user_query>"
        if isinstance(parts, list):
            query["content"] = [{"type": "text", "text": query["content"]}] + [
                p for p in parts if p.get("type") in {"image", "image_url", "input_image"}]
        result.append(query)
    hint = (f"\n\nFull verbatim rollouts of previous segments are available at {directory}/segment_*.md.  "
            f"See {directory}/INDEX.md for a table of contents.  Use read_file or grep to recover specific "
            "details (exact code, file paths, tool outputs) if this summary is insufficient.  Do NOT modify these files.")
    result.append({"role": "user", "content": CONTINUATION + "\n\n" + clean_summary(summary) + hint,
                   "agent_synthetic": "compaction_summary"})
    if state:
        result.append({"role": "user", "content": "<system-reminder>\n" + json_text(state) + "\n</system-reminder>",
                       "agent_synthetic": "state_reminder"})
    result.append({"role": "user", "content": AUTO_CONTINUE, "agent_synthetic": "auto_continue"})
    return result


def overflow_error(status: int, text: str) -> bool:
    if status == 429:
        return False
    m = text.lower()
    phrases = ("too long for this model", "prompt is too long", "maximum prompt length", "maximum context length",
               "maximum allowed number of bytes", "413 payload too large", "413 content too large",
               "413 request entity too large")
    slugs = ("request too large", "context_length_exceeded", "exceed_context_size_error", "payload_too_large", "request_too_large")
    return (status == 413 or any(p in m for p in phrases) or
            ("current message" in m and "exceeds budget" in m) or
            ("input length" in m and "exceeds the maximum allowed length" in m) or
            any(segment.startswith(slug) for segment in m.split(": ") for slug in slugs))


class CompactError(RuntimeError):
    def __init__(self, message: str, status: int = 0, *, deterministic: bool = False):
        super().__init__(message)
        self.status = status
        self.overflow = overflow_error(status, message)
        self.deterministic = deterministic or (400 <= status < 500 and status not in {408, 429})


def suppress_reason(error: CompactError) -> str:
    message = f"status {error.status}: {error}".lower()
    if any(word in message for word in ("spending-limit", "spending limit", "out of credits",
                                       "usage balance exhausted", "usage limit reached")):
        return "credit"
    if error.overflow:
        return "size"
    if "status 401" in message or "unauthorized" in message:
        return "auth"
    if "invalid_request_error" in message:
        return "schema"
    return "turn"


Sample = Callable[[list[dict[str, Any]], list[dict[str, Any]], str], Awaitable[str]]


class AgentCompactor:
    def __init__(self, store: SegmentStore, window: int | None, protocol: str, meter: TokenMeter,
                 record: Callable[[str, dict[str, Any]], None], stopped: Callable[[], bool],
                 state: dict[str, Any] | None = None):
        self.store, self.window = store, window or DEFAULT_WINDOW
        self.protocol, self.meter, self.record, self.stopped = protocol, meter, record, stopped
        self.suppressed = str((state or {}).get("suppressed") or "")
        # Turn suppression expires on a new user turn; schema/size are scoped
        # to route/config/window by the journal loader.
        if self.suppressed == "turn":
            self.suppressed = ""
        self.prefire: asyncio.Task | None = None
        self.cache: dict[str, Any] | None = copy.deepcopy((state or {}).get("prefire_cache"))
        self.runtime_state: dict[str, Any] = {}

    def export(self) -> dict[str, Any]:
        return {**self.runtime_state, "meter": self.meter.export(), "suppressed": self.suppressed,
                "window": self.window, "prefire_cache": self.cache}

    async def __aenter__(self) -> AgentCompactor:
        return self

    async def __aexit__(self, exc_type: Any, *args: Any) -> None:
        await self.close(cancel=exc_type is not None)
        self.record("context/state", self.export())

    def success(self) -> None:
        if self.suppressed == "credit":
            self.suppressed = ""

    async def close(self, *, cancel: bool = True) -> None:
        if self.prefire:
            if cancel and not self.prefire.done():
                self.prefire.cancel()
            await asyncio.gather(self.prefire, return_exceptions=True)
            self.prefire = None

    def maybe_prefire(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: Sample) -> None:
        if self.cache and fingerprint(messages[:self.cache["length"]]) != self.cache["fingerprint"]:
            self.cache = None
        if self.suppressed or self.prefire is not None or self.cache or len(messages) < 4:
            return
        if self.meter.used(messages) * 100 < self.window * PREFIRE_PERCENT:
            return
        split = split_two_pass(messages)
        if not 0 < split < len(messages):
            return
        snapshot = copy.deepcopy(messages[:split])
        signature = fingerprint(snapshot)
        async def prefire() -> None:
            self.record("context/prefire_start", {"prefix_messages": split})
            try:
                history = prepared(snapshot, strip_reasoning=self.protocol == "messages")
                history = prune_history(image_budget(history, self.protocol))
                raw = await sample(history + [{"role": "user", "content": SUMMARY_PROMPT}], tools, "prefire")
                note = note_for_pass2(raw)
                if note.strip():
                    self.cache = {"note": note, "length": split, "fingerprint": signature}
                self.record("context/prefire_end", {"cached": bool(self.cache)})
            except (CompactError, TimeoutError) as error:
                self.record("context/prefire_failed", {"error": str(error)})
        self.prefire = asyncio.create_task(prefire())

    async def compact(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: Sample,
                      runtime_state: dict[str, Any], *, force: bool = False) -> list[dict[str, Any]] | None:
        if self.suppressed or (not force and self.meter.used(messages) * 100 < self.window * THRESHOLD_PERCENT):
            return None
        before = self.meter.used(messages)
        self.record("context/compaction_start", {"tokens": before, "window": self.window, "threshold_percent": 85})
        if self.prefire:
            await self.prefire
            self.prefire = None
        raw = None
        cache, self.cache = self.cache, None
        if cache and fingerprint(messages[:cache["length"]]) == cache["fingerprint"]:
            prefix = [m for m in messages[:cache["length"]] if m.get("role") == "system"]
            history = prefix + [{"role": "user", "content": CONTINUATION + "\n\n" + clean_summary(cache["note"])}]
            history += prepared(messages[cache["length"]:], strip_reasoning=self.protocol == "messages")
            try:
                candidate = await sample(history + [{"role": "user", "content": SUMMARY_PROMPT}], tools, "pass2")
                if len(clean_summary(candidate)) >= MIN_SUMMARY_CHARS:
                    raw = candidate
            except (CompactError, TimeoutError) as error:
                self.record("context/pass2_failed", {"error": str(error)})
        if raw is None:
            raw = await self._single_pass(messages, tools, sample)
        if raw is None:
            return None
        if self.stopped():
            raise asyncio.CancelledError
        # Disk must succeed before replacing the active/persisted projection.
        archive = await asyncio.to_thread(self.store.save, messages, raw)
        result = rebuilt_history(messages, raw, archive["directory"], runtime_state)
        self.meter.reseed(result)
        if self.meter.used(result) * 100 >= self.window * THRESHOLD_PERCENT:
            self.suppressed = "size"
        self.record("context/compaction_end", {"tokens_before": before, "tokens_after": self.meter.used(result),
                                               "archive": archive, "summary_chars": len(clean_summary(raw))})
        return result

    async def _single_pass(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: Sample) -> str | None:
        verbatim = prepared(messages, strip_reasoning=self.protocol == "messages")
        tool_tokens = estimate_tools(tools)
        verbatim = image_budget(verbatim, self.protocol, tool_tokens)
        if self.meter.total > self.window // 2:
            verbatim = prune_history(verbatim)
        fitted_budget = max(0, self.window - SUMMARY_RESERVE - tool_tokens)
        stages = ["verbatim", "verbatim_fitted", "lossy"]
        if estimate_history(verbatim) > fitted_budget:
            stages.pop(0)
        for stage in stages:
            history = verbatim if stage == "verbatim" else fit_history(
                prepared(messages, lossy=True) if stage == "lossy" else verbatim,
                max(0, self.window * 7 // 10 - tool_tokens) if stage == "lossy" else fitted_budget)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if self.stopped():
                    raise asyncio.CancelledError
                self.record("context/compaction_attempt", {"stage": stage, "attempt": attempt})
                try:
                    raw = await sample(history + [{"role": "user", "content": SUMMARY_PROMPT}], tools, stage)
                    if len(clean_summary(raw)) >= MIN_SUMMARY_CHARS:
                        return raw
                    raise CompactError("empty/degenerate summary")
                except CompactError as error:
                    self.record("context/compaction_error", {"stage": stage, "attempt": attempt,
                                                              "status": error.status, "error": str(error)})
                    if error.overflow:
                        if stage != "lossy":
                            break
                        self.suppressed = "size"
                    elif error.deterministic:
                        self.suppressed = suppress_reason(error)
                    elif attempt == MAX_ATTEMPTS:
                        self.suppressed = "turn"
                    else:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    self.record("context/compaction_failed", {"suppressed": self.suppressed, "history_unchanged": True})
                    if self.suppressed == "auth":
                        raise error
                    return None
        return None
