"""Optional parser for MiniMax's leaked private tool-call markup.

Some OpenAI-compatible MiniMax gateways occasionally return a tool request in
``]<]minimax[>[...`` markup inside ``content`` instead of standard
``tool_calls``.  Native calls always win.  Set ``MINIMAX_TOOL_FALLBACK=0`` to
disable this adapter without changing the main request loop.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .custom_tool_normalization import normalize_tool_calls


_FALSE_VALUES = {"0", "false", "no", "off"}
_MARKER_TEXT = "]<]minimax[>["
_MARKER_RE = re.compile(re.escape(_MARKER_TEXT), re.I)
_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.I | re.S)
_INVOKE_RE = re.compile(r'<invoke\s+name="([^"<>]+)"\s*>(.*?)</invoke>', re.I | re.S)
_MAX_MARKER_LEN = len(_MARKER_TEXT)


def applies_to(model: str) -> bool:
    enabled = os.getenv("MINIMAX_TOOL_FALLBACK", "1").strip().casefold() not in _FALSE_VALUES
    return enabled and "minimax" in str(model or "").casefold()


def looks_like_minimax_markup(content: str) -> bool:
    return bool(content) and bool(_MARKER_RE.search(content))


def _normalized_markup(content: str) -> str:
    return _MARKER_RE.sub("", content or "")


def _clean_content(content: str) -> str:
    match = _MARKER_RE.search(content or "")
    return (content[: match.start()] if match else content).rstrip()


def _parameter(body: str, name: str) -> str | None:
    match = re.search(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", body, re.I | re.S)
    return match.group(1) if match else None


def parse_minimax_markup(content: str, id_prefix: str = "minimax") -> tuple[str, list[dict[str, Any]]]:
    """Convert leaked MiniMax markup into standard OpenAI tool calls."""
    normalized = _normalized_markup(content)
    calls: list[dict[str, Any]] = []
    for block in _TOOL_BLOCK_RE.findall(normalized):
        for name, body in _INVOKE_RE.findall(block):
            arguments: dict[str, Any] = {}
            for parameter in ("path", "content", "old_text", "new_text", "query"):
                value = _parameter(body, parameter)
                if value is not None:
                    arguments[parameter] = value
            replace_all = _parameter(body, "replace_all")
            if replace_all is not None:
                arguments["replace_all"] = replace_all.strip().casefold() in {"1", "true", "yes"}
            calls.append(
                {
                    "id": f"{id_prefix}_{len(calls)}_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            )
    return _clean_content(content), calls


def recover_tool_calls(
    content: str,
    native_calls: list[dict[str, Any]],
    *,
    id_prefix: str,
    tools_available: bool,
) -> tuple[str, list[dict[str, Any]]]:
    if not looks_like_minimax_markup(content):
        return content, normalize_tool_calls(native_calls)
    clean, parsed = parse_minimax_markup(content, id_prefix=id_prefix)
    if native_calls:
        return clean, normalize_tool_calls(native_calls)
    if tools_available:
        return clean, normalize_tool_calls(parsed)
    # A prose preamble such as "let me patch it" is not a usable final answer.
    # Empty content makes the main loop request a real answer with tools removed.
    return "", []


class MiniMaxStreamBuffer:
    """Prevent split MiniMax private markers from flashing in streamed text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_markup = False

    def feed(self, delta: str) -> str:
        self._buffer += delta or ""
        if self._in_markup or _MARKER_RE.search(self._buffer):
            match = _MARKER_RE.search(self._buffer)
            self._in_markup = True
            if match:
                output = self._buffer[: match.start()]
                self._buffer = self._buffer[match.start() :]
                return output
            return ""
        safe_length = max(0, len(self._buffer) - _MAX_MARKER_LEN)
        output, self._buffer = self._buffer[:safe_length], self._buffer[safe_length:]
        return output

    def flush(self) -> str:
        remainder = "" if self._in_markup else self._buffer
        self._buffer = ""
        self._in_markup = False
        return remainder
