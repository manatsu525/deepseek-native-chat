"""Optional DSML tool-call fallback for OpenCode-hosted DeepSeek V4 Flash.

The adapter is deliberately narrow: it is active only for OpenCode's Zen host
and the explicit DeepSeek V4 Flash model IDs below.  Native ``tool_calls``
always win.  Set ``OPENCODE_DSML_FALLBACK=0`` to disable the module globally,
or disable ``dsml_fallback_enabled`` in the Custom provider settings.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit


_FALSE_VALUES = {"0", "false", "no", "off"}
_OPENCODE_HOST = "opencode.ai"
_SUPPORTED_MODELS = {"deepseek-v4-flash", "deepseek-v4-flash-free"}

# The official tokenizer uses the full-width U+FF5C separator.  Gateways have
# also been observed to normalize it to one or more ASCII pipes.
_SEP = r"[|｜]+"
_TOOL_CALLS_RE = re.compile(
    rf"<{_SEP}DSML{_SEP}tool_calls>(.*?)</{_SEP}DSML{_SEP}tool_calls>", re.S
)
_INVOKE_RE = re.compile(
    rf'<{_SEP}DSML{_SEP}invoke\s+name="([^"]+)"\s*>(.*?)</{_SEP}DSML{_SEP}invoke>', re.S
)
_PARAM_RE = re.compile(
    rf'<{_SEP}DSML{_SEP}parameter\s+name="([^"]+)"'
    rf'(?:\s+string="(true|false)")?\s*>(.*?)</{_SEP}DSML{_SEP}parameter>',
    re.S,
)
_ORPHAN_RE = re.compile(rf"<{_SEP}DSML{_SEP}tool_calls>.*\Z", re.S)
_STRAY_TAG_RE = re.compile(rf"</?{_SEP}?DSML{_SEP}?[^>]*>?")
_HAS_DSML_RE = re.compile(rf"<{_SEP}?DSML{_SEP}?")
_MAX_MARKER_LEN = 32


def _globally_enabled() -> bool:
    return os.getenv("OPENCODE_DSML_FALLBACK", "1").strip().casefold() not in _FALSE_VALUES


def applies_to(base_url: str, model: str, setting_enabled: bool = True) -> bool:
    """Return whether this exact request is allowed to use the fallback."""
    host = (urlsplit(str(base_url or "")).hostname or "").casefold()
    model_name = str(model or "").strip().casefold().rsplit("/", 1)[-1]
    return bool(setting_enabled) and _globally_enabled() and host == _OPENCODE_HOST and model_name in _SUPPORTED_MODELS


def looks_like_dsml(content: str) -> bool:
    """Quickly detect DSML markup in ordinary assistant content."""
    return bool(content) and bool(_HAS_DSML_RE.search(content))


def _remove_dsml(content: str) -> str:
    clean = _TOOL_CALLS_RE.sub("", content)
    clean = _ORPHAN_RE.sub("", clean)
    return _STRAY_TAG_RE.sub("", clean)


def parse_dsml(content: str, id_prefix: str = "call") -> tuple[str, list[dict[str, Any]]]:
    """Split assistant content into clean text and OpenAI-format tool calls."""
    content = content or ""
    tool_calls: list[dict[str, Any]] = []

    for block in _TOOL_CALLS_RE.findall(content):
        for name, body in _INVOKE_RE.findall(block):
            arguments: dict[str, Any] = {}
            for parameter_name, is_string, raw in _PARAM_RE.findall(body):
                value: Any = raw.strip()
                if is_string == "false":
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
                arguments[parameter_name] = value
            tool_calls.append(
                {
                    "id": f"{id_prefix}_{len(tool_calls)}_{name}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

    return _remove_dsml(content).strip(), tool_calls


def normalize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair OpenCode tool argument types that disagree with the schema."""
    for call in calls:
        function = call.get("function") or {}
        if not isinstance(function, dict) or function.get("name") != "web_search":
            continue
        try:
            arguments = json.loads(str(function.get("arguments") or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(arguments, dict) or not isinstance(arguments.get("search_queries"), str):
            continue

        raw_queries = arguments["search_queries"].strip()
        try:
            decoded_queries = json.loads(raw_queries)
        except (json.JSONDecodeError, TypeError, ValueError):
            decoded_queries = raw_queries
        if isinstance(decoded_queries, list):
            queries = [str(item).strip() for item in decoded_queries if str(item).strip()]
        else:
            query = str(decoded_queries).strip()
            queries = [query] if query else []
        arguments["search_queries"] = queries
        function["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return calls


def recover_tool_calls(
    content: str,
    native_calls: list[dict[str, Any]],
    *,
    id_prefix: str,
    tools_available: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Clean leaked DSML and recover calls only when native calls are absent."""
    if not looks_like_dsml(content):
        return content, normalize_tool_calls(native_calls)
    clean, parsed_calls = parse_dsml(content, id_prefix=id_prefix)
    if native_calls:
        return clean, normalize_tool_calls(native_calls)
    return clean, normalize_tool_calls(parsed_calls) if tools_available else []


class DsmlStreamBuffer:
    """Hold possible split DSML markers so markup never flashes in the UI."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_dsml = False

    def feed(self, delta: str) -> str:
        self._buffer += delta or ""
        if self._in_dsml or _HAS_DSML_RE.search(self._buffer):
            self._in_dsml = True
            return ""
        safe_length = max(0, len(self._buffer) - _MAX_MARKER_LEN)
        output, self._buffer = self._buffer[:safe_length], self._buffer[safe_length:]
        return output

    def flush(self) -> str:
        remainder = self._buffer
        self._buffer = ""
        self._in_dsml = False
        return _remove_dsml(remainder) if looks_like_dsml(remainder) else remainder
