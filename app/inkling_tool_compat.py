"""Optional compatibility for Inkling's typed tool-call protocol.

Inkling may leak its native ``content_invoke_tool_json`` blocks through an
OpenAI-compatible gateway instead of returning ``tool_calls``.  It also
occasionally omits ``path`` from large patch calls.  For small workspaces we
therefore expose patch functions explicitly pre-bound to each existing file.
Set ``INKLING_TOOL_COMPAT=0`` to disable both behaviors.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from copy import deepcopy
from typing import Any

from .custom_tool_normalization import normalize_tool_calls


_FALSE_VALUES = {"0", "false", "no", "off"}
_MESSAGE = "<|message_model|>"
_INVOKE = "<|content_invoke_tool_json|>"
_END = "<|end_message|>"
_SAMPLING_END = "<|content_model_end_sampling|>"
_TEXT_BLOCK_RE = re.compile(r"<\|content_text\|>(.*?)<\|end_message\|>", re.S)
_CALL_BLOCK_RE = re.compile(
    r"<\|message_model\|>(?P<header>.*?)<\|content_invoke_tool_json\|>"
    r"(?P<payload>.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>)",
    re.S,
)
MAX_BOUND_FILES = 12


def applies_to(model: str) -> bool:
    enabled = os.getenv("INKLING_TOOL_COMPAT", "1").strip().casefold() not in _FALSE_VALUES
    return enabled and "inkling" in str(model or "").casefold()


def _clean_content(content: str) -> str:
    raw = str(content or "")
    prefix = raw.split(_MESSAGE, 1)[0]
    visible = _TEXT_BLOCK_RE.findall(raw)
    return "".join([prefix, *visible]).strip()


def parse_tool_markup(content: str, id_prefix: str = "inkling") -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in _CALL_BLOCK_RE.finditer(str(content or "")):
        try:
            payload = json.loads(match.group("payload").strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or match.group("header") or "").strip()
        arguments = payload.get("args")
        if not name or not isinstance(arguments, dict):
            continue
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
    if _MESSAGE not in str(content or "") and _INVOKE not in str(content or ""):
        return content, normalize_tool_calls(native_calls)
    clean, parsed = parse_tool_markup(content, id_prefix=id_prefix)
    if native_calls:
        return clean, normalize_tool_calls(native_calls)
    if parsed and tools_available:
        return clean, normalize_tool_calls(parsed)
    return clean if clean else "", []


def bind_patch_tools(
    tools: list[dict[str, Any]], paths: list[str]
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    """Replace generic patch schemas with path-free, explicitly bound tools."""
    if not paths or len(paths) > MAX_BOUND_FILES:
        return tools, {}
    result: list[dict[str, Any]] = []
    bindings: dict[str, tuple[str, str]] = {}
    for tool in tools:
        function = tool.get("function") or {}
        original_name = str(function.get("name") or "")
        if original_name not in {"apply_patch", "apply_patch_batch"}:
            result.append(tool)
            continue
        for path in paths:
            bound = deepcopy(tool)
            bound_function = bound["function"]
            path_key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
            bound_name = f"inkling_{original_name}_{path_key}"
            parameters = bound_function["parameters"]
            parameters["properties"].pop("path", None)
            parameters["required"] = [item for item in parameters.get("required") or [] if item != "path"]
            bound_function["name"] = bound_name
            bound_function["description"] = (
                f"{bound_function.get('description', '')} This function is already bound to the exact file "
                f"{path!r}; do not provide a path argument."
            )
            result.append(bound)
            bindings[bound_name] = (original_name, path)
    return result, bindings


class InklingStreamBuffer:
    """Hide split private Inkling blocks until they can be parsed after streaming."""

    def __init__(self) -> None:
        self._buffer = ""
        self._private = False

    def feed(self, delta: str) -> str:
        self._buffer += delta or ""
        if self._private or _MESSAGE in self._buffer:
            if _MESSAGE in self._buffer:
                prefix, self._buffer = self._buffer.split(_MESSAGE, 1)
                self._buffer = _MESSAGE + self._buffer
                self._private = True
                return prefix
            return ""
        safe_length = max(0, len(self._buffer) - len(_MESSAGE))
        output, self._buffer = self._buffer[:safe_length], self._buffer[safe_length:]
        return output

    def flush(self) -> str:
        remainder = "" if self._private else self._buffer
        self._buffer = ""
        self._private = False
        return remainder
