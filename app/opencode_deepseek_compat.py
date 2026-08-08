"""Narrow compatibility shim for OpenCode-hosted DeepSeek V4 models.

Some OpenCode DeepSeek V4 streams expose the model's native DSML tool markup
through ``delta.content`` while also returning a structured tool call.  Some
streams also serialize Parallel Search's ``search_queries`` array as a single
string.  This adapter fixes only those provider/model-specific wire quirks.

Removal is intentionally simple: delete this file and the marked hooks in
``mimo_local.py``.  At runtime, set ``OPENCODE_DEEPSEEK_COMPAT=0`` to disable
the adapter without changing or restarting from different code.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlsplit


_FALSE_VALUES = {"0", "false", "no", "off"}
_DSML_MARKERS = ("<｜DSML｜", "｜DSML｜>", "<|DSML|", "|DSML|>")


def _is_enabled() -> bool:
    return os.getenv("OPENCODE_DEEPSEEK_COMPAT", "1").strip().casefold() not in _FALSE_VALUES


class OpenCodeDeepSeekCompat:
    """Request-scoped adapter; inactive instances are pass-through."""

    def __init__(self, active: bool) -> None:
        self.active = active

    @classmethod
    def for_request(cls, base_url: str, model: str) -> "OpenCodeDeepSeekCompat":
        host = (urlsplit(str(base_url or "")).hostname or "").casefold()
        model_name = str(model or "").casefold().rsplit("/", 1)[-1]
        host_matches = host == "opencode.ai" or host.endswith(".opencode.ai")
        model_matches = model_name.startswith("deepseek-v4-")
        return cls(_is_enabled() and host_matches and model_matches)

    def preview_content(self, content: str, tools_offered: bool) -> str:
        """Buffer tool-enabled rounds so transient DSML never reaches the UI."""
        if self.active and tools_offered:
            return ""
        if self.active:
            # A broken provider can still print a forbidden tool request during
            # an answer-only retry. Hide both complete markers and an incomplete
            # marker suffix until enough streaming bytes arrive to classify it.
            marker_at = min(
                (content.find(marker) for marker in _DSML_MARKERS if marker in content),
                default=-1,
            )
            if marker_at >= 0:
                return content[:marker_at]
            for marker in _DSML_MARKERS:
                for suffix_size in range(1, min(len(marker), len(content) + 1)):
                    if content.endswith(marker[:suffix_size]):
                        return content[:-suffix_size]
        return content

    def contains_dsml(self, content: str) -> bool:
        return self.active and any(marker in str(content or "") for marker in _DSML_MARKERS)

    def final_content(self, content: str, calls: list[dict[str, Any]]) -> str:
        """Discard leaked DSML when the equivalent structured call exists."""
        if calls and self.contains_dsml(content):
            return ""
        return content

    def normalize_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Repair only the observed OpenCode web_search argument shape."""
        if not self.active:
            return calls
        for call in calls:
            function = call.get("function")
            if not isinstance(function, dict) or function.get("name") != "web_search":
                continue
            raw_arguments = function.get("arguments")
            try:
                arguments = json.loads(raw_arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(arguments, dict) or not isinstance(arguments.get("search_queries"), str):
                continue
            query = " ".join(arguments["search_queries"].split())
            arguments["search_queries"] = [query] if query else []
            function["arguments"] = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return calls
