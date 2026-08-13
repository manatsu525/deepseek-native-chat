"""Normalize common tool-call argument mismatches from Custom providers."""

from __future__ import annotations

import json
from typing import Any


def normalize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair tool argument types that disagree with the advertised schema."""
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
