"""Project native Responses items from the canonical executable tool calls."""
from __future__ import annotations

from typing import Any


def project_response_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = {str(call.get("id") or ""): call.get("function") or {}
             for call in message.get("tool_calls") or []}
    items = []
    for raw in message.get("responses_output_items") or []:
        if not isinstance(raw, dict) or not raw.get("type"):
            continue
        item = dict(raw)
        function = calls.get(str(item.get("call_id") or item.get("id") or ""))
        if item["type"] == "function_call" and function is not None:
            # Reassembled or compacted execution arguments take precedence.
            # IDs, statuses and opaque reasoning/signatures remain unchanged.
            item["arguments"] = str(function.get("arguments") or "{}")
        items.append(item)
    return items
