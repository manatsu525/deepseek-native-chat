"""User-controlled JSON extensions for Custom API requests."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


MAX_REQUEST_OVERRIDES_BYTES = 64 * 1024

# These fields describe the request envelope that the server must construct
# itself.  Allowing them to be replaced would either discard the conversation
# or make the SSE/tool loop impossible to parse.
REQUEST_OVERRIDE_RESERVED_FIELDS = frozenset(
    {"model", "input", "messages", "stream", "tools", "tool_choice"}
)
REQUEST_OVERRIDE_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def validate_request_overrides(value: Any) -> dict[str, Any]:
    """Validate and copy one model's top-level request extension object."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("高级请求 JSON 必须是一个 JSON 对象")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("高级请求 JSON 只能包含标准 JSON 值") from exc
    if len(encoded.encode("utf-8")) > MAX_REQUEST_OVERRIDES_BYTES:
        raise ValueError(f"高级请求 JSON 不能超过 {MAX_REQUEST_OVERRIDES_BYTES // 1024}KB")
    reserved = sorted(str(key) for key in value if str(key) in REQUEST_OVERRIDE_RESERVED_FIELDS)
    if reserved:
        raise ValueError("高级请求 JSON 不能覆盖服务端控制字段：" + ", ".join(reserved))
    return dict(value)


def _expand(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    if isinstance(value, tuple):
        return [_expand(item, context) for item in value]
    if not isinstance(value, str):
        return value

    exact = REQUEST_OVERRIDE_PLACEHOLDER.fullmatch(value)
    if exact and exact.group(1) in context:
        return context[exact.group(1)]
    return REQUEST_OVERRIDE_PLACEHOLDER.sub(
        lambda match: str(context.get(match.group(1), match.group(0))),
        value,
    )


def apply_request_overrides(
    payload: dict[str, Any],
    overrides: Any,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Expand placeholders and merge user fields over generated payload.

    The merge is intentionally shallow at the request-root level: setting a
    provider-specific object such as ``provider`` or ``providerOptions``
    replaces that whole root field, which makes the JSON editor predictable.
    """
    validated = validate_request_overrides(overrides)
    if not validated:
        return
    payload.update(_expand(validated, context or {}))
