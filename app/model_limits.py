"""What the provider says about a model: for now, its context window.

OpenAI-compatible gateways (Vercel AI Gateway, OpenRouter, many self-hosted
servers) list models under /models with a context size field. The lookup is
best-effort: any failure means "unknown" and the caller uses a safe default.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any

LOOKUP_TIMEOUT_SECONDS = 8
CACHE_TTL_SECONDS = 6 * 3600
WINDOW_FIELDS = ("context_window", "context_length", "max_context_length", "max_input_tokens", "max_model_len")

_cache: dict[str, tuple[float, int | None]] = {}
_lock = threading.Lock()


def _window_from_entry(entry: dict[str, Any]) -> int | None:
    for field in WINDOW_FIELDS:
        value = entry.get(field)
        if value is None and isinstance(entry.get("top_provider"), dict):
            value = entry["top_provider"].get(field)
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            continue
        if tokens > 0:
            return tokens
    return None


def parse_context_window(payload: Any, model: str) -> int | None:
    """Find ``model`` in a /models payload and return its context size in tokens."""
    entries = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return None
    wanted = model.strip().casefold()
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip().casefold() == wanted:
            return _window_from_entry(entry)
    return None


def context_window_tokens(base_url: str, api_key: str, model: str, *, fetch: Any = None) -> int | None:
    """Context window of ``model`` at ``base_url``, cached per process; None if unknown."""
    base = str(base_url or "").strip().rstrip("/")
    if not base or not model:
        return None
    key = f"{base}\n{model}"
    now = time.time()
    with _lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    result: int | None = None
    try:
        if fetch is None:
            request = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(request, timeout=LOOKUP_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        else:
            payload = fetch(base + "/models")
        result = parse_context_window(payload, model)
    except Exception:
        result = None
    with _lock:
        _cache[key] = (now, result)
    return result
