from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urldefrag

import httpx


MIMO_MODELS = {"mimo-v2.5-pro", "mimo-v2.5"}
MIMO_MAX_COMPLETION_TOKENS = 131072
# Keep the system instruction deliberately small. Search decisions belong to MiMo's
# native tool, not to a second client-side agent loop.
MIMO_SYSTEM_PROMPT = "You are MiMo, an AI assistant developed by Xiaomi."
DEFAULT_SETTINGS = {
    "max_keyword": 3,
    "limit": 5,
    "force_search": False,
    "country": "",
    "region": "",
    "city": "",
    "thinking": "enabled",
    "max_completion_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
}


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _settings(value: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_SETTINGS)
    if value:
        result.update(value)
    return result


async def list_models(base_url: str, api_key: str, timeout: int = 30) -> list[str]:
    headers = {"api-key": api_key}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(_url(base_url, "/models"), headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
    return sorted({str(item.get("id")) for item in data if item.get("id")})


def _source_from_annotation(annotation: dict[str, Any]) -> dict[str, str]:
    raw_url = str(annotation.get("url") or "")
    url = urldefrag(raw_url).url if raw_url else ""
    return {
        "url": url,
        "title": str(annotation.get("title") or annotation.get("site_name") or url),
        "summary": str(annotation.get("summary") or ""),
        "site_name": str(annotation.get("site_name") or ""),
        "publish_time": str(annotation.get("publish_time") or ""),
        "logo_url": str(annotation.get("logo_url") or ""),
    }


def _add_sources(found: dict[str, dict[str, str]], annotations: Any) -> None:
    if isinstance(annotations, dict):
        annotations = [annotations]
    if not isinstance(annotations, list):
        return
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        source = _source_from_annotation(annotation)
        if source["url"]:
            found[source["url"]] = source


def _normalize_usage(raw: dict[str, Any]) -> dict[str, Any]:
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return {
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
        "input_tokens_details": {"cached_tokens": int(prompt_details.get("cached_tokens") or 0)},
        "output_tokens_details": {"reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0)},
        "web_search_usage": raw.get("web_search_usage") or {},
    }


def _search_items(sources: list[dict[str, str]], usage: dict[str, Any], error: str = "") -> list[dict[str, Any]]:
    web_usage = usage.get("web_search_usage") or {}
    tool_count = int(web_usage.get("tool_usage") or (1 if sources else 0))
    if error and not tool_count:
        tool_count = 1
    if not tool_count:
        return []
    return [
        {
            "id": f"mimo-web-search-{index + 1}",
            "status": "failed" if error else "completed",
            "action": "search",
            "query": "MiMo 联网搜索",
            "url": "",
            "error": error,
        }
        for index in range(tool_count)
    ]


async def stream_response(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    stopped: Callable[[], bool],
    update: Callable[[dict[str, Any]], Awaitable[None]],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _settings(settings)
    tool: dict[str, Any] = {
        "type": "web_search",
        "max_keyword": int(config["max_keyword"]),
        "limit": int(config["limit"]),
        "force_search": bool(config["force_search"]),
    }
    location = {
        key: str(config[key]).strip()
        for key in ("country", "region", "city")
        if str(config.get(key) or "").strip()
    }
    if location:
        tool["user_location"] = {"type": "approximate", **location}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": MIMO_SYSTEM_PROMPT}, *messages],
        "tools": [tool],
        "tool_choice": "auto",
        "max_completion_tokens": int(config["max_completion_tokens"]),
        "stream": True,
        "thinking": {"type": config["thinking"]},
    }
    if config["thinking"] == "disabled":
        payload["temperature"] = float(config["temperature"])
        payload["top_p"] = float(config["top_p"])
    headers = {"api-key": api_key, "Content-Type": "application/json", "Accept": "text/event-stream"}
    answer = ""
    reasoning = ""
    usage: dict[str, Any] = {}
    sources: dict[str, dict[str, str]] = {}
    search_error = ""
    tool_calls: list[dict[str, Any]] = []
    limits = httpx.Timeout(timeout, connect=30)
    async with httpx.AsyncClient(timeout=limits) as client:
        async with client.stream("POST", _url(base_url, "/chat/completions"), headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:2000]
                raise RuntimeError(f"MiMo API {response.status_code}: {body}")
            async for line in response.aiter_lines():
                if stopped():
                    raise asyncio.CancelledError
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise RuntimeError(f"MiMo 响应失败: {data['error']}")
                raw_usage = data.get("usage")
                if isinstance(raw_usage, dict):
                    usage = _normalize_usage(raw_usage)
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    message = choice.get("message") or {}
                    answer += str(delta.get("content") or "")
                    reasoning += str(delta.get("reasoning_content") or "")
                    _add_sources(sources, delta.get("annotations"))
                    _add_sources(sources, message.get("annotations"))
                    _add_sources(sources, choice.get("annotations"))
                    if isinstance(message.get("tool_calls"), list):
                        tool_calls = message["tool_calls"]
                    search_error = str(delta.get("error_message") or message.get("error_message") or search_error)
                source_list = list(sources.values())
                searches = _search_items(source_list, usage, search_error)
                await update({"answer": answer, "reasoning": reasoning, "searches": searches, "usage": usage, "sources": source_list})
    source_list = list(sources.values())
    searches = _search_items(source_list, usage, search_error)
    return {"answer": answer, "reasoning": reasoning, "searches": searches, "sources": source_list, "usage": usage, "tool_calls": tool_calls, "response": {"tool_calls": tool_calls}}
