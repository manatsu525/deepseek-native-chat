from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urldefrag, urlsplit

import httpx


SYSTEM_PROMPT = """You are a helpful assistant. Use the server-side web search tool when current, uncertain, or source-backed information is needed. Search iteratively when useful, but stop once sufficient evidence is available and avoid redundant searches. Prefer diverse, relevant sources and clearly distinguish sourced facts from inference. Do not make more than five web searches for one answer."""


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


async def list_models(base_url: str, api_key: str, timeout: int = 30) -> list[str]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(_url(base_url, "/models"), headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        data = response.json().get("data", [])
    return sorted({str(item.get("id")) for item in data if item.get("id")})


def _search_from_item(item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action") or {}
    raw_url = action.get("url") or ""
    clean_url = urldefrag(raw_url).url if raw_url else ""
    return {
        "id": item.get("id", ""),
        "status": item.get("status", "searching"),
        "action": action.get("type", "search"),
        "query": action.get("query") or action.get("queries") or "",
        "url": clean_url,
    }


def _collect_sources(response: dict[str, Any]) -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for item in response.get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            raw_url = action.get("url") or ""
            if raw_url:
                url = urldefrag(raw_url).url
                host = urlsplit(url).netloc.removeprefix("www.")
                found[url] = {"title": host or url, "url": url}
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            for ann in part.get("annotations", []):
                url = ann.get("url") or (ann.get("url_citation") or {}).get("url")
                if not url:
                    continue
                title = ann.get("title") or (ann.get("url_citation") or {}).get("title") or url
                found[url] = {"title": title, "url": url}
    return list(found.values())


async def stream_response(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    effort: str,
    timeout: int,
    stopped: Callable[[], bool],
    update: Callable[[dict[str, Any]], Awaitable[None]],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": messages,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "reasoning": {"effort": effort},
        # DeepSeek V4 的服务端上限是 384K，超过会直接拒绝请求。
        "max_output_tokens": 393216,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    answer = ""
    reasoning = ""
    searches: dict[str, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    sources: list[dict[str, str]] = []
    completed: dict[str, Any] = {}
    limits = httpx.Timeout(timeout, connect=30)
    async with httpx.AsyncClient(timeout=limits) as client:
        async with client.stream("POST", _url(base_url, "/responses"), headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:2000]
                raise RuntimeError(f"DeepSeek API {response.status_code}: {body}")
            event_name = ""
            async for line in response.aiter_lines():
                if stopped():
                    raise asyncio.CancelledError
                if not line:
                    event_name = ""
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = data.get("type") or event_name
                if kind == "response.output_text.delta":
                    answer += data.get("delta", "")
                elif kind == "response.reasoning_text.delta":
                    reasoning += data.get("delta", "")
                elif kind in {"response.output_item.added", "response.output_item.done"}:
                    item = data.get("item") or {}
                    if item.get("type") == "web_search_call":
                        search = _search_from_item(item)
                        searches[search["id"] or str(len(searches))] = search
                elif kind.startswith("response.web_search_call."):
                    item = data.get("item") or {}
                    key = item.get("id") or data.get("item_id") or str(data.get("output_index", len(searches)))
                    search = searches.get(key, _search_from_item(item))
                    search["status"] = kind.rsplit(".", 1)[-1]
                    if item.get("action"):
                        search.update(_search_from_item(item))
                    searches[key] = search
                elif kind in {"response.completed", "response.incomplete"}:
                    completed = data.get("response") or {}
                    usage = completed.get("usage") or {}
                    sources = _collect_sources(completed)
                    for output_item in completed.get("output", []):
                        if output_item.get("type") == "web_search_call":
                            search = _search_from_item(output_item)
                            searches[search["id"] or str(len(searches))] = search
                elif kind == "response.failed":
                    failure = data.get("response", {}).get("error") or data.get("error") or data
                    raise RuntimeError(f"DeepSeek 响应失败: {failure}")
                await update({"answer": answer, "reasoning": reasoning, "searches": list(searches.values()), "usage": usage, "sources": sources})
    return {"answer": answer, "reasoning": reasoning, "searches": list(searches.values()), "sources": sources, "usage": usage, "response": completed}
