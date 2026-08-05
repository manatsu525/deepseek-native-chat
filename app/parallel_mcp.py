from __future__ import annotations

import json
from typing import Any

import httpx


PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
PARALLEL_MCP_PROTOCOL = "2025-03-26"


class ParallelMCPError(RuntimeError):
    pass


class ParallelMCPClient:
    """Small Streamable HTTP MCP client for Parallel's anonymous Search MCP."""

    def __init__(self, timeout: int = 90) -> None:
        self._timeout = httpx.Timeout(timeout, connect=10)
        self._client: httpx.AsyncClient | None = None
        self._session_id = ""
        self._request_id = 0

    async def __aenter__(self) -> "ParallelMCPClient":
        self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self

    async def _initialize(self) -> None:
        if not self._client:
            raise ParallelMCPError("Parallel Search MCP 客户端尚未启动")
        response = await self._client.post(
            PARALLEL_MCP_URL,
            headers=self._headers(include_session=False),
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": PARALLEL_MCP_PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "deepseek-native-chat", "version": "1.0.0"},
                },
            },
        )
        data = self._decode_response(response)
        self._raise_rpc_error(data)
        self._session_id = response.headers.get("mcp-session-id", "")
        if not self._session_id:
            raise ParallelMCPError("Parallel Search MCP 未返回会话 ID")
        ready = await self._client.post(
            PARALLEL_MCP_URL,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )
        if ready.status_code >= 400:
            raise ParallelMCPError(f"Parallel Search MCP 初始化确认失败（HTTP {ready.status_code}）")

    async def __aexit__(self, *args: Any) -> None:
        if not self._client:
            return
        if self._session_id:
            try:
                await self._client.delete(PARALLEL_MCP_URL, headers=self._headers())
            except Exception:
                pass
        await self._client.aclose()
        self._client = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            raise ParallelMCPError("Parallel Search MCP 客户端尚未启动")
        if not self._session_id:
            await self._initialize()
        response = await self._client.post(
            PARALLEL_MCP_URL,
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        data = self._decode_response(response)
        self._raise_rpc_error(data)
        result = data.get("result") or {}
        if result.get("isError"):
            message = self._content_text(result) or f"Parallel MCP 工具 {name} 执行失败"
            raise ParallelMCPError(message[:1000])
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        text = self._content_text(result)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ParallelMCPError(f"Parallel MCP 工具 {name} 返回了无法解析的结果") from exc
        if not isinstance(parsed, dict):
            raise ParallelMCPError(f"Parallel MCP 工具 {name} 返回格式无效")
        return parsed

    def _headers(self, *, include_session: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": PARALLEL_MCP_PROTOCOL,
        }
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            detail = response.text.strip().replace("\n", " ")[:500]
            suffix = f"：{detail}" if detail else ""
            raise ParallelMCPError(f"Parallel Search MCP HTTP {response.status_code}{suffix}")
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            events: list[dict[str, Any]] = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    try:
                        item = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        events.append(item)
            if not events:
                raise ParallelMCPError("Parallel Search MCP 返回了空 SSE 响应")
            data = next((item for item in reversed(events) if "result" in item or "error" in item), events[-1])
        else:
            data = response.json()
        if not isinstance(data, dict):
            raise ParallelMCPError("Parallel Search MCP 返回格式无效")
        return data

    @staticmethod
    def _raise_rpc_error(data: dict[str, Any]) -> None:
        error = data.get("error")
        if not error:
            return
        if isinstance(error, dict):
            message = str(error.get("message") or error)
        else:
            message = str(error)
        raise ParallelMCPError(f"Parallel Search MCP：{message[:1000]}")

    @staticmethod
    def _content_text(result: dict[str, Any]) -> str:
        return "\n".join(
            str(item.get("text") or "")
            for item in result.get("content") or []
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
