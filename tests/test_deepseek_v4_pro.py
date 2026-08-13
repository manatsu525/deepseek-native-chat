"""Offline checks for the official DeepSeek V4 Pro Responses route."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from app import deepseek, main


class _FakeResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def aiter_lines(self):
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"usage": {"output_tokens": 1}, "output": []}},
        ]
        for event in events:
            yield f"data: {json.dumps(event)}"
            yield ""


class _FakeClient:
    request: dict = {}

    def __init__(self, **_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def stream(self, method, url, *, headers, json):
        type(self).request = {"method": method, "url": url, "headers": headers, "json": json}
        return _FakeResponse()


class DeepSeekV4ProTests(unittest.TestCase):
    def test_pro_is_an_allowed_official_model(self) -> None:
        main.validate_provider_selection("deepseek", "deepseek-v4-pro")

    def test_pro_uses_responses_with_native_web_search(self) -> None:
        updates: list[dict] = []

        async def update(value):
            updates.append(value)

        with patch.object(deepseek.httpx, "AsyncClient", _FakeClient):
            result = asyncio.run(
                deepseek.stream_response(
                    base_url="https://api.deepseek.com",
                    api_key="offline-test-key",
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": "test"}],
                    effort="high",
                    timeout=30,
                    stopped=lambda: False,
                    update=update,
                )
            )

        request = _FakeClient.request
        self.assertEqual(request["url"], "https://api.deepseek.com/responses")
        self.assertEqual(request["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(request["json"]["tools"], [{"type": "web_search"}])
        self.assertEqual(request["json"]["tool_choice"], "auto")
        self.assertEqual(result["answer"], "ok")
        self.assertTrue(updates)


if __name__ == "__main__":
    unittest.main()
