"""Offline tests for user-defined Custom request JSON."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from app.custom_request import apply_request_overrides, validate_request_overrides
from app import mimo_local


class _FakeStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def aread(self):
        return b""

    async def aiter_lines(self):
        yield 'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "ok"})
        yield "data: [DONE]"


class _FakeAsyncClient:
    payloads = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def stream(self, method, url, **kwargs):
        self.__class__.payloads.append(kwargs["json"])
        return _FakeStreamResponse()


class CustomRequestOverrideTests(unittest.TestCase):
    def test_placeholders_expand_recursively_and_override_root_fields(self) -> None:
        payload = {"temperature": 1.0, "provider": {"allow_fallbacks": True}}
        apply_request_overrides(
            payload,
            {
                "session_id": "{{conversation_id}}",
                "provider": {"only": ["meta"]},
                "metadata": {"model": "{{model}}", "items": ["{{effort}}", 3]},
                "temperature": 0.2,
            },
            context={"conversation_id": "conversation-123", "model": "meta/muse", "effort": "high"},
        )

        self.assertEqual(payload["session_id"], "conversation-123")
        self.assertEqual(payload["provider"], {"only": ["meta"]})
        self.assertEqual(payload["metadata"], {"model": "meta/muse", "items": ["high", 3]})
        self.assertEqual(payload["temperature"], 0.2)

    def test_unknown_placeholders_are_left_untouched(self) -> None:
        payload: dict = {}
        apply_request_overrides(payload, {"note": "{{provider_specific_value}}"}, context={})
        self.assertEqual(payload["note"], "{{provider_specific_value}}")

    def test_envelope_fields_are_rejected(self) -> None:
        for field in ("model", "input", "messages", "stream", "tools", "tool_choice"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_request_overrides({field: "unsafe"})

    def test_stream_response_applies_overrides_to_generated_payload(self) -> None:
        _FakeAsyncClient.payloads = []
        updates = []

        async def update(value):
            updates.append(value)

        with patch.object(mimo_local.httpx, "AsyncClient", _FakeAsyncClient):
            result = asyncio.run(
                mimo_local.stream_response(
                    base_url="https://provider.invalid/v1",
                    api_key="test-token",
                    model="meta/muse-spark",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=5,
                    stopped=lambda: False,
                    update=update,
                    settings={
                        "thinking": "disabled",
                        "request_overrides": {
                            "session_id": "{{conversation_id}}",
                            "provider": {"only": ["meta"]},
                            "metadata": {"model": "{{model}}", "protocol": "{{api_protocol}}"},
                        },
                    },
                    conversation_id="conversation-42",
                    effort="xhigh",
                    web_enabled=False,
                    max_tool_rounds=0,
                    api_protocol="responses",
                )
            )

        self.assertEqual(result["answer"], "ok")
        self.assertTrue(updates)
        self.assertEqual(len(_FakeAsyncClient.payloads), 1)
        payload = _FakeAsyncClient.payloads[0]
        self.assertEqual(payload["session_id"], "conversation-42")
        self.assertEqual(payload["provider"], {"only": ["meta"]})
        self.assertEqual(payload["metadata"], {"model": "meta/muse-spark", "protocol": "responses"})
        self.assertEqual(payload["model"], "meta/muse-spark")
        self.assertIn("input", payload)


if __name__ == "__main__":
    unittest.main()
