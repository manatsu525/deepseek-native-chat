"""Offline regression tests for NanoGPT billing reservation retries."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from app import mimo_local


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self.body = body or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def aread(self) -> bytes:
        return json.dumps(self.body).encode()

    async def aiter_lines(self):
        if self.status_code < 400:
            yield 'data: {"choices":[{"delta":{"content":"OK"}}]}'
            yield "data: [DONE]"


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def stream(self, *args, **kwargs) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeParallelClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class NanoGPTBillingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def run_response(self, client: FakeClient):
        async def update(_state):
            return None

        with patch.object(mimo_local.httpx, "AsyncClient", return_value=client), patch.object(
            mimo_local, "ParallelMCPClient", FakeParallelClient
        ), patch.object(mimo_local.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = await mimo_local.stream_response(
                base_url="https://nano-gpt.com/api/v1",
                api_key="test-key",
                model="meta/muse-spark-1.2-contributor",
                messages=[{"role": "user", "content": "Reply OK"}],
                timeout=30,
                stopped=lambda: False,
                update=update,
                settings={"max_completion_tokens": 64},
                effort="low",
            )
        return result, sleep

    async def test_explicit_reservation_failure_retries_once(self) -> None:
        client = FakeClient(
            [
                FakeResponse(503, {"error": "Billing reservation failed", "code": "async_billing_reservation_failed"}),
                FakeResponse(200),
            ]
        )
        result, sleep = await self.run_response(client)
        self.assertEqual(result["answer"], "OK")
        self.assertEqual(client.calls, 2)
        sleep.assert_awaited_once_with(mimo_local.BILLING_RESERVATION_RETRY_DELAY)

    async def test_other_503_is_not_retried(self) -> None:
        client = FakeClient([FakeResponse(503, {"error": "upstream unavailable", "code": "service_unavailable"})])
        with self.assertRaisesRegex(RuntimeError, "service_unavailable"):
            await self.run_response(client)
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
