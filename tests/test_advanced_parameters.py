"""Preview/live parity and complete replacement, without model API calls."""
import unittest
from unittest.mock import patch

from app.custom_request import validate_advanced_request
from app.mimo_local import build_custom_request_parameters, stream_response
from test_responses_state import Transport


class AdvancedParameterTests(unittest.IsolatedAsyncioTestCase):
    async def send(self, protocol, config):
        events = {
            "responses": [{"type": "response.output_text.delta", "delta": "ok"},
                          {"type": "response.completed", "response": {"id": "resp_done", "output": []}}],
            "messages": [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}],
            "chat_completions": [{"choices": [{"delta": {"content": "ok"}}]}],
        }
        transport = Transport([events[protocol]])
        async def update(_):
            pass
        with patch("app.mimo_local.httpx.AsyncClient", return_value=transport):
            result = await stream_response(
                base_url="https://test.invalid/v1", api_key="test", model="test-model",
                messages=[{"role": "user", "content": "test"}], timeout=5, stopped=lambda: False,
                update=update, settings=config, api_protocol=protocol, effort="high",
                conversation_id="chat-123", web_enabled=False, max_tool_rounds=0,
            )
        self.assertEqual(result["answer"], "ok")
        return transport.payloads[0]

    async def test_all_protocols_preview_matches_actual_normal_parameters(self):
        for protocol in ("responses", "messages", "chat_completions"):
            for thinking in ("enabled", "disabled"):
                with self.subTest(protocol=protocol, thinking=thinking):
                    config = {"advanced_enabled": False, "thinking": thinking, "reasoning_effort_enabled": False,
                              "max_completion_tokens": 4096, "lowest_price_aggregators": ["vercel"]}
                    preview = build_custom_request_parameters("https://test.invalid/v1", "test-model", config, api_protocol=protocol)
                    payload = await self.send(protocol, config)
                    envelope = {"messages", "input", "system", "instructions", "stream", "tools", "tool_choice"}
                    self.assertEqual(preview, {key: value for key, value in payload.items() if key not in envelope})

    async def test_advanced_replaces_and_deletes_fields_in_every_protocol(self):
        for protocol in ("responses", "messages", "chat_completions"):
            with self.subTest(protocol=protocol):
                config = {"advanced_enabled": True, "thinking": "enabled", "reasoning_effort_enabled": True,
                          "lowest_price_aggregators": ["openrouter", "vercel"],
                          "advanced_request": {"model": "manual-route", "max_tokens": 123,
                                               "temperature": 0.1, "session_id": "{{conversation_id}}",
                                               "vendor": {"custom": [True, 1]}}}
                payload = await self.send(protocol, config)
                self.assertEqual(payload["model"], "manual-route")
                self.assertEqual(payload["temperature"], 0.1)
                self.assertEqual(payload["max_tokens"], 123)
                self.assertEqual(payload["session_id"], "chat-123")
                for missing in ("thinking", "reasoning", "reasoning_effort", "output_config", "top_p",
                                "max_completion_tokens", "max_output_tokens", "providerOptions", "store"):
                    self.assertNotIn(missing, payload)

    async def test_disabled_advanced_document_does_not_affect_request(self):
        config = {"advanced_enabled": False, "temperature_enabled": True, "temperature": 0.8, "advanced_request": {"temperature": 0.1},
                  "request_overrides": {"temperature": 0.2}}
        payload = await self.send("chat_completions", config)
        self.assertEqual(payload["temperature"], 0.8)

    async def test_sampling_parameters_are_opt_in(self):
        config = {"advanced_enabled": False, "thinking": "disabled", "temperature": 0.2, "top_p": 0.8,
                  "max_completion_tokens": 4096}
        for protocol in ("responses", "messages", "chat_completions"):
            with self.subTest(protocol=protocol):
                preview = build_custom_request_parameters(
                    "https://test.invalid/v1", "test-model", config, api_protocol=protocol
                )
                self.assertNotIn("temperature", preview)
                self.assertNotIn("top_p", preview)
                enabled = {**config, "temperature_enabled": True, "top_p_enabled": True}
                preview = build_custom_request_parameters(
                    "https://test.invalid/v1", "test-model", enabled, api_protocol=protocol
                )
                self.assertEqual(preview["temperature"], 0.2)
                self.assertEqual(preview["top_p"], 0.8)

    def test_advanced_validates_runtime_fields_and_json_types(self):
        for value in ([], {"messages": []}, {"previous_response_id": "old"}, {"stream": False},
                      {"model": ""}, {"store": "false"}, {"temperature": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_advanced_request(value)
