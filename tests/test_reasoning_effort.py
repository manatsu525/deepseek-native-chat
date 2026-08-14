"""Offline checks for the five-level reasoning effort control."""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.main import ChatBody, validate_effort
from app.mimo_local import _apply_thinking_options, _is_malformed_tool_call_error
from app.reasoning_effort import DEFAULT, LEVELS, normalize


class ReasoningEffortTests(unittest.TestCase):
    def test_levels_and_default_are_stable(self) -> None:
        self.assertEqual(LEVELS, ("low", "medium", "high", "xhigh", "max"))
        self.assertEqual(DEFAULT, "high")
        self.assertEqual(ChatBody(content="x", provider_id=1).effort, "high")

    def test_api_validation_accepts_only_the_five_levels(self) -> None:
        for level in LEVELS:
            self.assertEqual(validate_effort(level), level)
        with self.assertRaises(HTTPException):
            validate_effort("ultra")

    def test_internal_normalization_has_a_safe_default(self) -> None:
        self.assertEqual(normalize("low"), "low")
        self.assertEqual(normalize("unknown"), "high")

    def test_generic_custom_model_receives_the_selected_level(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://gateway.invalid/v1",
            "ordinary-model",
            "enabled",
            "xhigh",
            True,
            65536,
        )
        self.assertEqual(payload["reasoning_effort"], "xhigh")

    def test_nvidia_deepseek_uses_chat_template_kwargs(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://integrate.api.nvidia.com/v1",
            "deepseek-ai/deepseek-v4-flash",
            "enabled",
            "medium",
            True,
            65536,
        )
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"thinking": True, "reasoning_effort": "medium"},
        )

    def test_disabled_reasoning_effort_is_not_sent(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://gateway.invalid/v1",
            "ordinary-model",
            "enabled",
            "max",
            False,
            65536,
        )
        self.assertNotIn("reasoning_effort", payload)

    def test_nanogpt_muse_uses_only_supported_reasoning_effort(self) -> None:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            "https://nano-gpt.com/api/v1",
            "meta/muse-spark-1.2-contributor",
            "enabled",
            "high",
            True,
            65536,
        )
        self.assertEqual(payload, {"reasoning_effort": "high"})

    def test_nanogpt_muse_maps_max_and_can_disable_reasoning(self) -> None:
        maximum: dict = {}
        _apply_thinking_options(
            maximum,
            "https://nano-gpt.com/api/v1",
            "meta/muse-spark-1.2-contributor",
            "enabled",
            "max",
            True,
            65536,
        )
        self.assertEqual(maximum, {"reasoning_effort": "xhigh"})

        disabled: dict = {}
        _apply_thinking_options(
            disabled,
            "https://nano-gpt.com/api/v1",
            "meta/muse-spark-1.2-contributor",
            "disabled",
            "high",
            True,
            65536,
        )
        self.assertEqual(disabled, {"reasoning_effort": "none"})

    def test_only_structured_malformed_tool_call_error_is_recoverable(self) -> None:
        self.assertTrue(_is_malformed_tool_call_error({"code": "malformed_tool_call"}))
        self.assertFalse(_is_malformed_tool_call_error({"code": "rate_limit_exceeded"}))
        self.assertFalse(_is_malformed_tool_call_error("malformed_tool_call"))


if __name__ == "__main__":
    unittest.main()
