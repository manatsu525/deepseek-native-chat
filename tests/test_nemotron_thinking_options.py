"""Regression tests for NVIDIA reasoning controls on every Nemotron model."""

from __future__ import annotations

import unittest

from app.mimo_local import _apply_thinking_options


class NemotronThinkingOptionsTests(unittest.TestCase):
    def apply(self, base_url: str, model: str, thinking: str = "enabled") -> dict:
        payload: dict = {}
        _apply_thinking_options(
            payload,
            base_url,
            model,
            thinking,
            "high",
            True,
            65536,
        )
        return payload

    def test_opencode_lightning_uses_nvidia_thinking_syntax(self) -> None:
        payload = self.apply("https://opencode.ai/zen/v1", "nemotron-3.5-lightning-free")
        self.assertEqual(payload, {"chat_template_kwargs": {"enable_thinking": True}})

    def test_any_provider_and_case_with_nemotron_in_model_name_is_supported(self) -> None:
        payload = self.apply("https://gateway.invalid/v1", "Vendor/NEMOTRON-Custom")
        self.assertEqual(payload, {"chat_template_kwargs": {"enable_thinking": True}})

    def test_disabled_nemotron_explicitly_disables_thinking(self) -> None:
        payload = self.apply("https://opencode.ai/zen/v1", "nemotron-3.5-lightning-free", "disabled")
        self.assertEqual(payload, {"chat_template_kwargs": {"enable_thinking": False}})

    def test_nvidia_ultra_keeps_its_existing_budget_workaround(self) -> None:
        payload = self.apply("https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(
            payload,
            {
                "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True},
                "reasoning_budget": 16384,
            },
        )

    def test_non_nemotron_custom_models_keep_generic_controls(self) -> None:
        payload = self.apply("https://gateway.invalid/v1", "ordinary-model")
        self.assertEqual(payload, {"thinking": {"type": "enabled"}, "reasoning_effort": "high"})


if __name__ == "__main__":
    unittest.main()
