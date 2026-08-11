import unittest

from app.mimo import _settings
from app.mimo_local import _apply_thinking_options


class CustomReasoningOptionsTests(unittest.TestCase):
    def build_payload(self, include_reasoning_enabled: bool) -> dict:
        payload = {}
        _apply_thinking_options(
            payload,
            "https://ai-gateway.vercel.sh/v1",
            "deepseek/deepseek-v4-flash-0731",
            "enabled",
            "high",
            True,
            include_reasoning_enabled,
            65536,
        )
        return payload

    def test_default_setting_does_not_include_reasoning(self):
        settings = _settings({})
        self.assertFalse(settings["include_reasoning_enabled"])
        self.assertFalse(settings["dsml_fallback_enabled"])
        self.assertNotIn("include_reasoning", self.build_payload(False))

    def test_enabled_setting_sends_true(self):
        payload = self.build_payload(True)
        self.assertIs(payload["include_reasoning"], True)
        self.assertEqual(payload["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
