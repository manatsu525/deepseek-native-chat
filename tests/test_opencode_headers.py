from __future__ import annotations

import unittest

from app.mimo import custom_auth_headers, is_opencode_base_url


class OpenCodeHeaderTests(unittest.TestCase):
    def test_opencode_hosts_receive_requested_user_agent(self) -> None:
        for base_url in ("https://opencode.ai/zen/v1", "https://api.opencode.ai/v1"):
            self.assertTrue(is_opencode_base_url(base_url))
            headers = custom_auth_headers("secret", base_url=base_url, stream=True)
            self.assertEqual(headers["User-Agent"], "opencode/1.18.16")
            self.assertEqual(headers["Accept"], "text/event-stream")

    def test_other_and_lookalike_hosts_are_unchanged(self) -> None:
        for base_url in ("https://api.openai.com/v1", "https://opencode.ai.example.com/v1", ""):
            self.assertFalse(is_opencode_base_url(base_url))
            self.assertNotIn("User-Agent", custom_auth_headers("secret", base_url=base_url))


if __name__ == "__main__":
    unittest.main()
