import unittest

from app import model_limits


class ModelLimitsTests(unittest.TestCase):
    def setUp(self) -> None:
        model_limits._cache.clear()

    def test_parse_common_listing_shapes(self) -> None:
        vercel = {"data": [{"id": "meta/muse-spark-1.3-contributor", "context_window": 1048576, "max_tokens": 1048576}]}
        self.assertEqual(model_limits.parse_context_window(vercel, "meta/muse-spark-1.3-contributor"), 1048576)
        openrouter = {"data": [{"id": "x/y", "context_length": 131072, "top_provider": {"context_length": 65536}}]}
        self.assertEqual(model_limits.parse_context_window(openrouter, "X/Y"), 131072)
        nested = {"data": [{"id": "z", "top_provider": {"context_length": 8192}}]}
        self.assertEqual(model_limits.parse_context_window(nested, "z"), 8192)
        self.assertIsNone(model_limits.parse_context_window({"data": [{"id": "z"}]}, "z"))
        self.assertIsNone(model_limits.parse_context_window({"data": [{"id": "other", "context_window": 1}]}, "z"))
        self.assertIsNone(model_limits.parse_context_window("garbage", "z"))

    def test_lookup_is_cached_and_never_raises(self) -> None:
        calls = []

        def fetch(url):
            calls.append(url)
            return {"data": [{"id": "m", "context_window": 200_000}]}

        self.assertEqual(model_limits.context_window_tokens("https://gw.test/v1/", "k", "m", fetch=fetch), 200_000)
        self.assertEqual(model_limits.context_window_tokens("https://gw.test/v1", "k", "m", fetch=fetch), 200_000)
        self.assertEqual(calls, ["https://gw.test/v1/models"])

        def broken(url):
            raise OSError("down")

        self.assertIsNone(model_limits.context_window_tokens("https://other.test/v1", "k", "m", fetch=broken))
        # The failure is cached too, so a dead endpoint is not retried every job.
        self.assertIsNone(model_limits.context_window_tokens("https://other.test/v1", "k", "m", fetch=fetch))
        self.assertIsNone(model_limits.context_window_tokens("", "k", "m", fetch=fetch))


if __name__ == "__main__":
    unittest.main()
