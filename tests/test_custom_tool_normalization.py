import json
import unittest

from app.custom_tool_normalization import normalize_tool_calls


class CustomToolNormalizationTests(unittest.TestCase):
    def test_json_encoded_search_query_array_is_normalized(self):
        calls = [
            {
                "id": "nvidia-call",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps(
                        {
                            "objective": "核实资料",
                            "search_queries": '["NVIDIA DeepSeek Flash", "DeepSeek V4 Flash"]',
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ]

        normalized = normalize_tool_calls(calls)

        arguments = json.loads(normalized[0]["function"]["arguments"])
        self.assertEqual(arguments["search_queries"], ["NVIDIA DeepSeek Flash", "DeepSeek V4 Flash"])

    def test_plain_search_query_string_becomes_one_item_array(self):
        calls = [
            {
                "id": "custom-call",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"objective": "核实资料", "search_queries": "单个查询"}),
                },
            }
        ]

        normalize_tool_calls(calls)

        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["search_queries"], ["单个查询"])

    def test_valid_array_is_unchanged(self):
        original_arguments = json.dumps({"objective": "核实资料", "search_queries": ["查询一", "查询二"]})
        calls = [
            {
                "id": "valid-call",
                "type": "function",
                "function": {"name": "web_search", "arguments": original_arguments},
            }
        ]

        normalize_tool_calls(calls)

        self.assertEqual(calls[0]["function"]["arguments"], original_arguments)


if __name__ == "__main__":
    unittest.main()
