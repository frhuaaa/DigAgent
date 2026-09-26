from __future__ import annotations

import unittest
from unittest.mock import patch

from adapters.llm_client import AgentClient


class LlmParserTests(unittest.TestCase):
    def test_responses_parser_excludes_reasoning_text(self):
        payload = {
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "analysis"}]},
                {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]},
            ]
        }
        self.assertEqual(AgentClient._response_text(payload), '{"ok":true}')

    def test_unsafe_agent_response_is_rejected_and_retried(self):
        responses = [
            {"choices": [{"message": {"content": '{"value":"test_metrics are unavailable"}'}}]},
            {"choices": [{"message": {"content": '{"value":"validation evidence only"}'}}]},
        ]

        class FakeResponse:
            status_code = 200
            text = ""

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def post(self, *args, **kwargs):
                return FakeResponse(responses.pop(0))

        client = AgentClient.__new__(AgentClient)
        client.model_name = "fake-model"
        client.model_id = "fake-model"
        client.reasoning_effort = "medium"
        client.api = "chat_completions"
        client.base_url = "https://example.invalid"
        client._headers = {}
        schema = {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        }

        with patch("adapters.llm_client.httpx.Client", FakeClient), patch("adapters.llm_client.time.sleep"):
            result = client.complete_json("system", {"validation": True}, schema, "safe_retry")

        self.assertEqual(result, {"value": "validation evidence only"})
        self.assertEqual(responses, [])


if __name__ == "__main__":
    unittest.main()
