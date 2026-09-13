from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()

