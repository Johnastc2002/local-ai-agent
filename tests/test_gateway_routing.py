#!/usr/bin/env python3
"""Gateway routing logic — no vLLM required."""

import unittest

from gateway.cursor_protocol import (
    is_agent_request,
    is_tool_result_turn,
    is_user_turn,
    find_plan_tool,
)
from gateway.router import inject_icr_context


class GatewayRoutingTest(unittest.TestCase):
    def test_plan_request(self):
        body = {
            "messages": [{"role": "user", "content": "plan this"}],
            "tools": [{"type": "function", "function": {"name": "CreatePlan"}}],
        }
        self.assertTrue(is_user_turn(body))
        self.assertEqual(find_plan_tool(body["tools"]), "CreatePlan")
        self.assertFalse(is_agent_request(body))

    def test_agent_request(self):
        body = {
            "messages": [{"role": "user", "content": "fix bug"}],
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "StrReplace"}},
            ],
        }
        self.assertTrue(is_agent_request(body))
        self.assertIsNone(find_plan_tool(body["tools"]))

    def test_tool_result_turn(self):
        body = {
            "messages": [
                {"role": "user", "content": "read foo"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "file contents"},
            ],
        }
        self.assertTrue(is_tool_result_turn(body))
        self.assertFalse(is_user_turn(body))

    def test_icr_context_injected(self):
        msgs = [{"role": "user", "content": "hello"}]
        out = inject_icr_context(msgs, "refined plan text")
        self.assertIn("refined plan text", out[-1]["content"])
        self.assertIn("[ICR refined context]", out[-1]["content"])


if __name__ == "__main__":
    unittest.main()
