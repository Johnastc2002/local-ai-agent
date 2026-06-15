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
        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ]
        out = inject_icr_context(msgs, "refined plan text")
        self.assertEqual(len(out), 4)
        self.assertEqual(out[-1]["content"], "follow up")
        self.assertEqual(out[-2]["role"], "developer")
        self.assertIn("refined plan text", out[-2]["content"])
        self.assertIn("[ICR refined context]", out[-2]["content"])

    def test_trim_keeps_history_after_icr(self):
        from gateway.context_trim import trim_conversation_tail
        from gateway.router import inject_icr_context

        msgs = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "message one"},
            {"role": "assistant", "content": "reply one"},
            {"role": "user", "content": "message two"},
        ]
        icr = "x" * 50_000
        enriched = inject_icr_context(msgs, icr)
        out = trim_conversation_tail(enriched, budget_tokens=4000, min_tail_messages=4)
        roles = [m["role"] for m in out]
        self.assertIn("assistant", roles)
        self.assertEqual(roles.count("user"), 2)


if __name__ == "__main__":
    unittest.main()
