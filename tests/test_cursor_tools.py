#!/usr/bin/env python3
import unittest

from gateway.cursor_protocol import (
    cursor_tools,
    is_cursor_managed_tool,
    tool_name,
    tool_names_from_body,
)


class CursorToolsTest(unittest.TestCase):
    def test_function_tool_name(self):
        tool = {"type": "function", "function": {"name": "read_file"}}
        self.assertEqual(tool_name(tool), "read_file")

    def test_custom_tool_name(self):
        tool = {"type": "custom", "custom": {"name": "mcp_linear"}}
        self.assertEqual(tool_name(tool), "mcp_linear")

    def test_cursor_managed_any_sent_tool(self):
        body = {
            "tools": [
                {"type": "function", "function": {"name": "grep"}},
                {"type": "function", "function": {"name": "CreatePlan"}},
            ]
        }
        self.assertTrue(is_cursor_managed_tool("grep", body))
        self.assertFalse(is_cursor_managed_tool("CreatePlan", body))
        self.assertFalse(is_cursor_managed_tool("made_up", body))

    def test_icr_strips_plan_only(self):
        body = {
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "CreatePlan"}},
            ]
        }
        names = [tool_name(t) for t in cursor_tools(body)]
        self.assertEqual(names, ["read_file"])
        self.assertEqual(tool_names_from_body(body), ["read_file", "CreatePlan"])


if __name__ == "__main__":
    unittest.main()
