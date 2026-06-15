#!/usr/bin/env python3
"""Extract tool names from Cursor BYOK requests (function + custom formats)."""

from __future__ import annotations

from gateway.cursor_protocol import tool_name, tool_names_from_body


def tool_names(body: dict) -> list[str]:
    return tool_names_from_body(body, include_plan=True)


def classify_request(body: dict) -> str:
    names = [n.lower() for n in tool_names(body)]
    if any("plan" in n for n in names):
        return "plan"
    if any(
        n in (
            "write",
            "strreplace",
            "applypatch",
            "delete",
            "shell",
            "run_terminal_cmd",
            "edit_file",
        )
        for n in names
    ):
        return "agent"
    return "ask"
