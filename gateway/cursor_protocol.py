#!/usr/bin/env python3
"""Use Cursor BYOK request shape — messages, tools, tool-result round-trips."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llm import Message, content_to_text

PLAN_TOOL_HINTS = ("createplan", "create_plan", "updateplan", "update_plan")
PYTHON_TOOL_NAME = "python_virtual_filesystem"


def tool_name(tool: dict) -> str:
    fn = tool.get("function") or {}
    return str(fn.get("name") or "")


def is_plan_tool_name(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in PLAN_TOOL_HINTS)


def cursor_tools(body: dict, *, for_icr: bool = True) -> list[dict]:
    """Tools Cursor sent — Cursor executes these on the Mac when model calls them."""
    tools = list(body.get("tools") or [])
    if not for_icr:
        return tools
    # During ICR phases, don't let inner agents call CreatePlan early.
    return [t for t in tools if not is_plan_tool_name(tool_name(t))]


def cursor_tool_names(body: dict, *, for_icr: bool = True) -> set[str]:
    return {tool_name(t) for t in cursor_tools(body, for_icr=for_icr) if tool_name(t)}


def find_plan_tool(tools: list[dict] | None) -> str | None:
    if not tools:
        return None
    for tool in tools:
        name = tool_name(tool)
        if is_plan_tool_name(name):
            return name
    return None


def is_agent_request(body: dict) -> bool:
    tools = body.get("tools") or []
    return bool(tools) and find_plan_tool(tools) is None


def is_cursor_managed_tool(name: str, body: dict) -> bool:
    """Tool Cursor runs locally (read_file, grep, MCP, etc.)."""
    if name == PYTHON_TOOL_NAME:
        return False
    return name in cursor_tool_names(body, for_icr=True)


def seed_messages_from_cursor(body: dict) -> list[Message]:
    """
    Keep Cursor's context: system rules, @files, codebase snippets, user prompt.
    Drops prior assistant/tool turns from earlier ICR pauses — those are in session.
    """
    out: list[Message] = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        if role == "system":
            out.append(dict(msg))
        elif role == "user":
            out.append(dict(msg))
        elif role == "developer":
            out.append(dict(msg))
    if not any(m.get("role") == "user" for m in out):
        for msg in reversed(body.get("messages") or []):
            if msg.get("role") == "user":
                out.append(dict(msg))
                break
    return out


def extract_task(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        text = content_to_text(msg.get("content")).strip()
        if text:
            return text
    return ""


def is_user_turn(body: dict) -> bool:
    messages = body.get("messages") or []
    return bool(messages) and messages[-1].get("role") == "user"


def is_tool_result_turn(body: dict) -> bool:
    messages = body.get("messages") or []
    return bool(messages) and messages[-1].get("role") == "tool"


def conversation_key(body: dict) -> str:
    """Stable id for ICR session across tool round-trips."""
    seed_msgs = seed_messages_from_cursor(body)
    payload = json.dumps(seed_msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def pending_tool_call_ids(messages: list[Message]) -> list[str]:
    """Tool call ids from the last assistant message before trailing tool messages."""
    if not messages or messages[-1].get("role") != "tool":
        return []
    idx = len(messages) - 1
    while idx >= 0 and messages[idx].get("role") == "tool":
        idx -= 1
    if idx < 0 or messages[idx].get("role") != "assistant":
        return []
    assistant = messages[idx]
    ids: list[str] = []
    for tc in assistant.get("tool_calls") or []:
        if tc_id := tc.get("id"):
            ids.append(tc_id)
    return ids


def tool_results_since(messages: list[Message], after_index: int) -> list[Message]:
    return [dict(m) for m in messages[after_index + 1 :] if m.get("role") == "tool"]


def completion_from_assistant(body: dict, assistant: Message) -> dict:
    import time
    import uuid

    from llm import load_env

    model = body.get("model") or load_env().get("MODEL_NAME", "Qwen/Qwen3.6-27B")
    finish = "tool_calls" if assistant.get("tool_calls") else "stop"
    return {
        "id": f"chatcmpl-icr-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": assistant,
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
