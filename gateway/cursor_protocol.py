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
    """Extract tool name from OpenAI function or Cursor custom tool defs."""
    if tool.get("type") == "function":
        fn = tool.get("function") or {}
        return str(fn.get("name") or "")
    if custom := tool.get("custom"):
        if isinstance(custom, dict):
            return str(custom.get("name") or custom.get("type") or custom.get("id") or "")
        return str(custom)
    return str(tool.get("name") or "")


def tool_names_from_body(body: dict, *, include_plan: bool = True) -> list[str]:
    names: list[str] = []
    for tool in body.get("tools") or []:
        name = tool_name(tool)
        if not name:
            continue
        if not include_plan and is_plan_tool_name(name):
            continue
        names.append(name)
    return names


def is_plan_tool_name(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in PLAN_TOOL_HINTS)


def cursor_tools(body: dict, *, for_icr: bool = True) -> list[dict]:
    """
    Tools to attach to inner ICR vLLM calls.
    Copies Cursor's tool defs verbatim; only strips CreatePlan during ICR phases.
    """
    tools = [dict(t) for t in body.get("tools") or []]
    if not for_icr:
        return tools
    return [t for t in tools if not is_plan_tool_name(tool_name(t))]


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
    """
    True if Cursor sent this tool and should execute it on the Mac.
    Excludes our internal python sandbox and plan tools (gateway handles plan).
    """
    if not name or name == PYTHON_TOOL_NAME or is_plan_tool_name(name):
        return False
    return name in tool_names_from_body(body, include_plan=False)


def icr_request_options(body: dict) -> dict[str, Any]:
    """Forward tool-related fields Cursor sent (parity with passthrough)."""
    opts: dict[str, Any] = {}
    if "tool_choice" in body:
        opts["tool_choice"] = body["tool_choice"]
    if "parallel_tool_calls" in body:
        opts["parallel_tool_calls"] = body["parallel_tool_calls"]
    return opts


def seed_messages_from_cursor(body: dict) -> list[Message]:
    """Full Cursor thread — system, user, assistant, tool (for ICR + conversation key)."""
    return [dict(m) for m in body.get("messages") or []]


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
    seed_msgs = seed_messages_from_cursor(body)
    payload = json.dumps(seed_msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def pending_tool_call_ids(messages: list[Message]) -> list[str]:
    if not messages or messages[-1].get("role") != "tool":
        return []
    idx = len(messages) - 1
    while idx >= 0 and messages[idx].get("role") == "tool":
        idx -= 1
    if idx < 0 or messages[idx].get("role") != "assistant":
        return []
    assistant = messages[idx]
    return [tc.get("id") for tc in assistant.get("tool_calls") or [] if tc.get("id")]


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
