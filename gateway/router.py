#!/usr/bin/env python3
"""Detect Cursor Plan-mode requests and build CreatePlan responses."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from llm import content_to_text, load_env

Message = dict[str, Any]

PLAN_TOOL_HINTS = ("createplan", "create_plan", "updateplan", "update_plan")


def find_plan_tool(tools: list[dict] | None) -> str | None:
    if not tools:
        return None
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name") or ""
        if any(h in name.lower() for h in PLAN_TOOL_HINTS):
            return name
    return None


def is_plan_request(body: dict) -> tuple[bool, str | None]:
    from gateway.config import icr_mode

    plan_tool = find_plan_tool(body.get("tools"))
    mode = icr_mode()
    if mode == "off":
        return False, plan_tool
    if mode == "always":
        return True, plan_tool or "CreatePlan"
    return plan_tool is not None, plan_tool


def extract_task(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = content_to_text(msg.get("content")).strip()
        if text:
            return text
    return ""


def extract_attach_paths(messages: list[Message]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        text = content_to_text(msg.get("content"))
        for match in re.finditer(r"(?:^|[\s'\"`])(/(?:[\w.\-]+/)+[\w.\-]+\.\w+)", text):
            p = match.group(1)
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def extract_plan_todos(body: str) -> list[dict[str, str]]:
    todos: list[dict[str, str]] = []
    for line in body.splitlines():
        m = re.match(
            r"^#{1,3}\s+(?:Step\s+(\d+)\s*[:\—\-]\s*|(\d+)\.\s*)(.+)$",
            line.strip(),
        )
        if m:
            todos.append(
                {
                    "id": f"step-{len(todos) + 1}",
                    "content": m.group(3).strip(),
                    "status": "pending",
                }
            )
    if not todos:
        todos.append(
            {"id": "implement", "content": "Implement the plan below", "status": "pending"}
        )
    return todos


def build_plan_arguments(state, run_dir, task: str) -> dict[str, Any]:
    plan_md = state.current_best_generation or "(empty)"
    overview = plan_md.split("\n", 1)[0][:240]
    return {
        "name": task.strip().split("\n", 1)[0][:80] or "Implementation plan",
        "overview": overview,
        "plan": plan_md,
        "todos": extract_plan_todos(plan_md),
        "isProject": False,
    }


def completion_with_plan_tool(body: dict, *, plan_tool_name: str, arguments: dict) -> dict:
    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    model = body.get("model") or load_env().get("MODEL_NAME", "Qwen/Qwen3.6-27B")
    return {
        "id": f"chatcmpl-icr-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": plan_tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def stream_chunks(completion: dict) -> list[str]:
    cid = completion["id"]
    model = completion["model"]
    created = completion["created"]
    tool_calls = completion["choices"][0]["message"]["tool_calls"]
    chunks: list[str] = []

    def emit(obj: dict) -> None:
        chunks.append(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n")

    emit(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    emit(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_calls[0]["id"],
                                "type": "function",
                                "function": {"name": tool_calls[0]["function"]["name"], "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    args = tool_calls[0]["function"]["arguments"]
    emit(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args}}]},
                    "finish_reason": None,
                }
            ],
        }
    )
    emit(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
    )
    chunks.append("data: [DONE]\n\n")
    return chunks
