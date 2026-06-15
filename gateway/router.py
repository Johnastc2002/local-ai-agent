#!/usr/bin/env python3
"""Route Cursor requests through ICR and build OpenAI responses."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from llm import content_to_text, load_env
from gateway.cursor_protocol import find_plan_tool, is_agent_request as _is_agent_request

ICR_CONTEXT_MARKER = "[ICR refined context]"
ICR_MAX_CHARS = 8000

Message = dict[str, Any]


def is_user_turn(body: dict) -> bool:
    messages = body.get("messages") or []
    return bool(messages) and messages[-1].get("role") == "user"


def is_agent_request(body: dict) -> bool:
    """Agent with Cursor tools, but not Plan (CreatePlan)."""
    return _is_agent_request(body)


def inject_icr_context(messages: list[Message], icr_text: str) -> list[Message]:
    """
    Attach ICR output as a developer message before the latest user turn.
    Do NOT append to the last user message — that bloats it and trim drops prior turns.
    """
    out: list[Message] = [dict(m) for m in messages]
    icr = icr_text.strip()
    if len(icr) > ICR_MAX_CHARS:
        icr = icr[:ICR_MAX_CHARS] + "\n[... ICR truncated ...]"
    block = f"{ICR_CONTEXT_MARKER}\n{icr}"

    insert_at = len(out)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            insert_at = i
            break
    out.insert(insert_at, {"role": "developer", "content": block})
    return out


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


def completion_with_content(body: dict, content: str) -> dict:
    model = body.get("model") or load_env().get("MODEL_NAME", "Qwen/Qwen3.6-27B")
    return {
        "id": f"chatcmpl-icr-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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


def stream_text_chunks(completion: dict) -> list[str]:
    cid = completion["id"]
    model = completion["model"]
    created = completion["created"]
    text = completion["choices"][0]["message"]["content"] or ""
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
    step = max(1, len(text) // 40)
    for i in range(0, len(text), step):
        emit(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text[i : i + step]},
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
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    chunks.append("data: [DONE]\n\n")
    return chunks
