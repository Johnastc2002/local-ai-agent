#!/usr/bin/env python3
"""
Call a contextual agent — uses Cursor's tools (read_file, grep, MCP) when provided.

When the model calls a Cursor tool, raises CursorToolPause so the gateway can return
tool_calls to Cursor. Cursor executes on the Mac and sends tool results back.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from gateway.cursor_protocol import (
    PYTHON_TOOL_NAME,
    icr_request_options,
    is_cursor_managed_tool,
    tool_name,
)
from llm import (
    Message,
    assistant_message,
    chat_completions,
    content_to_text,
    new_tool_call_id,
    parse_assistant_message,
    tool_message,
)
from python_tool import (
    PYTHON_TOOL_DEF,
    PythonSandbox,
    format_tool_result,
)

MAX_TOOL_TURNS = 32


@dataclass
class AgentCallResult:
    text: str
    prompt_text: str
    final_text: str
    loop_messages: list[Message] = field(default_factory=list)


@dataclass
class PendingAgentCall:
    agent_name: str
    system_prompt: str
    working: list[Message]
    session_id: str
    transcript_parts: list[str]
    loop_start: int
    temperature: float
    top_p: float | None
    max_tokens: int
    seed_images: list[dict]
    cursor_tools: list[dict]
    body: dict
    request_options: dict[str, Any]


class CursorToolPause(Exception):
    """Model requested a Cursor-managed tool — return to Cursor for execution."""

    def __init__(self, assistant: Message, pending: PendingAgentCall):
        self.assistant = assistant
        self.pending = pending
        super().__init__("Cursor tool pause")


def _python_tools_enabled(env: dict[str, str]) -> bool:
    val = env.get("REFINE_PYTHON_TOOLS", os.environ.get("REFINE_PYTHON_TOOLS", "false"))
    return val.lower() in ("1", "true", "yes", "on")


def _parse_tool_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _tools_for_call(cursor_tools: list[dict], env: dict[str, str]) -> list[dict]:
    tools = list(cursor_tools)
    if _python_tools_enabled(env):
        names = {tool_name(t) for t in tools}
        if PYTHON_TOOL_NAME not in names:
            tools.append(PYTHON_TOOL_DEF)
    return tools


def _run_tool_loop(
    pending: PendingAgentCall,
    *,
    env: dict[str, str],
    model: str | None,
) -> AgentCallResult:
    working = list(pending.working)
    tools = _tools_for_call(pending.cursor_tools, env)
    sandbox = PythonSandbox(pending.session_id)
    transcript_parts = list(pending.transcript_parts)
    loop_start = pending.loop_start
    final_text = ""
    should_seed = True

    for _turn in range(MAX_TOOL_TURNS):
        body = chat_completions(
            working,
            system=pending.system_prompt,
            model=model,
            temperature=pending.temperature,
            top_p=pending.top_p,
            max_tokens=pending.max_tokens,
            tools=tools or None,
            tool_choice=pending.request_options.get("tool_choice", "auto" if tools else None),
            parallel_tool_calls=pending.request_options.get("parallel_tool_calls"),
            env=env,
        )
        response = parse_assistant_message(body)
        working.append(response)

        model_text = content_to_text(response.get("content"))
        tool_calls = response.get("tool_calls") or []

        if model_text:
            transcript_parts.append(model_text)

        if not tool_calls:
            final_text = model_text
            break

        cursor_calls: list[dict] = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            if is_cursor_managed_tool(name, pending.body):
                cursor_calls.append(tc)
                continue

            args = _parse_tool_args(fn.get("arguments"))
            tc_id = tc.get("id") or new_tool_call_id()

            if name == PYTHON_TOOL_NAME:
                code = str(args.get("code", "")).strip()
                if should_seed and pending.seed_images:
                    sandbox.seed_images(pending.seed_images)
                    should_seed = False
                if not code:
                    result = {
                        "ok": False,
                        "exitCode": 1,
                        "stdout": "",
                        "stderr": "",
                        "error": "Missing or invalid Python code in tool call.",
                        "durationMs": 0,
                    }
                else:
                    result = sandbox.execute(code)
                tool_content = format_tool_result(result)
                transcript_parts.append(f"```python\n{code}\n```\n\n{tool_content}")
                working.append(tool_message(tc_id, tool_content, name=name))
            else:
                working.append(
                    tool_message(
                        tc_id,
                        f"Tool not available in ICR loop: {name}",
                        name=name,
                    )
                )

        if cursor_calls:
            pause_assistant = dict(response)
            pause_assistant["tool_calls"] = cursor_calls
            if model_text and len(cursor_calls) < len(tool_calls):
                pause_assistant["content"] = model_text
            raise CursorToolPause(
                pause_assistant,
                PendingAgentCall(
                    agent_name=pending.agent_name,
                    system_prompt=pending.system_prompt,
                    working=working,
                    session_id=pending.session_id,
                    transcript_parts=transcript_parts,
                    loop_start=loop_start,
                    temperature=pending.temperature,
                    top_p=pending.top_p,
                    max_tokens=pending.max_tokens,
                    seed_images=pending.seed_images,
                    cursor_tools=pending.cursor_tools,
                    body=pending.body,
                    request_options=pending.request_options,
                ),
            )

    if not final_text:
        raise RuntimeError(
            f"{pending.agent_name} exceeded {MAX_TOOL_TURNS} tool turns without a final response."
        )

    display = "\n\n".join(transcript_parts).strip() or final_text
    return AgentCallResult(
        text=display,
        prompt_text=final_text,
        final_text=final_text,
        loop_messages=working[loop_start:],
    )


def call_contextual_agent(
    agent_name: str,
    messages: list[Message],
    system_prompt: str,
    *,
    session_id: str,
    seed_images: list[dict],
    cursor_tools: list[dict] | None = None,
    cursor_body: dict | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int = 8192,
    env: dict[str, str] | None = None,
) -> AgentCallResult:
    from llm import chat, load_env

    env = env or load_env()
    body = cursor_body or {"tools": cursor_tools or []}
    tools = _tools_for_call(cursor_tools or [], env)
    req_opts = icr_request_options(body)

    if not tools:
        text = chat(
            messages,
            system=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            env=env,
        )
        loop = [assistant_message(text)]
        return AgentCallResult(text=text, prompt_text=text, final_text=text, loop_messages=loop)

    pending = PendingAgentCall(
        agent_name=agent_name,
        system_prompt=system_prompt,
        working=list(messages),
        session_id=session_id,
        transcript_parts=[],
        loop_start=len(messages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed_images=seed_images,
        cursor_tools=cursor_tools or [],
        body=body,
        request_options=req_opts,
    )
    return _run_tool_loop(pending, env=env, model=model)


def continue_contextual_agent(
    pending: PendingAgentCall,
    tool_messages: list[Message],
    *,
    env: dict[str, str] | None = None,
    model: str | None = None,
) -> AgentCallResult:
    """Resume after Cursor executed tools and sent tool role messages."""
    from llm import load_env

    env = env or load_env()
    pending.working = list(pending.working) + list(tool_messages)
    return _run_tool_loop(pending, env=env, model=model)
