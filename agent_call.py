#!/usr/bin/env python3
"""
Call a contextual agent (mirrors ICR callContextualAgent + optional Python tool loop).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from llm import (
    Message,
    assistant_message,
    chat_completions,
    content_to_text,
    new_tool_call_id,
    parse_assistant_message,
    text_message,
    tool_message,
    user_message,
)
from python_tool import (
    PYTHON_TOOL_DEF,
    PYTHON_TOOL_NAME,
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


def _python_tools_enabled(env: dict[str, str]) -> bool:
    val = env.get("REFINE_PYTHON_TOOLS", os.environ.get("REFINE_PYTHON_TOOLS", "true"))
    return val.lower() in ("1", "true", "yes", "on")


def call_contextual_agent(
    agent_name: str,
    messages: list[Message],
    system_prompt: str,
    *,
    session_id: str,
    seed_images: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int = 8192,
    env: dict[str, str] | None = None,
) -> AgentCallResult:
    from llm import load_env

    env = env or load_env()
    working = list(messages)

    if not _python_tools_enabled(env):
        from llm import chat

        text = chat(
            working,
            system=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            env=env,
        )
        loop = [assistant_message(text)]
        return AgentCallResult(text=text, prompt_text=text, final_text=text, loop_messages=loop)

    sandbox = PythonSandbox(session_id)
    tools = [PYTHON_TOOL_DEF]
    transcript_parts: list[str] = []
    loop_start = len(working)
    final_text = ""
    should_seed = True

    for _turn in range(MAX_TOOL_TURNS):
        body = chat_completions(
            working,
            system=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
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

        for tc in tool_calls:
            fn = tc.get("function") or {}
            if fn.get("name") != PYTHON_TOOL_NAME:
                working.append(tool_message(
                    tc.get("id") or new_tool_call_id(),
                    f"Unknown tool: {fn.get('name')}",
                    name=fn.get("name"),
                ))
                continue

            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                code = str(args.get("code", "")).strip()
            except json.JSONDecodeError:
                code = ""

            if should_seed and seed_images:
                sandbox.seed_images(seed_images)
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

            working.append(tool_message(
                tc.get("id") or new_tool_call_id(),
                tool_content,
                name=PYTHON_TOOL_NAME,
            ))

    if not final_text:
        raise RuntimeError(
            f"{agent_name} exceeded {MAX_TOOL_TURNS} Python tool turns without a final response."
        )

    display = "\n\n".join(transcript_parts).strip() or final_text
    loop_messages = working[loop_start:]
    return AgentCallResult(
        text=display,
        prompt_text=final_text,
        final_text=final_text,
        loop_messages=loop_messages,
    )
