#!/usr/bin/env python3
"""
Call a contextual agent (mirrors ICR callContextualAgent + tool loop).

Tools: read_file / grep / list_dir (codebase) + python_virtual_filesystem.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from codebase_tool import (
    CODEBASE_TOOL_DEFS,
    CODEBASE_TOOL_NAMES,
    CodebaseTools,
    codebase_tools_enabled,
    format_codebase_result,
)
from llm import (
    Message,
    chat_completions,
    content_to_text,
    new_tool_call_id,
    parse_assistant_message,
    tool_message,
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


def _tools_for_agent(env: dict[str, str]) -> list[dict]:
    tools: list[dict] = []
    if codebase_tools_enabled(env):
        tools.extend(CODEBASE_TOOL_DEFS)
    if _python_tools_enabled(env):
        tools.append(PYTHON_TOOL_DEF)
    return tools


def _parse_tool_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


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
    from llm import assistant_message, chat, load_env

    env = env or load_env()
    working = list(messages)
    tools = _tools_for_agent(env)

    if not tools:
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
    codebase = CodebaseTools(env) if codebase_tools_enabled(env) else None
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
            name = fn.get("name") or ""
            args = _parse_tool_args(fn.get("arguments"))
            tc_id = tc.get("id") or new_tool_call_id()

            if name == PYTHON_TOOL_NAME:
                code = str(args.get("code", "")).strip()
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

            elif name in CODEBASE_TOOL_NAMES and codebase is not None:
                result = codebase.dispatch(name, args)
                tool_content = format_codebase_result(result)
                transcript_parts.append(f"[{name}]\n{tool_content}")

            else:
                tool_content = f"Unknown or unavailable tool: {name}"
                transcript_parts.append(tool_content)

            working.append(tool_message(tc_id, tool_content, name=name))

    if not final_text:
        raise RuntimeError(
            f"{agent_name} exceeded {MAX_TOOL_TURNS} tool turns without a final response."
        )

    display = "\n\n".join(transcript_parts).strip() or final_text
    loop_messages = working[loop_start:]
    return AgentCallResult(
        text=display,
        prompt_text=final_text,
        final_text=final_text,
        loop_messages=loop_messages,
    )
