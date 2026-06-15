#!/usr/bin/env python3
"""
OpenAI-compatible client for RunPod vLLM — same message shapes as OpenAI/Anthropic APIs.

Supports:
  - Text and multimodal content (image_url parts)
  - tool_calls / tool role messages
  - top_p, max_tokens, retries with backoff
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

Message = dict[str, Any]
ContentPart = dict[str, Any]

MAX_RETRIES = 2
INITIAL_DELAY_S = 2.0
BACKOFF_FACTOR = 1.5


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key in (
        "MODEL_NAME", "RUNPOD_API_KEY", "ICR_REPO", "CURSOR_PLAN_MAX_ITERATIONS",
        "REFINE_MEMORY_EVERY", "REFINE_POOL_SIZE", "REFINE_MAX_TOKENS",
        "REFINE_PYTHON_TOOLS", "REFINE_TEMPERATURE", "REFINE_TOP_P",
    ):
        if key not in env and os.environ.get(key):
            env[key] = os.environ[key]
    return env


def require(name: str, env: dict[str, str]) -> str:
    value = env.get(name) or os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in {ENV_FILE} or export it.")
    return value


def base_url(env: dict[str, str]) -> str:
    if os.environ.get("GATEWAY_ON_POD", "").lower() in ("1", "true", "yes"):
        upstream = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8000")
        return upstream.rstrip("/")
    explicit = env.get("RUNPOD_BASE_URL") or os.environ.get("RUNPOD_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    pod_id = require("RUNPOD_POD_ID", env)
    port = env.get("RUNPOD_PORT") or os.environ.get("RUNPOD_PORT", "8000")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def text_message(role: str, content: str) -> Message:
    return {"role": role, "content": content}


def user_message(content: str | list[ContentPart]) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: str, tool_calls: list[dict] | None = None) -> Message:
    msg: Message = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_message(tool_call_id: str, content: str, name: str | None = None) -> Message:
    msg: Message = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name:
        msg["name"] = name
    return msg


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                chunks.append("[image]")
        return "\n".join(chunks)
    return str(content)


def chat_completions(
    messages: list[Message],
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int = 8192,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    env = env or load_env()
    api_key = require("RUNPOD_API_KEY", env)
    model = model or env.get("MODEL_NAME", "Qwen/Qwen3.6-27B")
    url = f"{base_url(env)}/v1/chat/completions"

    full_messages: list[Message] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            last_error = RuntimeError(f"LLM HTTP {e.code}: {detail[:1200]}")
        except urllib.error.URLError as e:
            last_error = RuntimeError(f"LLM connection error: {e.reason}")

        if attempt < MAX_RETRIES:
            delay = INITIAL_DELAY_S * (BACKOFF_FACTOR ** attempt)
            time.sleep(delay)

    raise last_error or RuntimeError("LLM request failed")


def chat(
    messages: list[Message],
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int = 8192,
    env: dict[str, str] | None = None,
) -> str:
    body = chat_completions(
        messages,
        system=system,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        env=env,
    )
    choice = body["choices"][0]["message"]
    return content_to_text(choice.get("content"))


def parse_assistant_message(body: dict) -> Message:
    return body["choices"][0]["message"]


def new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
