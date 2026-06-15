#!/usr/bin/env python3
"""Trim Cursor messages so ICR + vLLM stay within context budget."""

from __future__ import annotations

from llm import Message, content_to_text, load_env


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def message_tokens(msg: Message) -> int:
    return estimate_tokens(content_to_text(msg.get("content")))


def context_input_budget(env: dict | None = None) -> int:
    """Tokens available for Cursor seed messages (reserve output + ICR system)."""
    env = env or load_env()
    max_len = int(env.get("MAX_MODEL_LEN", "32768"))
    max_out = int(env.get("REFINE_MAX_TOKENS", "2048"))
    reserve = max_out + 4096  # ICR agent system prompts + headroom
    return max(4096, max_len - reserve)


def trim_messages(messages: list[Message], budget_tokens: int) -> list[Message]:
    if not messages or budget_tokens <= 0:
        return list(messages)

    out: list[Message] = []
    used = 0

    # Latest user message is mandatory — trim its content if needed.
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in ("system", "user", "developer"):
            continue
        if i != last_user_idx and role == "user":
            continue

        text = content_to_text(msg.get("content"))
        need = estimate_tokens(text)
        remaining = budget_tokens - used

        if i == last_user_idx:
            if need > remaining:
                char_budget = max(800, remaining * 4)
                text = text[:char_budget] + "\n\n[... truncated for context limit ...]"
                need = estimate_tokens(text)
            out.append({"role": role, "content": text})
            used += need
            continue

        if need > remaining // 2:
            char_budget = max(400, (remaining // 2) * 4)
            text = text[:char_budget] + "\n\n[... truncated ...]"
            need = estimate_tokens(text)
        if need > remaining:
            continue
        out.append({"role": role, "content": text})
        used += need

    if not any(m.get("role") == "user" for m in out) and last_user_idx is not None:
        msg = dict(messages[last_user_idx])
        text = content_to_text(msg.get("content"))
        char_budget = min(len(text), budget_tokens * 4)
        out.append({"role": "user", "content": text[:char_budget]})
    return out


def trim_seed_from_cursor(body: dict, env: dict | None = None) -> list[Message]:
    from gateway.cursor_protocol import seed_messages_from_cursor

    env = env or load_env()
    seed = seed_messages_from_cursor(body)
    return trim_messages(seed, context_input_budget(env))
