#!/usr/bin/env python3
"""Trim Cursor messages so ICR + vLLM stay within context budget."""

from __future__ import annotations

from llm import Message, content_to_text, load_env


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def message_tokens(msg: Message) -> int:
    tokens = estimate_tokens(content_to_text(msg.get("content")))
    tool_calls = msg.get("tool_calls") or []
    tokens += sum(estimate_tokens(str(tc)) for tc in tool_calls)
    return tokens


def max_model_len(env: dict | None = None) -> int:
    env = env or load_env()
    return int(env.get("MAX_MODEL_LEN", "32768"))


def context_input_budget(env: dict | None = None, *, max_output: int | None = None) -> int:
    """Tokens available for input messages."""
    env = env or load_env()
    max_len = max_model_len(env)
    max_out = max_output if max_output is not None else int(env.get("REFINE_MAX_TOKENS", "2048"))
    reserve = max(512, max_out) + 2048  # tools schema + headroom
    return max(4096, max_len - reserve)


def _trim_content(msg: Message, char_budget: int) -> Message:
    item = dict(msg)
    text = content_to_text(item.get("content"))
    if len(text) > char_budget:
        item["content"] = text[:char_budget] + "\n[... truncated ...]"
    return item


def trim_conversation_tail(messages: list[Message], budget_tokens: int) -> list[Message]:
    """Keep prefix system/dev + newest tail (incl. tool turns) within budget."""
    if not messages or budget_tokens <= 0:
        return list(messages)

    prefix: list[Message] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer") and not rest:
            prefix.append(dict(msg))
        else:
            rest.append(dict(msg))

    prefix_budget = min(budget_tokens // 4, 4096)
    trimmed_prefix: list[Message] = []
    used = 0
    for msg in prefix:
        need = message_tokens(msg)
        if need > prefix_budget - used:
            msg = _trim_content(msg, max(400, (prefix_budget - used) * 4))
            need = message_tokens(msg)
        trimmed_prefix.append(msg)
        used += need

    tail_budget = budget_tokens - used
    kept_rev: list[Message] = []
    tail_used = 0
    for msg in reversed(rest):
        need = message_tokens(msg)
        if tail_used + need > tail_budget and kept_rev:
            break
        item = dict(msg)
        if tail_used + need > tail_budget:
            item = _trim_content(item, max(400, (tail_budget - tail_used) * 4))
            need = message_tokens(item)
        kept_rev.append(item)
        tail_used += need

    kept_rev.reverse()
    if not kept_rev and rest:
        kept_rev = [_trim_content(rest[-1], tail_budget * 4)]
    return trimmed_prefix + kept_rev


def trim_messages(messages: list[Message], budget_tokens: int) -> list[Message]:
    """Seed messages for ICR (system/user/dev, latest user)."""
    if not messages or budget_tokens <= 0:
        return list(messages)

    out: list[Message] = []
    used = 0

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


def trim_agent_history(
    messages: list[Message],
    budget_tokens: int,
    *,
    system_reserve: int = 0,
) -> list[Message]:
    """Keep the newest messages (incl. assistant) within token budget."""
    budget = max(1024, budget_tokens - system_reserve)
    return trim_conversation_tail(messages, budget)


def trim_body_for_vllm(body: dict, env: dict | None = None) -> dict:
    """Trim full Cursor request before vLLM (Agent passthrough after ICR)."""
    env = env or load_env()
    max_out = body.get("max_tokens")
    if not max_out or int(max_out) < 1:
        max_out = int(env.get("REFINE_MAX_TOKENS", "2048"))
    max_out = min(int(max_out), 8192)
    budget = context_input_budget(env, max_output=max_out)
    messages = trim_conversation_tail(body.get("messages") or [], budget)
    return {**body, "messages": messages, "max_tokens": max_out, "stream": False}


def trim_seed_from_cursor(body: dict, env: dict | None = None) -> list[Message]:
    from gateway.cursor_protocol import seed_messages_from_cursor

    env = env or load_env()
    seed = seed_messages_from_cursor(body)
    return trim_messages(seed, context_input_budget(env))
