#!/usr/bin/env python3
"""Trim Cursor messages so ICR + vLLM stay within context budget."""

from __future__ import annotations

from typing import Any

from llm import Message, content_to_text, load_env


def clamp_max_tokens(value: Any, env: dict | None = None) -> int:
    """vLLM rejects max_tokens < 1; Cursor often sends 0."""
    env = env or load_env()
    default = int(env.get("REFINE_MAX_TOKENS", "2048"))
    try:
        n = int(value) if value is not None else 0
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        n = default
    return min(n, 8192)


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
    reserve = max(512, max_out) + 1024  # tools schema + small headroom
    return max(8192, max_len - reserve)


def conversation_token_count(messages: list[Message], tools: list | None = None) -> int:
    import json

    total = sum(message_tokens(m) for m in messages)
    if tools:
        total += estimate_tokens(json.dumps(tools))
    return total


def estimate_usage(
    messages: list[Message],
    *,
    tools: list | None = None,
    completion_text: str = "",
    tool_calls: list | None = None,
) -> dict[str, int]:
    import json

    prompt = conversation_token_count(messages, tools)
    if tool_calls:
        comp = estimate_tokens(json.dumps(tool_calls))
    else:
        comp = estimate_tokens(completion_text)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "total_tokens": prompt + comp,
    }


def ensure_completion_usage(completion: dict, payload: dict) -> dict:
    """Cursor context indicator needs non-zero usage; vLLM sometimes omits it."""
    usage = completion.get("usage") or {}
    if usage.get("prompt_tokens", 0) > 0:
        return completion
    msg = completion["choices"][0]["message"]
    usage = estimate_usage(
        payload.get("messages") or [],
        tools=payload.get("tools"),
        completion_text=content_to_text(msg.get("content")),
        tool_calls=msg.get("tool_calls"),
    )
    out = dict(completion)
    out["usage"] = usage
    return out


def wants_usage_in_stream(body: dict) -> bool:
    opts = body.get("stream_options") or {}
    return bool(opts.get("include_usage"))


def _trim_content(msg: Message, char_budget: int) -> Message:
    item = dict(msg)
    text = content_to_text(item.get("content"))
    if len(text) > char_budget:
        item["content"] = text[:char_budget] + "\n[... truncated ...]"
    return item


def trim_conversation_tail(
    messages: list[Message],
    budget_tokens: int,
    *,
    min_tail_messages: int | None = None,
) -> list[Message]:
    """Keep prefix system/dev + newest tail (incl. tool turns) within budget."""
    env = load_env()
    if min_tail_messages is None:
        min_tail_messages = int(env.get("MIN_TAIL_MESSAGES", "20"))

    if not messages or budget_tokens <= 0:
        return list(messages)

    copied = [dict(m) for m in messages]
    if conversation_token_count(copied) <= budget_tokens:
        return copied

    prefix: list[Message] = []
    rest: list[Message] = []
    for msg in copied:
        if msg.get("role") in ("system", "developer") and not rest:
            prefix.append(dict(msg))
        else:
            rest.append(dict(msg))

    prefix_cap = int(env.get("SYSTEM_PREFIX_MAX_TOKENS", "8192"))
    prefix_budget = min(budget_tokens // 3, prefix_cap)
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
        item = dict(msg)
        if tail_used + need > tail_budget:
            if len(kept_rev) < min_tail_messages:
                char_budget = max(400, (tail_budget - tail_used) * 4)
                item = _trim_content(item, char_budget)
                need = message_tokens(item)
            elif kept_rev:
                break
            else:
                item = _trim_content(item, max(400, tail_budget * 4))
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
    """Normalize + trim full Cursor request before vLLM (all Agent passthrough paths)."""
    env = env or load_env()
    raw_out = body.get("max_tokens", body.get("max_completion_tokens"))
    max_out = clamp_max_tokens(raw_out, env)
    budget = context_input_budget(env, max_output=max_out)
    messages = sanitize_tool_chain(trim_conversation_tail(body.get("messages") or [], budget))
    messages = ensure_non_empty_messages(messages)
    return build_vllm_chat_payload(body, messages=messages, max_tokens=max_out, stream=False)


# Fields vLLM accepts on /v1/chat/completions — drop Cursor extras (e.g. stream_options).
_VLLM_CHAT_FIELDS = (
    "model",
    "messages",
    "max_tokens",
    "stream",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "logprobs",
    "top_logprobs",
    "n",
    "user",
)


def build_vllm_chat_payload(
    body: dict,
    *,
    messages: list[Message],
    max_tokens: int,
    stream: bool,
) -> dict:
    """Build a vLLM-safe payload; strip stream_options and other invalid Cursor fields."""
    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    for key in _VLLM_CHAT_FIELDS:
        if key in ("model", "messages", "max_tokens", "stream"):
            continue
        if key in body:
            out[key] = body[key]
    if not stream:
        out.pop("stream_options", None)
    return out


def ensure_non_empty_messages(messages: list[Message]) -> list[Message]:
    if messages:
        return messages
    return [{"role": "user", "content": ""}]


def _strip_incomplete_tool_calls(fixed: list[Message]) -> None:
    for k in range(len(fixed) - 1, -1, -1):
        if fixed[k].get("role") == "assistant" and fixed[k].get("tool_calls"):
            item = dict(fixed[k])
            item.pop("tool_calls", None)
            if not item.get("content"):
                item["content"] = ""
            fixed[k] = item
            return


def sanitize_tool_chain(messages: list[Message]) -> list[Message]:
    """Drop orphan tool messages and broken tool_calls pairs vLLM rejects with 400."""
    if not messages:
        return []
    fixed: list[Message] = []
    pending: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            if pending and fixed:
                _strip_incomplete_tool_calls(fixed)
            item = dict(msg)
            tcs = item.get("tool_calls") or []
            pending = {tc.get("id") for tc in tcs if tc.get("id")}
            fixed.append(item)
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid and tid in pending:
                fixed.append(dict(msg))
                pending.discard(tid)
        else:
            if pending and fixed:
                _strip_incomplete_tool_calls(fixed)
            pending = set()
            fixed.append(dict(msg))
    if pending and fixed:
        _strip_incomplete_tool_calls(fixed)
    return fixed


def trim_icr_loop_seed(body: dict, env: dict | None = None) -> list[Message]:
    """Trim isolated ICR loop seed (system + latest user task only)."""
    from gateway.cursor_protocol import icr_loop_seed_from_cursor

    env = env or load_env()
    seed = icr_loop_seed_from_cursor(body)
    return trim_messages(seed, context_input_budget(env))


def trim_seed_from_cursor(body: dict, env: dict | None = None) -> list[Message]:
    """Alias — ICR loop uses isolated seed, not full chat history."""
    return trim_icr_loop_seed(body, env)
