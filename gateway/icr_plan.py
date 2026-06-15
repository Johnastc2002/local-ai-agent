#!/usr/bin/env python3
"""ICR orchestration — Cursor messages/tools, pause/resume for tool round-trips."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent_call import (
    AgentCallResult,
    CursorToolPause,
    PendingAgentCall,
    continue_contextual_agent,
)
from attachments import load_seed_images
from gateway.cursor_protocol import (
    completion_from_assistant,
    conversation_key,
    cursor_tools,
    extract_task,
    icr_request_options,
    is_agent_request,
    pending_tool_call_ids,
    seed_messages_from_cursor,
)
from gateway.icr_session import IcrSession, PendingAgent
from gateway.router import (
    build_plan_arguments,
    completion_with_content,
    completion_with_plan_tool,
    inject_icr_context,
    stream_chunks,
    stream_text_chunks,
)
from icr_prompts import load_icr_prompts
from llm import load_env
from refine import RefineState, resume_contextual_loop, run_contextual_loop

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


class IcrPaused(Exception):
    """Return Cursor tool_calls to the IDE."""

    def __init__(self, completion: dict, session: IcrSession):
        self.completion = completion
        self.session = session
        super().__init__("ICR paused for Cursor tools")


def _pending_from_call(p: PendingAgentCall) -> PendingAgent:
    return PendingAgent(
        agent_name=p.agent_name,
        system_prompt=p.system_prompt,
        working=p.working,
        session_id=p.session_id,
        transcript_parts=p.transcript_parts,
        loop_start=p.loop_start,
        temperature=p.temperature,
        seed_images=p.seed_images,
    )


def _pending_to_call(p: PendingAgent, body: dict, env: dict) -> PendingAgentCall:
    return PendingAgentCall(
        agent_name=p.agent_name,
        system_prompt=p.system_prompt,
        working=p.working,
        session_id=p.session_id,
        transcript_parts=p.transcript_parts,
        loop_start=p.loop_start,
        temperature=p.temperature,
        top_p=float(env.get("REFINE_TOP_P", "0.95")),
        max_tokens=int(env.get("REFINE_MAX_TOKENS", "8192")),
        seed_images=p.seed_images,
        cursor_tools=cursor_tools(body),
        body=body,
        request_options=icr_request_options(body),
    )


def _loop_kwargs(body: dict, env: dict) -> dict[str, Any]:
    messages = body.get("messages") or []
    task = extract_task(body)
    if not task:
        raise ValueError("No user message in request")

    attach_paths: list[Path] = []
    seed_images = load_seed_images(attach_paths)
    prompts = load_icr_prompts()
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")

    seed = seed_messages_from_cursor(body)
    if not seed:
        from llm import user_message

        seed = [user_message(task)]

    return {
        "task": task,
        "initial_user_content": task,
        "seed_messages": seed,
        "seed_images": seed_images,
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "prompts": prompts,
        "env": env,
        "max_iterations": int(env.get("CURSOR_PLAN_MAX_ITERATIONS", "8")),
        "memory_every": int(env.get("REFINE_MEMORY_EVERY", "10")),
        "pool_size": int(env.get("REFINE_POOL_SIZE", "12")),
        "temperature": float(env.get("REFINE_TEMPERATURE", "0.7")),
        "top_p": float(env.get("REFINE_TOP_P", "0.95")),
        "max_tokens": int(env.get("REFINE_MAX_TOKENS", "8192")),
        "cursor_tools": cursor_tools(body),
        "cursor_body": body,
    }


def _handle_pause(body: dict, exc: CursorToolPause, checkpoint: dict) -> None:
    key = conversation_key(body)
    tool_ids = [tc.get("id") for tc in exc.assistant.get("tool_calls") or [] if tc.get("id")]
    session = IcrSession(
        conversation_key=key,
        body=body,
        task=extract_task(body),
        run_id=checkpoint.get("run_id", ""),
        run_dir=checkpoint.get("run_dir", ""),
        pending=_pending_from_call(exc.pending),
        checkpoint=checkpoint,
        pending_tool_ids=tool_ids,
    )
    session.save()
    completion = completion_from_assistant(body, exc.assistant)
    raise IcrPaused(completion, session)


def run_icr_state(body: dict) -> RefineState:
    env = load_env()
    kwargs = _loop_kwargs(body, env)
    try:
        return run_contextual_loop(**kwargs)
    except CursorToolPause as exc:
        snapshot = getattr(exc, "loop_snapshot", {}) or {}
        _handle_pause(body, exc, snapshot)


def resume_icr_state(body: dict, session: IcrSession) -> RefineState:
    env = load_env()
    messages = body.get("messages") or []
    tool_ids = pending_tool_call_ids(messages)
    if not tool_ids:
        raise ValueError("No tool results to resume ICR")

    # Collect tool messages matching pending ids
    tool_msgs: list[dict] = []
    seen = set(session.pending_tool_ids)
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if msg.get("tool_call_id") in seen:
            tool_msgs.append(dict(msg))

    pending_call = _pending_to_call(session.pending, body, env)
    try:
        agent_result = continue_contextual_agent(pending_call, tool_msgs, env=env)
    except CursorToolPause as exc:
        snapshot = getattr(exc, "loop_snapshot", {}) or session.checkpoint
        _handle_pause(body, exc, snapshot)
    session.clear()
    try:
        return resume_contextual_loop(
            session.checkpoint,
            session.pending.agent_name,
            agent_result,
        )
    except CursorToolPause as exc:
        snapshot = getattr(exc, "loop_snapshot", {}) or {}
        _handle_pause(body, exc, snapshot)


def enrich_body_with_icr(body: dict, state: RefineState) -> dict:
    icr_text = state.current_best_generation or "(empty)"
    messages = inject_icr_context(body.get("messages") or [], icr_text)
    return {**body, "messages": messages}


def finish_icr(body: dict, state: RefineState, plan_tool: str | None) -> dict:
    if plan_tool:
        task = extract_task(body)
        run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
        args = build_plan_arguments(state, run_dir, task)
        return completion_with_plan_tool(body, plan_tool_name=plan_tool, arguments=args)
    return completion_with_content(body, state.current_best_generation or "(empty)")


def run_icr_plan(body: dict, *, plan_tool_name: str) -> dict:
    state = run_icr_state(body)
    task = extract_task(body)
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    args = build_plan_arguments(state, run_dir, task)
    return completion_with_plan_tool(body, plan_tool_name=plan_tool_name, arguments=args)


def run_icr_plan_stream(body: dict, *, plan_tool_name: str) -> list[str]:
    return stream_chunks(run_icr_plan(body, plan_tool_name=plan_tool_name))


def run_icr_answer(body: dict) -> dict:
    state = run_icr_state(body)
    return completion_with_content(body, state.current_best_generation or "(empty)")


def run_icr_answer_stream(body: dict) -> list[str]:
    return stream_text_chunks(run_icr_answer(body))
