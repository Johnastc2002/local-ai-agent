#!/usr/bin/env python3
"""Run ICR pipeline for all Cursor modes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from codebase_paths import codebase_root
from codebase_tool import codebase_tools_enabled
from attachments import load_seed_images
from gateway.cursor_context import build_icr_user_content, harvest_paths
from gateway.router import (
    build_plan_arguments,
    completion_with_content,
    completion_with_plan_tool,
    extract_task,
    inject_icr_context,
    stream_chunks,
    stream_text_chunks,
)
from icr_prompts import load_icr_prompts
from llm import load_env
from refine import RefineState, run_contextual_loop

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def run_icr_state(body: dict) -> RefineState:
    task = extract_task(body.get("messages") or [])
    if not task:
        raise ValueError("No user message in request")

    env = load_env()
    if codebase_tools_enabled(env) and codebase_root(env) is None:
        raise ValueError(
            "CODEBASE_ROOT is missing on pod. Clone your project, e.g.\n"
            "  git clone <repo> /workspace/bobot-xs-v1\n"
            "Set CODEBASE_ROOT and CODEBASE_HOST_ROOT in .env"
        )

    messages = body.get("messages") or []
    attach_paths = harvest_paths(messages, env)

    max_iterations = int(env.get("CURSOR_PLAN_MAX_ITERATIONS", "8"))
    memory_every = int(env.get("REFINE_MEMORY_EVERY", "10"))
    pool_size = int(env.get("REFINE_POOL_SIZE", "12"))
    temperature = float(env.get("REFINE_TEMPERATURE", "0.7"))
    top_p = float(env.get("REFINE_TOP_P", "0.95"))
    max_tokens = int(env.get("REFINE_MAX_TOKENS", "8192"))

    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    initial_content = build_icr_user_content(task, messages, env)
    seed_images = load_seed_images(attach_paths)
    prompts = load_icr_prompts()

    return run_contextual_loop(
        task,
        initial_content,
        seed_images,
        run_dir.name,
        run_dir,
        prompts=prompts,
        env=env,
        max_iterations=max_iterations,
        memory_every=memory_every,
        pool_size=pool_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def enrich_body_with_icr(body: dict, state: RefineState) -> dict:
    icr_text = state.current_best_generation or "(empty)"
    messages = inject_icr_context(body.get("messages") or [], icr_text)
    return {**body, "messages": messages}


def run_icr_plan(body: dict, *, plan_tool_name: str) -> dict:
    state = run_icr_state(body)
    task = extract_task(body.get("messages") or [])
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    args = build_plan_arguments(state, run_dir, task)
    return completion_with_plan_tool(body, plan_tool_name=plan_tool_name, arguments=args)


def run_icr_plan_stream(body: dict, *, plan_tool_name: str) -> list[str]:
    completion = run_icr_plan(body, plan_tool_name=plan_tool_name)
    return stream_chunks(completion)


def run_icr_answer(body: dict) -> dict:
    state = run_icr_state(body)
    text = state.current_best_generation or "(empty)"
    return completion_with_content(body, text)


def run_icr_answer_stream(body: dict) -> list[str]:
    completion = run_icr_answer(body)
    return stream_text_chunks(completion)
