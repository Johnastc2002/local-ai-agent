#!/usr/bin/env python3
"""
Iterative Contextual Refinement — faithful port of ICR ContextualCore.ts

Loop (while running):
  1. Main Generator
  2. Iterative Agent (critique)
  3. Strategic Pool Agent
  4. Feed critique + pool → Main Generator; memory every 10 turns
  Exit: Strategic Pool outputs <<<Exit>>>

Uses full prompts from Iterative-Contextual-Refinements and OpenAI-compatible
multimodal + tool API (RunPod vLLM).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent_call import AgentCallResult, CursorToolPause, call_contextual_agent
from attachments import build_initial_user_content, load_seed_images
from icr_prompts import load_icr_prompts
from llm import Message, user_message
from python_tool import session_id_for_agent

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

_stop_requested = False


def _handle_sigint(_signum, _frame):
    global _stop_requested
    _stop_requested = True
    print("\nStopping after current step...", file=sys.stderr)


signal.signal(signal.SIGINT, _handle_sigint)


@dataclass
class RefineState:
    task: str
    started_at: str
    iteration_count: int = 0
    initial_main_generation: str = ""
    current_best_generation: str = ""
    current_best_suggestions: str = ""
    current_strategic_pool: str = ""
    current_memory: str = ""
    memory_snapshots: list[dict] = field(default_factory=list)
    all_iterative_suggestions: list[str] = field(default_factory=list)
    all_strategic_pools: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    status: str = "running"
    exit_reason: str | None = None

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(json.dumps(asdict(self), indent=2))
        if self.current_best_generation:
            (run_dir / "latest.md").write_text(self.current_best_generation)


def _prompt_text(result: AgentCallResult) -> str:
    return result.prompt_text or result.text


def _loop_snapshot(
    *,
    phase: str,
    task: str,
    state: RefineState,
    main_msgs: list[Message],
    iterative_msgs: list[Message],
    strategic_msgs: list[Message],
    iteration: int,
    turns_since_condense: int,
    run_id: str,
    run_dir: Path,
    prompts: dict[str, str],
    env: dict[str, str],
    max_iterations: int,
    memory_every: int,
    pool_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed_images: list[dict],
    cursor_tools: list[dict] | None,
    cursor_body: dict | None,
    seed_messages: list[Message] | None,
    main_prompt_text: str = "",
    suggestions_prompt_text: str = "",
    main_loop_messages: list[Message] | None = None,
    suggestions_loop_messages: list[Message] | None = None,
    strat_loop_messages: list[Message] | None = None,
) -> dict:
    return {
        "phase": phase,
        "task": task,
        "state": asdict(state),
        "main_msgs": main_msgs,
        "iterative_msgs": iterative_msgs,
        "strategic_msgs": strategic_msgs,
        "iteration": iteration,
        "turns_since_condense": turns_since_condense,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "prompts": prompts,
        "env": env,
        "max_iterations": max_iterations,
        "memory_every": memory_every,
        "pool_size": pool_size,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed_images": seed_images,
        "cursor_tools": cursor_tools or [],
        "cursor_body": cursor_body or {},
        "seed_messages": seed_messages or [],
        "main_prompt_text": main_prompt_text,
        "suggestions_prompt_text": suggestions_prompt_text,
        "main_loop_messages": main_loop_messages or [],
        "suggestions_loop_messages": suggestions_loop_messages or [],
        "strat_loop_messages": strat_loop_messages or [],
    }


def _agent(
    phase: str,
    snapshot_kwargs: dict,
    *args,
    **kwargs,
) -> AgentCallResult:
    from gateway.context_trim import context_input_budget, estimate_tokens, trim_agent_history
    from llm import load_env

    if len(args) >= 3:
        messages = list(args[1])
        system_prompt = args[2]
        env = kwargs.get("env") or load_env()
        budget = context_input_budget(env)
        messages = trim_agent_history(
            messages,
            budget,
            system_reserve=estimate_tokens(system_prompt) + 512,
        )
        args = (args[0], messages, *args[2:])
    try:
        return call_contextual_agent(*args, **kwargs)
    except CursorToolPause as exc:
        exc.loop_snapshot = _loop_snapshot(phase=phase, **snapshot_kwargs)
        raise


def run_contextual_loop(
    task: str,
    initial_user_content: str | list,
    seed_images: list[dict],
    run_id: str,
    run_dir: Path,
    *,
    prompts: dict[str, str],
    env: dict[str, str],
    max_iterations: int,
    memory_every: int,
    pool_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed_messages: list[Message] | None = None,
    cursor_tools: list[dict] | None = None,
    cursor_body: dict | None = None,
    resume_checkpoint: dict | None = None,
    resume_agent: tuple[str, AgentCallResult] | None = None,
) -> RefineState:
    if resume_checkpoint:
        ck = resume_checkpoint
        state = RefineState(**ck["state"])
        main_msgs = list(ck["main_msgs"])
        iterative_msgs = list(ck["iterative_msgs"])
        strategic_msgs = list(ck["strategic_msgs"])
        iteration = ck["iteration"]
        turns_since_condense = ck["turns_since_condense"]
        run_id = ck["run_id"]
        run_dir = Path(ck["run_dir"])
        prompts = ck["prompts"]
        env = ck["env"]
        max_iterations = ck["max_iterations"]
        memory_every = ck["memory_every"]
        pool_size = ck["pool_size"]
        temperature = ck["temperature"]
        top_p = ck["top_p"]
        max_tokens = ck["max_tokens"]
        seed_images = ck["seed_images"]
        cursor_tools = ck.get("cursor_tools")
        cursor_body = ck.get("cursor_body")
        seed_messages = ck.get("seed_messages")
        task = ck["task"]
    else:
        state = RefineState(task=task, started_at=datetime.now(timezone.utc).isoformat())
        iteration = 0
        turns_since_condense = 0

    if not resume_checkpoint:
        if seed_messages:
            main_msgs = list(seed_messages)
            iterative_msgs = list(seed_messages)
            strategic_msgs = list(seed_messages)
        else:
            initial_msg = user_message(initial_user_content)
            main_msgs = [initial_msg]
            iterative_msgs = [initial_msg]
            strategic_msgs = [initial_msg]

    resuming = resume_agent is not None
    active_ck = resume_checkpoint

    while not _stop_requested:
        if resuming:
            resuming = False
        else:
            iteration += 1
            if max_iterations > 0 and iteration > max_iterations:
                state.status = "completed"
                state.exit_reason = f"max iterations ({max_iterations})"
                break

        state.iteration_count = iteration
        print(f"\n{'='*60}\nIteration {iteration}\n{'='*60}", file=sys.stderr)

        snap = lambda phase, **extra: _loop_snapshot(  # noqa: E731
            phase=phase,
            task=task,
            state=state,
            main_msgs=main_msgs,
            iterative_msgs=iterative_msgs,
            strategic_msgs=strategic_msgs,
            iteration=iteration,
            turns_since_condense=turns_since_condense,
            run_id=run_id,
            run_dir=run_dir,
            prompts=prompts,
            env=env,
            max_iterations=max_iterations,
            memory_every=memory_every,
            pool_size=pool_size,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed_images=seed_images,
            cursor_tools=cursor_tools,
            cursor_body=cursor_body,
            seed_messages=seed_messages,
            **extra,
        )

        resume_phase = resume_agent[0] if resume_agent else None
        skip_main = resume_phase in (
            "Iterative Agent",
            "Strategic Pool Agent",
            "Memory Agent",
        )

        if resume_phase == "Main Generator":
            main_result = resume_agent[1]
            resume_agent = None
        elif not skip_main:
            print("[1/4] Main Generator...", file=sys.stderr)
            main_result = _agent(
                "Main Generator",
                snap("Main Generator"),
                "Main Generator",
                main_msgs,
                prompts["main_generator"],
                session_id=session_id_for_agent(run_id, "main-generator"),
                seed_images=seed_images,
                cursor_tools=cursor_tools,
                cursor_body=cursor_body,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                env=env,
            )

        if skip_main and active_ck:
            main_prompt_text = active_ck.get("main_prompt_text", "")
            main_loop_messages = active_ck.get("main_loop_messages", [])
        elif not skip_main:
            main_generation = main_result.text
            main_prompt_text = _prompt_text(main_result)
            main_loop_messages = main_result.loop_messages or [{"role": "assistant", "content": main_generation}]

            if iteration == 1:
                state.initial_main_generation = main_generation
            state.current_best_generation = main_generation
            state.messages.append({
                "role": "main_generator",
                "iteration": iteration,
                "content": main_generation,
            })
            state.save(run_dir)

            main_msgs.extend(main_loop_messages)
            iterative_msgs.extend(main_loop_messages)

        if _stop_requested:
            break

        skip_critique = resume_phase == "Strategic Pool Agent"

        if resume_phase == "Iterative Agent":
            critique_result = resume_agent[1]
            resume_agent = None
        elif not skip_critique:
            print("[2/4] Iterative Agent...", file=sys.stderr)
            iterative_msgs.append(user_message(
                "Please critique the solution and tool executions you just generated above. "
                "If no tools were used, critique the generation text."
            ))
            critique_result = _agent(
                "Iterative Agent",
                snap("Iterative Agent", main_prompt_text=main_prompt_text, main_loop_messages=main_loop_messages),
                "Iterative Agent",
                iterative_msgs,
                prompts["iterative_agent"],
                session_id=session_id_for_agent(run_id, "iterative-agent"),
                seed_images=seed_images,
                cursor_tools=cursor_tools,
                cursor_body=cursor_body,
                temperature=max(0.2, temperature - 0.3),
                top_p=top_p,
                max_tokens=max_tokens,
                env=env,
            )
        if skip_critique and active_ck:
            suggestions_prompt_text = active_ck.get("suggestions_prompt_text", "")
            suggestions_loop_messages = active_ck.get("suggestions_loop_messages", [])
        elif not skip_critique:
            suggestions = critique_result.text
            suggestions_prompt_text = _prompt_text(critique_result)
            suggestions_loop_messages = critique_result.loop_messages or [{"role": "assistant", "content": suggestions}]

            state.current_best_suggestions = suggestions
            state.all_iterative_suggestions.append(suggestions)
            state.messages.append({"role": "iterative_agent", "iteration": iteration, "content": suggestions})
            iterative_msgs.extend(suggestions_loop_messages)
            state.save(run_dir)

        if _stop_requested:
            break

        if resume_phase == "Strategic Pool Agent":
            pool_result = resume_agent[1]
            resume_agent = None
        else:
            print("[3/4] Strategic Pool Agent...", file=sys.stderr)
            strat_observation = "\n".join([
                "## Observation: Current Main Generation",
                main_prompt_text,
                "",
                "## Observation: Solution Critique",
                suggestions_prompt_text,
                "",
                "## Deep Analysis Task",
                "Study the solution and tool executions above carefully:",
                "- What unexplored strategic territories remain?",
                "",
                "## Strategic Pool Evolution Task",
                f"Based on your deep observation, UPDATE and EVOLVE your strategic pool with {pool_size} strategies:",
                "- If a strategy was well-explored, replace it with something more orthogonal",
                "- If a strategy was ignored or poorly attempted, keep it but reframe more compellingly",
                "- If they're fixated on one approach, propose radical departures",
                "- Progressively expand into more unexpected domains with each iteration",
                "- Focus on what they HAVEN'T tried, not what they have",
                "",
                f"Generate {pool_size} evolved strategies that push exploration further.",
            ])
            strategic_msgs.append(user_message(strat_observation))

            pool_result = _agent(
                "Strategic Pool Agent",
                snap(
                    "Strategic Pool Agent",
                    main_prompt_text=main_prompt_text,
                    main_loop_messages=main_loop_messages,
                    suggestions_prompt_text=suggestions_prompt_text,
                    suggestions_loop_messages=suggestions_loop_messages,
                ),
                "Strategic Pool Agent",
                strategic_msgs,
                prompts["strategic_pool"],
                session_id=session_id_for_agent(run_id, "strategic-pool"),
                seed_images=seed_images,
                cursor_tools=cursor_tools,
                cursor_body=cursor_body,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                env=env,
            )
        strategic_pool = pool_result.text
        strategic_pool_prompt_text = _prompt_text(pool_result)
        strat_loop_messages = pool_result.loop_messages or [{"role": "assistant", "content": strategic_pool}]

        if (pool_result.final_text or strategic_pool).strip() == "<<<Exit>>>":
            state.status = "completed"
            state.exit_reason = "Strategic Pool: Solution Critique found no flaws 3 times consecutively"
            state.messages.append({"role": "system", "content": state.exit_reason})
            state.save(run_dir)
            print("[exit] <<<Exit>>>", file=sys.stderr)
            break

        state.current_strategic_pool = strategic_pool
        state.all_strategic_pools.append(strategic_pool)
        state.messages.append({"role": "strategic_pool_agent", "iteration": iteration, "content": strategic_pool})
        strategic_msgs.extend(strat_loop_messages)

        # --- 4: LOOP PREP (L375-480) ---
        combined_critique = "\n".join([
            suggestions_prompt_text,
            "",
            "---",
            "",
            "## Strategic Pool",
            f"The following {pool_size} strategies have been generated to expand your solution exploration:",
            "",
            strategic_pool_prompt_text,
        ])
        main_msgs.append(user_message(combined_critique))

        turns_since_condense += 1
        if turns_since_condense >= memory_every:
            print("[memory] Memory Agent...", file=sys.stderr)
            complete = [
                m for m in state.messages
                if m.get("role") in ("main_generator", "iterative_agent")
            ]
            memory_parts = [
                f"Initial User Request:\n{task}",
                "",
            ]
            for idx, snap in enumerate(state.memory_snapshots):
                memory_parts.append(
                    f"Memory V{idx + 1}:\n{snap['memory']}\n\n"
                    f"Final Main Generation after Memory V{idx + 1}:\n{snap['finalGeneration']}"
                )
                memory_parts.append("")
            memory_parts.append("Recent Iterations to Analyze:")
            for m in complete:
                label = "Main Generation" if m["role"] == "main_generator" else "Critique"
                memory_parts.append(
                    f"[Iteration {m['iteration']}] {label}:\n{m['content']}"
                )
            memory_parts.append("")
            memory_parts.append(
                "Task: Create an evolving memory document summarizing what worked and "
                "what didn't based on these iteration texts."
            )

            mem_result = _agent(
                "Memory Agent",
                snap("Memory Agent"),
                "Memory Agent",
                [user_message("\n".join(memory_parts))],
                prompts["memory_agent"],
                session_id=session_id_for_agent(run_id, "memory-agent"),
                seed_images=[],
                cursor_tools=cursor_tools,
                cursor_body=cursor_body,
                temperature=0.3,
                top_p=top_p,
                max_tokens=max_tokens,
                env=env,
            )
            memory_text = mem_result.text
            state.current_memory = memory_text
            state.memory_snapshots.append({
                "memory": memory_text,
                "finalGeneration": state.current_best_generation,
                "condensePoint": iteration,
            })
            state.messages.append({"role": "memory_agent", "iteration": iteration, "content": memory_text})

            memory_block = user_message(f"Memory Summary (What worked and what didn't):\n{memory_text}")
            initial_req = user_message(f"Initial User Request:\n{task}")

            main_msgs = [
                initial_req,
                memory_block,
                user_message("Latest Context:\n"),
                *main_loop_messages,
                user_message(combined_critique),
            ]
            iterative_msgs = [
                initial_req,
                memory_block,
                user_message("Latest Context:\n"),
                *main_loop_messages,
                *suggestions_loop_messages,
            ]
            strategic_msgs = [
                initial_req,
                memory_block,
                user_message("Latest Strategic Pool Context:\n"),
                *strat_loop_messages,
            ]
            turns_since_condense = 0
            state.save(run_dir)

        time.sleep(1.0)

        main_msgs.append(user_message(
            "Now implement the next iteration of the solution based on the critique and "
            "the strategies you just generated above. Ensure you fully resolve the issues "
            "raised in the critique."
        ))

        active_ck = None
        resume_agent = None

    if _stop_requested and state.status == "running":
        state.status = "stopped"
        state.exit_reason = "user interrupt"

    state.save(run_dir)
    return state


def resume_contextual_loop(
    checkpoint: dict,
    agent_name: str,
    agent_result: AgentCallResult,
) -> RefineState:
    return run_contextual_loop(
        task=checkpoint["task"],
        initial_user_content=checkpoint["task"],
        seed_images=checkpoint["seed_images"],
        run_id=checkpoint["run_id"],
        run_dir=Path(checkpoint["run_dir"]),
        prompts=checkpoint["prompts"],
        env=checkpoint["env"],
        max_iterations=checkpoint["max_iterations"],
        memory_every=checkpoint["memory_every"],
        pool_size=checkpoint["pool_size"],
        temperature=checkpoint["temperature"],
        top_p=checkpoint["top_p"],
        max_tokens=checkpoint["max_tokens"],
        seed_messages=checkpoint.get("seed_messages"),
        cursor_tools=checkpoint.get("cursor_tools"),
        cursor_body=checkpoint.get("cursor_body"),
        resume_checkpoint=checkpoint,
        resume_agent=(agent_name, agent_result),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ICR Contextual mode on RunPod (OpenAI-compatible API)")
    parser.add_argument("task", nargs="?", help="Task / problem statement")
    parser.add_argument("--file", "-f", help="Read task from file")
    parser.add_argument("--attach", "-a", action="append", default=[], help="Attach file (text or image); repeatable")
    parser.add_argument("--max-iterations", type=int, default=None, help="0 = unlimited like ICR (default 0)")
    parser.add_argument("--memory-every", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--output", "-o", help="Write final result to file")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-python-tools", action="store_true")
    args = parser.parse_args()

    from llm import load_env

    env = load_env()
    if args.no_python_tools:
        env["REFINE_PYTHON_TOOLS"] = "false"

    max_iter = args.max_iterations
    if max_iter is None:
        max_iter = int(env.get("REFINE_MAX_ITERATIONS", "0"))

    memory_every = args.memory_every or int(env.get("REFINE_MEMORY_EVERY", "10"))
    pool_size = args.pool_size or int(env.get("REFINE_POOL_SIZE", "12"))
    temperature = args.temperature if args.temperature is not None else float(env.get("REFINE_TEMPERATURE", "0.7"))
    top_p = args.top_p if args.top_p is not None else float(env.get("REFINE_TOP_P", "0.95"))
    max_tokens = args.max_tokens or int(env.get("REFINE_MAX_TOKENS", "8192"))

    if args.file:
        task = Path(args.file).read_text().strip()
    elif args.task:
        task = args.task.strip()
    else:
        parser.error("Provide a task string or --file")

    attach_paths = [Path(p).expanduser() for p in args.attach]

    icr_path = env.get("ICR_REPO") or str(ROOT.parent / "Iterative-Contextual-Refinements")
    print(f"Prompts: ICR ContextualPrompts.ts ({icr_path})", file=sys.stderr)
    print(f"Attachments: {len(attach_paths)} file(s)", file=sys.stderr)

    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Run: {run_dir}", file=sys.stderr)

    initial_content = build_initial_user_content(task, attach_paths)
    seed_images = load_seed_images(attach_paths)
    prompts = load_icr_prompts()

    state = run_contextual_loop(
        task,
        initial_content,
        seed_images,
        run_dir.name,
        run_dir,
        prompts=prompts,
        env=env,
        max_iterations=max_iter,
        memory_every=memory_every,
        pool_size=pool_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    print(f"\nDone: {state.status} — {state.exit_reason or 'ok'}", file=sys.stderr)
    print(f"Iterations: {state.iteration_count}", file=sys.stderr)
    print(f"Latest: {run_dir / 'latest.md'}", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(state.current_best_generation)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print("\n" + state.current_best_generation)


if __name__ == "__main__":
    main()
