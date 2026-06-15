#!/usr/bin/env python3
"""Load system prompts from Iterative-Contextual-Refinements (full ICR text)."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ICR_PROMPT_VARS = {
    "main_generator": "MAIN_GENERATOR_SYSTEM_PROMPT",
    "iterative_agent": "ITERATIVE_AGENT_SYSTEM_PROMPT",
    "strategic_pool": "STRATEGIC_POOL_AGENT_SYSTEM_PROMPT",
    "memory_agent": "MEMORY_AGENT_SYSTEM_PROMPT",
}

FALLBACK_DIR = ROOT / "prompts"


def icr_repo_path() -> Path:
    env = os.environ.get("ICR_REPO", "")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT.parent / "Iterative-Contextual-Refinements").resolve()


def _extract_template_literal(source: str, var_name: str) -> str:
    pattern = rf"export const {re.escape(var_name)}\s*=\s*`"
    match = re.search(pattern, source)
    if not match:
        raise ValueError(f"Could not find {var_name} in ICR ContextualPrompts.ts")
    start = match.end()
    i = start
    while i < len(source):
        if source[i] == "`":
            return source[start:i]
        i += 1
    raise ValueError(f"Unterminated template literal for {var_name}")


def load_icr_prompts() -> dict[str, str]:
    icr_file = icr_repo_path() / "Contextual" / "ContextualPrompts.ts"
    if icr_file.exists():
        source = icr_file.read_text(encoding="utf-8")
        return {key: _extract_template_literal(source, var) for key, var in ICR_PROMPT_VARS.items()}

    # Fallback to local shortened prompts
    prompts: dict[str, str] = {}
    for key in ICR_PROMPT_VARS:
        path = FALLBACK_DIR / f"{key}.md"
        if not path.exists():
            raise FileNotFoundError(
                f"ICR repo not found at {icr_repo_path()} and no fallback {path}. "
                "Clone Iterative-Contextual-Refinements or set ICR_REPO in .env"
            )
        prompts[key] = path.read_text(encoding="utf-8")
    return prompts
