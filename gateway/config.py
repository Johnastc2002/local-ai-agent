#!/usr/bin/env python3
"""Gateway configuration from .env."""

from __future__ import annotations

import os

from llm import base_url, load_env


def proxy_port() -> int:
    return int(os.environ.get("PROXY_PORT", "8787"))


def vllm_upstream() -> str:
    explicit = os.environ.get("VLLM_UPSTREAM", "").strip()
    if explicit:
        return explicit.rstrip("/")
    # Gateway runs on the RunPod pod next to vLLM (see scripts/install-on-pod.sh)
    if os.environ.get("GATEWAY_ON_POD", "").lower() in ("1", "true", "yes"):
        port = os.environ.get("RUNPOD_PORT", "8000")
        return f"http://127.0.0.1:{port}"
    return base_url(load_env()).rstrip("/")


def icr_mode() -> str:
    return os.environ.get("ICR_MODE", "auto").lower()


def icr_ask_mode() -> str:
    """Ask mode: icr (slow, refined) or passthrough (fast, direct vLLM)."""
    return os.environ.get("ICR_ASK", "passthrough").lower()


def icr_agent_mode() -> str:
    """Agent first user turn: icr or passthrough before edit loop."""
    return os.environ.get("ICR_AGENT", "icr").lower()


def plan_max_iterations() -> int:
    env = load_env()
    return int(env.get("CURSOR_PLAN_MAX_ITERATIONS", "8"))
