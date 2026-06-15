#!/usr/bin/env python3
"""Resolve codebase paths (Cursor Mac paths → pod CODEBASE_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def codebase_root(env: dict[str, str] | None = None) -> Path | None:
    from llm import load_env

    env = env or load_env()
    raw = env.get("CODEBASE_ROOT") or os.environ.get("CODEBASE_ROOT", "")
    if not raw.strip():
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def host_root(env: dict[str, str] | None = None) -> Path | None:
    from llm import load_env

    env = env or load_env()
    raw = env.get("CODEBASE_HOST_ROOT") or os.environ.get("CODEBASE_HOST_ROOT", "")
    if not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def remap_to_codebase(path_str: str, env: dict[str, str] | None = None) -> str:
    """Map a Cursor/Mac absolute path to a path under CODEBASE_ROOT."""
    root = codebase_root(env)
    if root is None:
        return path_str
    p = Path(path_str.strip())
    host = host_root(env)
    if host and p.is_absolute():
        try:
            rel = p.resolve().relative_to(host.resolve())
            return str(root / rel)
        except ValueError:
            pass
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    return str(root / p)


def resolve_in_codebase(path_str: str, env: dict[str, str] | None = None) -> Path:
    root = codebase_root(env)
    if root is None:
        raise ValueError("CODEBASE_ROOT is not configured or missing on pod")

    mapped = remap_to_codebase(path_str, env)
    candidate = Path(mapped).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path escapes CODEBASE_ROOT: {path_str}")
    return resolved
