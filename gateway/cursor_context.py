#!/usr/bin/env python3
"""Harvest codebase paths and inline file context from Cursor BYOK requests."""

from __future__ import annotations

import re
from pathlib import Path

from attachments import build_initial_user_content
from codebase_paths import codebase_root, remap_to_codebase
from llm import Message, content_to_text

# Cursor / IDE inline file blocks (several shapes seen in the wild)
_FILE_BLOCK_PATTERNS = [
    re.compile(r'<file\s+path="([^"]+)">\s*(.*?)\s*</file>', re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:[\w+-]+:)?([^\n`]+)\n(.*?)```", re.DOTALL),
]

_PATH_IN_TEXT = re.compile(
    r"(?:^|[\s'\"`])((?:/[A-Za-z0-9._\-/]+)+\.(?:kt|java|py|ts|tsx|js|jsx|md|json|xml|gradle|kts))\b"
)

_FILENAME_MENTION = re.compile(
    r"\b([A-Za-z0-9_\-/]+\.(?:kt|java|py|ts|tsx|js|jsx|md|json|xml|gradle|kts))\b"
)


def _inline_blocks_from_text(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for pattern in _FILE_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            path, body = match.group(1).strip(), match.group(2).strip()
            if path and body and len(body) > 20:
                found.append((path, body))
    return found


def harvest_paths(messages: list[Message], env: dict) -> list[Path]:
    """Paths to attach from message text (remapped to pod CODEBASE_ROOT)."""
    root = codebase_root(env)
    if root is None:
        return []

    seen: set[str] = set()
    paths: list[Path] = []

    def add(raw: str) -> None:
        mapped = remap_to_codebase(raw, env)
        if mapped in seen:
            return
        seen.add(mapped)
        p = Path(mapped)
        if p.is_file():
            paths.append(p)

    for msg in messages:
        text = content_to_text(msg.get("content"))
        for path, _body in _inline_blocks_from_text(text):
            add(path)
        for match in _PATH_IN_TEXT.finditer(text):
            add(match.group(1))
        for match in _FILENAME_MENTION.finditer(text):
            name = match.group(1)
            if "/" not in name:
                hits = list(root.rglob(Path(name).name))[:3]
                for hit in hits:
                    add(str(hit))
            else:
                add(name)
    return paths


def build_icr_user_content(task: str, messages: list[Message], env: dict) -> str | list:
    """Task + attached pod files + inline Cursor file blocks."""
    attach_paths = harvest_paths(messages, env)
    base = build_initial_user_content(task, attach_paths)

    inline_parts: list[str] = []
    for msg in messages:
        text = content_to_text(msg.get("content"))
        for path, body in _inline_blocks_from_text(text):
            if len(body) > 120_000:
                body = body[:120_000] + "\n[... truncated ...]"
            inline_parts.append(
                f"--- Cursor inline file: {path} ---\n{body}\n--- end {path} ---"
            )

    root = codebase_root(env)
    if root:
        inline_parts.insert(
            0,
            f"Codebase root on pod: {root}\n"
            "Use read_file, grep, and list_dir tools to inspect the project before answering.",
        )

    if not inline_parts:
        return base

    extra = "\n\n".join(inline_parts)
    if isinstance(base, str):
        return f"{base}\n\n{extra}"
    return [*base, {"type": "text", "text": f"\n\n{extra}"}]
