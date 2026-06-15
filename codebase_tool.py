#!/usr/bin/env python3
"""Codebase read/grep/list tools for ICR agents (Opus-style code access on pod)."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from codebase_paths import codebase_root, resolve_in_codebase

READ_FILE_NAME = "read_file"
GREP_NAME = "grep"
LIST_DIR_NAME = "list_dir"

MAX_READ_BYTES = 200_000
MAX_GREP_MATCHES = 80
MAX_LIST_ENTRIES = 200

READ_FILE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": READ_FILE_NAME,
        "description": (
            "Read a file from the project codebase. Path may be relative to project root "
            "or an absolute path under CODEBASE_ROOT."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "offset": {
                    "type": "integer",
                    "description": "1-based start line (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (optional)",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

GREP_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": GREP_NAME,
        "description": (
            "Search the codebase with ripgrep. Use to find symbols, classes, usages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or plain text pattern"},
                "path": {
                    "type": "string",
                    "description": "File or directory under project root (default: whole repo)",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter, e.g. *.kt",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}

LIST_DIR_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": LIST_DIR_NAME,
        "description": "List files and directories under a project path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory relative to project root (default: .)",
                },
            },
            "additionalProperties": False,
        },
    },
}

CODEBASE_TOOL_DEFS = [READ_FILE_TOOL_DEF, GREP_TOOL_DEF, LIST_DIR_TOOL_DEF]
CODEBASE_TOOL_NAMES = {READ_FILE_NAME, GREP_NAME, LIST_DIR_NAME}


def codebase_tools_enabled(env: dict[str, str]) -> bool:
    val = env.get("REFINE_CODEBASE_TOOLS", os.environ.get("REFINE_CODEBASE_TOOLS", "auto"))
    val = val.lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return codebase_root(env) is not None
    # auto
    return codebase_root(env) is not None


class CodebaseTools:
    def __init__(self, env: dict[str, str]):
        self.env = env
        self.root = codebase_root(env)
        if self.root is None:
            raise ValueError("CODEBASE_ROOT not set")

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> dict:
        try:
            resolved = resolve_in_codebase(path, self.env)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not resolved.is_file():
            return {"ok": False, "error": f"Not a file: {resolved}"}
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": f"Binary or non-UTF-8 file: {resolved}"}
        if len(text.encode()) > MAX_READ_BYTES:
            text = text[:MAX_READ_BYTES] + "\n[... truncated ...]"
        lines = text.splitlines()
        start = max(1, int(offset or 1))
        end = len(lines) if limit is None else min(len(lines), start + int(limit) - 1)
        slice_lines = lines[start - 1 : end]
        numbered = "\n".join(f"{i + start:6}|{line}" for i, line in enumerate(slice_lines))
        rel = resolved.relative_to(self.root.resolve()) if self.root else resolved
        return {
            "ok": True,
            "path": str(rel),
            "lines": f"{start}-{end} of {len(lines)}",
            "content": numbered,
        }

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
    ) -> dict:
        try:
            target = resolve_in_codebase(path or ".", self.env)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", str(MAX_GREP_MATCHES)]
        if glob:
            cmd.extend(["--glob", glob])
        cmd.extend([pattern, str(target)])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            return self._grep_python(pattern, target, glob)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "grep timed out"}

        out = proc.stdout.strip()
        if proc.returncode not in (0, 1):
            err = (proc.stderr or proc.stdout or "grep failed").strip()
            return {"ok": False, "error": err}
        if not out:
            return {"ok": True, "matches": 0, "content": "(no matches)"}
        lines = out.splitlines()[:MAX_GREP_MATCHES]
        return {"ok": True, "matches": len(lines), "content": "\n".join(lines)}

    def _grep_python(self, pattern: str, target: Path, glob: str | None) -> dict:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {"ok": False, "error": f"Invalid pattern: {e}"}
        root = self.root.resolve()
        files: list[Path] = []
        if target.is_file():
            files = [target]
        else:
            for p in target.rglob("*"):
                if not p.is_file():
                    continue
                if glob and not fnmatch.fnmatch(p.name, glob.lstrip("*/")):
                    continue
                files.append(p)
        hits: list[str] = []
        for fp in files:
            if len(hits) >= MAX_GREP_MATCHES:
                break
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        rel = fp.relative_to(root)
                        hits.append(f"{rel}:{i}:{line[:300]}")
                        if len(hits) >= MAX_GREP_MATCHES:
                            break
            except (UnicodeDecodeError, OSError):
                continue
        if not hits:
            return {"ok": True, "matches": 0, "content": "(no matches)"}
        return {"ok": True, "matches": len(hits), "content": "\n".join(hits)}

    def list_dir(self, path: str = ".") -> dict:
        try:
            resolved = resolve_in_codebase(path or ".", self.env)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not resolved.is_dir():
            return {"ok": False, "error": f"Not a directory: {resolved}"}
        entries: list[str] = []
        for child in sorted(resolved.iterdir()):
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append("...")
                break
            suffix = "/" if child.is_dir() else ""
            try:
                rel = child.relative_to(self.root.resolve())
            except ValueError:
                rel = child
            entries.append(f"{rel}{suffix}")
        return {"ok": True, "path": str(resolved.relative_to(self.root.resolve())), "entries": entries}

    def dispatch(self, name: str, args: dict[str, Any]) -> dict:
        if name == READ_FILE_NAME:
            return self.read_file(
                str(args.get("path", "")),
                args.get("offset"),
                args.get("limit"),
            )
        if name == GREP_NAME:
            return self.grep(
                str(args.get("pattern", "")),
                str(args.get("path") or "."),
                args.get("glob"),
            )
        if name == LIST_DIR_NAME:
            return self.list_dir(str(args.get("path") or "."))
        return {"ok": False, "error": f"Unknown codebase tool: {name}"}


def format_codebase_result(result: dict) -> str:
    if not result.get("ok"):
        return f"Error: {result.get('error', 'unknown')}"
    if "content" in result and isinstance(result["content"], str):
        extra = {k: v for k, v in result.items() if k not in ("ok", "content")}
        head = json.dumps(extra, ensure_ascii=False) if extra else "ok"
        return f"{head}\n\n{result['content']}"
    if "entries" in result:
        return json.dumps(result, indent=2, ensure_ascii=False)
    return json.dumps(result, indent=2, ensure_ascii=False)
