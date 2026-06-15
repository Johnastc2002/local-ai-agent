#!/usr/bin/env python3
"""Agent-scoped Python sandbox (simplified port of ICR ContextualPythonToolRuntime)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VFS_ROOT = Path(tempfile.gettempdir()) / "local-ai-agent-python-vfs"
DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 300

PYTHON_TOOL_NAME = "python_virtual_filesystem"
PYTHON_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": PYTHON_TOOL_NAME,
        "description": (
            "Execute Python code inside the agent virtual filesystem. "
            "Use for calculations, simulations, algorithm tests, image inspection, plots. "
            "Uploaded images are available by filename in the working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use relative paths for image files.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def _safe_session_id(session_id: str) -> bool:
    return bool(session_id) and len(session_id) <= 80 and all(
        c.isalnum() or c in "-_" for c in session_id
    )


class PythonSandbox:
    def __init__(self, session_id: str):
        if not _safe_session_id(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        self.session_id = session_id
        self.workspace = VFS_ROOT / session_id
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._seeded = False

    def seed_images(self, seeds: list[dict]) -> None:
        if self._seeded:
            return
        for seed in seeds:
            name = seed.get("name") or "uploaded-image.png"
            raw = __import__("base64").standard_b64decode(seed["base64"])
            (self.workspace / name).write_bytes(raw)
        self._seeded = True

    def execute(self, code: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
        timeout_s = min(max(timeout_s, 1), MAX_TIMEOUT_S)
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "MPLBACKEND": "Agg",
                },
            )
            duration_ms = int((time.time() - start) * 1000)
            return {
                "ok": proc.returncode == 0,
                "exitCode": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "error": None,
                "durationMs": duration_ms,
                "timedOut": False,
            }
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.time() - start) * 1000)
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return {
                "ok": False,
                "exitCode": -1,
                "stdout": stdout,
                "stderr": stderr,
                "error": f"Execution timed out after {timeout_s}s",
                "durationMs": duration_ms,
                "timedOut": True,
            }
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            return {
                "ok": False,
                "exitCode": 1,
                "stdout": "",
                "stderr": "",
                "error": str(e),
                "durationMs": duration_ms,
                "timedOut": False,
            }


def format_tool_result(result: dict) -> str:
    lines = [
        f"Python execution {'completed' if result['ok'] else 'failed'}.",
        f"exit_code={result['exitCode']}",
        f"duration_ms={result['durationMs']}",
    ]
    if result.get("stdout", "").strip():
        lines.append(f"stdout:\n{result['stdout'].strip()}")
    if result.get("stderr", "").strip():
        lines.append(f"stderr:\n{result['stderr'].strip()}")
    if result.get("error"):
        lines.append(f"error:\n{result['error']}")
    return "\n\n".join(lines)


def session_id_for_agent(run_id: str, agent_name: str) -> str:
    safe = agent_name.lower().replace(" ", "-")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in safe)
    return f"ctx-{run_id}-{safe}"[:80]
