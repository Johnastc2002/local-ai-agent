#!/usr/bin/env python3
"""Log Cursor request shape when GATEWAY_AUDIT=1."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from gateway.tool_audit import classify_request, tool_names

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "runs" / "audit"


def audit_enabled() -> bool:
    return os.environ.get("GATEWAY_AUDIT", "").lower() in ("1", "true", "yes", "on")


def log_request(body: dict) -> None:
    if not audit_enabled():
        return
    kind = classify_request(body)
    names = tool_names(body)
    line = (
        f"[audit] {kind} tools={names} stream={body.get('stream')} "
        f"msgs={len(body.get('messages') or [])}"
    )
    print(line, file=sys.stderr)
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = AUDIT_DIR / f"{ts}_{kind}.json"
        path.write_text(
            json.dumps(
                {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "tool_names": names,
                    "stream": body.get("stream"),
                    "model": body.get("model"),
                    "message_roles": [m.get("role") for m in body.get("messages") or []],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
