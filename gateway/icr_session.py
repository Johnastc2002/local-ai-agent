#!/usr/bin/env python3
"""Persist ICR state while Cursor executes tools on the Mac (standard BYOK loop)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "runs" / "sessions"


@dataclass
class PendingAgent:
    agent_name: str
    system_prompt: str
    working: list[dict]
    session_id: str
    transcript_parts: list[str]
    loop_start: int
    temperature: float
    seed_images: list[dict]


@dataclass
class IcrSession:
    conversation_key: str
    body: dict
    task: str
    run_id: str
    run_dir: str
    pending: PendingAgent
    checkpoint: dict[str, Any] = field(default_factory=dict)
    pending_tool_ids: list[str] = field(default_factory=list)

    def path(self) -> Path:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        return SESSIONS / f"{self.conversation_key}.json"

    def save(self) -> None:
        self.path().write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, conversation_key: str) -> IcrSession | None:
        path = SESSIONS / f"{conversation_key}.json"
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_by_tool_ids(cls, tool_ids: list[str]) -> IcrSession | None:
        if not tool_ids or not SESSIONS.exists():
            return None
        want = set(tool_ids)
        for path in sorted(SESSIONS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            pending = set(data.get("pending_tool_ids") or [])
            if pending and pending == want:
                return cls.from_dict(data)
        return None

    def clear(self) -> None:
        path = self.path()
        if path.exists():
            path.unlink()

    def to_dict(self) -> dict:
        return {
            "conversation_key": self.conversation_key,
            "body": self.body,
            "task": self.task,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "pending": {
                "agent_name": self.pending.agent_name,
                "system_prompt": self.pending.system_prompt,
                "working": self.pending.working,
                "session_id": self.pending.session_id,
                "transcript_parts": self.pending.transcript_parts,
                "loop_start": self.pending.loop_start,
                "temperature": self.pending.temperature,
                "seed_images": self.pending.seed_images,
            },
            "checkpoint": self.checkpoint,
            "pending_tool_ids": self.pending_tool_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IcrSession:
        p = data["pending"]
        return cls(
            conversation_key=data["conversation_key"],
            body=data["body"],
            task=data["task"],
            run_id=data["run_id"],
            run_dir=data["run_dir"],
            pending=PendingAgent(
                agent_name=p["agent_name"],
                system_prompt=p["system_prompt"],
                working=p["working"],
                session_id=p["session_id"],
                transcript_parts=p["transcript_parts"],
                loop_start=p["loop_start"],
                temperature=p["temperature"],
                seed_images=p.get("seed_images") or [],
            ),
            checkpoint=data.get("checkpoint") or {},
            pending_tool_ids=data.get("pending_tool_ids") or [],
        )
