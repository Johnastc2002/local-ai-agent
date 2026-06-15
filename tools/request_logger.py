#!/usr/bin/env python3
"""
Log Cursor OpenAI requests to captures/ then forward unchanged to vLLM.

Phase 0 tool — point Cursor BYOK at http://127.0.0.1:8787/v1 while this runs.

  python tools/request_logger.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gateway.config import proxy_port, vllm_upstream  # noqa: E402
from gateway.passthrough import forward  # noqa: E402

CAPTURES = ROOT / "captures"
app = FastAPI(title="Cursor Request Logger", version="1.0")


def _redact(obj: object) -> object:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.lower() in ("authorization", "api_key", "apikey"):
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, str):
        return re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", obj)
    return obj


from gateway.tool_audit import classify_request, tool_names


def _save_capture(kind: str, body: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = CAPTURES / f"{ts}_{kind}.json"
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "tool_names": tool_names(body),
        "stream": body.get("stream"),
        "model": body.get("model"),
        "body": _redact(body),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def _classify(body: dict) -> str:
    return classify_request(body)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "request_logger", "upstream": vllm_upstream(), "captures": str(CAPTURES)}


@app.get("/v1/models")
async def models(request: Request):
    return await forward("GET", "/v1/models", request, None)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw = await request.body()
    try:
        body = json.loads(raw)
        kind = _classify(body)
        path = _save_capture(kind, body)
        print(f"Captured {kind} → {path}", file=sys.stderr)
    except json.JSONDecodeError:
        pass
    return await forward("POST", "/v1/chat/completions", request, raw)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    body = await request.body() if request.method in ("POST", "PUT") else None
    return await forward(request.method, f"/{path}", request, body)


def main() -> int:
    port = proxy_port()
    print(f"Request logger http://0.0.0.1:{port}/v1", file=sys.stderr)
    print(f"  Upstream: {vllm_upstream()}/v1", file=sys.stderr)
    print(f"  Captures: {CAPTURES}", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
