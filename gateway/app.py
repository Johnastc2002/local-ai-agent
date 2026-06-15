#!/usr/bin/env python3
"""
OpenAI-compatible ICR gateway for Cursor BYOK.

Run on the RunPod pod (not your Mac):

  bash scripts/install-on-pod.sh

Cursor BYOK → https://<POD_ID>-8787.proxy.runpod.net/v1

Every user turn runs the ICR pipeline first:
  Plan  → ICR → CreatePlan tool_calls
  Agent → ICR → enriched context → vLLM (tools still work)
  Ask   → ICR → answer text

Set ICR_MODE=off to passthrough everything (debug only).
"""

from __future__ import annotations

import json
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import icr_mode, proxy_port, vllm_upstream
from gateway.icr_plan import (
    enrich_body_with_icr,
    run_icr_answer,
    run_icr_answer_stream,
    run_icr_plan,
    run_icr_plan_stream,
    run_icr_state,
)
from gateway.passthrough import forward
from gateway.router import find_plan_tool, is_agent_request, is_user_turn

app = FastAPI(title="ICR Gateway", version="1.0")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream": vllm_upstream(),
        "icr_mode": icr_mode(),
    }


@app.get("/v1/models")
async def models(request: Request):
    return await forward("GET", "/v1/models", request, None)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return await forward("POST", "/v1/chat/completions", request, raw)

    if icr_mode() == "off":
        return await forward("POST", "/v1/chat/completions", request, raw)

    # Mid-turn tool/assistant messages: continue agent chain without re-running ICR.
    if not is_user_turn(body):
        return await forward("POST", "/v1/chat/completions", request, raw)

    plan_tool = find_plan_tool(body.get("tools"))

    try:
        if plan_tool:
            if body.get("stream"):
                chunks = run_icr_plan_stream(body, plan_tool_name=plan_tool)
                return StreamingResponse(iter(chunks), media_type="text/event-stream")
            return JSONResponse(run_icr_plan(body, plan_tool_name=plan_tool))

        if is_agent_request(body):
            state = run_icr_state(body)
            enriched = enrich_body_with_icr(body, state)
            enriched_raw = json.dumps(enriched).encode()
            return await forward("POST", "/v1/chat/completions", request, enriched_raw)

        if body.get("stream"):
            chunks = run_icr_answer_stream(body)
            return StreamingResponse(iter(chunks), media_type="text/event-stream")
        return JSONResponse(run_icr_answer(body))
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"ICR pipeline failed: {exc}", "type": "icr_error"}},
            status_code=502,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    body = await request.body() if request.method in ("POST", "PUT") else None
    return await forward(request.method, f"/{path}", request, body)


def main() -> int:
    port = proxy_port()
    print(f"ICR gateway http://0.0.0.0:{port}/v1", file=sys.stderr)
    print(f"  Upstream vLLM: {vllm_upstream()}/v1", file=sys.stderr)
    print(f"  ICR_MODE: {icr_mode()} (all user turns → ICR)", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
