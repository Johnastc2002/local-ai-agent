#!/usr/bin/env python3
"""
OpenAI-compatible ICR gateway for Cursor BYOK.

ICR uses Cursor's standard tool protocol:
  1. Cursor sends messages + tools (read_file, grep, MCP, …)
  2. Model returns tool_calls → gateway returns them to Cursor
  3. Cursor executes on your Mac → sends tool results back
  4. Gateway resumes ICR where it paused

Set ICR_MODE=off to passthrough everything (debug only).
"""

from __future__ import annotations

import json
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import icr_mode, proxy_port, vllm_upstream
from gateway.cursor_protocol import is_tool_result_turn, is_user_turn, pending_tool_call_ids
from gateway.icr_plan import (
    IcrPaused,
    enrich_body_with_icr,
    finish_icr,
    resume_icr_state,
    run_icr_answer,
    run_icr_answer_stream,
    run_icr_plan,
    run_icr_plan_stream,
    run_icr_state,
)
from gateway.icr_session import IcrSession
from gateway.passthrough import forward
from gateway.router import find_plan_tool, is_agent_request, stream_chunks, stream_text_chunks

app = FastAPI(title="ICR Gateway", version="1.0")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream": vllm_upstream(),
        "icr_mode": icr_mode(),
        "cursor_tools": "passthrough",
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

    plan_tool = find_plan_tool(body.get("tools"))

    if is_tool_result_turn(body):
        session = IcrSession.load_by_tool_ids(pending_tool_call_ids(body.get("messages") or []))
        if session:
            try:
                state = resume_icr_state(body, session)
                if plan_tool:
                    completion = finish_icr(body, state, plan_tool)
                    if body.get("stream"):
                        return StreamingResponse(iter(stream_chunks(completion)), media_type="text/event-stream")
                    return JSONResponse(completion)
                if is_agent_request(body):
                    enriched_raw = json.dumps(enrich_body_with_icr(body, state)).encode()
                    return await forward("POST", "/v1/chat/completions", request, enriched_raw)
                completion = finish_icr(body, state, None)
                if body.get("stream"):
                    return StreamingResponse(iter(stream_text_chunks(completion)), media_type="text/event-stream")
                return JSONResponse(completion)
            except IcrPaused as paused:
                return JSONResponse(paused.completion)
        return await forward("POST", "/v1/chat/completions", request, raw)

    if not is_user_turn(body):
        return await forward("POST", "/v1/chat/completions", request, raw)

    try:
        if plan_tool:
            if body.get("stream"):
                chunks = run_icr_plan_stream(body, plan_tool_name=plan_tool)
                return StreamingResponse(iter(chunks), media_type="text/event-stream")
            return JSONResponse(run_icr_plan(body, plan_tool_name=plan_tool))

        if is_agent_request(body):
            state = run_icr_state(body)
            enriched_raw = json.dumps(enrich_body_with_icr(body, state)).encode()
            return await forward("POST", "/v1/chat/completions", request, enriched_raw)

        if body.get("stream"):
            chunks = run_icr_answer_stream(body)
            return StreamingResponse(iter(chunks), media_type="text/event-stream")
        return JSONResponse(run_icr_answer(body))
    except IcrPaused as paused:
        return JSONResponse(paused.completion)
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
    print(f"  ICR_MODE: {icr_mode()} (Cursor tools → IDE executes)", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
