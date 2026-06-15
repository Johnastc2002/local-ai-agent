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

import asyncio
import json
import logging
import sys
import traceback

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.audit_log import log_request
from gateway.config import icr_mode, proxy_port, vllm_upstream
from llm import load_env
from gateway.cursor_protocol import is_tool_result_turn, is_user_turn, pending_tool_call_ids
from gateway.icr_plan import (
    IcrPaused,
    enrich_body_with_icr,
    finish_icr,
    resume_icr_state,
    run_icr_plan,
    run_icr_state,
)
from gateway.icr_session import IcrSession
from gateway.context_trim import wants_usage_in_stream
from gateway.passthrough import forward, forward_chat_completion
from gateway.router import find_plan_tool, is_agent_request, stream_chunks, stream_text_chunks

app = FastAPI(title="ICR Gateway", version="1.0")
log = logging.getLogger("icr.gateway")


def _gateway_error_response(exc: Exception) -> JSONResponse:
    detail = str(exc)
    if hasattr(exc, "response") and exc.response is not None:
        try:
            detail = exc.response.text[:2000]
        except Exception:
            pass
    if isinstance(exc, RuntimeError) and detail:
        log.error("Gateway request failed: %s", detail)
        print(f"Gateway request failed: {detail}", file=sys.stderr)
    else:
        log.error("Gateway request failed: %s\n%s", detail, traceback.format_exc())
        print(f"Gateway request failed: {detail}", file=sys.stderr)
        traceback.print_exc()
    return JSONResponse(
        {"error": {"message": detail, "type": "gateway_error"}},
        status_code=502,
    )


async def _forward_chat(body: dict, request: Request) -> Response:
    try:
        return await forward_chat_completion(body, request)
    except Exception as exc:
        return _gateway_error_response(exc)


def _stream_completion(body: dict, completion: dict) -> StreamingResponse:
    include_usage = wants_usage_in_stream(body)
    msg = completion["choices"][0]["message"]
    if msg.get("tool_calls"):
        chunks = stream_chunks(completion, include_usage=include_usage)
    else:
        chunks = stream_text_chunks(completion, include_usage=include_usage)
    return StreamingResponse(iter(chunks), media_type="text/event-stream")


def _paused_response(body: dict, completion: dict) -> JSONResponse | StreamingResponse:
    if not body.get("stream"):
        return JSONResponse(completion)
    return _stream_completion(body, completion)


@app.get("/health")
async def health():
    from gateway.context_trim import max_model_len

    env = load_env()
    return {
        "status": "ok",
        "upstream": vllm_upstream(),
        "icr_mode": icr_mode(),
        "cursor_tools": "passthrough",
        "max_model_len": max_model_len(env),
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
        return await _forward_chat(body, request)

    log_request(body)
    plan_tool = find_plan_tool(body.get("tools"))

    if is_tool_result_turn(body):
        session = IcrSession.load_by_tool_ids(pending_tool_call_ids(body.get("messages") or []))
        if session:
            try:
                state = await asyncio.to_thread(resume_icr_state, body, session)
                if plan_tool:
                    completion = finish_icr(body, state, plan_tool)
                    if body.get("stream"):
                        return _stream_completion(body, completion)
                    return JSONResponse(completion)
                enriched = enrich_body_with_icr(body, state)
                return await _forward_chat(enriched, request)
            except IcrPaused as paused:
                return _paused_response(body, paused.completion)
        return await _forward_chat(body, request)

    if not is_user_turn(body):
        return await _forward_chat(body, request)

    try:
        if plan_tool:
            if body.get("stream"):
                completion = await asyncio.to_thread(run_icr_plan, body, plan_tool_name=plan_tool)
                return _stream_completion(body, completion)
            completion = await asyncio.to_thread(run_icr_plan, body, plan_tool_name=plan_tool)
            return JSONResponse(completion)

        if is_agent_request(body):
            state = await asyncio.to_thread(run_icr_state, body)
            enriched = enrich_body_with_icr(body, state)
            return await _forward_chat(enriched, request)

        # Ask — same as Cursor: ICR enriches, vLLM answers with full thread history
        state = await asyncio.to_thread(run_icr_state, body)
        enriched = enrich_body_with_icr(body, state)
        return await _forward_chat(enriched, request)
    except IcrPaused as paused:
        return _paused_response(body, paused.completion)
    except Exception as exc:
        return _gateway_error_response(exc)


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
