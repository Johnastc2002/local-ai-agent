#!/usr/bin/env python3
"""
OpenAI-compatible ICR gateway for Cursor BYOK.

Run on the RunPod pod (not your Mac):

  bash scripts/install-on-pod.sh

Cursor BYOK → https://<POD_ID>-8787.proxy.runpod.net/v1
  Plan (CreatePlan in tools) → ICR pipeline
  Agent / Ask                → passthrough vLLM
"""

from __future__ import annotations

import json
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import icr_mode, proxy_port, vllm_upstream
from gateway.icr_plan import run_icr_plan, run_icr_plan_stream
from gateway.passthrough import forward
from gateway.router import is_plan_request

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

    run_icr, plan_tool = is_plan_request(body)
    if not run_icr or not plan_tool:
        return await forward("POST", "/v1/chat/completions", request, raw)

    try:
        if body.get("stream"):
            chunks = run_icr_plan_stream(body, plan_tool_name=plan_tool)
            return StreamingResponse(iter(chunks), media_type="text/event-stream")
        return JSONResponse(run_icr_plan(body, plan_tool_name=plan_tool))
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
    print(f"  ICR_MODE: {icr_mode()}", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
