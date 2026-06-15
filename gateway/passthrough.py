#!/usr/bin/env python3
"""Forward OpenAI requests to vLLM unchanged."""

from __future__ import annotations

import json
import logging
import sys

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from gateway.config import vllm_upstream

log = logging.getLogger(__name__)


def _headers(request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    if auth := request.headers.get("authorization"):
        out["Authorization"] = auth
    ct = request.headers.get("content-type")
    if ct:
        out["Content-Type"] = ct
    return out


def _is_streaming(body: bytes | None, request: Request) -> bool:
    if request.headers.get("accept") == "text/event-stream":
        return True
    if body and (b'"stream":true' in body or b'"stream": true' in body):
        return True
    return False


def _parse_body(body: bytes | None) -> dict:
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


async def chat_completion_json(body: dict, request: Request) -> tuple[dict, dict]:
    """Non-streaming completion from vLLM (reliable for localhost upstream)."""
    from gateway.context_trim import ensure_completion_usage, trim_body_for_vllm

    payload = trim_body_for_vllm({**body, "stream": False})
    url = f"{vllm_upstream()}/v1/chat/completions"
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=_headers(request), json=payload)
        if resp.status_code >= 400:
            detail = resp.text[:2000]
            log.error(
                "vLLM %s (max_tokens=%s, messages=%s): %s",
                resp.status_code,
                payload.get("max_tokens"),
                len(payload.get("messages") or []),
                detail,
            )
            print(f"vLLM {resp.status_code}: {detail}", file=sys.stderr)
            raise RuntimeError(detail)
        completion = ensure_completion_usage(resp.json(), payload)
        return completion, payload


async def forward(
    method: str,
    path: str,
    request: Request,
    body: bytes | None,
) -> Response:
    url = f"{vllm_upstream()}{path}"
    headers = _headers(request)
    params = dict(request.query_params)
    timeout = httpx.Timeout(600.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if _is_streaming(body, request):
            req = client.build_request(method, url, headers=headers, params=params, content=body)
            resp = await client.send(req, stream=True)

            async def stream():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                except httpx.ReadError:
                    log.warning("vLLM stream closed early for %s", path)
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream(),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

        resp = await client.request(method, url, headers=headers, params=params, content=body)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )


async def forward_chat_completion(body: dict, request: Request) -> Response:
    """
    Chat completion via vLLM. If client asked for stream, buffer then emit SSE chunks.
    Avoids fragile long-lived passthrough streams (RunPod proxy / client timeouts).
    """
    from gateway.context_trim import ensure_completion_usage, trim_body_for_vllm, wants_usage_in_stream
    from gateway.router import stream_chunks, stream_text_chunks

    want_stream = bool(body.get("stream"))
    include_usage = wants_usage_in_stream(body)
    if want_stream:
        completion, payload = await chat_completion_json(body, request)
        msg = completion["choices"][0]["message"]
        if msg.get("tool_calls"):
            chunks = stream_chunks(completion, include_usage=include_usage)
        else:
            chunks = stream_text_chunks(completion, include_usage=include_usage)
        return StreamingResponse(iter(chunks), media_type="text/event-stream")

    payload = trim_body_for_vllm({**body, "stream": False})
    payload_bytes = json.dumps(payload).encode()
    resp = await forward("POST", "/v1/chat/completions", request, payload_bytes)
    if resp.status_code == 200:
        try:
            completion = json.loads(resp.body)
            completion = ensure_completion_usage(completion, payload)
            return Response(
                content=json.dumps(completion).encode(),
                status_code=200,
                media_type="application/json",
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return resp

