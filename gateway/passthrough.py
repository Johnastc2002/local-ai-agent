#!/usr/bin/env python3
"""Forward OpenAI requests to vLLM unchanged."""

from __future__ import annotations

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from gateway.config import vllm_upstream


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
                async for chunk in resp.aiter_bytes():
                    yield chunk

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
