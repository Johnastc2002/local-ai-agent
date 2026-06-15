#!/usr/bin/env python3
"""
Smoke-test vLLM and/or ICR gateway on RunPod.

  python smoke_test.py                 # vLLM direct (:8000)
  python smoke_test.py --gateway       # gateway (:8787) passthrough
  python smoke_test.py --gateway --plan-smoke   # ICR Plan route (slow)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")

    profile = env.get("MODEL_PROFILE", "")
    if profile:
        profile_file = ROOT / "config" / "models" / f"{profile}.env"
        if profile_file.is_file():
            for line in profile_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def require(name: str, env: dict[str, str]) -> str:
    value = env.get(name) or os.environ.get(name, "")
    if not value:
        print(f"Missing {name} in .env", file=sys.stderr)
        sys.exit(1)
    return value


def vllm_base(env: dict[str, str]) -> str:
    explicit = env.get("RUNPOD_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    pod_id = require("RUNPOD_POD_ID", env)
    port = env.get("RUNPOD_PORT", "8000")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def gateway_base(env: dict[str, str]) -> str:
    explicit = env.get("GATEWAY_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    pod_id = require("RUNPOD_POD_ID", env)
    port = env.get("GATEWAY_PORT", "8787")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def get_json(url: str, api_key: str, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def post_json(url: str, api_key: str, payload: dict, stream: bool = False, timeout: int = 120) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                chunks = []
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk.decode(errors="replace"))
                return resp.status, "".join(chunks)
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def test_health(base: str, api_key: str) -> bool:
    status, body = get_json(f"{base}/health", api_key)
    if status != 200:
        print(f"FAIL GET /health — HTTP {status}: {body[:300]}")
        return False
    print(f"OK  GET /health — {body[:200]}")
    return True


def test_models(base: str, api_key: str) -> bool:
    status, body = get_json(f"{base}/v1/models", api_key)
    if status != 200:
        print(f"FAIL GET /v1/models — HTTP {status}: {body[:400]}")
        return False
    print("OK  GET /v1/models")
    print(body[:600])
    return True


def test_chat(base: str, api_key: str, model: str) -> bool:
    status, body = post_json(
        f"{base}/v1/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    if status != 200:
        print(f"FAIL POST /v1/chat/completions — HTTP {status}: {body[:400]}")
        return False
    try:
        text = json.loads(body)["choices"][0]["message"]["content"]
        print(f"OK  POST /v1/chat/completions — reply: {text!r}")
        return True
    except (KeyError, json.JSONDecodeError) as e:
        print(f"FAIL POST /v1/chat/completions — bad response: {e}\n{body[:400]}")
        return False


def test_tool_stream(base: str, api_key: str, model: str) -> bool:
    status, body = post_json(
        f"{base}/v1/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the calculator tool to compute 19 + 23. Call the tool only.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate a math expression",
                        "parameters": {
                            "type": "object",
                            "properties": {"expr": {"type": "string"}},
                            "required": ["expr"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 256,
            "stream": True,
        },
        stream=True,
    )
    if status != 200:
        print(f"FAIL streaming tool test — HTTP {status}: {body[:400]}")
        return False
    if "tool_calls" in body:
        print("OK  streaming tool test — tool_calls present")
        return True
    if "<tool_call>" in body:
        print("FAIL streaming tool test — raw XML (fix TOOL_CALL_PARSER on pod)")
        return False
    print(f"WARN streaming tool test — no tool_calls\n{body[:500]}")
    return True


def test_plan_route(base: str, api_key: str, model: str) -> bool:
    print("Plan smoke: ICR via gateway (may take several minutes)...")
    status, body = post_json(
        f"{base}/v1/chat/completions",
        api_key,
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "Plan a one-step refactor of a hello function."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "CreatePlan",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "overview": {"type": "string"},
                                "plan": {"type": "string"},
                                "todos": {"type": "array"},
                                "isProject": {"type": "boolean"},
                            },
                        },
                    },
                }
            ],
        },
        timeout=900,
    )
    if status != 200:
        print(f"FAIL plan route — HTTP {status}: {body[:500]}")
        return False
    try:
        msg = json.loads(body)["choices"][0]["message"]
        if msg.get("tool_calls"):
            name = msg["tool_calls"][0]["function"]["name"]
            print(f"OK  plan route — tool_calls returned ({name})")
            return True
        print(f"FAIL plan route — no tool_calls: {body[:500]}")
        return False
    except (KeyError, json.JSONDecodeError) as e:
        print(f"FAIL plan route — {e}\n{body[:400]}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", action="store_true", help="Test via ICR gateway (:8787)")
    parser.add_argument("--plan-smoke", action="store_true", help="Run ICR Plan route (slow)")
    parser.add_argument("--chat-only", action="store_true")
    args = parser.parse_args()

    env = load_env()
    api_key = require("RUNPOD_API_KEY", env)
    model = env.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    base = gateway_base(env) if args.gateway else vllm_base(env)
    label = "gateway" if args.gateway else "vLLM"

    print(f"Target:   {label}")
    print(f"Endpoint: {base}")
    print(f"Model:    {model}\n")

    ok = True
    if args.gateway:
        ok = test_health(base, api_key) and ok
    ok = test_models(base, api_key) and ok
    ok = test_chat(base, api_key, model) and ok
    if not args.chat_only:
        ok = test_tool_stream(base, api_key, model) and ok
    if args.plan_smoke:
        if not args.gateway:
            print("--plan-smoke requires --gateway", file=sys.stderr)
            sys.exit(1)
        ok = test_plan_route(base, api_key, model) and ok

    if ok:
        print("\nAll checks passed.")
        if args.gateway:
            print(f"  Cursor BYOK URL:   {base}/v1")
        print(f"  Cursor model name: {model}")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
