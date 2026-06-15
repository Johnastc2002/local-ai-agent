#!/usr/bin/env python3
"""
RunPod pod lifecycle helper for Cursor BYOK with Qwen3.6-27B.

Usage:
  python runpod.py list
  python runpod.py start
  python runpod.py stop
  python runpod.py status
  python runpod.py url          # print vLLM OpenAI base URL
  python runpod.py gateway-url  # print ICR gateway URL for Cursor BYOK
  python runpod.py wait         # block until /v1/models responds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
RUNPOD_REST = "https://rest.runpod.io/v1"


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
    return env


def require(name: str, env: dict[str, str]) -> str:
    value = env.get(name) or os.environ.get(name, "")
    if not value:
        print(f"Missing {name}. Set it in {ENV_FILE} or export it.", file=sys.stderr)
        sys.exit(1)
    return value


def api_request(method: str, path: str, api_key: str, body: dict | None = None) -> Any:
    url = f"{RUNPOD_REST}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        print(f"RunPod API error {e.code}: {detail}", file=sys.stderr)
        sys.exit(1)


def base_url(env: dict[str, str]) -> str:
    explicit = env.get("RUNPOD_BASE_URL") or os.environ.get("RUNPOD_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    pod_id = require("RUNPOD_POD_ID", env)
    port = env.get("RUNPOD_PORT") or os.environ.get("RUNPOD_PORT", "8000")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def openai_base_url(env: dict[str, str]) -> str:
    return f"{base_url(env)}/v1"


def gateway_base_url(env: dict[str, str]) -> str:
    explicit = env.get("GATEWAY_BASE_URL") or os.environ.get("GATEWAY_BASE_URL", "")
    if explicit:
        return explicit.rstrip("/")
    pod_id = require("RUNPOD_POD_ID", env)
    port = env.get("GATEWAY_PORT") or os.environ.get("GATEWAY_PORT", "8787")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def gateway_openai_url(env: dict[str, str]) -> str:
    return f"{gateway_base_url(env)}/v1"


def cmd_list(api_key: str) -> None:
    pods = api_request("GET", "/pods", api_key)
    if not isinstance(pods, list):
        print(json.dumps(pods, indent=2))
        return
    if not pods:
        print("No pods found.")
        return
    print(f"{'ID':<22} {'NAME':<24} {'STATUS':<12} DESIRED")
    print("-" * 70)
    for pod in pods:
        pod_id = pod.get("id", "?")
        name = pod.get("name", "?")
        status = (pod.get("runtime") or {}).get("status") or pod.get("desiredStatus", "?")
        desired = pod.get("desiredStatus", "?")
        print(f"{pod_id:<22} {name:<24} {str(status):<12} {desired}")


def cmd_start(env: dict[str, str], api_key: str) -> None:
    pod_id = require("RUNPOD_POD_ID", env)
    api_request("POST", f"/pods/{pod_id}/start", api_key)
    print(f"Started pod {pod_id}")
    print(f"vLLM:    {openai_base_url(env)}")
    print(f"Cursor:  {gateway_openai_url(env)}  (after gateway started on pod)")


def cmd_stop(env: dict[str, str], api_key: str) -> None:
    pod_id = require("RUNPOD_POD_ID", env)
    api_request("POST", f"/pods/{pod_id}/stop", api_key)
    print(f"Stopped pod {pod_id}")


def cmd_status(env: dict[str, str], api_key: str) -> None:
    pod_id = require("RUNPOD_POD_ID", env)
    pod = api_request("GET", f"/pods/{pod_id}", api_key)
    runtime = pod.get("runtime") or {}
    print(json.dumps(
        {
            "id": pod.get("id"),
            "name": pod.get("name"),
            "desiredStatus": pod.get("desiredStatus"),
            "runtimeStatus": runtime.get("status"),
            "proxyUrl": base_url(env),
            "cursorOpenAiBaseUrl": openai_base_url(env),
            "gatewayUrl": gateway_base_url(env),
            "cursorGatewayBaseUrl": gateway_openai_url(env),
            "model": env.get("MODEL_NAME", "Qwen/Qwen3.6-27B"),
        },
        indent=2,
    ))


def cmd_url(env: dict[str, str]) -> None:
    print(openai_base_url(env))


def cmd_gateway_url(env: dict[str, str]) -> None:
    print(gateway_openai_url(env))


def probe_models(url: str, api_key: str, timeout: int = 10) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{url}/v1/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()[:500]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def probe_health(url: str, api_key: str, timeout: int = 10) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{url}/health",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()[:500]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def cmd_wait(env: dict[str, str], api_key: str, timeout_sec: int, interval_sec: int) -> None:
    url = base_url(env)
    deadline = time.time() + timeout_sec
    attempt = 0
    print(f"Waiting for {url}/v1/models (timeout {timeout_sec}s)...")
    while time.time() < deadline:
        attempt += 1
        status, _body = probe_models(url, api_key)
        if status == 200:
            print(f"Ready after {attempt} attempt(s).")
            print(f"vLLM (debug):     {openai_base_url(env)}")
            print(f"Cursor BYOK →     {gateway_openai_url(env)}")
            print("(On pod run: bash scripts/install-on-pod.sh)")
            return
        print(f"  attempt {attempt}: HTTP {status} — retrying in {interval_sec}s...")
        time.sleep(interval_sec)
    print("Timed out. Pod may still be loading the model (first boot can take 10+ min).", file=sys.stderr)
    sys.exit(1)


def cmd_wait_all(env: dict[str, str], api_key: str, timeout_sec: int, interval_sec: int) -> None:
    cmd_wait(env, api_key, timeout_sec, interval_sec)
    gurl = gateway_base_url(env)
    deadline = time.time() + timeout_sec
    attempt = 0
    print(f"\nWaiting for {gurl}/health ...")
    while time.time() < deadline:
        attempt += 1
        status, _body = probe_health(gurl, api_key)
        if status == 200:
            print(f"Gateway ready after {attempt} attempt(s).")
            print(f"Cursor BYOK → {gateway_openai_url(env)}")
            return
        print(f"  attempt {attempt}: HTTP {status} — retrying in {interval_sec}s...")
        time.sleep(interval_sec)
    print("Gateway timed out. On pod run: bash scripts/install-on-pod.sh", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod pod helper for Cursor BYOK")
    parser.add_argument(
        "command",
        choices=["list", "start", "stop", "status", "url", "gateway-url", "wait", "wait-all"],
        help="Action to perform",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds to wait for wait command (default: 900)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Poll interval for wait command (default: 15)",
    )
    args = parser.parse_args()

    env = load_env()
    api_key = require("RUNPOD_API_KEY", env)

    if args.command == "list":
        cmd_list(api_key)
    elif args.command == "start":
        cmd_start(env, api_key)
    elif args.command == "stop":
        cmd_stop(env, api_key)
    elif args.command == "status":
        cmd_status(env, api_key)
    elif args.command == "url":
        cmd_url(env)
    elif args.command == "gateway-url":
        cmd_gateway_url(env)
    elif args.command == "wait":
        cmd_wait(env, api_key, args.timeout, args.interval)
    elif args.command == "wait-all":
        cmd_wait_all(env, api_key, args.timeout, args.interval)


if __name__ == "__main__":
    main()
