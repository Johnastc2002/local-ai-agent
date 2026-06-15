#!/usr/bin/env bash
# Run ON the RunPod pod — tail vLLM + gateway logs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tail -f "$ROOT/runs/vllm.log" "$ROOT/runs/gateway.log"
