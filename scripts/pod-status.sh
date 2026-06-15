#!/usr/bin/env bash
# Run ON the RunPod pod — check vLLM + gateway health.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VLLM_PORT="${RUNPOD_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8787}"
LOG_DIR="$ROOT/runs"

check() {
  local name="$1" url="$2"
  if curl -fsS "$url" >/dev/null 2>&1; then
    echo "OK   $name  $url"
  else
    echo "FAIL $name  $url"
  fi
}

echo "=== Pod status ==="
check "vLLM"    "http://127.0.0.1:${VLLM_PORT}/v1/models"
check "gateway" "http://127.0.0.1:${GATEWAY_PORT}/health"

for label in vllm gateway; do
  pid_file="$LOG_DIR/${label}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "PID  $label  $pid (running)"
    else
      echo "PID  $label  $pid (not running)"
    fi
  fi
done

echo ""
echo "Logs: tail -f ${LOG_DIR}/vllm.log ${LOG_DIR}/gateway.log"
echo "Restart: bash scripts/install-on-pod.sh"
