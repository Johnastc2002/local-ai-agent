#!/usr/bin/env bash
# Run ON the RunPod pod — stop vLLM + gateway.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/runs"

stop_pid() {
  local name="$1" pid_file="$LOG_DIR/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "Stopped $name (pid $pid)"
    else
      echo "$name not running (stale pid $pid)"
    fi
    rm -f "$pid_file"
  else
    echo "No pid file for $name"
  fi
}

stop_pid gateway
stop_pid vllm
echo "Done. Start again: MODEL_PROFILE=\${MODEL_PROFILE:-test} bash scripts/start-services.sh"
