#!/usr/bin/env bash
# Run ON the RunPod pod — starts vLLM (:8000) + ICR gateway (:8787).
#
#   bash scripts/install-on-pod.sh
#   MODEL_PROFILE=production bash scripts/install-on-pod.sh
#
# RunPod PyTorch pods have no Docker. This script uses native Python + vLLM.
# Optional Docker path: bash optional/install-docker.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${MODEL_PROFILE:-test}"
ENV_FILE=".env"
PROFILE_FILE="config/models/${PROFILE}.env"
LOG_DIR="$ROOT/runs"
VLLM_PID="$LOG_DIR/vllm.pid"
GATEWAY_PID="$LOG_DIR/gateway.pid"

if [[ ! -f "$ENV_FILE" ]]; then
  cp .env.example "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example"
fi
if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "Missing $PROFILE_FILE"
  exit 1
fi

ICR_HOST="${ICR_REPO_HOST:-/workspace/Iterative-Contextual-Refinements}"
if [[ ! -d "$ICR_HOST" ]]; then
  echo "Cloning ICR repo → $ICR_HOST"
  git clone --depth 1 https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements "$ICR_HOST"
fi

mkdir -p "$LOG_DIR"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1090
source "$PROFILE_FILE"
set +a

# Codebase on pod (ICR agents read/grep this — required for real planning)
CODEBASE_DIR="${CODEBASE_ROOT:-/workspace/bobot-xs-v1}"
if [[ -n "${CODEBASE_GIT_URL:-}" ]] && [[ ! -d "$CODEBASE_DIR/.git" ]]; then
  echo "Cloning codebase → $CODEBASE_DIR"
  git clone --depth 1 "$CODEBASE_GIT_URL" "$CODEBASE_DIR"
fi
if [[ ! -d "$CODEBASE_DIR" ]]; then
  echo ""
  echo "WARN: Codebase missing at $CODEBASE_DIR"
  echo "      ICR cannot read your project. Either:"
  echo "        git clone <your-repo> $CODEBASE_DIR"
  echo "      or set CODEBASE_GIT_URL in .env and re-run install."
  echo ""
else
  echo "Codebase: $CODEBASE_DIR"
fi

export ICR_REPO_HOST="$ICR_HOST"
export ICR_REPO="$ICR_HOST"
export GATEWAY_ON_POD=1
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://127.0.0.1:${RUNPOD_PORT:-8000}}"
export PROXY_PORT="${GATEWAY_PORT:-8787}"

echo "Profile: $PROFILE"
echo "Model:   $MODEL_NAME"
echo "ICR:     $ICR_HOST"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt

if ! python -c "import vllm" 2>/dev/null; then
  echo "Installing vLLM (first time — may take several minutes)..."
  pip install -q vllm
fi

vllm_port="${RUNPOD_PORT:-8000}"
if curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
  echo "vLLM already listening on :${vllm_port}"
else
  if [[ -f "$VLLM_PID" ]] && kill -0 "$(cat "$VLLM_PID")" 2>/dev/null; then
    echo "Stopping previous vLLM (pid $(cat "$VLLM_PID"))"
    kill "$(cat "$VLLM_PID")" || true
    sleep 2
  fi

  echo "Starting vLLM on :${vllm_port} (log: ${LOG_DIR}/vllm.log)"
  nohup vllm serve "$MODEL_NAME" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    --host 0.0.0.0 \
    --port "$vllm_port" \
    >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$VLLM_PID"

  echo "Waiting for vLLM (first boot downloads model — 5–15 min)..."
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready."
      break
    fi
    sleep 10
  done
  if ! curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
    echo "vLLM not ready — check: tail -f ${LOG_DIR}/vllm.log"
    tail -20 "$LOG_DIR/vllm.log" || true
    exit 1
  fi
fi

if [[ -f "$GATEWAY_PID" ]] && kill -0 "$(cat "$GATEWAY_PID")" 2>/dev/null; then
  echo "Stopping previous gateway (pid $(cat "$GATEWAY_PID"))"
  kill "$(cat "$GATEWAY_PID")" || true
  sleep 1
fi

echo "Starting gateway on :${PROXY_PORT} (log: ${LOG_DIR}/gateway.log)"
nohup python -m gateway.app >"$LOG_DIR/gateway.log" 2>&1 &
echo $! >"$GATEWAY_PID"

sleep 2
if curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
  echo ""
  echo "=== Pod ready ==="
  echo "  vLLM:    http://127.0.0.1:${vllm_port}"
  echo "  gateway: http://127.0.0.1:${PROXY_PORT}"
  echo "  status:  bash scripts/pod-status.sh"
  echo "  logs:    bash scripts/pod-logs.sh"
  echo "  Mac:     make ready"
else
  echo "Gateway failed — check: tail -f ${LOG_DIR}/gateway.log"
  tail -20 "$LOG_DIR/gateway.log" || true
  exit 1
fi
