#!/usr/bin/env bash
# Run ON the RunPod pod — native vLLM + gateway (no Docker).
# Standard RunPod PyTorch templates do not include Docker.
#
#   bash scripts/install-on-pod-native.sh
#   MODEL_PROFILE=production bash scripts/install-on-pod-native.sh

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
  echo "Missing .env — copy .env.example (HF_TOKEN optional)"
  exit 1
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

export ICR_REPO_HOST="$ICR_HOST"
export ICR_REPO="$ICR_HOST"
export GATEWAY_ON_POD=1
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://127.0.0.1:${RUNPOD_PORT:-8000}}"
export PROXY_PORT="${GATEWAY_PORT:-8787}"

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

  echo "Starting vLLM :${vllm_port} model=${MODEL_NAME} (logs: ${LOG_DIR}/vllm.log)"
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
    echo "vLLM not ready yet — tail ${LOG_DIR}/vllm.log"
    tail -20 "$LOG_DIR/vllm.log" || true
    exit 1
  fi
fi

if [[ -f "$GATEWAY_PID" ]] && kill -0 "$(cat "$GATEWAY_PID")" 2>/dev/null; then
  echo "Stopping previous gateway (pid $(cat "$GATEWAY_PID"))"
  kill "$(cat "$GATEWAY_PID")" || true
  sleep 1
fi

echo "Starting ICR gateway :${PROXY_PORT} → ${VLLM_UPSTREAM} (logs: ${LOG_DIR}/gateway.log)"
nohup python -m gateway.app >"$LOG_DIR/gateway.log" 2>&1 &
echo $! >"$GATEWAY_PID"

sleep 2
if curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
  echo ""
  echo "Done."
  echo "  vLLM:    http://127.0.0.1:${vllm_port}"
  echo "  gateway: http://127.0.0.1:${PROXY_PORT}"
  echo "  logs:    tail -f ${LOG_DIR}/vllm.log ${LOG_DIR}/gateway.log"
  echo "  Mac:     make ready"
else
  echo "Gateway failed — tail ${LOG_DIR}/gateway.log"
  tail -20 "$LOG_DIR/gateway.log" || true
  exit 1
fi
