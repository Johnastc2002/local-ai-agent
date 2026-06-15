#!/usr/bin/env bash
# Run ON the RunPod pod — starts vLLM (:8000) + ICR gateway (:8787).
#
#   bash scripts/install-on-pod.sh
#   MODEL_PROFILE=production bash scripts/install-on-pod.sh
#   MODEL_PROFILE=production-500k bash scripts/install-on-pod.sh   # H200, ~524k context
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

# Codebase on pod — optional fallback only (standard path: Cursor tools on Mac)
# CODEBASE_ROOT=/workspace/bobot-xs-v1
# CODEBASE_HOST_ROOT=/Users/tough/hobby/bobot-xs-v1
# REFINE_CODEBASE_TOOLS=off
export ICR_REPO_HOST="$ICR_HOST"
export ICR_REPO="$ICR_HOST"
export GATEWAY_ON_POD=1
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://127.0.0.1:${RUNPOD_PORT:-8000}}"
export PROXY_PORT="${GATEWAY_PORT:-8787}"
# vLLM on localhost does not validate this; llm.py still requires a value
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-local}"

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

echo "Running unit tests..."
python -m unittest discover -s tests -p 'test_*.py' -q

if ! python -c "import vllm" 2>/dev/null; then
  echo "Installing vLLM (first time — may take several minutes)..."
  pip install -q vllm
fi

vllm_port="${RUNPOD_PORT:-8000}"
want_len="${MAX_MODEL_LEN:-8192}"
want_model="$MODEL_NAME"
need_vllm_start=1
if curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
  read -r got_model got_len <<< "$(curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" | python3 -c "
import json, sys
d = json.load(sys.stdin)['data'][0]
print(d.get('id',''), d.get('max_model_len',''))
" 2>/dev/null || echo '  ')"
  if [[ "$got_model" == "$want_model" && "$got_len" == "$want_len" ]]; then
    need_vllm_start=0
    echo "vLLM already listening on :${vllm_port} (${got_model}, max_len=${got_len})"
  else
    echo "vLLM config changed (have ${got_model}@${got_len}, want ${want_model}@${want_len}) — restarting"
    if [[ -f "$VLLM_PID" ]] && kill -0 "$(cat "$VLLM_PID")" 2>/dev/null; then
      kill "$(cat "$VLLM_PID")" || true
      sleep 3
    fi
  fi
fi

if [[ "$need_vllm_start" -eq 1 ]]; then
  if curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
    : # killed above
  elif [[ -f "$VLLM_PID" ]] && kill -0 "$(cat "$VLLM_PID")" 2>/dev/null; then
    echo "Stopping previous vLLM (pid $(cat "$VLLM_PID"))"
    kill "$(cat "$VLLM_PID")" || true
    sleep 2
  fi

  echo "Starting vLLM on :${vllm_port} (log: ${LOG_DIR}/vllm.log, max-model-len=${want_len})"

  vllm_args=(
    serve "$MODEL_NAME"
    --max-model-len "${want_len}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --host 0.0.0.0
    --port "$vllm_port"
  )
  if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
    vllm_args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  fi
  if [[ "${ENABLE_AUTO_TOOL_CHOICE:-}" == "true" ]]; then
    vllm_args+=(--enable-auto-tool-choice)
  fi
  if [[ -n "${REASONING_PARSER:-}" ]]; then
    vllm_args+=(--reasoning-parser "$REASONING_PARSER")
  fi
  if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
    vllm_args+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
  fi
  if [[ "${VLLM_LANGUAGE_MODEL_ONLY:-}" == "true" ]]; then
    vllm_args+=(--language-model-only)
  fi
  if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" && "${VLLM_TENSOR_PARALLEL_SIZE}" -gt 1 ]]; then
    vllm_args+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
  fi
  yarn_file="config/models/${PROFILE}-yarn.json"
  if [[ -f "$yarn_file" ]]; then
    vllm_args+=(--hf-overrides "$(tr -d '\n' <"$yarn_file")")
  fi
  if [[ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" == "1" ]]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  fi

  nohup vllm "${vllm_args[@]}" >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$VLLM_PID"

  echo "Waiting for vLLM (first boot downloads model — 5–30 min for large models)..."
  vllm_wait=120
  if [[ "${MAX_MODEL_LEN:-0}" -gt 100000 ]]; then
    vllm_wait=240
  fi
  for _ in $(seq 1 "$vllm_wait"); do
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

echo "Waiting for gateway..."
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "Gateway ready."
    break
  fi
  sleep 2
done
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
