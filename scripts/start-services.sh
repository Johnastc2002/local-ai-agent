#!/usr/bin/env bash
# Start vLLM + gateway ONLY. Model must already be in HF cache.
# Does NOT download weights — fails fast if cache is missing.
#
#   bash scripts/pod-stop.sh
#   MODEL_PROFILE=production-500k bash scripts/start-services.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "=== Start services ==="
echo "Profile: $PROFILE"
echo "Model:   $MODEL_NAME"

if ! pod_model_is_cached; then
  echo ""
  echo "Model NOT in cache ($HUGGINGFACE_HUB_CACHE)."
  echo "Download first (CPU pod OK):"
  echo "  export HF_HOME=$HF_HOME"
  echo "  MODEL_PROFILE=$PROFILE bash scripts/download-model.sh"
  exit 1
fi
echo "Model cache: OK"

bash "$ROOT/scripts/check-gpu-env.sh"

pod_activate_venv
if ! python -c "import vllm" 2>/dev/null; then
  echo "vLLM not installed — run: bash scripts/setup-pod.sh" >&2
  exit 1
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
    :
  elif [[ -f "$VLLM_PID" ]] && kill -0 "$(cat "$VLLM_PID")" 2>/dev/null; then
    echo "Stopping previous vLLM (pid $(cat "$VLLM_PID"))"
    kill "$(cat "$VLLM_PID")" || true
    sleep 2
  fi

  echo "Starting vLLM on :${vllm_port} (log: ${LOG_DIR}/vllm.log, max-model-len=${want_len})"

  vllm_args=(
    serve "$MODEL_NAME"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --host 0.0.0.0
    --port "$vllm_port"
  )
  if [[ "${MAX_MODEL_LEN}" == "-1" ]]; then
    vllm_args+=(--max-model-len -1)
  else
    vllm_args+=(--max-model-len "${want_len}")
  fi
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
  if [[ "${VLLM_ENFORCE_EAGER:-}" == "true" ]]; then
    vllm_args+=(--enforce-eager)
  fi
  if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" && "${VLLM_TENSOR_PARALLEL_SIZE}" -gt 1 ]]; then
    vllm_args+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
  fi
  yarn_file="config/models/${PROFILE}-yarn.json"
  if [[ "${VLLM_USE_YARN:-}" == "1" && -f "$yarn_file" ]]; then
    vllm_args+=(--hf-overrides "$(tr -d '\n' <"$yarn_file")")
    if [[ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" == "1" ]]; then
      export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    fi
  fi

  echo "vLLM command: vllm ${vllm_args[*]}"

  nohup vllm "${vllm_args[@]}" >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$VLLM_PID"

  echo "Waiting for vLLM to load from cache (no download)..."
  vllm_wait=60
  if [[ "${MAX_MODEL_LEN:-0}" -gt 100000 ]]; then
    vllm_wait=120
  fi
  for _ in $(seq 1 "$vllm_wait"); do
    if curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready."
      break
    fi
    sleep 10
  done
  if ! curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" >/dev/null 2>&1; then
    echo "vLLM not ready — root cause from log:"
    grep -E "ValueError|RuntimeError|CUDA out of memory|OOM|Failed|error|too old" "$LOG_DIR/vllm.log" | tail -15 || true
    echo "--- last 30 lines ---"
    tail -30 "$LOG_DIR/vllm.log" || true
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
  echo "  stop:    bash scripts/pod-stop.sh"
  echo "  restart: bash scripts/start-services.sh"
else
  echo "Gateway failed — check: tail -f ${LOG_DIR}/gateway.log"
  tail -20 "$LOG_DIR/gateway.log" || true
  exit 1
fi
