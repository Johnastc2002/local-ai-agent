#!/usr/bin/env bash
# Shared env for pod scripts. Source from other scripts — do not run directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file: source scripts/pod-env.sh" >&2
  exit 1
fi

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PROFILE="${MODEL_PROFILE:-test}"
ENV_FILE="$ROOT/.env"
PROFILE_FILE="$ROOT/config/models/${PROFILE}.env"
LOG_DIR="$ROOT/runs"
VLLM_PID="$LOG_DIR/vllm.pid"
GATEWAY_PID="$LOG_DIR/gateway.pid"

if [[ ! -f "$ENV_FILE" ]]; then
  cp .env.example "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example"
fi
if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "Missing $PROFILE_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1090
source "$PROFILE_FILE"
set +a

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$LOG_DIR"

ICR_HOST="${ICR_REPO_HOST:-/workspace/Iterative-Contextual-Refinements}"
export ICR_REPO_HOST="$ICR_HOST"
export ICR_REPO="$ICR_HOST"
export GATEWAY_ON_POD=1
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://127.0.0.1:${RUNPOD_PORT:-8000}}"
export PROXY_PORT="${GATEWAY_PORT:-8787}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-local}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

pod_model_slug() {
  echo "models--${MODEL_NAME//\//--}"
}

pod_model_hub_dir() {
  echo "${HUGGINGFACE_HUB_CACHE}/$(pod_model_slug)"
}

pod_clear_download_locks() {
  local slug lock_dir
  slug="$(pod_model_slug)"
  lock_dir="${HUGGINGFACE_HUB_CACHE}/.locks/${slug}"
  if [[ -d "$lock_dir" ]]; then
    echo "Removing stale download locks: $lock_dir"
    rm -rf "$lock_dir"
  fi
}

pod_model_is_cached() {
  python3 -c "
from huggingface_hub import try_to_load_from_cache
import sys
path = try_to_load_from_cache(
    '${MODEL_NAME}',
    'config.json',
    cache_dir='${HUGGINGFACE_HUB_CACHE}',
)
sys.exit(0 if path else 1)
" 2>/dev/null
}

pod_model_cache_path() {
  python3 -c "
from huggingface_hub import try_to_load_from_cache
path = try_to_load_from_cache(
    '${MODEL_NAME}',
    'config.json',
    cache_dir='${HUGGINGFACE_HUB_CACHE}',
)
print(path or '')
"
}

pod_activate_venv() {
  if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
}

# Prefer `hf` (huggingface_hub >= 0.34); fall back to legacy huggingface-cli.
pod_hf() {
  if command -v hf >/dev/null 2>&1; then
    hf "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli "$@"
  else
    python3 -m huggingface_hub.cli.hf "$@"
  fi
}
