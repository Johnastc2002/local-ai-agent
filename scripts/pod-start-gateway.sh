#!/usr/bin/env bash
# Run ON the RunPod pod (not your Mac).
# Starts ICR gateway on :8787 → vLLM on localhost:8000.
#
# Prereqs on pod:
#   - vLLM already listening on 8000
#   - this repo cloned (e.g. /workspace/local-ai-agent)
#   - ICR_REPO clone present
#   - port 8787 exposed in RunPod template
#
# Usage (SSH / web terminal on pod):
#   cd /workspace/local-ai-agent
#   bash scripts/pod-start-gateway.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export GATEWAY_ON_POD=1
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://127.0.0.1:8000}"
export PROXY_PORT="${PROXY_PORT:-8787}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo "ICR gateway on 0.0.0.0:${PROXY_PORT} → ${VLLM_UPSTREAM}"
echo "Cursor BYOK → https://\${RUNPOD_POD_ID}-${PROXY_PORT}.proxy.runpod.net/v1"
exec python -m gateway.app
