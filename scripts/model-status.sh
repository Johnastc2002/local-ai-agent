#!/usr/bin/env bash
# Show whether model weights are in HF cache.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "Model:    $MODEL_NAME"
echo "HF cache: $HUGGINGFACE_HUB_CACHE"

if pod_model_is_cached; then
  echo "Status:   CACHED"
  du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true
  pod_model_cache_path | head -1
else
  echo "Status:   NOT CACHED"
  echo "Run: MODEL_PROFILE=$PROFILE bash scripts/download-model.sh"
  exit 1
fi
