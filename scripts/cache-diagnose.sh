#!/usr/bin/env bash
# Where is the HF model cache? Run on pod when du looks stuck.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "=== HF cache diagnostic ==="
echo "Model:              $MODEL_NAME"
echo "HF_HOME:            $HF_HOME"
echo "HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
echo ""

show_dir() {
  local label="$1" path="$2"
  if [[ -d "$path" ]]; then
    echo "$label"
    du -sh "$path" 2>/dev/null || echo "  (empty or unreadable)"
    find "$path" -maxdepth 3 -type d -name 'models--*' 2>/dev/null | head -5 | sed 's/^/  /'
  else
    echo "$label — does not exist"
  fi
  echo ""
}

show_dir "Volume cache (/workspace):" "$HUGGINGFACE_HUB_CACHE"
show_dir "Home cache (~/.cache):" "${HOME}/.cache/huggingface/hub"
show_dir "Root cache (/root/.cache):" "/root/.cache/huggingface/hub"

echo "Download process:"
if pgrep -af 'huggingface-cli download' 2>/dev/null; then
  echo "  (running — if size only grows under ~/.cache, export HF_HOME=/workspace/.cache/huggingface)"
else
  echo "  none running"
fi
echo ""

if pod_model_is_cached; then
  echo "model-status: CACHED (download script will skip — no size change expected)"
  pod_model_cache_path | head -1
else
  echo "model-status: NOT CACHED"
fi

echo ""
echo "Fix wrong cache location:"
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo "  MODEL_PROFILE=$PROFILE bash scripts/download-model.sh"
