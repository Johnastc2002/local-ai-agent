#!/usr/bin/env bash
# Where is the HF model cache? Run on pod when du looks stuck.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "=== HF cache diagnostic ==="
echo "Profile:            $PROFILE  (set MODEL_PROFILE=production-500k if wrong)"
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

show_dir "Volume cache:" "$HUGGINGFACE_HUB_CACHE"
show_dir "Home cache:" "${HOME}/.cache/huggingface/hub"

hub_dir="$(pod_model_hub_dir)"
lock_dir="${HUGGINGFACE_HUB_CACHE}/.locks/$(pod_model_slug)"

if [[ -d "$lock_dir" ]] && ! pod_model_is_cached; then
  echo "STUCK: stale lock files (partial download died). Locks at:"
  echo "  $lock_dir"
  echo ""
fi

if [[ -d "$hub_dir" ]] && ! pod_model_is_cached; then
  partial="$(du -sh "$hub_dir" 2>/dev/null | cut -f1 || echo '?')"
  echo "PARTIAL: $hub_dir ($partial) — need ~28GB+, not 28M"
  echo ""
fi

echo "Download process:"
if pgrep -af 'hf download|huggingface-cli download' 2>/dev/null; then
  echo "  running"
else
  echo "  none running"
fi
echo ""

if pod_model_is_cached; then
  echo "Status: CACHED"
  pod_model_cache_path | head -1
else
  echo "Status: NOT CACHED"
  echo ""
  echo "Fix:"
  echo "  export HF_HOME=/workspace/.cache/huggingface"
  echo "  cd /workspace/local-ai-agent && git pull"
  echo "  MODEL_PROFILE=production-500k bash scripts/download-model.sh"
  echo ""
  echo "If still stuck:"
  echo "  CLEAN=1 MODEL_PROFILE=production-500k bash scripts/download-model.sh"
fi

if [[ -f "$LOG_DIR/download.log" ]]; then
  echo ""
  echo "Last download log ($LOG_DIR/download.log):"
  tail -10 "$LOG_DIR/download.log" 2>/dev/null || true
fi
