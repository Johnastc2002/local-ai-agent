#!/usr/bin/env bash
# Download model weights to HF cache ONLY — no GPU, no vLLM start.
# Safe to run on a cheap CPU pod with /workspace volume attached.
#
#   export HF_HOME=/workspace/.cache/huggingface
#   MODEL_PROFILE=production-500k bash scripts/download-model.sh
#
# Re-run is a no-op if already cached (use FORCE=1 to re-fetch).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "=== Download model (no vLLM) ==="
echo "Profile:  $PROFILE"
echo "Model:    $MODEL_NAME"
echo "HF cache: $HUGGINGFACE_HUB_CACHE"

if pod_model_is_cached && [[ "${FORCE:-}" != "1" ]]; then
  echo "Already cached — skipping download."
  du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true
  pod_model_cache_path | head -1
  exit 0
fi

pip install -U pip huggingface_hub tqdm
if pip install hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "Using hf_transfer (parallel downloads)"
else
  echo "hf_transfer not available — standard download"
fi

export HF_HUB_DISABLE_PROGRESS_BARS=0

echo ""
echo "Downloading (per-file progress below; resume supported)..."
echo "Tip: open a second terminal and run: watch -n5 du -sh $HUGGINGFACE_HUB_CACHE"
echo ""

huggingface-cli download "$MODEL_NAME" \
  --cache-dir "$HUGGINGFACE_HUB_CACHE" \
  --resume-download \
  --max-workers 8

echo ""
echo "Done."
echo "Cache size:"
du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true
echo ""
echo "Next (GPU pod): MODEL_PROFILE=$PROFILE bash scripts/start-services.sh"
