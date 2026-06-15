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
  echo "Already cached — skipping download (du will NOT change)."
  bash "$ROOT/scripts/cache-diagnose.sh"
  exit 0
fi

pod_activate_venv
pip install -U pip "huggingface_hub[cli]" tqdm
if [[ "${HF_TRANSFER:-}" == "1" ]] && pip install hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "Using hf_transfer (parallel downloads)"
else
  echo "Standard download (set HF_TRANSFER=1 to try parallel)"
fi

export HF_HUB_DISABLE_PROGRESS_BARS=0

echo ""
echo "Downloading via: $(command -v hf || command -v huggingface-cli || echo 'python -m huggingface_hub.cli.hf')"
echo "Cache monitor (embedded — also run: bash scripts/cache-diagnose.sh)"
echo ""

cache_watch() {
  while [[ -f "$1" ]]; do
    vol="$(du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null | cut -f1 || echo '?')"
    home="$(du -sh "${HOME}/.cache/huggingface/hub" 2>/dev/null | cut -f1 || echo '?')"
    echo "[cache] volume=$vol  home=$home  $(date +%H:%M:%S)"
    sleep 10
  done
}
WATCH_FLAG="$LOG_DIR/.download-watch"
: >"$WATCH_FLAG"
cache_watch "$WATCH_FLAG" &
WATCH_PID=$!
cleanup_watch() { rm -f "$WATCH_FLAG"; kill "$WATCH_PID" 2>/dev/null || true; }
trap cleanup_watch EXIT

pod_hf download "$MODEL_NAME" \
  --cache-dir "$HUGGINGFACE_HUB_CACHE" \
  --resume-download \
  --max-workers "${HF_MAX_WORKERS:-4}"

cleanup_watch
trap - EXIT

echo ""
echo "Done."
bash "$ROOT/scripts/cache-diagnose.sh"
echo ""
echo "Next (GPU pod): MODEL_PROFILE=$PROFILE bash scripts/start-services.sh"
