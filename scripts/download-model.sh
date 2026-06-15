#!/usr/bin/env bash
# Download model weights to HF cache ONLY — no GPU, no vLLM start.
#
#   export HF_HOME=/workspace/.cache/huggingface
#   MODEL_PROFILE=production-500k bash scripts/download-model.sh
#
# Stuck at ~28M with .locks? This script clears locks and resumes automatically.
# Nuclear reset: CLEAN=1 MODEL_PROFILE=production-500k bash scripts/download-model.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

DOWNLOAD_LOG="$LOG_DIR/download.log"

echo "=== Download model (no vLLM) ==="
echo "Profile:  $PROFILE"
echo "Model:    $MODEL_NAME"
echo "HF cache: $HUGGINGFACE_HUB_CACHE"
echo "Log:      $DOWNLOAD_LOG"

if pod_model_is_cached && [[ "${FORCE:-}" != "1" ]]; then
  echo "Already cached — skipping download."
  bash "$ROOT/scripts/cache-diagnose.sh"
  exit 0
fi

if [[ "${CLEAN:-}" == "1" ]]; then
  echo "CLEAN=1 — removing partial cache for $(pod_model_slug)"
  rm -rf "$(pod_model_hub_dir)" "${HUGGINGFACE_HUB_CACHE}/.locks/$(pod_model_slug)"
fi

pod_clear_download_locks

hub_dir="$(pod_model_hub_dir)"
if [[ -d "$hub_dir" ]]; then
  partial="$(du -sh "$hub_dir" 2>/dev/null | cut -f1 || echo '?')"
  echo "Partial cache present ($partial) — resuming download..."
fi

pod_activate_venv
pip install -U pip "huggingface_hub[cli]" tqdm

if ! pod_hf version >/dev/null 2>&1; then
  echo "ERROR: hf CLI not working after install." | tee -a "$DOWNLOAD_LOG"
  pod_hf version 2>&1 | tee -a "$DOWNLOAD_LOG" || true
  exit 1
fi
echo "hf: $(pod_hf version 2>&1 | head -1)"

if [[ "${HF_TRANSFER:-}" == "1" ]] && pip install hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "Using hf_transfer"
fi

export HF_HUB_DISABLE_PROGRESS_BARS=0

echo ""
echo "Starting: hf download $MODEL_NAME"
echo ""

cache_watch() {
  while [[ -f "$1" ]]; do
    vol="$(du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null | cut -f1 || echo '?')"
    echo "[cache] $vol  $(date +%H:%M:%S)"
    sleep 10
  done
}
WATCH_FLAG="$LOG_DIR/.download-watch"
: >"$WATCH_FLAG"
cache_watch "$WATCH_FLAG" &
WATCH_PID=$!
cleanup_watch() { rm -f "$WATCH_FLAG"; kill "$WATCH_PID" 2>/dev/null || true; }
trap cleanup_watch EXIT

set +e
pod_hf download "$MODEL_NAME" \
  --cache-dir "$HUGGINGFACE_HUB_CACHE" \
  --resume-download \
  --max-workers "${HF_MAX_WORKERS:-4}" \
  2>&1 | tee -a "$DOWNLOAD_LOG"
dl_status=${PIPESTATUS[0]}
set -e

cleanup_watch
trap - EXIT

if [[ "$dl_status" -ne 0 ]]; then
  echo ""
  echo "DOWNLOAD FAILED (exit $dl_status). Last lines:"
  tail -20 "$DOWNLOAD_LOG"
  exit "$dl_status"
fi

if ! pod_model_is_cached; then
  echo ""
  echo "DOWNLOAD FINISHED but model still NOT CACHED — incomplete."
  bash "$ROOT/scripts/cache-diagnose.sh"
  exit 1
fi

echo ""
echo "Done — model cached."
du -sh "$HUGGINGFACE_HUB_CACHE"
echo "Next: MODEL_PROFILE=$PROFILE bash scripts/start-services.sh"
