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

pip install -q -U pip huggingface_hub

echo "Downloading (resume supported) — can take 10–30 min for 27B..."
python3 - <<PY
import os
from huggingface_hub import snapshot_download

model = os.environ["MODEL_NAME"]
cache = os.environ["HUGGINGFACE_HUB_CACHE"]
path = snapshot_download(
    repo_id=model,
    cache_dir=cache,
    resume_download=True,
)
print(f"Done: {path}")
PY

echo ""
echo "Cache size:"
du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true
echo ""
echo "Next (GPU pod): MODEL_PROFILE=$PROFILE bash scripts/start-services.sh"
