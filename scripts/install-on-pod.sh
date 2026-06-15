#!/usr/bin/env bash
# Run ON the RunPod pod — starts vLLM + ICR gateway via Docker Compose.
#
#   bash scripts/install-on-pod.sh
#   MODEL_PROFILE=production bash scripts/install-on-pod.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${MODEL_PROFILE:-test}"
ENV_FILE=".env"
PROFILE_FILE="config/models/${PROFILE}.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env — copy .env.example and set RUNPOD_API_KEY, HF_TOKEN if needed"
  exit 1
fi
if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "Missing $PROFILE_FILE"
  exit 1
fi

ICR_HOST="${ICR_REPO_HOST:-/workspace/Iterative-Contextual-Refinements}"
if [[ ! -d "$ICR_HOST" ]]; then
  echo "Cloning ICR repo → $ICR_HOST"
  git clone --depth 1 https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements "$ICR_HOST"
fi
export ICR_REPO_HOST="$ICR_HOST"

if ! command -v docker >/dev/null; then
  echo "Docker not found — using native install (normal on RunPod PyTorch pods)."
  exec bash "$ROOT/scripts/install-on-pod-native.sh"
fi

echo "Profile: $PROFILE"
echo "ICR:     $ICR_HOST"
echo "Starting vLLM + gateway (first boot may take 5–15 min while model downloads)..."

docker compose --env-file "$ENV_FILE" --env-file "$PROFILE_FILE" up -d --build

echo ""
docker compose --env-file "$ENV_FILE" --env-file "$PROFILE_FILE" ps
echo ""
echo "Logs:  docker compose logs -f"
echo "Mac:   make ready"
