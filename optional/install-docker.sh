#!/usr/bin/env bash
# Optional — only if your pod has Docker + nvidia-container-runtime.
# Normal RunPod PyTorch pods: use bash scripts/install-on-pod.sh instead.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null; then
  echo "Docker not found. Use: bash scripts/install-on-pod.sh"
  exit 1
fi

PROFILE="${MODEL_PROFILE:-test}"
ENV_FILE=".env"
PROFILE_FILE="config/models/${PROFILE}.env"

[[ -f "$ENV_FILE" ]] || cp .env.example "$ENV_FILE"
[[ -f "$PROFILE_FILE" ]] || { echo "Missing $PROFILE_FILE"; exit 1; }

docker compose -f optional/docker-compose.yml \
  --env-file "$ENV_FILE" --env-file "$PROFILE_FILE" \
  up -d --build

docker compose -f optional/docker-compose.yml \
  --env-file "$ENV_FILE" --env-file "$PROFILE_FILE" ps
