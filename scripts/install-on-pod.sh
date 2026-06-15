#!/usr/bin/env bash
# Full pod bootstrap: setup deps + start (does NOT download model).
#
# Prefer the split workflow:
#   bash scripts/download-model.sh   # once per volume
#   bash scripts/setup-pod.sh        # once per venv
#   bash scripts/start-services.sh   # every restart
#
#   MODEL_PROFILE=production-500k bash scripts/install-on-pod.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "$ROOT/scripts/setup-pod.sh"
bash "$ROOT/scripts/start-services.sh"
