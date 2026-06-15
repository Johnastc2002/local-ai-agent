#!/usr/bin/env bash
# Install Python deps + vLLM. No model download, no service start.
#
#   MODEL_PROFILE=production-500k bash scripts/setup-pod.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pod-env.sh"

echo "=== Setup pod deps ==="
echo "Profile: $PROFILE"
echo "Model:   $MODEL_NAME"
echo "ICR:     $ICR_HOST"

if [[ ! -d "$ICR_HOST" ]]; then
  echo "Cloning ICR repo → $ICR_HOST"
  git clone --depth 1 https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements "$ICR_HOST"
fi

bash "$ROOT/scripts/check-gpu-env.sh"

pod_activate_venv
pip install -q -U pip
pip install -q -r requirements.txt

echo "Running unit tests..."
python -m unittest discover -s tests -p 'test_*.py' -q

need_vllm_install=0
if ! python -c "import vllm" 2>/dev/null; then
  need_vllm_install=1
elif ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Broken vLLM/torch (CUDA unavailable) — reinstalling..."
  need_vllm_install=1
fi
if [[ "$need_vllm_install" -eq 1 ]]; then
  echo "Installing vLLM..."
  bash "$ROOT/scripts/install-vllm.sh"
fi
python -c "import vllm; print(f'vLLM {vllm.__version__}')"

echo ""
echo "Setup done. Model download: bash scripts/download-model.sh"
echo "Start services:           bash scripts/start-services.sh"
