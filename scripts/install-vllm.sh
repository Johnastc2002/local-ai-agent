#!/usr/bin/env bash
# Install vLLM wheel matching host CUDA driver (called from install-on-pod.sh).
set -euo pipefail

min_ver="${VLLM_MIN_VERSION:-0.17.0}"
pin_ver="${VLLM_PIN_VERSION:-0.23.0}"

driver="$(python3 -c "
import ctypes
lib = ctypes.CDLL('libcuda.so')
v = ctypes.c_int()
lib.cuDriverGetVersion(ctypes.byref(v))
print(v.value)
")"

install_cu129_wheel() {
  local ver="$1"
  local url="https://github.com/vllm-project/vllm/releases/download/v${ver}/vllm-${ver}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  echo "Installing vLLM ${ver} (cu129) from release wheel..."
  pip install -q "$url"
}

if [[ "$driver" -ge 12090 ]]; then
  install_cu129_wheel "$pin_ver"
elif [[ "$driver" -ge 12080 ]]; then
  echo "Driver ${driver} (CUDA 12.8): installing vLLM via PyTorch cu128 index..."
  pip install -q "vllm>=${min_ver}" --extra-index-url https://download.pytorch.org/whl/cu128
else
  echo "Driver ${driver} too old — run scripts/check-gpu-env.sh" >&2
  exit 1
fi

python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, cuda={torch.version.cuda}')"
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'"
