#!/usr/bin/env bash
# Fail fast before model download / vLLM install if the host cannot run modern vLLM.
set -euo pipefail

min_driver="${MIN_CUDA_DRIVER:-12090}"

driver="$(python3 -c "
import ctypes
lib = ctypes.CDLL('libcuda.so')
v = ctypes.c_int()
if lib.cuDriverGetVersion(ctypes.byref(v)) != 0:
    raise SystemExit('cuDriverGetVersion failed')
print(v.value)
")"

echo "CUDA driver API version: ${driver} (need >= ${min_driver} for vLLM 0.17+ / Qwen3.6)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if [[ "$driver" -lt "$min_driver" ]]; then
  cat >&2 <<EOF

FATAL: This RunPod host driver is too old for pip vLLM (Qwen3.6 needs CUDA 12.9+ wheels).

  Host driver API: ${driver}
  Required:        >= ${min_driver}  (CUDA 12.9)

Do NOT run pip install vllm here — it will download ~30GB then crash.

Fix (costs ~1 min, not another \$10):
  1. Stop this pod (keep /workspace volume if attached).
  2. Deploy H200 with image:
       runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
     (NOT runpod/pytorch:2.4.0-py3.11-cuda12.4.1 — that image is too old)
  3. Before install, probe the new host:
       bash scripts/check-gpu-env.sh
     If still < ${min_driver}, terminate and redeploy (different physical host).
  4. git pull && rm -rf .venv && MODEL_PROFILE=production-500k bash scripts/install-on-pod.sh

EOF
  exit 1
fi

echo "GPU driver OK for vLLM."
