# Optional Docker Compose path

**Most RunPod pods:** use `bash scripts/install-on-pod.sh` (native Python + vLLM).

Use this folder only if your pod template includes Docker and GPU passthrough.

```bash
bash optional/install-docker.sh
```
