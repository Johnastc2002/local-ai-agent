# local-ai-agent

**Cursor BYOK → RunPod → ICR gateway.** Plan mode runs the iterative refinement pipeline; Agent/Ask passthrough to vLLM.

**Full runbook:** [docs/MANUAL.md](docs/MANUAL.md) — setup, daily use, troubleshooting. Read that; you shouldn't need anything else.

## Three steps

**Mac** — `cp .env.example .env` → fill `RUNPOD_API_KEY`, `RUNPOD_POD_ID`

**Pod** — three separate steps (download once, restart without re-download):

```bash
export HF_HOME=/workspace/.cache/huggingface
git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent
cd /workspace/local-ai-agent

# 1. Download weights once (CPU pod OK)
MODEL_PROFILE=production-500k bash scripts/download-model.sh

# 2. Setup deps once (GPU pod)
MODEL_PROFILE=production-500k bash scripts/setup-pod.sh

# 3. Start / restart anytime (no download)
MODEL_PROFILE=production-500k bash scripts/start-services.sh
```

After a bug fix: `bash scripts/pod-stop.sh && git pull && bash scripts/start-services.sh`

**Mac** — `make ready` → paste `make cursor-config` into Cursor Settings → Models

## Architecture

```
Cursor → https://<pod>-8787.proxy.runpod.net/v1
           └─ every user turn → ICR pipeline first
                ├─ Plan  → CreatePlan tool_calls
                ├─ Agent → ICR context + vLLM tools
                └─ Ask   → ICR answer text
```

## Makefile

| Command | Where |
|---------|--------|
| `make ready` | Mac — start, test, print Cursor settings |
| `make pod-up` | Mac — print pod install commands |
| `make test-gateway` | Mac — quick smoke test |
| `make stop` | Mac — stop pod |

Default model: **`Qwen/Qwen2.5-3B-Instruct`** (`MODEL_PROFILE=test`). Production: `Qwen/Qwen3.6-27B`. Long context: `production-500k` (~524k, H200).
