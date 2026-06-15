# local-ai-agent

**Cursor BYOK → RunPod → ICR gateway.** Plan mode runs the iterative refinement pipeline; Agent/Ask passthrough to vLLM.

**Full runbook:** [docs/MANUAL.md](docs/MANUAL.md) — setup, daily use, troubleshooting. Read that; you shouldn't need anything else.

## Three steps

**Mac** — `cp .env.example .env` → fill `RUNPOD_API_KEY`, `RUNPOD_POD_ID`

**Pod** — RunPod web terminal:

```bash
git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent
cd /workspace/local-ai-agent && bash scripts/install-on-pod.sh
```

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

Default model: **`Qwen/Qwen2.5-3B-Instruct`** (`MODEL_PROFILE=test`). Production: `Qwen/Qwen3.6-27B`.
