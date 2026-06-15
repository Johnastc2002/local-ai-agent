# local-ai-agent

**Qwen on RunPod + ICR gateway.** Your Mac: edit `.env`, then `make ready`. Pod: one install script.

## Test model (weak GPU)

**`Qwen/Qwen2.5-3B-Instruct`** — ~6–8 GB VRAM, good enough to verify Plan → ICR → Agent pipeline.

Production: set `MODEL_PROFILE=production` → **Qwen/Qwen3.6-27B** (needs strong GPU).

See **[docs/MANUAL.md](docs/MANUAL.md)** for the full step-by-step operator guide (setup, testing, troubleshooting).

Quick start:

**1. Mac — `.env` only**

```bash
cp .env.example .env
# RUNPOD_API_KEY, RUNPOD_POD_ID
# MODEL_PROFILE=test   ← default, small model
```

**2. Pod — clone + install** (RunPod web terminal)

```bash
git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent
cd /workspace/local-ai-agent
cp .env.example .env
bash scripts/install-on-pod.sh
```

That starts **vLLM** (:8000) + **ICR gateway** (:8787). RunPod PyTorch pods have no Docker — install uses native Python automatically. First boot downloads the model (5–15 min).

Production model on pod:

```bash
MODEL_PROFILE=production bash scripts/install-on-pod.sh
```

**3. Mac — test everything**

```bash
make ready
```

Does: start pod → wait for vLLM + gateway → smoke test gateway → print Cursor BYOK settings.

Optional full ICR Plan test (slow):

```bash
make test-pipeline
```

## Cursor

Output of `make cursor-config` — paste **Override OpenAI Base URL** and **Custom model** into Cursor Settings → Models.

## Makefile cheatsheet

| Command | Where |
|---------|--------|
| `make pod-up` | Shows pod install one-liner |
| `make ready` | Mac: full test flow |
| `make test-gateway` | Mac: quick gateway check |
| `make test-pipeline` | Mac: ICR Plan smoke (minutes) |
| `make cursor-config` | Print Cursor settings |

## Architecture

```
Cursor → https://<pod>-8787.proxy.runpod.net/v1  (gateway)
              ├─ Plan  → ICR pipeline
              └─ Agent → vLLM :8000
```
