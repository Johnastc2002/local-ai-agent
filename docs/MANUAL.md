# ICR Gateway — Operator Manual

Use this document to set up and test **Cursor + RunPod + ICR gateway** from scratch, or to pick up again later without re-explaining the architecture.

**Repo:** [github.com/Johnastc2002/local-ai-agent](https://github.com/Johnastc2002/local-ai-agent)  
**Mac checkout:** `/Users/tough/hobby/local-ai-agent`  
**Coding project example:** `/Users/tough/hobby/bobot-xs-v1`

---

## 1. What you are building

```
┌─────────────┐     HTTPS      ┌──────────────────────────────────┐
│  Cursor     │ ─────────────► │  RunPod pod                      │
│  (your Mac) │   BYOK :8787   │  ┌──────────┐    ┌─────────────┐ │
│             │                │  │ gateway  │───►│ vLLM :8000  │ │
│ Plan/Agent  │                │  │ ICR loop │    │ Qwen model  │ │
└─────────────┘                │  └──────────┘    └─────────────┘ │
                               └──────────────────────────────────┘
```

**Principles:**

- **Cursor is 100% standard** — Plan / Agent / Ask, BYOK, CreatePlan, Build. No hooks, no rules, no MCP required.
- **Nothing runs on your Mac except Cursor** (plus optional `make` commands to control the pod).
- **The only special behavior** is on RunPod: when Cursor **Plan mode** sends a `CreatePlan` tool request, the **gateway** runs the **ICR pipeline** (Main → Critique → Strategic Pool, repeat) instead of a single model forward pass.
- **Agent / Ask** requests pass straight through gateway → vLLM unchanged.

---

## 2. Prerequisites

| Item | Notes |
|------|--------|
| **Cursor Pro** | Required for BYOK |
| **RunPod account** | Dedicated pod (not serverless for this setup) |
| **RunPod API key** | Settings → API Keys |
| **Pod with NVIDIA GPU** | Native vLLM + gateway (Docker optional) |
| **Pod ports exposed** | **8000** (vLLM), **8787** (gateway) |
| **Git on pod** | To clone this repo |
| **Mac terminal** | For `make start`, `make ready`, etc. |

You do **not** need Python or an LLM on your Mac for the Cursor workflow.

---

## 3. Model profiles

Start with the **test** profile on a weaker GPU. Switch to **production** when the pipeline works.

| Profile | Model | VRAM (approx) | Config file |
|---------|--------|---------------|-------------|
| **test** (default) | `Qwen/Qwen2.5-3B-Instruct` | ~6–8 GB @ 8k context | `config/models/test.env` |
| **production** | `Qwen/Qwen3.6-27B` | Strong GPU (e.g. A40/A100) | `config/models/production.env` |

Set on **Mac** `.env`:

```bash
MODEL_PROFILE=test        # or production
```

The same profile must be used when running `install-on-pod.sh` on the pod (`MODEL_PROFILE=...`).

---

## 4. One-time setup

### 4.1 RunPod pod template

Ensure the pod template exposes:

| Port | Service |
|------|---------|
| **8000** | vLLM |
| **8787** | ICR gateway |

Standard **RunPod PyTorch** pods have no Docker — `install-on-pod.sh` falls back to native vLLM + Python gateway automatically.

### 4.2 Mac — `.env`

```bash
cd /Users/tough/hobby/local-ai-agent
cp .env.example .env
```

Edit `.env` — **only secrets required:**

```bash
RUNPOD_API_KEY=rp_...
RUNPOD_POD_ID=your_pod_id
MODEL_PROFILE=test
```

Optional: `HF_TOKEN` if HuggingFace downloads are slow or gated.

Get pod ID from RunPod console URL or:

```bash
make list
```

### 4.3 Pod — clone repo and install (once per volume)

Open **RunPod web terminal** (or SSH) on the pod:

```bash
git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent
cd /workspace/local-ai-agent
cp .env.example .env
# Optional: add HF_TOKEN to .env on pod
bash scripts/install-on-pod.sh
```

What this does:

1. Clones **Iterative-Contextual-Refinements** to `/workspace/Iterative-Contextual-Refinements` if missing (ICR prompts).
2. Starts **vLLM** on `:8000` and **ICR gateway** on `:8787` (native Python, or Docker Compose if Docker exists).
3. First boot downloads the model — **allow 5–15 minutes**.

Check status on pod:

```bash
cd /workspace/local-ai-agent
curl -fsS http://127.0.0.1:8000/v1/models | head
curl -fsS http://127.0.0.1:8787/health
tail -f runs/vllm.log runs/gateway.log
```

**After pod reboot**, run install again (or add compose to pod start command):

```bash
cd /workspace/local-ai-agent && bash scripts/install-on-pod.sh
```

Production model on pod:

```bash
MODEL_PROFILE=production bash scripts/install-on-pod.sh
```

### 4.4 Cursor — BYOK (once)

From Mac:

```bash
cd /Users/tough/hobby/local-ai-agent
make cursor-config
```

In **Cursor Settings → Models**:

| Field | Value |
|-------|--------|
| **Override OpenAI Base URL** | `make gateway-url` output, e.g. `https://<POD_ID>-8787.proxy.runpod.net/v1` |
| **OpenAI API Key** | Your RunPod API key |
| **Custom model** | Must match profile — test: `Qwen/Qwen2.5-3B-Instruct`, production: `Qwen/Qwen3.6-27B` |

If connections fail: **Settings → Network → HTTP Compatibility Mode → HTTP/1.1**.

Disable other paid models if you only want RunPod.

---

## 5. Daily workflow (Mac)

```bash
cd /Users/tough/hobby/local-ai-agent
make ready
```

This runs:

1. `make start` — wake RunPod pod  
2. `make wait-all` — wait until vLLM (:8000) and gateway (:8787) respond  
3. `make test-gateway` — smoke test through gateway  
4. `make cursor-config` — print Cursor settings reminder  

If the pod was stopped, ensure gateway is still running on the pod (see §4.3 reboot note).

Stop pod when done (saves money):

```bash
make stop
```

---

## 6. Testing checklist

Run in order. Do not skip to Plan until Ask/Agent work.

### 6.1 Automated (Mac)

| Command | What it verifies | Speed |
|---------|------------------|-------|
| `make test` | vLLM direct (:8000) | ~30s |
| `make test-gateway` | Gateway health, models, chat, tools | ~1 min |
| `make test-pipeline` | Full ICR Plan route (CreatePlan → ICR → tool_calls) | **Several minutes** |

### 6.2 Manual (Cursor + bobot-xs-v1)

Open **bobot-xs-v1**, select your RunPod custom model.

**A. Ask** (passthrough)

- Mode: **Ask**
- Prompt: *What does VoiceChatManager.kt do?*
- Expect: normal answer

**B. Agent** (passthrough)

- Mode: **Agent**
- Prompt: *Add a one-line comment at the top of BotConfig.kt*
- Expect: file edit applies

**C. Plan** (ICR — the real test)

- Mode: **Plan** (Shift+Tab)
- Prompt: *Refactor the voice chat state machine in BotConfig.kt*
- Expect: long wait (ICR on pod), plan UI / CreatePlan flow
- Review plan → **Build** in Agent mode to implement

On pod during Plan, GPU should show sustained activity (many vLLM calls). Logs: `local-ai-agent/runs/` on pod volume.

---

## 7. Makefile reference

| Command | Where | Purpose |
|---------|--------|---------|
| `make help` | Mac | List commands |
| `make pod-up` | Mac | Print pod install one-liner |
| `make ready` | Mac | Full daily test flow |
| `make start` / `make stop` | Mac | RunPod pod power |
| `make status` | Mac | Pod + URLs JSON |
| `make wait-all` | Mac | Wait vLLM + gateway |
| `make test-gateway` | Mac | Gateway smoke test |
| `make test-pipeline` | Mac | ICR Plan smoke test |
| `make cursor-config` | Mac | Print Cursor BYOK fields |
| `make gateway-url` | Mac | Gateway OpenAI base URL |
| `make url` | Mac | Raw vLLM URL (debug only) |
| `make refine TASK='...'` | Mac | Optional CLI ICR (calls RunPod) |

---

## 8. Cursor modes (expected behavior)

| Cursor mode | Gateway behavior |
|-------------|------------------|
| **Plan** | Detects `CreatePlan` in tools → runs ICR → returns plan as `tool_calls` |
| **Agent** | Passthrough → vLLM (edits, shell, etc.) |
| **Ask** | Passthrough → vLLM |

You do not tell the agent to plan — Plan mode triggers ICR automatically at the API layer.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `make ready` fails on wait | Pod stopped or vLLM still loading | `make start`, wait 10+ min on first boot, `docker compose logs -f vllm` on pod |
| Gateway connection refused | Gateway container down or port 8787 not exposed | On pod: `bash scripts/install-on-pod.sh` |
| Cursor can't connect | Wrong URL (used :8000 instead of :8787) | Use `make gateway-url` |
| Ask/Agent fail, gateway OK | Wrong model name in Cursor | Must match `MODEL_NAME` in active profile |
| Plan returns 502 | ICR failed (ICR repo, vLLM, OOM) | `docker compose logs gateway`; check `ICR_REPO_HOST` mount |
| Plan acts like plain chat | CreatePlan not in request | See `docs/cursor-request-capture.md`; optional logger |
| Tool calls fail (Agent) | Wrong `TOOL_CALL_PARSER` | test: `hermes`, production: `qwen3_xml` — set in model profile |
| OOM on pod | Model too large for GPU | Stay on `MODEL_PROFILE=test` or reduce `MAX_MODEL_LEN` |

### Useful pod commands

```bash
cd /workspace/local-ai-agent
docker compose ps
docker compose logs -f
docker compose restart gateway
docker compose down && bash scripts/install-on-pod.sh
```

### Useful Mac commands

```bash
make status
curl -s "$(make -s gateway-url | sed 's|/v1||')/health"
```

---

## 10. Upgrade to production model

1. Mac `.env`: `MODEL_PROFILE=production`
2. Pod:

   ```bash
   cd /workspace/local-ai-agent
   MODEL_PROFILE=production bash scripts/install-on-pod.sh
   ```

3. Cursor custom model → `Qwen/Qwen3.6-27B`
4. `make ready` and re-test

Requires a GPU that fits 27B (see RunPod template sizing).

---

## 11. Architecture files (for debugging)

| Path | Role |
|------|------|
| `docker-compose.yml` | vLLM + gateway on pod |
| `scripts/install-on-pod.sh` | Pod bootstrap |
| `gateway/app.py` | OpenAI-compatible entry |
| `gateway/router.py` | Detect Plan vs passthrough |
| `gateway/icr_plan.py` | Run `refine.py` loop, return CreatePlan |
| `refine.py` | ICR Contextual loop |
| `config/models/test.env` | Small model settings |
| `config/models/production.env` | 27B settings |
| `smoke_test.py` | Mac smoke tests |
| `runpod.py` | Pod start/stop/wait |

---

## 12. Quick reference card

```
ONE-TIME
  Mac:  cp .env.example .env  →  RUNPOD_API_KEY, RUNPOD_POD_ID
  Pod:  git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent && bash scripts/install-on-pod.sh
  Cursor: make cursor-config → paste BYOK

DAILY
  make ready
  (use Cursor Plan / Agent on bobot-xs-v1)
  make stop   # when done

TEST PIPELINE
  make test-gateway
  make test-pipeline

PRODUCTION
  MODEL_PROFILE=production on Mac + pod install script
```

---

## 13. Related docs

- [cursor-byok.md](cursor-byok.md) — short BYOK summary  
- [cursor-request-capture.md](cursor-request-capture.md) — Phase 0 CreatePlan capture (optional)  
- [iterative-thinking.md](iterative-thinking.md) — ICR loop details  
- [config/gateway-deploy.example](../config/gateway-deploy.example) — pod deploy notes  

---

*Last aligned with: Docker Compose setup, `MODEL_PROFILE=test` default, gateway on pod :8787.*
