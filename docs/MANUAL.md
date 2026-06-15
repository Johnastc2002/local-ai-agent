# ICR Gateway — Operator Manual

**Everything you need.** Read this once, bookmark it, done.

**Repo:** https://github.com/Johnastc2002/local-ai-agent

---

## What this does

```
Cursor (Mac) ──BYOK :8787──► RunPod pod
                               gateway
                                 │
                                 └─ every user turn → ICR pipeline first
                                      ├─ Plan  → CreatePlan tool_calls
                                      ├─ Agent → ICR + vLLM tools
                                      └─ Ask   → ICR answer text
```

- **Mac:** Cursor + `make` commands only. No local LLM.
- **Pod:** vLLM + gateway. No Docker required (RunPod PyTorch pods don't have it).
- **Cursor:** Standard Plan / Agent / Ask. No hooks, rules, or MCP.
- **ICR runs on every new user message** (Plan, Agent, Ask). Agent tool-result turns skip ICR and continue normally.

---

## 1. RunPod pod template (create once)

| Setting | Value |
|---------|--------|
| **Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **Container disk** | 40 GB |
| **Volume disk** | 80 GB |
| **Volume mount** | `/workspace` |
| **Expose HTTP ports** | `8000,8787` |

Rent any GPU that fits your model profile (see §5).

---

## 2. Mac setup (once)

```bash
cd /Users/tough/hobby/local-ai-agent   # or your clone path
cp .env.example .env
```

Edit `.env` — **only these two are required on Mac:**

```bash
RUNPOD_API_KEY=rp_...
RUNPOD_POD_ID=your_pod_id
MODEL_PROFILE=test
```

Get pod ID from the RunPod console URL or `make list`.

---

## 3. Pod setup (once per volume)

Open **RunPod web terminal** on your pod:

```bash
git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent
cd /workspace/local-ai-agent
bash scripts/install-on-pod.sh
```

The script creates `.env`, clones ICR prompts, clones **your codebase** (see §3b), installs vLLM, starts both services.

### 3b. Codebase on pod (required)

ICR agents use `read_file`, `grep`, and `list_dir` on the pod — same idea as Opus reading your repo in Cursor.

On pod `.env`:

```bash
CODEBASE_ROOT=/workspace/bobot-xs-v1
CODEBASE_HOST_ROOT=/Users/tough/hobby/bobot-xs-v1   # Mac paths from Cursor → pod
CODEBASE_GIT_URL=https://github.com/you/bobot-xs-v1.git   # optional auto-clone
```

Or clone manually once:

```bash
git clone https://github.com/you/bobot-xs-v1.git /workspace/bobot-xs-v1
```

Without this, ICR runs blind and **will fail** with `CODEBASE_ROOT is missing`.

**First boot:** model download takes **5–15 minutes**. Watch:

```bash
bash scripts/pod-logs.sh
```

When ready:

```bash
bash scripts/pod-status.sh
```

Both lines should show `OK`.

---

## 4. Cursor BYOK (once)

On Mac:

```bash
make cursor-config
```

Paste into **Cursor → Settings → Models**:

| Field | Value |
|-------|--------|
| **Override OpenAI Base URL** | `https://<POD_ID>-8787.proxy.runpod.net/v1` |
| **OpenAI API Key** | Your RunPod API key (same as Mac `.env`) |
| **Custom model** | `Qwen/Qwen2.5-3B-Instruct` (test) or `Qwen/Qwen3.6-27B` (production) |

If connection fails: **Settings → Network → HTTP Compatibility Mode → HTTP/1.1**

Use **:8787** (gateway), not :8000 (raw vLLM).

---

## 5. Model profiles

| Profile | Model | VRAM | When |
|---------|--------|------|------|
| **test** (default) | `Qwen/Qwen2.5-3B-Instruct` | ~6–8 GB | Pipeline testing |
| **production** | `Qwen/Qwen3.6-27B` | Strong GPU | Real work |

Same profile on **Mac `.env`** and **pod install**:

```bash
MODEL_PROFILE=production bash scripts/install-on-pod.sh
```

---

## 6. Daily workflow (Mac)

```bash
make ready          # start pod, wait, test gateway, print Cursor settings
# … use Cursor on your project …
make stop           # stop pod (save money)
```

---

## 7. After pod reboot

Services don't auto-start. On pod:

```bash
cd /workspace/local-ai-agent
git pull            # get latest scripts
bash scripts/install-on-pod.sh
```

Optional: add that line to your pod **Start Command** in RunPod template.

---

## 8. Testing

**Automated (Mac):**

| Command | Checks |
|---------|--------|
| `make test-gateway` | Gateway health + chat (~1 min) |
| `make test-pipeline` | Full ICR Plan route (several min) |

**Manual (Cursor):**

1. **Ask** — simple question → ICR-refined answer (slow first time)
2. **Agent** — small edit → ICR first, then tools/edits (slow per user message)
3. **Plan** — real task → ICR → plan UI, then **Build** in Agent

| Mode | ICR? |
|------|------|
| Plan | Yes → CreatePlan |
| Agent (new user message) | Yes → then vLLM |
| Agent (tool results) | No — continues chain |
| Ask | Yes → answer text |

Debug passthrough only: set `ICR_MODE=off` in pod `.env` and restart gateway.

---

## 9. Commands cheat sheet

### Mac

| Command | Purpose |
|---------|---------|
| `make ready` | Full daily startup + test |
| `make start` / `make stop` | Power pod |
| `make status` | Pod info + URLs |
| `make cursor-config` | Print BYOK fields |
| `make gateway-url` | Gateway URL only |
| `make test-gateway` | Smoke test |
| `make pod-up` | Print pod install block |

### Pod

| Command | Purpose |
|---------|---------|
| `bash scripts/install-on-pod.sh` | Install / restart everything |
| `bash scripts/pod-status.sh` | Health check |
| `bash scripts/pod-logs.sh` | Tail logs |
| `bash scripts/pod-stop.sh` | Stop services |

---

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Docker not found` | **Ignore** — use `bash scripts/install-on-pod.sh` (native). Don't install Docker. |
| `make ready` hangs | Pod stopped or vLLM still loading. `make start`, wait 10+ min, `bash scripts/pod-logs.sh` on pod |
| Cursor can't connect | Wrong URL — use `:8787` not `:8000`. Run `make gateway-url` |
| Wrong model errors | Custom model in Cursor must match `MODEL_NAME` in active profile |
| Plan feels like plain chat | Wrong mode, or `ICR_MODE=off` — check `curl localhost:8787/health` |
| Agent very slow | Expected — ICR runs before every new user message |
| OOM on pod | Stay on `MODEL_PROFILE=test` or use bigger GPU |
| vLLM install fails | Paste last 30 lines of `runs/vllm.log` |

---

## 11. Architecture files

| Path | Role |
|------|------|
| `scripts/install-on-pod.sh` | Pod bootstrap (main entry) |
| `gateway/router.py` | ICR routing + response shaping |
| `refine.py` | ICR loop |
| `config/models/test.env` | Small model settings |
| `config/models/production.env` | 27B settings |
| `optional/` | Docker Compose (advanced, rarely needed) |

---

## 12. Copy-paste card

```
ONE-TIME
  RunPod template: pytorch image, /workspace volume, ports 8000+8787
  Mac:  cp .env.example .env → RUNPOD_API_KEY, RUNPOD_POD_ID
  Pod:  git clone …/local-ai-agent.git /workspace/local-ai-agent
        cd /workspace/local-ai-agent && bash scripts/install-on-pod.sh
  Cursor: make cursor-config → paste BYOK (:8787, RunPod key, model name)

DAILY
  make ready
  make stop

REBOOT
  cd /workspace/local-ai-agent && bash scripts/install-on-pod.sh
```
