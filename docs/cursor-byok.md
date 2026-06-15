# Cursor BYOK — automated setup

## One-time

**Mac:** `cp .env.example .env` → fill `RUNPOD_API_KEY`, `RUNPOD_POD_ID`

**Pod:** clone repo, then:

```bash
bash scripts/install-on-pod.sh
```

Uses Docker Compose + **`Qwen/Qwen2.5-3B-Instruct`** (`MODEL_PROFILE=test`).

Pod template ports: **8000**, **8787**.

## Daily (Mac)

```bash
make ready
make cursor-config   # paste into Cursor
```

## Tests

| Command | What |
|---------|------|
| `make test-gateway` | Fast — passthrough + tools |
| `make test-pipeline` | Slow — full ICR Plan smoke |

## Production model

On pod:

```bash
MODEL_PROFILE=production bash scripts/install-on-pod.sh
```

On Mac `.env`: `MODEL_PROFILE=production`

Requires strong GPU (Qwen3.6-27B).
