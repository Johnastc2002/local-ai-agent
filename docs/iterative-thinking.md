# Iterative thinking (ICR Contextual port)

This module runs **Iterative-Contextual-Refinements Contextual mode** against RunPod using the **same OpenAI-compatible API** as hosted providers.

## What changed from v1

| Before | Now |
|---|---|
| Short local prompts | **Full** `ContextualPrompts.ts` via `ICR_REPO` |
| Text-only | **Multimodal** `--attach` (files + images) |
| No tools | **Python tool loop** (`python_virtual_filesystem`) |
| Max 20 iterations | **Unlimited** until `<<<Exit>>>` (ICR default) |
| Pool size 5 | **Pool size 12** (ICR default) |
| Simple HTTP | **OpenAI message format** + tools + retries |

## Loop (matches ContextualCore.ts)

1. **Main Generator** — produce/revise artifact (optional Python tools)
2. **Iterative Agent** — exactly 5 critical questions, no fixes
3. **Strategic Pool** — N orthogonal strategies; may emit `<<<Exit>>>`
4. Feed critique + pool to Main Generator
5. Every **10** turns: **Memory Agent** condenses and resets histories

## RunPod = model API

`llm.py` sends the same payloads OpenAI expects:

```json
{
  "model": "Qwen/Qwen3.6-27B",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "stdout: ..."}
  ],
  "tools": [...],
  "tool_choice": "auto"
}
```

Configure RunPod vLLM with `TOOL_CALL_PARSER=qwen3_xml` for tool calling.

## Usage with code context

```bash
python refine.py \
  "Refactor voice chat state machine for clarity and testability" \
  --attach /path/to/BotConfig.kt \
  --attach /path/to/VoiceChatManager.kt \
  -o refactor-plan.md
```

Then in Cursor (RunPod BYOK): `Implement refactor-plan.md`.

## Still not in this port

- **Deepthink** (parallel DFS/BFS, hypotheses, Final Judge) — run ICR web UI against RunPod
- **Agentic** mode — use Cursor Agent
- Per-agent model overrides — single `MODEL_NAME` for all agents (ICR allows per-agent models)

## Requirements

- Clone [Iterative-Contextual-Refinements](https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements) and set `ICR_REPO` in `.env`
- RunPod pod with Qwen3.6-27B + tool calling + optional vision for images
