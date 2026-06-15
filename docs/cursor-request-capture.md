# Cursor request capture (Phase 0)

Validate Plan vs Agent request shapes **before** enabling ICR routing in the gateway.

## Procedure (on RunPod pod)

1. vLLM on :8000, gateway or logger on :8787
2. Cursor BYOK → `make gateway-url` (from Mac)
3. Run Plan / Agent / Ask once each on bobot-xs-v1
4. Inspect `captures/*.json` on the pod — filenames are auto-tagged `plan`, `agent`, or `ask`.

Example prompts:

| Mode | Example |
|---|---|
| **Plan** | Refactor voice chat state machine in BotConfig.kt |
| **Agent** | Add a one-line comment to BotConfig.kt |
| **Ask** | What does VoiceChatManager.kt do? |

## CreatePlan schema (baseline — verify against captures)

From [Cursor ACP `cursor/create_plan`](https://cursor.com/docs/cli/acp) and Plan mode behavior. The gateway expects Cursor BYOK to send a **function tool** in `tools[]` whose name contains `CreatePlan` / `create_plan`.

### Request signals (Plan mode)

```json
{
  "model": "Qwen/Qwen3.6-27B",
  "stream": true,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Refactor BotConfig.kt ..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "CreatePlan",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "overview": {"type": "string"},
            "plan": {"type": "string"},
            "todos": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": {"type": "string"},
                  "content": {"type": "string"},
                  "status": {"type": "string"}
                }
              }
            },
            "isProject": {"type": "boolean"}
          }
        }
      }
    }
  ]
}
```

**Verify from your capture:** exact `name` string, full `parameters`, whether `stream` is true.

### Response shape (ICR gateway returns)

When Plan tool is present, gateway **does not** passthrough. It runs ICR and returns:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_...",
        "type": "function",
        "function": {
          "name": "CreatePlan",
          "arguments": "{\"name\":\"...\",\"overview\":\"...\",\"plan\":\"...\",\"todos\":[...],\"isProject\":false}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Tool name in response must **match** the name from the request capture.

### Agent mode (passthrough)

Captures tagged `agent` should include edit tools (`Write`, `StrReplace`, etc.) and **no** plan tool. Gateway forwards these unchanged to vLLM.

### Ask mode (passthrough)

Captures tagged `ask` — read-only or minimal tools. Gateway forwards unchanged.

## Gate

Before enabling ICR routing in production:

- [ ] At least one `captures/*_plan.json` exists
- [ ] Plan tool `name` confirmed
- [ ] Agent/Ask captures confirm passthrough path

If capture differs from baseline, update [`gateway/router.py`](../gateway/router.py) `PLAN_TOOL_HINTS` and [`gateway/icr_plan.py`](../gateway/icr_plan.py) argument builder.

## Sample capture location

```
local-ai-agent/captures/
  20260615T120000_plan.json
  20260615T120100_agent.json
  20260615T120200_ask.json
```

Captures redact `Authorization` headers and API keys automatically.
