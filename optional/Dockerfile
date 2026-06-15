# Gateway image (vLLM uses official image in compose)
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ ./gateway/
COPY llm.py agent_call.py icr_prompts.py python_tool.py attachments.py refine.py \
     codebase_tool.py codebase_paths.py ./

ENV GATEWAY_ON_POD=1
ENV VLLM_UPSTREAM=http://vllm:8000
ENV PROXY_PORT=8787
ENV ICR_REPO=/icr

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8787/health || exit 1

CMD ["python", "-m", "gateway.app"]
