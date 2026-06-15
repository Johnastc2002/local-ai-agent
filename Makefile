.PHONY: help ready verify test test-gateway test-pipeline cursor-config logger \
       start stop wait wait-all status list url gateway-url \
       pod-up pod-up-prod refine test-tools

help:
	@echo "ICR gateway on RunPod — you edit .env once, then:"
	@echo ""
	@echo "  make verify        Local checks before deploy (run this first)"
	@echo "  make pod-up        Print pod install command (run in RunPod terminal)"
	@echo "  make ready         Mac: start pod + wait + test gateway + print Cursor settings"
	@echo "  make test          Test vLLM (:8000)"
	@echo "  make test-gateway  Test gateway passthrough (:8787)"
	@echo "  make test-pipeline Test ICR Plan route (slow, uses MODEL_PROFILE)"
	@echo "  make test-tools     Unit test Cursor tool parsing"
	@echo "  make logger         Mac: capture Cursor requests to captures/"
	@echo "  make cursor-config Print Cursor BYOK settings"
	@echo ""
	@echo "  make start | stop | status | wait | wait-all"
	@echo ""
	@echo "Model profiles: MODEL_PROFILE=test (default, Qwen2.5-3B) or production"

# --- Mac workflow ---

verify:
	bash scripts/verify-ready.sh

ready: start wait-all test-gateway cursor-config

test:
	python3 smoke_test.py

test-gateway:
	python3 smoke_test.py --gateway

test-tools:
	python3 -m unittest discover -s tests -p 'test_*.py' -v

logger:
	python3 tools/request_logger.py

test-pipeline:
	python3 smoke_test.py --gateway --plan-smoke

cursor-config:
	@echo ""
	@echo "=== Cursor Settings → Models ==="
	@echo "Override OpenAI Base URL:"
	@python3 runpod.py gateway-url
	@echo "API Key:              (your RUNPOD_API_KEY from .env)"
	@python3 -c "import re; e=open('.env').read() if __import__('pathlib').Path('.env').exists() else ''; \
m=re.search(r'^MODEL_PROFILE=(\\w+)', e, re.M); p=m.group(1) if m else 'test'; \
lines=open(f'config/models/{p}.env').read() if __import__('pathlib').Path(f'config/models/{p}.env').exists() else ''; \
mn=re.search(r'^MODEL_NAME=(.+)', lines, re.M); print('Custom model:        ', mn.group(1) if mn else 'see config/models/')"
	@echo ""

pod-up:
	@echo "=== RunPod pod template ==="
	@echo "  Image:   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
	@echo "  Volume:  /workspace (80 GB)"
	@echo "  Ports:   8000,8787"
	@echo ""
	@echo "=== Run ONCE in RunPod web terminal ==="
	@echo ""
	@echo "  git clone https://github.com/Johnastc2002/local-ai-agent.git /workspace/local-ai-agent"
	@echo "  cd /workspace/local-ai-agent && bash scripts/install-on-pod.sh"
	@echo ""
	@echo "  bash scripts/pod-status.sh    # check"
	@echo "  bash scripts/pod-logs.sh      # watch first boot"
	@echo ""
	@echo "Production model:"
	@echo "  MODEL_PROFILE=production bash scripts/install-on-pod.sh"
	@echo ""
	@echo "Full runbook: docs/MANUAL.md"

pod-up-prod:
	@echo "  cd /workspace/local-ai-agent && MODEL_PROFILE=production bash scripts/install-on-pod.sh"

# --- RunPod control ---

list:
	python3 runpod.py list

start:
	python3 runpod.py start

stop:
	python3 runpod.py stop

status:
	python3 runpod.py status

url:
	python3 runpod.py url

gateway-url:
	python3 runpod.py gateway-url

wait:
	python3 runpod.py wait

wait-all:
	python3 runpod.py wait-all

refine:
	@test -n "$(TASK)" || (echo "Usage: make refine TASK='...' [ATTACH=path/to/file]" && exit 1)
	python3 refine.py "$(TASK)" $(if $(ATTACH),--attach $(ATTACH),)
