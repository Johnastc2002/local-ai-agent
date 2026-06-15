#!/usr/bin/env bash
# Local readiness check — run before pod deploy or Cursor BYOK.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== local-ai-agent readiness ==="

echo "[1/3] Python compile..."
find . -name '*.py' -not -path './.venv/*' -print0 | xargs -0 python3 -m py_compile

echo "[2/3] Unit tests..."
python3 -m unittest discover -s tests -p 'test_*.py' -v

echo "[3/3] Required files..."
for f in gateway/app.py refine.py agent_call.py scripts/install-on-pod.sh config/models/test.env; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done

if [[ -f .env ]] && grep -q '^RUNPOD_POD_ID=' .env 2>/dev/null; then
  echo ""
  echo "Optional pod check (make test-gateway)..."
  if make test-gateway 2>/dev/null; then
    echo "Pod gateway: OK"
  else
    echo "Pod gateway: not reachable (run make ready when pod is stopped)"
  fi
fi

echo ""
echo "READY — deploy with:"
echo "  git push origin master   # if not pushed yet"
echo "  # on pod:"
echo "  cd /workspace/local-ai-agent && git pull && bash scripts/install-on-pod.sh"
echo "  # on Mac:"
echo "  make ready && make cursor-config"
