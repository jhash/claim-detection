#!/usr/bin/env bash
# Run the API locally without Docker. Useful for fast iteration.
#
# Usage:
#   ./scripts/serve_local.sh           # sync mode (no Redis, no worker)
#   QUEUE=1 ./scripts/serve_local.sh   # queued mode (assumes local Redis on :6379)

set -eu
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}

echo "starting API on http://$HOST:$PORT (QUEUE=${QUEUE:-0})"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
