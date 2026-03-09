#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=/tmp/rag_backend.log
PID_FILE=/tmp/rag_backend.pid

echo "Starting backend..."
RETRIEVER_TYPE=hybrid HYBRID_ALPHA=0.7 uvicorn backend.main:app --host 127.0.0.1 --port 8000 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
PID=$(cat "$PID_FILE")

cleanup() {
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping backend..."
    kill "$PID" || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Waiting for backend..."
READY=0
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "Backend failed to become ready."
  echo "==== Backend log ===="
  cat "$LOG_FILE"
  exit 1
fi

echo "Backend is ready."
echo "Running evaluation..."

python eval/evaluate.py \
  --base_url http://127.0.0.1:8000 \
  --dataset eval/v1.jsonl \
  --modes baseline \
  --timeout_s 120