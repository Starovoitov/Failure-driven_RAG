#!/usr/bin/env bash
set -euo pipefail

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
EXTERNAL_PORT="${EXTERNAL_PORT:-5173}"

if [ "${FRONTEND_PORT}" != "${EXTERNAL_PORT}" ]; then
  echo "FRONTEND_PORT (${FRONTEND_PORT}) and EXTERNAL_PORT (${EXTERNAL_PORT}) differ."
  echo "Using FRONTEND_PORT for in-container Vite server and EXTERNAL_PORT for host mapping."
fi

echo "Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}"
python -m uvicorn api.server:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "Starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}"
npm --prefix /app/FE run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" --strictPort &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping services..."
  kill "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
}

trap cleanup INT TERM
wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
cleanup
