#!/bin/sh
# entrypoint.sh — Start both the Streamlit dashboard and the daily scheduler.
#
# The scheduler runs as a background process; Streamlit runs in the foreground
# so Docker can track its health and PID correctly.

set -e

echo "[entrypoint] Starting scheduler in background..."
python3 -m scheduler.daily_runner &

echo "[entrypoint] Starting Streamlit dashboard on port ${PORT:-8501}..."
exec streamlit run dashboard/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
