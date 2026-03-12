#!/bin/sh
# entrypoint.sh — Start both the Streamlit dashboard and the daily scheduler.

echo "[entrypoint] PORT=$PORT"
echo "[entrypoint] Starting scheduler in background..."
python3 -m scheduler.daily_runner 2>&1 &

echo "[entrypoint] Starting Streamlit on port ${PORT:-8501}..."
exec streamlit run dashboard/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
