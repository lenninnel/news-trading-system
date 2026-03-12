#!/bin/sh
# entrypoint.sh — Start both the Streamlit dashboard and the daily scheduler.

echo "[entrypoint] Starting up..."
echo "[entrypoint] PORT=${PORT:-8501}"
echo "[entrypoint] PYTHONPATH=/app"

export PYTHONPATH=/app

# Start scheduler in background (non-fatal if it fails)
echo "[entrypoint] Launching scheduler in background..."
python3 -m scheduler.daily_runner 2>&1 &

echo "[entrypoint] Starting Streamlit dashboard on port ${PORT:-8501}..."
exec streamlit run dashboard/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
