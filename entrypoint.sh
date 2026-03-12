#!/bin/sh
# entrypoint.sh — Start Streamlit dashboard + daily scheduler.
# Railway injects $PORT at container runtime.

export STREAMLIT_SERVER_PORT="${PORT:-8501}"
echo "[entrypoint] Listening on port $STREAMLIT_SERVER_PORT"

python3 -m scheduler.daily_runner 2>&1 &

exec streamlit run dashboard/app.py \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
