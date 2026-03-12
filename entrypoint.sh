#!/bin/sh
# entrypoint.sh — Start Streamlit dashboard + daily scheduler.
echo "[entrypoint] STREAMLIT_SERVER_PORT=$STREAMLIT_SERVER_PORT"

python3 -m scheduler.daily_runner 2>&1 &

exec streamlit run dashboard/app.py \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
