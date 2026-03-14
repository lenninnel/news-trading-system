#!/bin/sh
export STREAMLIT_SERVER_PORT="${PORT:-8501}"
echo "[entrypoint] Listening on port $STREAMLIT_SERVER_PORT"

echo "[entrypoint] /data check: exists=$(test -d /data && echo yes || echo no) writable=$(test -w /data && echo yes || echo no)"
ls -la /data/ 2>/dev/null || echo "[entrypoint] /data not accessible"

echo "[entrypoint] Initialising database tables..."
python3 -c "from storage.database import Database; db=Database(); print(f'[entrypoint] DB path: {db.db_path}')"

# Start Streamlit FIRST so the healthcheck passes while the pipeline runs
echo "[entrypoint] Starting Streamlit..."
streamlit run dashboard/app.py \
    --server.port="$STREAMLIT_SERVER_PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false &
STREAMLIT_PID=$!

# Give Streamlit a moment to bind the port
sleep 3

echo "[entrypoint] Running initial pipeline in background..."
python3 -m scheduler.daily_runner --now --workers 2 2>&1 &

echo "[entrypoint] Starting background scheduler..."
python3 -m scheduler.daily_runner 2>&1 &

# Wait on Streamlit (the main process)
wait $STREAMLIT_PID
