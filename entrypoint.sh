#!/bin/sh
export STREAMLIT_SERVER_PORT="${PORT:-8501}"
echo "[entrypoint] Listening on port $STREAMLIT_SERVER_PORT"

echo "[entrypoint] /data check: exists=$(test -d /data && echo yes || echo no) writable=$(test -w /data && echo yes || echo no)"
ls -la /data/ 2>/dev/null || echo "[entrypoint] /data not accessible"

echo "[entrypoint] Initialising database tables..."
python3 -c "from storage.database import Database; db=Database(); print(f'[entrypoint] DB path: {db.db_path}')"

echo "[entrypoint] Running initial pipeline..."
python3 -m scheduler.daily_runner --now 2>&1

echo "[entrypoint] Starting background scheduler..."
python3 -m scheduler.daily_runner 2>&1 &

echo "[entrypoint] Starting Streamlit..."
exec streamlit run dashboard/app.py \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
