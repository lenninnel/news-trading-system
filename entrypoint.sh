#!/bin/sh
export STREAMLIT_SERVER_PORT="${PORT:-8501}"
echo "[entrypoint] Listening on port $STREAMLIT_SERVER_PORT"

echo "[entrypoint] /data check: exists=$(test -d /data && echo yes || echo no) writable=$(test -w /data && echo yes || echo no)"
ls -la /data/ 2>/dev/null || echo "[entrypoint] /data not accessible"

echo "[entrypoint] Initialising database tables..."
python3 -c "from storage.database import Database; db=Database(); print(f'[entrypoint] DB path: {db.db_path}')"

# Start Streamlit FIRST so the healthcheck passes while the pipeline runs
echo "[entrypoint] Starting Streamlit on port $STREAMLIT_SERVER_PORT..."
streamlit run dashboard/app.py \
    --server.port="$STREAMLIT_SERVER_PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true &
STREAMLIT_PID=$!

# Wait for Streamlit to become ready (up to 60s)
echo "[entrypoint] Waiting for Streamlit healthcheck..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$STREAMLIT_SERVER_PORT/_stcore/health" > /dev/null 2>&1; then
        echo "[entrypoint] Streamlit ready after ${i}s"
        break
    fi
    sleep 1
done

# Clean ghost positions/trades from previous runs (belt-and-suspenders).
# The daemon also runs this at startup, but doing it here catches issues
# even if the daemon import fails.
echo "[entrypoint] Cleaning ghost trades from DB..."
python3 scripts/clean_ghost_trades.py --apply 2>&1 || echo "[entrypoint] Ghost cleanup skipped (non-fatal)"

# Start ONLY the daemon scheduler — it handles all runs including
# an immediate first run if within trading hours.
# Do NOT also run --now separately — that causes duplicate trades.
echo "[entrypoint] Starting daemon scheduler (4 runs/day, weekdays UTC)..."
python3 -m scheduler.daily_runner --daemon 2>&1 &
DAEMON_PID=$!

# Keep container alive — only restart if Streamlit dies.
# If the daemon crashes, trading stops but the dashboard stays up
# (prevents restart loops that trigger ghost trades on every boot).
wait $STREAMLIT_PID
