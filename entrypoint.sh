#!/bin/sh
echo "[entrypoint] /data check: exists=$(test -d /data && echo yes || echo no) writable=$(test -w /data && echo yes || echo no)"
ls -la /data/ 2>/dev/null || echo "[entrypoint] /data not accessible"

# ── Deploy overlap guard ────────────────────────────────────────
# If a previous container is still draining, wait for it to die.
LOCKFILE="/data/daemon.lock"
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(head -1 "$LOCKFILE" 2>/dev/null)
    echo "[entrypoint] Lock file exists (old PID=$OLD_PID). Waiting up to 30s for old container to die..."
    waited=0
    while [ $waited -lt 30 ]; do
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[entrypoint] Old process gone after ${waited}s"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if [ $waited -ge 30 ]; then
        echo "[entrypoint] Timeout waiting for old process — continuing anyway"
    fi
fi
# Write our PID + timestamp
echo "$$" > "$LOCKFILE"
date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$LOCKFILE"
echo "[entrypoint] Lock written (PID=$$)"

echo "[entrypoint] Initialising database tables..."
python3 -c "from storage.database import Database; db=Database(); print(f'[entrypoint] DB path: {db.db_path}')"

# Start Streamlit on fixed internal port (NOT $PORT — that's for FastAPI)
echo "[entrypoint] Starting Streamlit on internal port 8501..."
streamlit run dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true &
STREAMLIT_PID=$!

# Wait for Streamlit to become ready (up to 60s)
echo "[entrypoint] Waiting for Streamlit healthcheck..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:8501/_stcore/health" > /dev/null 2>&1; then
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

# Start daemon scheduler in background
echo "[entrypoint] Starting daemon scheduler (4 runs/day, weekdays UTC)..."
python3 -m scheduler.daily_runner --daemon 2>&1 &

# Start FastAPI on Railway's public port (foreground, PID 1).
# FastAPI serves /api/* directly and proxies everything else to Streamlit on :8501.
echo "[entrypoint] Starting FastAPI on public port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
