"""
FastAPI health-check server for Railway / Render uptime monitoring.

Endpoints
---------
GET /health
    Returns 200 OK with a JSON payload summarising system state.
    Returns 503 if the database is unreachable.

GET /ready
    Kubernetes-style readiness probe — returns 200 when all critical
    services (DB + Anthropic key) are available.

Usage
-----
    uvicorn deployment.health_server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="News Trading System — Health API", version="1.0.0")

_start_time = time.time()


def _check_database() -> tuple[bool, str]:
    """Try a lightweight DB query; return (ok, message)."""
    try:
        from storage.database import Database
        db = Database()
        db.get_recent_runs(limit=1)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_anthropic() -> tuple[bool, str]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return True, "key set"
    return False, "ANTHROPIC_API_KEY not set"


def _check_newsapi() -> tuple[bool, str]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if key:
        return True, "key set"
    return False, "NEWSAPI_KEY not set"


def _last_scheduler_run() -> str | None:
    """Return ISO timestamp of the most recent scheduler run, or None."""
    try:
        from storage.database import Database
        db = Database()
        rows = db._select(
            "SELECT run_at FROM scheduler_logs ORDER BY id DESC LIMIT 1"
        )
        return rows[0]["run_at"] if rows else None
    except Exception:
        return None


@app.get("/health")
def health() -> JSONResponse:
    """Full system health check."""
    db_ok, db_msg = _check_database()
    anthropic_ok, anthropic_msg = _check_anthropic()
    newsapi_ok, newsapi_msg = _check_newsapi()

    all_ok = db_ok and anthropic_ok

    payload = {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - _start_time, 1),
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "checks": {
            "database": {"ok": db_ok, "message": db_msg},
            "anthropic_key": {"ok": anthropic_ok, "message": anthropic_msg},
            "newsapi_key": {"ok": newsapi_ok, "message": newsapi_msg},
        },
        "last_scheduler_run": _last_scheduler_run(),
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(content=payload, status_code=status_code)


@app.get("/ready")
def ready(response: Response) -> dict:
    """Readiness probe — used by Railway to decide if traffic can be routed."""
    db_ok, _ = _check_database()
    anthropic_ok, _ = _check_anthropic()
    if db_ok and anthropic_ok:
        return {"ready": True}
    response.status_code = 503
    return {"ready": False}


@app.get("/")
def root() -> dict:
    """Root endpoint — confirms the health server is running."""
    return {
        "service": "news-trading-health",
        "status": "running",
        "docs": "/docs",
    }
