"""
Read-only FastAPI layer for the News Trading System.

Serves as the bridge between the trading bot and the React dashboard.
All endpoints are GET-only — the API never controls the trading system.

Start:  uvicorn api.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import DB_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_START_TIME = time.monotonic()

app = FastAPI(title="News Trading API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://news-trading-system-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB helpers (read-only, no locks needed)
# ---------------------------------------------------------------------------

def _resolve_db_path() -> str:
    railway_dir = "/data"
    if os.path.isdir(railway_dir) and os.access(railway_dir, os.W_OK):
        return os.path.join(railway_dir, "news_trading.db")
    return DB_PATH


_DB_PATH = _resolve_db_path()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query and return a list of dicts."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = _query(sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Schedule constants (mirrored from scheduler — no imports from scheduler)
# ---------------------------------------------------------------------------

_SCHEDULE = [
    {"name": "XETRA_OPEN", "hour": 7, "minute": 0},
    {"name": "US_OPEN", "hour": 14, "minute": 30},
    {"name": "MIDDAY", "hour": 18, "minute": 0},
    {"name": "EOD", "hour": 22, "minute": 15},
]


def _load_watchlist() -> list[str]:
    """Load watchlist from config/watchlist.yaml without importing scheduler."""
    try:
        import yaml

        path = os.path.join(_PROJECT_ROOT, "config", "watchlist.yaml")
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("watchlist", [])
    except Exception:
        return []


def _last_and_next_session() -> tuple[dict | None, dict | None]:
    """Determine last completed and next upcoming session based on UTC time."""
    now = datetime.now(timezone.utc)
    today_minutes = now.hour * 60 + now.minute

    last_session = None
    next_session = None

    for entry in _SCHEDULE:
        entry_minutes = entry["hour"] * 60 + entry["minute"]
        if today_minutes >= entry_minutes:
            last_session = entry
        elif next_session is None:
            next_session = entry

    return last_session, next_session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
def status() -> dict:
    uptime = int(time.monotonic() - _START_TIME)
    mode = os.environ.get("TRADING_MODE", "paper_local")
    watchlist = _load_watchlist()

    last, nxt = _last_and_next_session()
    today = date.today()

    # Check signal_events for the actual last run
    last_event = _query_one(
        "SELECT session, timestamp FROM signal_events ORDER BY id DESC LIMIT 1"
    )

    last_session_name = None
    last_run_at = None
    if last_event:
        last_session_name = last_event.get("session")
        last_run_at = last_event.get("timestamp")
    elif last:
        last_session_name = last["name"]

    next_session_name = None
    next_run_at = None
    if nxt:
        next_session_name = nxt["name"]
        next_run_at = datetime(
            today.year, today.month, today.day,
            nxt["hour"], nxt["minute"],
            tzinfo=timezone.utc,
        ).isoformat()

    return {
        "running": True,
        "uptime_seconds": uptime,
        "last_session": last_session_name,
        "last_run_at": last_run_at,
        "next_session": next_session_name,
        "next_run_at": next_run_at,
        "watchlist": watchlist,
        "mode": mode,
    }


@app.get("/api/signals")
def signals(
    limit: int = Query(50, ge=1, le=500),
    ticker: str | None = Query(None),
    days: int = Query(7, ge=1, le=365),
) -> list[dict]:
    clauses = ["timestamp >= ?"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    params: list = [cutoff]

    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())

    where = "WHERE " + " AND ".join(clauses)

    rows = _query(
        f"SELECT id, timestamp, session, ticker, strategy, signal, "
        f"confidence, rsi, sma_ratio, volume_ratio, "
        f"sentiment_score, news_score, social_score, "
        f"bull_case, bear_case, debate_outcome, "
        f"price_at_signal, outcome_3d_pct, outcome_5d_pct, outcome_10d_pct, "
        f"trade_executed "
        f"FROM signal_events {where} "
        f"ORDER BY id DESC LIMIT ?",
        tuple(params + [limit]),
    )

    # Normalise trade_executed to bool
    for r in rows:
        r["trade_executed"] = bool(r.get("trade_executed"))
    return rows


@app.get("/api/portfolio")
def portfolio() -> dict:
    # Try Alpaca first
    alpaca_data = _try_alpaca_portfolio()
    if alpaca_data:
        return alpaca_data

    # Fallback to local DB
    positions = _query("SELECT * FROM portfolio_positions ORDER BY ticker")

    total_value = sum(p.get("current_value", 0) or 0 for p in positions)

    # Daily PnL from trade_history
    today_str = date.today().isoformat()
    pnl_row = _query_one(
        "SELECT COALESCE(SUM(pnl), 0) AS daily_pnl FROM trade_history "
        "WHERE date(created_at) = date(?)",
        (today_str,),
    )
    daily_pnl = pnl_row["daily_pnl"] if pnl_row else 0.0
    daily_pnl_pct = (daily_pnl / total_value * 100) if total_value else 0.0

    pos_list = []
    for p in positions:
        shares = p.get("shares", 0)
        avg_price = p.get("avg_price", 0)
        current_value = p.get("current_value", 0) or 0
        current_price = (current_value / shares) if shares else 0
        cost = avg_price * shares if shares else 0
        pnl_pct = ((current_value - cost) / cost * 100) if cost else 0
        pos_list.append({
            "ticker": p.get("ticker"),
            "shares": shares,
            "entry": avg_price,
            "current": round(current_price, 2),
            "pnl_pct": round(pnl_pct, 1),
        })

    # Cash estimate: account balance minus invested
    total_invested = sum(
        (p.get("avg_price", 0) or 0) * (p.get("shares", 0) or 0)
        for p in positions
    )
    account_row = _query_one(
        "SELECT account_balance FROM risk_calculations ORDER BY id DESC LIMIT 1"
    )
    account_balance = account_row["account_balance"] if account_row else 10_000.0
    cash = account_balance - total_invested

    return {
        "value": round(total_value, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl_pct, 2),
        "positions": pos_list,
        "cash": round(cash, 2),
    }


def _try_alpaca_portfolio() -> dict | None:
    """Try reading portfolio from Alpaca API. Returns dict or None."""
    try:
        import alpaca_trade_api as tradeapi

        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        mode = os.environ.get("ALPACA_MODE", "paper").lower()
        if not key or not secret or "DISABLED" in key:
            return None
        base_url = (
            "https://api.alpaca.markets"
            if mode == "live"
            else "https://paper-api.alpaca.markets"
        )
        api = tradeapi.REST(
            key_id=key, secret_key=secret,
            base_url=base_url, api_version="v2",
        )
        account = api.get_account()
        positions = api.list_positions()

        portfolio_value = float(account.portfolio_value)
        last_equity = float(account.last_equity)
        daily_pnl = portfolio_value - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0

        pos_list = []
        for p in positions:
            pos_list.append({
                "ticker": p.symbol,
                "shares": int(p.qty),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
                "pnl_pct": round(float(p.unrealized_plpc) * 100, 1),
            })

        cash = float(account.cash)

        return {
            "value": round(portfolio_value, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "positions": pos_list,
            "cash": round(cash, 2),
        }
    except Exception:
        return None


@app.get("/api/performance")
def performance() -> dict:
    today_str = date.today().isoformat()

    # Total trades
    trades_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM trade_history"
    )
    total_trades = trades_row["cnt"] if trades_row else 0

    # Win rate
    win_rate = None
    if total_trades > 0:
        wins_row = _query_one(
            "SELECT COUNT(*) AS cnt FROM trade_history WHERE pnl > 0"
        )
        wins = wins_row["cnt"] if wins_row else 0
        win_rate = round(wins / total_trades * 100, 1)

    # Signals today
    signals_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM combined_signals "
        "WHERE date(created_at) = date(?)",
        (today_str,),
    )
    signals_today = signals_row["cnt"] if signals_row else 0

    strong_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM combined_signals "
        "WHERE date(created_at) = date(?) AND combined_signal = 'STRONG BUY'",
        (today_str,),
    )
    strong_buy_today = strong_row["cnt"] if strong_row else 0

    # Sessions today (from signal_events)
    sessions_rows = _query(
        "SELECT DISTINCT session FROM signal_events "
        "WHERE date(timestamp) = date(?)",
        (today_str,),
    )
    sessions_today = len(sessions_rows)

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe": None,
        "max_drawdown": None,
        "signals_today": signals_today,
        "strong_buy_today": strong_buy_today,
        "sessions_today": sessions_today,
        "sessions_total": 4,
    }


# ---------------------------------------------------------------------------
# Streamlit reverse proxy (catch-all — must be last route)
# ---------------------------------------------------------------------------

_STREAMLIT_BASE = "http://localhost:8501"


@app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE"])
async def proxy_streamlit(path: str, request: Request) -> StreamingResponse:
    """Forward non-/api/* requests to Streamlit running on port 8501."""
    import httpx

    url = f"{_STREAMLIT_BASE}/{path}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            params=dict(request.query_params),
            headers=headers,
            content=await request.body(),
        )
    # Strip hop-by-hop headers
    excluded = {"transfer-encoding", "connection", "content-encoding"}
    resp_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in excluded
    }
    return StreamingResponse(
        content=iter([resp.content]),
        status_code=resp.status_code,
        headers=resp_headers,
    )
