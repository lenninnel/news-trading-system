"""
Read-only FastAPI layer for the News Trading System.

Serves as the bridge between the trading bot and the React dashboard.
All endpoints are GET-only — the API never controls the trading system.

Start:  uvicorn api.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import DB_PATH  # noqa: E402
from config.sessions import SCHEDULE as _SCHEDULE  # noqa: E402

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
        "http://195.201.124.154:3000",
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


def _add_business_days(ts: datetime, n: int) -> datetime:
    """Return *ts* shifted forward by *n* weekdays (Mon–Fri).

    Holidays are not skipped — outcome_tracker's yfinance ±3-day fetch
    window absorbs holiday edge cases at fill time. Mirrors the
    frontend's addTradingDays helper so the two stay aligned.
    """
    result = ts
    added = 0
    while added < n:
        result = result + timedelta(days=1)
        if result.weekday() < 5:
            added += 1
    return result


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

    # Check signal_events for the actual last run.
    # Filter out PositionManager/etc. rows that have session=NULL —
    # we want the last row that actually belonged to a named session.
    last_event = _query_one(
        "SELECT session, timestamp FROM signal_events "
        "WHERE session IS NOT NULL ORDER BY id DESC LIMIT 1"
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

    # Sessions that actually ran today (UTC). Row exists = session
    # started. We don't currently track completion separately; if a
    # session crashed mid-flight that's a different alarm path.
    today_utc = datetime.now(timezone.utc).date().isoformat()
    sessions_today_rows = _query(
        "SELECT session, started_at FROM session_runs "
        "WHERE run_date = ? ORDER BY started_at ASC",
        (today_utc,),
    )
    sessions_today = [
        {"session": r["session"], "started_at": r["started_at"]}
        for r in sessions_today_rows
    ]

    return {
        "running": True,
        "uptime_seconds": uptime,
        "last_session": last_session_name,
        "last_run_at": last_run_at,
        "next_session": next_session_name,
        "next_run_at": next_run_at,
        "watchlist": watchlist,
        "mode": mode,
        "sessions_today": sessions_today,
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

    # Normalise trade_executed to bool; compute outcome_Nd_due_at on
    # the fly from timestamp + business-day offset (no schema column).
    for r in rows:
        r["trade_executed"] = bool(r.get("trade_executed"))
        try:
            ts = datetime.fromisoformat(r["timestamp"])
        except (TypeError, ValueError):
            r["outcome_3d_due_at"] = None
            r["outcome_5d_due_at"] = None
            r["outcome_10d_due_at"] = None
            continue
        r["outcome_3d_due_at"] = _add_business_days(ts, 3).isoformat()
        r["outcome_5d_due_at"] = _add_business_days(ts, 5).isoformat()
        r["outcome_10d_due_at"] = _add_business_days(ts, 10).isoformat()
    return rows


@app.get("/api/portfolio")
def portfolio() -> dict:
    positions = _query("SELECT * FROM portfolio_positions ORDER BY ticker")

    total_value = sum(p.get("current_value", 0) or 0 for p in positions)

    account_row = _query_one(
        "SELECT account_balance FROM risk_calculations ORDER BY id DESC LIMIT 1"
    )
    account_balance = account_row["account_balance"] if account_row else None

    # Daily PnL from trade_history
    today_str = date.today().isoformat()
    pnl_row = _query_one(
        "SELECT COALESCE(SUM(pnl), 0) AS daily_pnl FROM trade_history "
        "WHERE date(created_at) = date(?)",
        (today_str,),
    )
    daily_pnl = pnl_row["daily_pnl"] if pnl_row else 0.0
    daily_pnl_pct = (daily_pnl / account_balance * 100) if account_balance else 0.0

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
    if account_balance is None:
        logger.warning(
            "portfolio endpoint: no risk_calculations rows, cash defaulting to 0"
        )
        cash = 0.0
    else:
        cash = account_balance - total_invested

    return {
        "value": round(total_value, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl_pct, 2),
        "positions": pos_list,
        "cash": round(cash, 2),
    }


@app.get("/api/trades")
def trades(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """Return closed trade history from the trade_history table."""
    rows = _query(
        "SELECT ticker, action, shares, price, stop_loss, take_profit, pnl, created_at "
        "FROM trade_history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    trade_list = []
    for r in rows:
        shares = r.get("shares", 0)
        price = r.get("price", 0)
        value = shares * price if shares and price else 0
        # Calculate pnl_pct: find the matching BUY for this SELL
        pnl = r.get("pnl", 0) or 0
        pnl_pct = (pnl / value * 100) if value and pnl else 0
        trade_list.append({
            "ticker": r.get("ticker"),
            "action": r.get("action"),
            "shares": shares,
            "price": round(price, 2),
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 1),
            "stop_loss": r.get("stop_loss"),
            "take_profit": r.get("take_profit"),
            "timestamp": r.get("created_at"),
        })
    return {"trades": trade_list}


@app.get("/api/performance")
def performance() -> dict:
    today_str = date.today().isoformat()

    # Total CLOSED trades. trade_history contains both BUY (open
    # position) and SELL (closed) rows; only SELL with realized pnl
    # belongs in the win-rate denominator.
    trades_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM trade_history "
        "WHERE action = 'SELL' AND pnl IS NOT NULL"
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
        "sessions_total": len(_SCHEDULE),
    }


@app.get("/api/strategy-performance")
def get_strategy_performance() -> list[dict]:
    """Return the latest per-strategy performance metrics."""
    rows = _query(
        "SELECT strategy_name, run_date, sharpe_30d, win_rate_30d, avg_rr, "
        "signal_count, avg_confidence "
        "FROM strategy_perf_daily "
        "WHERE run_date = (SELECT MAX(run_date) FROM strategy_perf_daily) "
        "ORDER BY strategy_name"
    )
    return rows


# Caps mirrored from execution/portfolio_manager.py — duplicated here to keep
# the read-only API free of trading-code imports.
_MAX_POSITIONS    = 8
_MAX_PER_STRATEGY = 4
_MAX_PER_SECTOR   = 4
_MAX_DEPLOYED_PCT = 0.60
_MAX_SECTOR_PCT   = 0.40


@app.get("/api/state")
def state() -> dict:
    """Consolidated current-state view for dashboard monitoring.

    Joins live positions with metadata (strategy/sector) and the most-recent
    BUY row (for SL/TP), aggregates by sector and strategy, and reports
    deployment vs MAX_DEPLOYED_PCT.
    """
    account_row = _query_one(
        "SELECT account_balance FROM risk_calculations ORDER BY id DESC LIMIT 1"
    )
    account_balance = (account_row or {}).get("account_balance") or 10_000.0

    rows = _query(
        """
        SELECT
            pp.ticker, pp.shares, pp.avg_price, pp.current_value, pp.updated_at,
            pm.strategy, pm.sector, pm.entry_date,
            (SELECT th.stop_loss   FROM trade_history th
              WHERE th.ticker = pp.ticker AND th.action = 'BUY'
              ORDER BY th.id DESC LIMIT 1) AS stop_loss,
            (SELECT th.take_profit FROM trade_history th
              WHERE th.ticker = pp.ticker AND th.action = 'BUY'
              ORDER BY th.id DESC LIMIT 1) AS take_profit
        FROM portfolio_positions pp
        LEFT JOIN position_metadata pm ON pm.ticker = pp.ticker
        WHERE pp.shares > 0
        ORDER BY pp.ticker
        """
    )

    positions = []
    for r in rows:
        shares        = r.get("shares") or 0
        avg_price     = r.get("avg_price") or 0.0
        current_value = r.get("current_value") or 0.0
        # Fall back to entry when DB has no quote update yet (current_value=0).
        current_price = (current_value / shares) if (shares and current_value) else avg_price
        cost          = avg_price * shares
        market_value  = current_price * shares
        pnl           = market_value - cost
        pnl_pct       = (pnl / cost * 100) if cost else 0.0
        sl            = r.get("stop_loss")
        tp            = r.get("take_profit")
        sl_dist_pct   = ((current_price - sl) / current_price * 100) if (sl and current_price) else None
        tp_dist_pct   = ((tp - current_price) / current_price * 100) if (tp and current_price) else None
        positions.append({
            "ticker":           r.get("ticker"),
            "shares":           shares,
            "entry":            round(avg_price, 2),
            "current":          round(current_price, 2),
            "market_value":     round(market_value, 2),
            "pnl":              round(pnl, 2),
            "pnl_pct":          round(pnl_pct, 2),
            "strategy":         r.get("strategy"),
            "sector":           r.get("sector"),
            "entry_date":       r.get("entry_date"),
            "stop_loss":        sl,
            "take_profit":      tp,
            "sl_distance_pct":  round(sl_dist_pct, 2) if sl_dist_pct is not None else None,
            "tp_distance_pct":  round(tp_dist_pct, 2) if tp_dist_pct is not None else None,
            "updated_at":       r.get("updated_at"),
        })

    sectors_agg: dict[str, dict] = {}
    for p in positions:
        sec = p.get("sector") or "Other"
        bucket = sectors_agg.setdefault(sec, {"count": 0, "value": 0.0})
        bucket["count"] += 1
        bucket["value"] += p["market_value"]
    sectors = [
        {
            "sector":    k,
            "count":     v["count"],
            "value":     round(v["value"], 2),
            "pct":       round((v["value"] / account_balance * 100) if account_balance else 0, 2),
            "max_count": _MAX_PER_SECTOR,
            "max_pct":   round(_MAX_SECTOR_PCT * 100, 2),
        }
        for k, v in sorted(sectors_agg.items())
    ]

    strats_agg: dict[str, int] = {}
    for p in positions:
        s = p.get("strategy") or "Unknown"
        strats_agg[s] = strats_agg.get(s, 0) + 1
    strategies = [
        {"strategy": k, "count": v, "max_count": _MAX_PER_STRATEGY}
        for k, v in sorted(strats_agg.items())
    ]

    deployed_value = sum(p["market_value"] for p in positions)
    deployment_pct = (deployed_value / account_balance * 100) if account_balance else 0.0

    # Drawdown halt feature (commit 55d0086) tracks the all-time peak in
    # portfolio_peak (single-row halt state) and logs transitions to
    # drawdown_events. The old placeholder table `drawdown_state` was
    # never created, so this used to always return null. Use the live
    # account_balance (latest risk_calculations row, fed from IBKR
    # NetLiquidation) as the "current" so drawdown_pct stays fresh between
    # halt events. Keep the legacy keys (peak/current/drawdown_pct/
    # halt_triggered) plus halt-context fields for the dashboard tile.
    drawdown: dict | None = None
    peak_row = _query_one("SELECT * FROM portfolio_peak WHERE id = 1")
    if peak_row:
        peak_val = peak_row.get("peak_value") or 0.0
        current = account_balance if peak_val > 0 else None
        dd_pct = (
            (peak_val - current) / peak_val
            if peak_val > 0 and current is not None
            else None
        )
        drawdown = {
            "peak":             peak_val,
            "peak_observed_at": peak_row.get("peak_observed_at"),
            "current":          current,
            "drawdown_pct":     dd_pct,
            "halt_triggered":   bool(peak_row.get("halted") or 0),
            "halted_at":        peak_row.get("halted_at"),
            "halted_value":     peak_row.get("halted_value"),
            "halt_reason":      peak_row.get("halt_reason"),
        }

    updated_row = _query_one("SELECT MAX(updated_at) AS u FROM portfolio_positions")
    updated_at = (updated_row or {}).get("u")

    return {
        "account_balance": round(account_balance, 2),
        "deployment": {
            "value":   round(deployed_value, 2),
            "pct":     round(deployment_pct, 2),
            "max_pct": round(_MAX_DEPLOYED_PCT * 100, 2),
        },
        "position_count": len(positions),
        "max_positions":  _MAX_POSITIONS,
        "sectors":        sectors,
        "strategies":     strategies,
        "positions":      positions,
        "drawdown":       drawdown,
        "updated_at":     updated_at,
    }


