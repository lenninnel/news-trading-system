"""
MCP server for the News Trading System — dual-mode backend.

Exposes 5 read-only tools (get_portfolio, get_signals, get_performance,
get_status, get_signal_detail) that Claude (or any MCP client) can call
to inspect the trading system. The tool *surface* is identical across
both backends; only where the data comes from changes.

Backend selection
-----------------
* ``NTS_API_URL`` set    → HTTP mode.
  The tools make async ``httpx`` calls against the NTS FastAPI service
  (``api.main``). This is the mode Claude Desktop runs in: it spawns
  the server in stdio transport on the Mac and points it at the live
  Hetzner API at ``http://195.201.124.154:8001``, so tools read the
  *production* database without needing a local SQLite copy.

* ``NTS_API_URL`` unset  → SQLite mode.
  The tools open ``$DB_PATH`` directly with sqlite3 and run read-only
  queries. This is the mode the Hetzner-side ``nts-mcp.service`` uses
  — it lives on the same box as the DB so the HTTP hop is pointless.

The tool functions are also importable as plain async coroutines for
ad-hoc scripting::

    import asyncio
    from mcp_server.nts_mcp import get_portfolio, get_status
    print(asyncio.run(get_portfolio()))
    print(asyncio.run(get_status()))

Usage as a server
-----------------
    python -m mcp_server.nts_mcp                 # SSE on :8003 (default)
    python -m mcp_server.nts_mcp --stdio         # stdio transport
    python -m mcp_server.nts_mcp --port 9000     # custom SSE port

Environment
-----------
    NTS_API_URL   Base URL of the NTS FastAPI service. When set, every
                  tool uses HTTP (httpx) instead of SQLite.
    DB_PATH       SQLite path (SQLite mode only). Defaults to
                  config.settings.DB_PATH, then to a project-root fallback.
    NTS_MCP_PORT  Override SSE port at runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

# Ensure the project root is importable when running under systemd with
# ``python -m mcp_server.nts_mcp`` — the working directory should already
# be the project root, but add it to sys.path defensively so
# ``config.settings`` can always be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Constants ────────────────────────────────────────────────────────────

# Default SSE port. 8000 → nts-api uvicorn (8001), proprietary-data-pipeline
# (8002), then us. Streamlit takes 8501. Keep this in sync with
# deployment/nts-mcp.service.
DEFAULT_SSE_PORT = 8003

# HTTP client timeout (seconds). 15 is generous — the FastAPI endpoints are
# all read-only SQLite queries so they return in sub-second under load.
HTTP_TIMEOUT = 15

# Session schedule mirrored from scheduler/daily_runner.py. Used by
# get_status in SQLite mode to compute next_run_at. HTTP mode gets this
# from the API response. Updated 2026-04-11 to include D9 PREMARKET_SCAN.
_SCHEDULE = [
    {"name": "XETRA_PRE",      "hour": 6,  "minute": 45},
    {"name": "XETRA_OPEN",     "hour": 7,  "minute": 0},
    {"name": "PREMARKET_SCAN", "hour": 13, "minute": 0},
    {"name": "US_PRE",         "hour": 13, "minute": 15},
    {"name": "PEAD_OPEN",      "hour": 13, "minute": 45},
    {"name": "US_OPEN",        "hour": 14, "minute": 30},
    {"name": "MIDDAY",         "hour": 18, "minute": 0},
    {"name": "EOD",            "hour": 22, "minute": 15},
]

# ── Mode detection ───────────────────────────────────────────────────────


def _http_mode() -> bool:
    """True iff the caller has opted into HTTP mode via ``NTS_API_URL``.

    Read on every call (not at module load) so tests can monkeypatch the
    env before invoking a tool without reloading the module.
    """
    return bool(os.environ.get("NTS_API_URL", "").strip())


def _api_base() -> str:
    """Return the configured API base URL with any trailing slash trimmed."""
    return os.environ.get("NTS_API_URL", "").strip().rstrip("/")


# ── HTTP helper ──────────────────────────────────────────────────────────


async def _api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET *path* from the configured NTS API and return the parsed JSON.

    Raises the usual ``httpx`` exceptions on failure — callers should
    catch ``httpx.HTTPError`` (the superclass covering connect errors,
    status errors, and timeouts) and render a human-readable error.
    """
    url = f"{_api_base()}{path}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _http_error(exc: Exception) -> str:
    """Convert an httpx exception into a friendly single-line MCP error."""
    base = _api_base() or "<unset>"
    if isinstance(exc, httpx.ConnectError):
        return f"Error: cannot reach NTS API at {base} — connection refused."
    if isinstance(exc, httpx.TimeoutException):
        return f"Error: NTS API at {base} timed out after {HTTP_TIMEOUT}s."
    if isinstance(exc, httpx.HTTPStatusError):
        return (
            f"Error: NTS API at {base} returned "
            f"HTTP {exc.response.status_code}."
        )
    return f"Error: NTS API call failed ({type(exc).__name__}: {exc})."


# ── SQLite helpers ───────────────────────────────────────────────────────


def _resolve_db_path() -> str:
    """Return the SQLite path to open.

    Priority:
      1. ``DB_PATH`` environment variable (set in ``.env``)
      2. ``config.settings.DB_PATH`` import
      3. Literal fallback ``news_trading.db`` in the project root
    """
    env_val = os.environ.get("DB_PATH", "").strip()
    if env_val:
        return env_val
    try:
        from config.settings import DB_PATH
        return DB_PATH
    except Exception:
        return str(_PROJECT_ROOT / "news_trading.db")


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query and return a list of dicts.

    Silently returns an empty list on any sqlite error — the MCP surface
    must never raise into the client.
    """
    try:
        conn = sqlite3.connect(_resolve_db_path())
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        log.warning("nts_mcp query failed (%s): %s", type(exc).__name__, exc)
        return []
    except Exception as exc:
        log.warning("nts_mcp unexpected error: %s", exc)
        return []


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = _query(sql, params)
    return rows[0] if rows else None


def _load_watchlist() -> list[str]:
    """Read config/watchlist.yaml, falling back to an empty list."""
    try:
        import yaml

        path = _PROJECT_ROOT / "config" / "watchlist.yaml"
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("us_tickers") or cfg.get("watchlist") or []
    except Exception:
        return []


def _last_and_next_session() -> tuple[dict | None, dict | None]:
    """Compute last-completed and next-upcoming session from UTC clock."""
    now = datetime.now(timezone.utc)
    now_minutes = now.hour * 60 + now.minute

    last, nxt = None, None
    for entry in _SCHEDULE:
        entry_minutes = entry["hour"] * 60 + entry["minute"]
        if now_minutes >= entry_minutes:
            last = entry
        elif nxt is None:
            nxt = entry
    return last, nxt


def _signal_events_has_column(column: str) -> bool:
    """Return True iff signal_events has *column*. Cheap PRAGMA probe.

    Used so get_signal_detail can degrade gracefully against an older DB
    that's missing columns added by recent migrations (currently
    ``macro_context_used`` from the MacroContextAgent rollout).
    """
    try:
        rows = _query("PRAGMA table_info(signal_events)")
        return any((r.get("name") or "") == column for r in rows)
    except Exception:
        return False


def _db_available() -> bool:
    """Lightweight DB reachability probe — just runs ``SELECT 1``."""
    try:
        conn = sqlite3.connect(_resolve_db_path())
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return True
    except Exception:
        return False


# ── Fetch layer — returns structured data, branches on mode ─────────────
#
# Each _fetch_* function returns a dict or list in the SAME shape as the
# corresponding FastAPI endpoint (see api/main.py). The shared format
# functions below then render the same data regardless of source.


async def _fetch_portfolio() -> dict:
    if _http_mode():
        data = await _api_get("/api/portfolio")
        return data if isinstance(data, dict) else {}
    return _sql_portfolio()


def _sql_portfolio() -> dict:
    """SQLite equivalent of GET /api/portfolio.

    Mirrors api/main.py's shape exactly: {value, cash, daily_pnl,
    daily_pnl_pct, positions[]}.
    """
    positions = _query(
        "SELECT ticker, shares, avg_price, current_value "
        "FROM portfolio_positions ORDER BY ticker"
    )
    total_value = sum((p.get("current_value") or 0) for p in positions)

    today_str = date.today().isoformat()
    pnl_row = _query_one(
        "SELECT COALESCE(SUM(pnl), 0) AS daily_pnl FROM trade_history "
        "WHERE date(created_at) = date(?)",
        (today_str,),
    )
    daily_pnl = (pnl_row or {}).get("daily_pnl") or 0.0
    daily_pnl_pct = (daily_pnl / total_value * 100) if total_value else 0.0

    pos_list: list[dict] = []
    for p in positions:
        shares = p.get("shares") or 0
        avg_price = p.get("avg_price") or 0.0
        current_value = p.get("current_value") or 0.0
        current_price = (current_value / shares) if shares else 0.0
        cost = avg_price * shares if shares else 0
        pnl_pct = ((current_value - cost) / cost * 100) if cost else 0.0
        pos_list.append({
            "ticker": p.get("ticker"),
            "shares": shares,
            "entry": round(avg_price, 2),
            "current": round(current_price, 2),
            "pnl_pct": round(pnl_pct, 1),
        })

    total_invested = sum(
        (p.get("avg_price") or 0) * (p.get("shares") or 0)
        for p in positions
    )
    account_row = _query_one(
        "SELECT account_balance FROM risk_calculations "
        "ORDER BY id DESC LIMIT 1"
    )
    account_balance = (account_row or {}).get("account_balance") or 10_000.0
    cash = account_balance - total_invested

    return {
        "value": round(total_value, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl_pct, 2),
        "positions": pos_list,
        "cash": round(cash, 2),
    }


async def _fetch_signals(days: int, strategy: str, limit: int) -> list[dict]:
    if _http_mode():
        params = {"days": days, "limit": limit}
        data = await _api_get("/api/signals", params=params)
        if not isinstance(data, list):
            return []
        # The FastAPI doesn't support a strategy filter — do it client-side.
        if strategy:
            up = strategy.upper()
            data = [s for s in data if (s.get("strategy") or "").upper() == up]
        return data
    return _sql_signals(days, strategy, limit)


def _sql_signals(days: int, strategy: str, limit: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    clauses = ["timestamp >= ?"]
    params: list[Any] = [cutoff]
    if strategy:
        clauses.append("UPPER(COALESCE(strategy, '')) = ?")
        params.append(strategy.upper())
    where = " AND ".join(clauses)

    return _query(
        f"SELECT timestamp, session, ticker, strategy, signal, "
        f"confidence, price_at_signal, debate_outcome, "
        f"outcome_3d_pct, outcome_5d_pct, trade_executed "
        f"FROM signal_events WHERE {where} "
        f"ORDER BY id DESC LIMIT ?",
        tuple(params + [limit]),
    )


async def _fetch_performance() -> dict:
    if _http_mode():
        data = await _api_get("/api/performance")
        return data if isinstance(data, dict) else {}
    return _sql_performance()


def _sql_performance() -> dict:
    """SQLite equivalent of GET /api/performance + per-strategy breakdown.

    The per-strategy `strategies` field is a SQLite-only extra — the
    FastAPI endpoint doesn't return it. Format layer shows it only when
    present.
    """
    today_str = date.today().isoformat()

    trades_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM trade_history"
    )
    total_trades = (trades_row or {}).get("cnt") or 0

    win_rate: float | None = None
    if total_trades > 0:
        wins_row = _query_one(
            "SELECT COUNT(*) AS cnt FROM trade_history WHERE pnl > 0"
        )
        wins = (wins_row or {}).get("cnt") or 0
        win_rate = round(wins / total_trades * 100, 1)

    signals_row = _query_one(
        "SELECT COUNT(*) AS cnt FROM signal_events "
        "WHERE date(timestamp) = date(?)",
        (today_str,),
    )
    signals_today = (signals_row or {}).get("cnt") or 0

    sessions_rows = _query(
        "SELECT DISTINCT session FROM signal_events "
        "WHERE date(timestamp) = date(?) AND session IS NOT NULL",
        (today_str,),
    )
    sessions_today = len(sessions_rows)

    seven_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=7)
    ).isoformat()
    strat_rows = _query(
        "SELECT strategy, COUNT(*) AS cnt, AVG(confidence) AS avg_conf "
        "FROM signal_events "
        "WHERE timestamp >= ? AND strategy IS NOT NULL "
        "GROUP BY strategy ORDER BY cnt DESC",
        (seven_days_ago,),
    )

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "signals_today": signals_today,
        "sessions_today": sessions_today,
        # Extra, SQLite-only:
        "strategies": strat_rows,
    }


async def _fetch_status() -> dict:
    if _http_mode():
        data = await _api_get("/api/status")
        return data if isinstance(data, dict) else {}
    return _sql_status()


def _sql_status() -> dict:
    mode = os.environ.get("TRADING_MODE", "paper_local")
    watchlist = _load_watchlist()

    last, nxt = _last_and_next_session()
    today = date.today()

    last_event = _query_one(
        "SELECT session, timestamp FROM signal_events "
        "ORDER BY id DESC LIMIT 1"
    )
    if last_event:
        last_name = last_event.get("session")
        last_run_at = last_event.get("timestamp")
    elif last:
        last_name = last["name"]
        last_run_at = None
    else:
        last_name = None
        last_run_at = None

    if nxt:
        next_name = nxt["name"]
        next_run_at = datetime(
            today.year, today.month, today.day,
            nxt["hour"], nxt["minute"],
            tzinfo=timezone.utc,
        ).isoformat()
    else:
        next_name = None
        next_run_at = None

    return {
        "running": _db_available(),
        "mode": mode,
        "last_session": last_name,
        "last_run_at": last_run_at,
        "next_session": next_name,
        "next_run_at": next_run_at,
        "watchlist": watchlist,
    }


async def _fetch_signal_detail(ticker_upper: str) -> list[dict]:
    if _http_mode():
        params = {"ticker": ticker_upper, "days": 30, "limit": 10}
        data = await _api_get("/api/signals", params=params)
        return data if isinstance(data, list) else []
    return _sql_signal_detail(ticker_upper)


def _sql_signal_detail(ticker_upper: str) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    # macro_context_used was added in a migration alongside
    # MacroContextAgent. A DB that hasn't seen the migration yet (e.g.
    # the MCP server pointed at a stale file) will raise
    # OperationalError on the bare SELECT. Probe first and leave it off
    # the projection if absent — the tool still renders everything else.
    has_macro_col = _signal_events_has_column("macro_context_used")
    columns = (
        "timestamp, session, strategy, signal, confidence, "
        "rsi, sentiment_score, news_score, social_score, "
        "bull_case, bear_case, debate_outcome, price_at_signal, "
        "outcome_3d_pct, outcome_5d_pct, outcome_10d_pct, trade_executed"
    )
    if has_macro_col:
        columns += ", macro_context_used"

    return _query(
        f"SELECT {columns} "
        "FROM signal_events "
        "WHERE ticker = ? AND timestamp >= ? "
        "ORDER BY id DESC LIMIT 10",
        (ticker_upper, cutoff),
    )


# ── Format layer — pure, no I/O, works on either backend's data ─────────


def _format_portfolio(data: dict) -> str:
    if not data:
        return "Portfolio Summary\n" + "─" * 40 + "\n  No portfolio data available."

    total_value = data.get("value") or 0
    cash = data.get("cash") or 0
    daily_pnl = data.get("daily_pnl") or 0
    daily_pnl_pct = data.get("daily_pnl_pct") or 0
    positions = data.get("positions") or []

    lines = [
        "Portfolio Summary",
        "─" * 40,
        f"  Total value:  ${total_value:,.2f}",
        f"  Cash:         ${cash:,.2f}",
        f"  Daily P&L:    ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)",
        "",
    ]
    if positions:
        lines.append(f"Positions ({len(positions)}):")
        for p in positions:
            ticker = p.get("ticker") or "???"
            shares = p.get("shares") or 0
            entry = p.get("entry") or 0.0
            current = p.get("current") or 0.0
            pnl_pct = p.get("pnl_pct") or 0.0
            lines.append(
                f"  {ticker:<6} {shares} shares  "
                f"entry=${entry:.2f}  now=${current:.2f}  "
                f"pnl={pnl_pct:+.1f}%"
            )
    else:
        lines.append("No open positions.")
    return "\n".join(lines)


def _format_signals(rows: list[dict], days: int) -> str:
    if not rows:
        return f"No signals found in the last {days} day(s)."
    lines = [f"Found {len(rows)} signal(s) in the last {days} day(s):\n"]
    for s in rows:
        ts = (s.get("timestamp") or "")[:16]
        ticker = s.get("ticker") or "???"
        signal = s.get("signal") or "?"
        conf = s.get("confidence") or 0
        price = s.get("price_at_signal")
        debate = s.get("debate_outcome") or ""
        o3d = s.get("outcome_3d_pct")
        price_str = f"${price:.2f}" if price else "n/a"
        outcome_str = f"{o3d:+.1f}%" if o3d is not None else "pending"
        lines.append(
            f"  {ts}  {ticker:<6} {signal:<12} "
            f"conf={conf:.2f}  price={price_str}  "
            f"debate={debate or '-'}  3d={outcome_str}"
        )
    return "\n".join(lines)


def _format_performance(data: dict) -> str:
    total_trades = data.get("total_trades") or 0
    win_rate = data.get("win_rate")
    signals_today = data.get("signals_today") or 0
    sessions_today = data.get("sessions_today") or 0
    strategies = data.get("strategies") or []  # SQLite-only extra

    lines = [
        "Trading Performance",
        "─" * 40,
        f"  Total trades:     {total_trades}",
        "  Win rate:         "
        + (f"{win_rate:.1f}%" if win_rate is not None else "n/a"),
        f"  Signals today:    {signals_today}",
        f"  Sessions today:   {sessions_today}",
    ]
    if strategies:
        lines.append("")
        lines.append("  Strategy breakdown (last 7d):")
        for r in strategies:
            name = r.get("strategy") or "?"
            cnt = r.get("cnt") or 0
            avg = r.get("avg_conf") or 0
            lines.append(
                f"    {name:<20} {cnt:>4} signals  avg_conf={avg:.2f}"
            )
    return "\n".join(lines)


def _format_status(data: dict) -> str:
    running = data.get("running", False)
    mode = data.get("mode") or "unknown"
    last_name = data.get("last_session")
    last_run_at = data.get("last_run_at")
    next_name = data.get("next_session")
    next_run_at = data.get("next_run_at")
    watchlist = data.get("watchlist") or []

    lines = [
        "NTS System Status",
        "─" * 40,
        f"  Running:       {'Yes' if running else 'No (DB unreachable)'}",
        f"  Mode:          {mode}",
        f"  Last session:  {last_name or 'n/a'}"
        + (f" ({(last_run_at or '')[:16]})" if last_run_at else ""),
        f"  Next session:  {next_name or 'n/a'}"
        + (f" ({(next_run_at or '')[:16]})" if next_run_at else ""),
        "  Watchlist:     "
        + (", ".join(watchlist) if watchlist else "empty"),
    ]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"\n  Queried at:    {now_utc}")
    return "\n".join(lines)


def _format_signal_detail(ticker_upper: str, rows: list[dict]) -> str:
    if not rows:
        return f"No recent signals found for {ticker_upper}."

    lines = [
        f"Signal Detail: {ticker_upper}",
        f"Found {len(rows)} signal(s) in the last 30 days",
        "═" * 50,
    ]
    for s in rows:
        ts = (s.get("timestamp") or "")[:16]
        signal = s.get("signal") or "?"
        conf = s.get("confidence") or 0
        strategy = s.get("strategy") or "?"
        price = s.get("price_at_signal")
        rsi = s.get("rsi")
        sentiment = s.get("sentiment_score")
        news = s.get("news_score")
        social = s.get("social_score")
        bull = s.get("bull_case")
        bear = s.get("bear_case")
        debate = s.get("debate_outcome")
        macro_used = s.get("macro_context_used")
        o3d = s.get("outcome_3d_pct")
        o5d = s.get("outcome_5d_pct")
        o10d = s.get("outcome_10d_pct")
        executed = s.get("trade_executed")

        lines.append("")
        lines.append("─" * 50)
        lines.append(f"  Time:       {ts}")
        lines.append(f"  Signal:     {signal}  (confidence: {conf:.2f})")
        lines.append(f"  Strategy:   {strategy}")
        if price is not None:
            lines.append(f"  Price:      ${price:.2f}")

        scores: list[str] = []
        if rsi is not None:
            scores.append(f"RSI={rsi:.1f}")
        if sentiment is not None:
            scores.append(f"sentiment={sentiment:.2f}")
        if news is not None:
            scores.append(f"news={news:.2f}")
        if social is not None:
            scores.append(f"social={social:.2f}")
        if scores:
            lines.append(f"  Scores:     {', '.join(scores)}")

        if bull or bear:
            lines.append("")
            lines.append("  Bull/Bear Debate:")
            if bull:
                lines.append(f"    Bull: {bull}")
            if bear:
                lines.append(f"    Bear: {bear}")
            if debate:
                lines.append(f"    Outcome: {debate}")
            if macro_used:
                lines.append("    Macro context: used")

        outcomes: list[str] = []
        if o3d is not None:
            outcomes.append(f"3d={o3d:+.2f}%")
        if o5d is not None:
            outcomes.append(f"5d={o5d:+.2f}%")
        if o10d is not None:
            outcomes.append(f"10d={o10d:+.2f}%")
        if outcomes:
            lines.append(f"  Outcomes:   {', '.join(outcomes)}")

        if executed is not None:
            lines.append(f"  Executed:   {'Yes' if executed else 'No'}")
    return "\n".join(lines)


# ── MCP server + tool registration ───────────────────────────────────────

server = FastMCP("nts-trading")


@server.tool()
async def get_portfolio() -> str:
    """Current portfolio positions and P&L.

    In HTTP mode: fetches ``/api/portfolio`` from ``$NTS_API_URL``.
    In SQLite mode: reads ``portfolio_positions`` + ``trade_history`` +
    ``risk_calculations`` directly.
    """
    try:
        data = await _fetch_portfolio()
    except httpx.HTTPError as exc:
        return _http_error(exc)
    return _format_portfolio(data)


@server.tool()
async def get_signals(days: int = 7, strategy: str = "", limit: int = 50) -> str:
    """Recent signal_events rows.

    Args:
        days: Look-back window in days (1-365). Default 7.
        strategy: Optional strategy filter (e.g. "Combined", "PEAD").
            Empty string = all strategies.
        limit: Max rows to return (1-500). Default 50.
    """
    days = max(1, min(365, days))
    limit = max(1, min(500, limit))
    try:
        rows = await _fetch_signals(days, strategy, limit)
    except httpx.HTTPError as exc:
        return _http_error(exc)
    return _format_signals(rows, days)


@server.tool()
async def get_performance() -> str:
    """Aggregate trading performance and per-strategy breakdown."""
    try:
        data = await _fetch_performance()
    except httpx.HTTPError as exc:
        return _http_error(exc)
    return _format_performance(data)


@server.tool()
async def get_status() -> str:
    """System running state, last/next session, and current watchlist."""
    try:
        data = await _fetch_status()
    except httpx.HTTPError as exc:
        return _http_error(exc)
    return _format_status(data)


@server.tool()
async def get_signal_detail(ticker: str) -> str:
    """Last 10 signals for a ticker, including bull/bear debate cases.

    Args:
        ticker: Stock symbol, case-insensitive.
    """
    ticker_upper = (ticker or "").upper().strip()
    if not ticker_upper:
        return "Error: ticker argument is required."
    try:
        rows = await _fetch_signal_detail(ticker_upper)
    except httpx.HTTPError as exc:
        return _http_error(exc)
    return _format_signal_detail(ticker_upper, rows)


# ── Entry point ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NTS MCP Server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport instead of SSE (for local MCP clients).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NTS_MCP_PORT", DEFAULT_SSE_PORT)),
        help=f"SSE listener port (default {DEFAULT_SSE_PORT}, env NTS_MCP_PORT).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    mode = "HTTP→" + _api_base() if _http_mode() else f"SQLite@{_resolve_db_path()}"
    log.info("nts_mcp backend: %s", mode)

    if args.stdio:
        log.info("Starting nts_mcp (stdio transport)")
        server.run(transport="stdio")
    else:
        # FastMCP.run() reads its port from the settings object, not a
        # kwarg. Push the CLI value into settings before handing off.
        server.settings.port = args.port
        server.settings.host = "0.0.0.0"
        log.info("Starting nts_mcp (SSE transport on :%d)", args.port)
        server.run(transport="sse")


if __name__ == "__main__":
    main()
