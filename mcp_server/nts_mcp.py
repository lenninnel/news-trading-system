"""
MCP server for the News Trading System — direct SQLite backend.

Exposes 5 read-only tools that Claude (or any MCP client) can call to
query signals, portfolio, status, performance, and per-ticker signal
detail. Unlike the earlier HTTP-proxy version, every tool reads the
SQLite database at ``$DB_PATH`` directly, so the MCP server has no
runtime dependency on the FastAPI service.

The tool functions are also importable as plain async coroutines for
ad-hoc scripting::

    import asyncio
    from mcp_server.nts_mcp import get_portfolio, get_status
    print(asyncio.run(get_portfolio()))
    print(asyncio.run(get_status()))

Usage as a server
-----------------
    python -m mcp_server.nts_mcp                 # SSE on :8002 (default)
    python -m mcp_server.nts_mcp --stdio         # stdio transport
    python -m mcp_server.nts_mcp --port 9000     # custom SSE port

Environment
-----------
    DB_PATH      SQLite path. Defaults to config.settings.DB_PATH
                 (``news_trading.db`` in the project root if unset).
    NTS_MCP_PORT Override SSE port at runtime.
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

# Default SSE port chosen so it doesn't collide with the existing FastAPI
# (8001) or Streamlit (8501) on the Hetzner box.
DEFAULT_SSE_PORT = 8002

# Session schedule mirrored from scheduler/daily_runner.py. Only the subset
# needed for get_status — see that module for the canonical list. Updated
# 2026-04-11 to include PREMARKET_SCAN (D9) at 13:00.
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

# ── DB helpers ───────────────────────────────────────────────────────────


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


# ── MCP server ───────────────────────────────────────────────────────────

server = FastMCP("nts-trading")


# ── Tool: get_portfolio ──────────────────────────────────────────────────


@server.tool()
async def get_portfolio() -> str:
    """Current portfolio positions and P&L.

    Reads ``portfolio_positions`` for holdings and ``trade_history`` for
    day-over-day realised P&L. Cash is estimated from the most recent
    ``risk_calculations.account_balance`` row minus invested capital.

    Returns a human-readable summary plus per-position rows.
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

    lines = [
        "Portfolio Summary",
        "─" * 40,
        f"  Total value:  ${total_value:,.2f}",
        f"  Cash:         ${cash:,.2f}",
        f"  Daily P&L:    ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)",
        "",
    ]

    if pos_list:
        lines.append(f"Positions ({len(pos_list)}):")
        for p in pos_list:
            lines.append(
                f"  {p['ticker']:<6} {p['shares']} shares  "
                f"entry=${p['entry']:.2f}  now=${p['current']:.2f}  "
                f"pnl={p['pnl_pct']:+.1f}%"
            )
    else:
        lines.append("No open positions.")

    return "\n".join(lines)


# ── Tool: get_signals ────────────────────────────────────────────────────


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
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    clauses = ["timestamp >= ?"]
    params: list[Any] = [cutoff]
    if strategy:
        clauses.append("UPPER(COALESCE(strategy, '')) = ?")
        params.append(strategy.upper())
    where = " AND ".join(clauses)

    rows = _query(
        f"SELECT timestamp, session, ticker, strategy, signal, "
        f"confidence, price_at_signal, debate_outcome, "
        f"outcome_3d_pct, outcome_5d_pct, trade_executed "
        f"FROM signal_events WHERE {where} "
        f"ORDER BY id DESC LIMIT ?",
        tuple(params + [limit]),
    )

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


# ── Tool: get_performance ────────────────────────────────────────────────


@server.tool()
async def get_performance() -> str:
    """Aggregate trading performance and per-strategy breakdown."""
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

    # Per-strategy breakdown (last 7 days)
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

    lines = [
        "Trading Performance",
        "─" * 40,
        f"  Total trades:     {total_trades}",
        f"  Win rate:         "
        + (f"{win_rate:.1f}%" if win_rate is not None else "n/a"),
        f"  Signals today:    {signals_today}",
        f"  Sessions today:   {sessions_today}",
    ]
    if strat_rows:
        lines.append("")
        lines.append("  Strategy breakdown (last 7d):")
        for r in strat_rows:
            name = r.get("strategy") or "?"
            cnt = r.get("cnt") or 0
            avg = r.get("avg_conf") or 0
            lines.append(
                f"    {name:<20} {cnt:>4} signals  avg_conf={avg:.2f}"
            )
    return "\n".join(lines)


# ── Tool: get_status ─────────────────────────────────────────────────────


@server.tool()
async def get_status() -> str:
    """System running state, last/next session, and current watchlist."""
    mode = os.environ.get("TRADING_MODE", "paper_local")
    watchlist = _load_watchlist()

    last, nxt = _last_and_next_session()
    today = date.today()

    # "Real" last session from signal_events, falling back to schedule
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

    running = _db_available()

    lines = [
        "NTS System Status",
        "─" * 40,
        f"  Running:       {'Yes' if running else 'No (DB unreachable)'}",
        f"  Mode:          {mode}",
        f"  Last session:  {last_name or 'n/a'}"
        + (f" ({last_run_at[:16]})" if last_run_at else ""),
        f"  Next session:  {next_name or 'n/a'}"
        + (f" ({next_run_at[:16]})" if next_run_at else ""),
        f"  Watchlist:     "
        + (", ".join(watchlist) if watchlist else "empty"),
    ]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"\n  Queried at:    {now_utc}")
    return "\n".join(lines)


def _db_available() -> bool:
    """Lightweight DB reachability probe — just lists a system table."""
    try:
        conn = sqlite3.connect(_resolve_db_path())
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return True
    except Exception:
        return False


# ── Tool: get_signal_detail ──────────────────────────────────────────────


@server.tool()
async def get_signal_detail(ticker: str) -> str:
    """Last 10 signals for a ticker, including bull/bear debate cases.

    Args:
        ticker: Stock symbol, case-insensitive.
    """
    ticker_upper = (ticker or "").upper().strip()
    if not ticker_upper:
        return "Error: ticker argument is required."

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rows = _query(
        "SELECT timestamp, session, strategy, signal, confidence, "
        "rsi, sentiment_score, news_score, social_score, "
        "bull_case, bear_case, debate_outcome, price_at_signal, "
        "outcome_3d_pct, outcome_5d_pct, outcome_10d_pct, trade_executed, "
        "macro_context_used "
        "FROM signal_events "
        "WHERE ticker = ? AND timestamp >= ? "
        "ORDER BY id DESC LIMIT 10",
        (ticker_upper, cutoff),
    )

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
