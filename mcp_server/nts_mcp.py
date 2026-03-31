"""
MCP server for the News Trading System (FastMCP edition).

Exposes 5 read-only tools that proxy the existing FastAPI endpoints,
so Claude can query signals, portfolio, status, performance, and
detailed signal analysis directly in conversation.

Usage:
    python -m mcp_server.nts_mcp               # stdio transport (default)
    NTS_API_URL=http://localhost:8001 python -m mcp_server.nts_mcp

Environment:
    NTS_API_URL  Base URL of the NTS FastAPI service.
                 Default: https://news-trading-system-production.up.railway.app
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NTS_API_BASE = os.environ.get(
    "NTS_API_URL",
    "https://news-trading-system-production.up.railway.app",
).rstrip("/")

_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = FastMCP("nts-trading")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _api_get(path: str, params: dict[str, Any] | None = None) -> dict | list:
    """GET request to the NTS API. Returns parsed JSON."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{NTS_API_BASE}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _fmt_json(data: Any) -> str:
    """Pretty-print JSON for tool output."""
    return json.dumps(data, indent=2, default=str)


def _fmt_signal_row(s: dict) -> str:
    """Format one signal row as a readable summary line."""
    ts = s.get("timestamp", "")[:16]
    ticker = s.get("ticker", "???")
    signal = s.get("signal", "?")
    conf = s.get("confidence", 0)
    price = s.get("price_at_signal")
    price_str = f"${price:.2f}" if price else "n/a"
    return f"  {ts}  {ticker:<6} {signal:<12} conf={conf}  price={price_str}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@server.tool()
async def get_signals(days: int = 7, strategy: str = "") -> str:
    """Get recent trading signals from the news trading system.

    Args:
        days: Look-back window in days (1-365). Default 7.
        strategy: Filter by strategy name (e.g. "Combined"). Empty = all.
    """
    try:
        params: dict[str, Any] = {"days": days, "limit": 50}
        data = await _api_get("/api/signals", params=params)

        if not isinstance(data, list):
            return _fmt_json(data)

        # Client-side strategy filter (API doesn't support it)
        if strategy:
            upper = strategy.upper()
            data = [s for s in data if (s.get("strategy") or "").upper() == upper]

        if not data:
            return f"No signals found in the last {days} day(s)."

        lines = [f"Found {len(data)} signal(s) in the last {days} day(s):\n"]
        for s in data:
            lines.append(_fmt_signal_row(s))

        lines.append(f"\nFull JSON:\n{_fmt_json(data)}")
        return "\n".join(lines)

    except httpx.ConnectError:
        return f"Error: Cannot reach NTS API at {NTS_API_BASE}. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"Error: API returned {exc.response.status_code}"


@server.tool()
async def get_portfolio() -> str:
    """Get current portfolio positions and P&L."""
    try:
        data = await _api_get("/api/portfolio")

        if not isinstance(data, dict):
            return _fmt_json(data)

        positions = data.get("positions", [])
        value = data.get("value", 0)
        cash = data.get("cash", 0)
        daily_pnl = data.get("daily_pnl", 0)
        daily_pnl_pct = data.get("daily_pnl_pct", 0)

        lines = [
            "Portfolio Summary",
            "─" * 40,
            f"  Total value:  ${value:,.2f}",
            f"  Cash:         ${cash:,.2f}",
            f"  Daily P&L:    ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)",
            "",
        ]

        if positions:
            lines.append(f"Positions ({len(positions)}):")
            for p in positions:
                ticker = p.get("ticker", "???")
                shares = p.get("shares", 0)
                entry = p.get("entry", 0)
                current = p.get("current", 0)
                pnl_pct = p.get("pnl_pct", 0)
                lines.append(
                    f"  {ticker:<6} {shares} shares  "
                    f"entry=${entry:.2f}  now=${current:.2f}  "
                    f"pnl={pnl_pct:+.1f}%"
                )
        else:
            lines.append("No open positions.")

        return "\n".join(lines)

    except httpx.ConnectError:
        return f"Error: Cannot reach NTS API at {NTS_API_BASE}. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"Error: API returned {exc.response.status_code}"


@server.tool()
async def get_status() -> str:
    """Get system health, last session time, next session."""
    try:
        data = await _api_get("/api/status")

        if not isinstance(data, dict):
            return _fmt_json(data)

        running = data.get("running", False)
        mode = data.get("mode", "unknown")
        uptime = data.get("uptime_seconds", 0)
        last_session = data.get("last_session", "n/a")
        last_run = data.get("last_run_at", "n/a")
        next_session = data.get("next_session", "n/a")
        next_run = data.get("next_run_at", "n/a")
        watchlist = data.get("watchlist", [])

        hours = uptime // 3600
        mins = (uptime % 3600) // 60

        lines = [
            "NTS System Status",
            "─" * 40,
            f"  Running:       {'Yes' if running else 'No'}",
            f"  Mode:          {mode}",
            f"  Uptime:        {hours}h {mins}m",
            f"  Last session:  {last_session} ({last_run})",
            f"  Next session:  {next_session} ({next_run})",
            f"  Watchlist:     {', '.join(watchlist) if watchlist else 'empty'}",
        ]

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"\n  Queried at:    {now_utc}")
        return "\n".join(lines)

    except httpx.ConnectError:
        return f"Error: Cannot reach NTS API at {NTS_API_BASE}. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"Error: API returned {exc.response.status_code}"


@server.tool()
async def get_performance() -> str:
    """Get per-strategy performance metrics."""
    try:
        data = await _api_get("/api/performance")

        if not isinstance(data, dict):
            return _fmt_json(data)

        total = data.get("total_trades", 0)
        win_rate = data.get("win_rate")
        signals_today = data.get("signals_today", 0)
        strong_buys = data.get("strong_buy_today", 0)
        sessions_today = data.get("sessions_today", 0)
        sessions_total = data.get("sessions_total", 4)

        lines = [
            "Trading Performance",
            "─" * 40,
            f"  Total trades:     {total}",
            f"  Win rate:         {f'{win_rate:.1f}%' if win_rate is not None else 'n/a'}",
            f"  Signals today:    {signals_today}",
            f"  Strong buys:      {strong_buys}",
            f"  Sessions today:   {sessions_today} / {sessions_total}",
        ]
        return "\n".join(lines)

    except httpx.ConnectError:
        return f"Error: Cannot reach NTS API at {NTS_API_BASE}. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"Error: API returned {exc.response.status_code}"


@server.tool()
async def get_signal_detail(ticker: str) -> str:
    """Get detailed signal analysis for a specific ticker including bull/bear debate.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
    """
    try:
        params = {"ticker": ticker.upper(), "days": 30, "limit": 10}
        data = await _api_get("/api/signals", params=params)

        if not isinstance(data, list) or not data:
            return f"No recent signals found for {ticker.upper()}."

        lines = [
            f"Signal Detail: {ticker.upper()}",
            f"Found {len(data)} signal(s) in the last 30 days",
            "═" * 50,
        ]

        for s in data:
            ts = s.get("timestamp", "")[:16]
            signal = s.get("signal", "?")
            conf = s.get("confidence", 0)
            strategy = s.get("strategy", "?")
            price = s.get("price_at_signal")
            rsi = s.get("rsi")
            sentiment = s.get("sentiment_score")
            news = s.get("news_score")
            social = s.get("social_score")
            bull = s.get("bull_case")
            bear = s.get("bear_case")
            debate = s.get("debate_outcome")
            outcome_3d = s.get("outcome_3d_pct")
            outcome_5d = s.get("outcome_5d_pct")
            outcome_10d = s.get("outcome_10d_pct")
            executed = s.get("trade_executed")

            lines.append(f"\n{'─' * 50}")
            lines.append(f"  Time:       {ts}")
            lines.append(f"  Signal:     {signal}  (confidence: {conf})")
            lines.append(f"  Strategy:   {strategy}")
            if price:
                lines.append(f"  Price:      ${price:.2f}")

            # Scores
            scores = []
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

            # Bull/Bear debate
            if bull or bear:
                lines.append("")
                lines.append("  Bull/Bear Debate:")
                if bull:
                    lines.append(f"    Bull: {bull}")
                if bear:
                    lines.append(f"    Bear: {bear}")
                if debate:
                    lines.append(f"    Outcome: {debate}")

            # Outcomes
            outcomes = []
            if outcome_3d is not None:
                outcomes.append(f"3d={outcome_3d:+.2f}%")
            if outcome_5d is not None:
                outcomes.append(f"5d={outcome_5d:+.2f}%")
            if outcome_10d is not None:
                outcomes.append(f"10d={outcome_10d:+.2f}%")
            if outcomes:
                lines.append(f"  Outcomes:   {', '.join(outcomes)}")

            if executed is not None:
                lines.append(f"  Executed:   {'Yes' if executed else 'No'}")

        return "\n".join(lines)

    except httpx.ConnectError:
        return f"Error: Cannot reach NTS API at {NTS_API_BASE}. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"Error: API returned {exc.response.status_code}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run()
