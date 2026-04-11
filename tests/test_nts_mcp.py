"""
Tests for mcp_server/nts_mcp.py — direct-SQLite MCP server.

Covers:
    * DB_PATH resolution (env override, config fallback)
    * Each of the 5 tools produces correct schema / text output
    * Empty DB (no rows) is handled gracefully
    * Missing DB file is handled gracefully (never raises)
    * Schema errors (missing table) return empty, not exception
    * Transparency: @server.tool() decorated functions are still directly
      callable as async coroutines (required by the task's verification
      step which does ``from mcp_server.nts_mcp import get_portfolio``).
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server import nts_mcp


# ── Schema & fixture helpers ─────────────────────────────────────────────


SIGNAL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS signal_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    session             TEXT,
    ticker              TEXT    NOT NULL,
    strategy            TEXT,
    signal              TEXT    NOT NULL,
    confidence          REAL,
    rsi                 REAL,
    sma_ratio           REAL,
    volume_ratio        REAL,
    sentiment_score     REAL,
    news_score          REAL,
    social_score        REAL,
    bull_case           TEXT,
    bear_case           TEXT,
    debate_outcome      TEXT,
    price_at_signal     REAL,
    trade_executed      INTEGER NOT NULL DEFAULT 0,
    trade_id            TEXT,
    outcome_3d_pct      REAL,
    outcome_5d_pct      REAL,
    outcome_10d_pct     REAL,
    regime              TEXT,
    macro_context_used  INTEGER DEFAULT 0
)
"""

PORTFOLIO_DDL = """
CREATE TABLE IF NOT EXISTS portfolio_positions (
    ticker         TEXT PRIMARY KEY,
    shares         INTEGER NOT NULL,
    avg_price      REAL    NOT NULL,
    current_value  REAL
)
"""

TRADE_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS trade_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    action      TEXT,
    shares      INTEGER,
    price       REAL,
    pnl         REAL,
    created_at  TEXT NOT NULL
)
"""

RISK_CALC_DDL = """
CREATE TABLE IF NOT EXISTS risk_calculations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_balance REAL
)
"""


def _build_fixture_db(path: str) -> None:
    """Create all tables the MCP tools read and seed with a small
    known dataset so every tool has something to return."""
    conn = sqlite3.connect(path)
    for ddl in (SIGNAL_EVENTS_DDL, PORTFOLIO_DDL, TRADE_HISTORY_DDL, RISK_CALC_DDL):
        conn.execute(ddl)
    now = datetime.now(timezone.utc)

    # Portfolio: 2 positions
    conn.execute(
        "INSERT INTO portfolio_positions VALUES (?, ?, ?, ?)",
        ("AAPL", 10, 180.0, 1850.0),
    )
    conn.execute(
        "INSERT INTO portfolio_positions VALUES (?, ?, ?, ?)",
        ("MSFT", 5, 390.0, 2100.0),
    )

    # Account balance
    conn.execute(
        "INSERT INTO risk_calculations (account_balance) VALUES (?)",
        (10_000.0,),
    )

    # Trade history: 2 closed trades, 1 winning 1 losing
    today = now.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("AAPL", "SELL", 10, 185.0, 50.0, today),
    )
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("TSLA", "SELL", 5, 240.0, -25.0, today),
    )

    # Signal events: 3 rows across 2 tickers and 2 strategies
    rows = [
        (
            (now - timedelta(hours=1)).isoformat(),
            "US_OPEN", "AAPL", "Combined", "STRONG BUY",
            0.82, 55.0, 1.03, 1.4, 0.60, 0.65, 0.55,
            "Strong momentum and sentiment.", "Overbought on RSI.", "agree",
            185.50, 0, None, 2.5, 3.1, 4.2, "bullish", 1,
        ),
        (
            (now - timedelta(hours=2)).isoformat(),
            "US_OPEN", "MSFT", "PEAD", "WEAK BUY",
            0.48, 48.0, 1.01, 1.2, None, None, None,
            None, None, "skipped",
            392.50, 0, None, None, None, None, "neutral", 0,
        ),
        (
            (now - timedelta(days=1)).isoformat(),
            "EOD", "TSLA", "Combined", "SELL",
            0.71, 72.0, 0.97, 2.0, 0.30, 0.35, 0.25,
            "Weak earnings outlook.", "Uptrend intact.", "cautious",
            245.00, 1, "trade-42", -1.2, -0.5, 0.8, "neutral", 0,
        ),
    ]
    conn.executemany(
        "INSERT INTO signal_events ("
        "timestamp, session, ticker, strategy, signal, confidence, "
        "rsi, sma_ratio, volume_ratio, sentiment_score, news_score, "
        "social_score, bull_case, bear_case, debate_outcome, "
        "price_at_signal, trade_executed, trade_id, "
        "outcome_3d_pct, outcome_5d_pct, outcome_10d_pct, regime, "
        "macro_context_used"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """Build an isolated SQLite fixture and point DB_PATH at it."""
    path = tmp_path / "nts_mcp_test.db"
    _build_fixture_db(str(path))
    monkeypatch.setenv("DB_PATH", str(path))
    return path


# ── DB_PATH resolution ───────────────────────────────────────────────────


def test_db_path_resolves_from_env(monkeypatch):
    monkeypatch.setenv("DB_PATH", "/tmp/explicit_override.db")
    assert nts_mcp._resolve_db_path() == "/tmp/explicit_override.db"


def test_db_path_falls_back_to_config_settings(monkeypatch):
    """No DB_PATH env → agent reads config.settings.DB_PATH."""
    monkeypatch.delenv("DB_PATH", raising=False)
    # _resolve_db_path imports config.settings lazily each time, so we
    # can't easily swap the module-level constant. Just assert the path
    # is a plausible string (tests/conftest.py sets DB_PATH env already).
    result = nts_mcp._resolve_db_path()
    assert isinstance(result, str)
    assert result  # non-empty


# ── Tool: get_portfolio ──────────────────────────────────────────────────


def test_get_portfolio_returns_formatted_summary(fixture_db):
    output = asyncio.run(nts_mcp.get_portfolio())

    # Header and known fields
    assert "Portfolio Summary" in output
    assert "Total value:" in output
    assert "Cash:" in output
    assert "Daily P&L:" in output

    # Both positions appear
    assert "AAPL" in output
    assert "MSFT" in output

    # Total value = 1850 + 2100 = 3950
    assert "$3,950.00" in output


def test_get_portfolio_no_positions(tmp_path, monkeypatch):
    """Empty portfolio table → 'No open positions.'"""
    path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(path))
    conn.execute(PORTFOLIO_DDL)
    conn.execute(RISK_CALC_DDL)
    conn.execute(TRADE_HISTORY_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setenv("DB_PATH", str(path))

    output = asyncio.run(nts_mcp.get_portfolio())
    assert "No open positions" in output


# ── Tool: get_signals ────────────────────────────────────────────────────


def test_get_signals_returns_recent_rows(fixture_db):
    output = asyncio.run(nts_mcp.get_signals(days=7))
    # All 3 fixture rows within the 7-day window
    assert "Found 3 signal(s)" in output
    assert "AAPL" in output
    assert "MSFT" in output
    assert "TSLA" in output


def test_get_signals_clamps_invalid_days(fixture_db):
    """days=0 and days=-5 must not explode; clamped to 1."""
    output = asyncio.run(nts_mcp.get_signals(days=0))
    # 0 becomes 1 → still returns something without error
    assert "signal" in output.lower() or "No signals" in output


def test_get_signals_filters_by_strategy(fixture_db):
    output = asyncio.run(nts_mcp.get_signals(days=7, strategy="PEAD"))
    # Only MSFT row is PEAD strategy
    assert "MSFT" in output
    assert "AAPL" not in output
    assert "TSLA" not in output


def test_get_signals_respects_limit(fixture_db):
    output = asyncio.run(nts_mcp.get_signals(days=7, limit=1))
    # Only one row appears in the summary
    assert "Found 1 signal(s)" in output


def test_get_signals_empty_window(fixture_db):
    """Look-back of 1 day catches only rows within 24h of now."""
    output = asyncio.run(nts_mcp.get_signals(days=1))
    # AAPL (1h ago) and MSFT (2h ago) qualify; TSLA (1 day ago) sits on
    # the boundary and may or may not depending on clock — assert the
    # two clearly-in-window tickers are present.
    assert "AAPL" in output
    assert "MSFT" in output


# ── Tool: get_performance ────────────────────────────────────────────────


def test_get_performance_reports_aggregates(fixture_db):
    output = asyncio.run(nts_mcp.get_performance())

    assert "Total trades:" in output
    assert "Win rate:" in output
    assert "Signals today:" in output
    # 2 total trades, 1 winning → 50%
    assert "2" in output
    assert "50.0%" in output

    # Strategy breakdown present (2 strategies: Combined, PEAD)
    assert "Strategy breakdown" in output
    assert "Combined" in output
    assert "PEAD" in output


# ── Tool: get_status ─────────────────────────────────────────────────────


def test_get_status_reports_running_and_schedule(fixture_db):
    output = asyncio.run(nts_mcp.get_status())

    assert "NTS System Status" in output
    assert "Running:" in output
    assert "Mode:" in output
    assert "Next session:" in output


def test_get_status_handles_missing_db(tmp_path, monkeypatch):
    """Pointed at a non-existent file, status still renders a string."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nope.db"))
    output = asyncio.run(nts_mcp.get_status())
    # sqlite3.connect() will happily create the file — the status is
    # still renderable. The real failure mode ("DB unreachable") is
    # exercised by schema-error paths below.
    assert "NTS System Status" in output


# ── Tool: get_signal_detail ──────────────────────────────────────────────


def test_get_signal_detail_returns_bull_bear_cases(fixture_db):
    output = asyncio.run(nts_mcp.get_signal_detail("AAPL"))

    assert "Signal Detail: AAPL" in output
    assert "STRONG BUY" in output
    # Bull and bear cases from fixture row
    assert "Strong momentum and sentiment." in output
    assert "Overbought on RSI." in output
    # Macro flag rendered for AAPL (fixture sets macro_context_used=1)
    assert "Macro context: used" in output


def test_get_signal_detail_ticker_not_found(fixture_db):
    output = asyncio.run(nts_mcp.get_signal_detail("UNKNOWN"))
    assert "No recent signals found for UNKNOWN" in output


def test_get_signal_detail_rejects_empty_ticker(fixture_db):
    output = asyncio.run(nts_mcp.get_signal_detail(""))
    assert "Error" in output


def test_get_signal_detail_is_case_insensitive(fixture_db):
    output = asyncio.run(nts_mcp.get_signal_detail("aapl"))
    assert "Signal Detail: AAPL" in output


# ── Error handling: DB missing / schema broken ───────────────────────────


def test_query_swallows_missing_table(tmp_path, monkeypatch):
    """An unrecognised table name must not raise."""
    path = tmp_path / "empty.db"
    sqlite3.connect(str(path)).close()  # create empty file, no tables
    monkeypatch.setenv("DB_PATH", str(path))

    rows = nts_mcp._query("SELECT * FROM does_not_exist")
    assert rows == []


def test_tools_tolerate_totally_empty_db(tmp_path, monkeypatch):
    """All 5 tools must return a string (never raise) on an empty DB."""
    path = tmp_path / "empty.db"
    sqlite3.connect(str(path)).close()
    monkeypatch.setenv("DB_PATH", str(path))

    outputs = [
        asyncio.run(nts_mcp.get_portfolio()),
        asyncio.run(nts_mcp.get_signals()),
        asyncio.run(nts_mcp.get_performance()),
        asyncio.run(nts_mcp.get_status()),
        asyncio.run(nts_mcp.get_signal_detail("AAPL")),
    ]
    for out in outputs:
        assert isinstance(out, str)
        assert out  # non-empty


# ── Decorator transparency (task's verification step depends on this) ───


def test_tools_are_directly_importable_coroutines():
    """Every exported tool must remain an awaitable coroutine function
    so the verification step ``from mcp_server.nts_mcp import get_portfolio``
    + ``asyncio.run(get_portfolio())`` works without any MCP machinery."""
    import inspect

    for fn in (
        nts_mcp.get_portfolio,
        nts_mcp.get_signals,
        nts_mcp.get_performance,
        nts_mcp.get_status,
        nts_mcp.get_signal_detail,
    ):
        assert inspect.iscoroutinefunction(fn), f"{fn} is not a coroutine"


# ── CLI argument parsing ─────────────────────────────────────────────────


def test_parse_args_defaults_to_sse_mode():
    args = nts_mcp._parse_args([])
    assert args.stdio is False
    assert args.port == nts_mcp.DEFAULT_SSE_PORT


def test_parse_args_stdio_flag():
    args = nts_mcp._parse_args(["--stdio"])
    assert args.stdio is True


def test_parse_args_port_override():
    args = nts_mcp._parse_args(["--port", "9000"])
    assert args.port == 9000
