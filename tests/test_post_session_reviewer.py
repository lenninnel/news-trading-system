"""
Tests for PostSessionReviewer and its scheduler integration.

Covers:
    1. ENABLE_POST_SESSION_REVIEW flag gates the whole pipeline.
    2. Only EOD and US_OPEN sessions trigger a review; everything else
       short-circuits to "".
    3. Happy path: mock Claude returns a formatted block → agent returns
       it verbatim.
    4. SDK-level timeout (exception bubbling out of messages.create) →
       fire-and-forget empty string, never raises.
    5. _gather_state is resilient to:
         * DB init failing
         * Empty DB (no positions, no trades, no signal_events)
         * get_sector returning None
    6. Concentration flag fires only when 2+ open positions share a
       (non-null) sector.
    7. Stale position flag computed from MIN(created_at) in trade_history
       when `days_held > STALE_POSITION_DAYS`.
    8. Low-conviction flag pulled from the signals list (trade_executed=True
       AND confidence < LOW_CONVICTION_THRESHOLD).
    9. Prompt builder lays out all state cleanly.
   10. Scheduler integration: `_run_post_session_review` logs to
       signal_events with strategy="PostSessionReviewer" and sends via
       Telegram. Any failure in either step is swallowed.
   11. A failing reviewer does not raise into the scheduler — verified
       by calling `_run_post_session_review` with a patched agent that
       explodes.

All Claude / HTTP calls are monkeypatched — no real API traffic.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents import post_session_reviewer as psr
from agents.post_session_reviewer import PostSessionReviewer


# ── Fake Claude response ─────────────────────────────────────────────────


SAMPLE_REVIEW = (
    "📊 EOD REVIEW — 2026-04-11\n\n"
    "🔄 Sessions: 7/7 completed\n"
    "💼 Positions: 2 open | P&L: $+125.50\n"
    "📈 Trades: 3 executed today\n\n"
    "SIGNALS: Strong bull signals on META and NVDA led the day.\n\n"
    "⚠️ FLAGS:\n"
    "- Concentration: None\n"
    "- Stale position: None\n"
    "- Low conviction: None\n\n"
    "TOMORROW: Watch for Fed minutes at 14:00 ET."
)


def _fake_claude_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    )


# ── Feature flag gating ──────────────────────────────────────────────────


def test_is_enabled_defaults_false(monkeypatch):
    monkeypatch.delenv("ENABLE_POST_SESSION_REVIEW", raising=False)
    assert psr.is_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "TRUE", "Yes"])
def test_is_enabled_truthy(monkeypatch, val):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", val)
    assert psr.is_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "", "maybe"])
def test_is_enabled_falsy(monkeypatch, val):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", val)
    assert psr.is_enabled() is False


def test_disabled_flag_short_circuits(monkeypatch):
    monkeypatch.delenv("ENABLE_POST_SESSION_REVIEW", raising=False)
    mock_client = MagicMock()
    reviewer = PostSessionReviewer(client=mock_client)

    result = asyncio.run(reviewer.review("EOD", ["AAPL"], []))
    assert result == ""
    # Critically: the expensive Claude call must not be reached
    assert mock_client.messages.create.call_count == 0


# ── Reviewable session filter ────────────────────────────────────────────


@pytest.mark.parametrize("session", [
    "XETRA_PRE", "XETRA_OPEN", "PREMARKET_SCAN",
    "US_PRE", "PEAD_OPEN", "MIDDAY", "UNKNOWN",
])
def test_non_reviewable_session_returns_empty(monkeypatch, session):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", "true")
    mock_client = MagicMock()
    reviewer = PostSessionReviewer(client=mock_client)

    result = asyncio.run(reviewer.review(session, ["AAPL"], []))
    assert result == ""
    assert mock_client.messages.create.call_count == 0


@pytest.mark.parametrize("session", ["EOD", "US_OPEN"])
def test_reviewable_sessions_call_claude(monkeypatch, session, tmp_path):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", "true")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.db"))
    # The gather step will fail gracefully (no DB) → empty state.
    # We still expect the Claude call to go through.
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_REVIEW)
    reviewer = PostSessionReviewer(client=mock_client)

    result = asyncio.run(reviewer.review(session, ["AAPL"], []))
    assert result == SAMPLE_REVIEW
    assert mock_client.messages.create.call_count == 1


# ── Claude call parameters ───────────────────────────────────────────────


def test_review_passes_haiku_model_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", "true")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.db"))
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_REVIEW)
    reviewer = PostSessionReviewer(client=mock_client)

    asyncio.run(reviewer.review("EOD", ["AAPL"], []))

    call = mock_client.messages.create.call_args
    assert "haiku" in call.kwargs["model"].lower()
    assert call.kwargs["max_tokens"] == psr.MAX_TOKENS
    assert call.kwargs["timeout"] == psr.TIMEOUT_SECONDS
    assert "system" in call.kwargs
    # Review system prompt mentions the fixed format
    assert "EOD REVIEW" in call.kwargs["system"]


# ── Exception / timeout handling ────────────────────────────────────────


def test_review_swallows_claude_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", "true")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.db"))
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API down")
    reviewer = PostSessionReviewer(client=mock_client)

    # Must not raise — fire-and-forget
    result = asyncio.run(reviewer.review("EOD", ["AAPL"], []))
    assert result == ""


def test_review_swallows_sdk_timeout(monkeypatch, tmp_path):
    """Simulate the Anthropic SDK raising a timeout."""
    monkeypatch.setenv("ENABLE_POST_SESSION_REVIEW", "true")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.db"))

    class FakeTimeout(Exception):
        pass

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = FakeTimeout("HTTP timeout")
    reviewer = PostSessionReviewer(client=mock_client)

    result = asyncio.run(reviewer.review("EOD", ["AAPL"], []))
    assert result == ""


# ── _extract_text ────────────────────────────────────────────────────────


def test_extract_text_multiple_blocks():
    msg = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="line one"),
        SimpleNamespace(type="tool_use"),
        SimpleNamespace(type="text", text="line two"),
    ])
    assert PostSessionReviewer._extract_text(msg) == "line one\nline two"


def test_extract_text_empty_content():
    msg = SimpleNamespace(content=[])
    assert PostSessionReviewer._extract_text(msg) == ""


# ── _gather_state DB fixtures ────────────────────────────────────────────


PORTFOLIO_DDL = """
CREATE TABLE portfolio_positions (
    ticker         TEXT PRIMARY KEY,
    shares         INTEGER NOT NULL,
    avg_price      REAL    NOT NULL,
    current_value  REAL,
    updated_at     TEXT
)
"""

TRADE_HISTORY_DDL = """
CREATE TABLE trade_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT NOT NULL,
    action     TEXT,
    shares     INTEGER,
    price      REAL,
    stop_loss  REAL,
    take_profit REAL,
    pnl        REAL DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

SIGNAL_EVENTS_DDL = """
CREATE TABLE signal_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    session      TEXT,
    ticker       TEXT NOT NULL,
    strategy     TEXT,
    signal       TEXT NOT NULL,
    confidence   REAL,
    bull_case    TEXT
)
"""


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """Build a seeded DB with 2 positions, 3 trades today, 1 session done."""
    path = tmp_path / "psr_test.db"
    conn = sqlite3.connect(str(path))
    conn.execute(PORTFOLIO_DDL)
    conn.execute(TRADE_HISTORY_DDL)
    conn.execute(SIGNAL_EVENTS_DDL)

    now = datetime.now(timezone.utc)
    twenty_days_ago = (now - timedelta(days=20)).isoformat()
    today_iso = now.isoformat()

    # Positions: META is fresh (opened today), NVDA is stale (20d old)
    conn.execute(
        "INSERT INTO portfolio_positions VALUES (?, ?, ?, ?, ?)",
        ("META", 10, 500.0, 5250.0, today_iso),
    )
    conn.execute(
        "INSERT INTO portfolio_positions VALUES (?, ?, ?, ?, ?)",
        ("NVDA", 5, 800.0, 4200.0, today_iso),
    )

    # Trade history for age calculation
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES ('META', 'BUY', 10, 500.0, 0.0, ?)",
        (today_iso,),
    )
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES ('NVDA', 'BUY', 5, 800.0, 0.0, ?)",
        (twenty_days_ago,),
    )
    # Two today trades (for the today_trades count)
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES ('META', 'BUY', 10, 500.0, 25.0, ?)",
        (today_iso,),
    )
    conn.execute(
        "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
        "VALUES ('XOM', 'SELL', 3, 120.0, -15.0, ?)",
        (today_iso,),
    )

    # One session completed today (signal_events row with a session)
    conn.execute(
        "INSERT INTO signal_events (timestamp, session, ticker, signal) "
        "VALUES (?, 'US_OPEN', 'META', 'BUY')",
        (today_iso,),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DB_PATH", str(path))
    return path


def test_gather_state_computes_positions_with_age(fixture_db):
    state = psr._gather_state("EOD", ["META", "NVDA"], [])

    tickers = {p["ticker"] for p in state["positions"]}
    assert tickers == {"META", "NVDA"}

    meta = next(p for p in state["positions"] if p["ticker"] == "META")
    nvda = next(p for p in state["positions"] if p["ticker"] == "NVDA")

    # META: entry=500, current_value=5250 over 10 shares → current=525, pnl=+5%
    assert meta["entry"] == 500.0
    assert meta["current"] == 525.0
    assert meta["pnl_pct"] == 5.0
    assert meta["days_held"] == 0

    # NVDA: entry=800, current_value=4200 over 5 → current=840, pnl=+5%
    assert nvda["entry"] == 800.0
    assert nvda["current"] == 840.0
    assert nvda["pnl_pct"] == 5.0
    assert nvda["days_held"] == 20


def test_gather_state_today_trades_count(fixture_db):
    state = psr._gather_state("EOD", [], [])
    # 3 today trades: META BUY (age row, pnl=0) + META BUY (pnl=25) +
    # XOM SELL (pnl=-15). NVDA BUY was 20 days ago so not included.
    assert len(state["today_trades"]) == 3
    assert state["daily_pnl"] == pytest.approx(10.0)  # 0 + 25 - 15


def test_gather_state_sessions_done(fixture_db):
    state = psr._gather_state("EOD", [], [])
    # One distinct session in signal_events today
    assert state["sessions_done"] == 1


def test_gather_state_stale_flag(fixture_db):
    state = psr._gather_state("EOD", [], [])
    stale = {p["ticker"] for p in state["stale"]}
    # NVDA is 20 days old > 15 day threshold
    assert "NVDA" in stale
    assert "META" not in stale


def test_gather_state_low_conviction_flag(fixture_db):
    signals = [
        {"ticker": "AAPL", "confidence": 0.80, "trade_executed": True},  # high, executed
        {"ticker": "MSFT", "confidence": 0.20, "trade_executed": True},  # LOW, executed
        {"ticker": "TSLA", "confidence": 0.15, "trade_executed": False}, # low but not executed
    ]
    state = psr._gather_state("EOD", [], signals)
    low = [s["ticker"] for s in state["low_conviction"]]
    assert low == ["MSFT"]  # only executed low-conf trades


def test_gather_state_concentration_flag(fixture_db, monkeypatch):
    """2+ open positions in the same sector trigger the flag."""
    # get_sector is a module-level function in storage.database that
    # reads config/sector_map.json. Monkeypatch the reviewer's locally
    # re-imported reference via the sector map lookup.
    def fake_get_sector(ticker):
        return "technology" if ticker in ("META", "NVDA") else None

    monkeypatch.setattr("storage.database.get_sector", fake_get_sector)

    state = psr._gather_state("EOD", [], [])
    assert "technology" in state["concentrated"]
    assert set(state["concentrated"]["technology"]) == {"META", "NVDA"}


def test_gather_state_no_concentration_when_single(fixture_db, monkeypatch):
    """A lone position in a sector should NOT trigger concentration."""
    def fake_get_sector(ticker):
        # META is tech, NVDA is healthcare → no bucket has 2+
        return {"META": "technology", "NVDA": "healthcare"}.get(ticker)

    monkeypatch.setattr("storage.database.get_sector", fake_get_sector)

    state = psr._gather_state("EOD", [], [])
    assert state["concentrated"] == {}


def test_gather_state_empty_db_returns_empty_state(tmp_path, monkeypatch):
    """Pointing at a missing DB path must not raise."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nope.db"))
    state = psr._gather_state("EOD", [], [])
    assert state["positions"] == []
    assert state["today_trades"] == []
    assert state["sessions_done"] == 0
    assert state["stale"] == []
    assert state["low_conviction"] == []


# ── _build_user_prompt ───────────────────────────────────────────────────


def test_build_user_prompt_includes_all_state():
    state = {
        "positions": [
            {"ticker": "META", "entry": 500.0, "current": 525.0,
             "pnl_pct": 5.0, "days_held": 1},
        ],
        "today_trades": [
            {"ticker": "META", "action": "BUY", "shares": 10,
             "price": 500.0, "pnl": 0.0},
        ],
        "concentrated": {"technology": ["META", "NVDA"]},
        "stale": [{"ticker": "NVDA", "days_held": 22}],
        "low_conviction": [{"ticker": "MSFT", "confidence": 0.18}],
        "sessions_done": 5,
        "daily_pnl": 125.5,
    }
    signals = [
        {"ticker": "META", "signal": "STRONG BUY", "confidence": 0.82,
         "debate_outcome": "agree", "trade_executed": False},
    ]
    prompt = psr._build_user_prompt("EOD", ["META"], signals, state)

    # All the key data must appear
    assert "SESSION: EOD" in prompt
    assert "META" in prompt
    assert "STRONG BUY" in prompt
    assert "technology" in prompt
    assert "NVDA (22d)" in prompt
    assert "MSFT (0.18)" in prompt
    assert "sessions_done=5" in prompt
    assert "daily_pnl_str=$+125.50" in prompt


def test_build_user_prompt_empty_sections_render_placeholder():
    state = {
        "positions": [],
        "today_trades": [],
        "concentrated": {},
        "stale": [],
        "low_conviction": [],
        "sessions_done": 0,
        "daily_pnl": 0.0,
    }
    prompt = psr._build_user_prompt("EOD", [], [], state)
    assert "(no open positions)" in prompt
    assert "(none)" in prompt  # trades
    assert "(no sector with 2+ positions)" in prompt
    assert "STALE POSITIONS" in prompt  # section header still present
    assert ": None" in prompt  # stale/low-conviction show "None"


# ── Scheduler integration ────────────────────────────────────────────────


def test_scheduler_run_post_session_review_happy_path(monkeypatch):
    """_run_post_session_review wires reviewer → telegram → signal_log."""
    from scheduler.daily_runner import DailyScheduler

    # Stub TelegramNotifier
    tg = MagicMock()

    scheduler = DailyScheduler.__new__(DailyScheduler)
    scheduler._tg = tg

    # Stub the agent so we don't touch Claude / DB
    fake_reviewer = MagicMock()

    async def fake_review(session, tickers, signals):
        return "📊 EOD REVIEW — stub text"

    fake_reviewer.review = fake_review
    monkeypatch.setattr(
        "agents.post_session_reviewer.PostSessionReviewer",
        lambda *a, **k: fake_reviewer,
    )

    # Stub SignalLogger so we can assert it was called with the right fields
    mock_log = MagicMock()
    monkeypatch.setattr(
        "analytics.signal_logger.SignalLogger",
        lambda *a, **k: mock_log,
    )

    batch = {
        "results": [
            {"ticker": "META", "combined_signal": "BUY", "confidence": 0.75,
             "execution": {"trade_id": "t-1"}},
            {"ticker": "NVDA", "combined_signal": "HOLD", "confidence": 0.30,
             "execution": {}},
        ],
    }

    scheduler._run_post_session_review("EOD", ["META", "NVDA"], batch)

    # Telegram sent the review text
    tg._send.assert_called_once()
    sent_text = tg._send.call_args[0][0]
    assert "EOD REVIEW" in sent_text

    # SignalLogger got one row with the right fields
    mock_log.log.assert_called_once()
    logged = mock_log.log.call_args[0][0]
    assert logged["session"] == "EOD"
    assert logged["strategy"] == "PostSessionReviewer"
    assert logged["signal"] == "REVIEW"
    assert logged["ticker"] == "SESSION"  # sentinel
    assert "EOD REVIEW" in logged["bull_case"]


def test_scheduler_review_swallows_agent_failure(monkeypatch):
    """If the agent call raises, the helper must return normally."""
    from scheduler.daily_runner import DailyScheduler

    scheduler = DailyScheduler.__new__(DailyScheduler)
    scheduler._tg = MagicMock()

    class BrokenReviewer:
        async def review(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "agents.post_session_reviewer.PostSessionReviewer",
        lambda *a, **k: BrokenReviewer(),
    )

    # Must not raise
    scheduler._run_post_session_review(
        "EOD", ["META"],
        {"results": [{"ticker": "META", "combined_signal": "BUY",
                      "confidence": 0.7, "execution": {}}]},
    )

    # Telegram should NOT have been called (review was empty)
    scheduler._tg._send.assert_not_called()


def test_scheduler_review_empty_text_no_telegram(monkeypatch):
    """Reviewer returns "" (disabled) → no Telegram, no signal log."""
    from scheduler.daily_runner import DailyScheduler

    scheduler = DailyScheduler.__new__(DailyScheduler)
    scheduler._tg = MagicMock()

    fake_reviewer = MagicMock()

    async def empty_review(session, tickers, signals):
        return ""

    fake_reviewer.review = empty_review
    monkeypatch.setattr(
        "agents.post_session_reviewer.PostSessionReviewer",
        lambda *a, **k: fake_reviewer,
    )

    mock_log = MagicMock()
    monkeypatch.setattr(
        "analytics.signal_logger.SignalLogger",
        lambda *a, **k: mock_log,
    )

    scheduler._run_post_session_review("EOD", [], {"results": []})

    scheduler._tg._send.assert_not_called()
    mock_log.log.assert_not_called()


def test_scheduler_review_telegram_failure_still_logs(monkeypatch):
    """Telegram send raises → signal_events still gets the row."""
    from scheduler.daily_runner import DailyScheduler

    scheduler = DailyScheduler.__new__(DailyScheduler)
    scheduler._tg = MagicMock()
    scheduler._tg._send.side_effect = RuntimeError("telegram dead")

    fake_reviewer = MagicMock()

    async def fake_review(session, tickers, signals):
        return "review body"

    fake_reviewer.review = fake_review
    monkeypatch.setattr(
        "agents.post_session_reviewer.PostSessionReviewer",
        lambda *a, **k: fake_reviewer,
    )

    mock_log = MagicMock()
    monkeypatch.setattr(
        "analytics.signal_logger.SignalLogger",
        lambda *a, **k: mock_log,
    )

    scheduler._run_post_session_review("EOD", [], {"results": []})

    # Logging still happened despite Telegram failure
    mock_log.log.assert_called_once()
    logged = mock_log.log.call_args[0][0]
    assert logged["bull_case"] == "review body"
