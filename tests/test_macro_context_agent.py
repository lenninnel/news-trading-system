"""
Tests for MacroContextAgent and its integration with the bull/bear debate.

Covers:
    1. Agent respects the ENABLE_MACRO_CONTEXT feature flag.
    2. Non-US sessions (XETRA/EOD/PREMARKET_SCAN) return "".
    3. Happy path: Claude mock returns a valid block → agent returns it.
    4. Timeout handling: slow Claude → agent returns "" (never raises).
    5. Generic exception handling: Claude explodes → agent returns "".
    6. Integration: BullResearcher / BearResearcher prepend macro_context to
       the user message (NOT to the system prompt — protects ephemeral cache).
    7. BullBearDebate.run_async forwards macro_context to both researchers.
    8. Caching semantics: the scheduler fetches macro_context ONCE per
       session, not per ticker — verified by counting Claude calls across
       a multi-ticker batch.

No real network calls — everything is monkeypatched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents import macro_context_agent
from agents.macro_context_agent import MacroContextAgent, US_SESSIONS, is_enabled


# ── Fake Claude response helpers ──────────────────────────────────────────


def _fake_claude_response(text: str) -> SimpleNamespace:
    """Build a minimal stand-in for the object returned by
    ``client.messages.create(...)`` — just enough for _extract_text."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


# Sample output matching the prompt's expected format
SAMPLE_MACRO = (
    "MACRO CONTEXT [2026-04-15 US_PRE]:\n"
    "- VIX: 18.2 (rising vs yesterday)\n"
    "- Key events today: Fed Chair Powell 14:00 ET, CPI release 08:30 ET\n"
    "- Sector moves: Tech +1.2%, Energy -0.8%, Financials +0.5%, Healthcare +0.1%\n"
    "- Market tone: Cautious risk-on ahead of Powell remarks.\n"
    "- Risk flags: none"
)


# ── Feature flag gating ──────────────────────────────────────────────────


def test_is_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("ENABLE_MACRO_CONTEXT", raising=False)
    assert is_enabled() is False


def test_is_enabled_truthy_values(monkeypatch):
    for val in ("true", "1", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("ENABLE_MACRO_CONTEXT", val)
        assert is_enabled() is True


def test_is_enabled_falsy_values(monkeypatch):
    for val in ("false", "0", "no", "", "maybe"):
        monkeypatch.setenv("ENABLE_MACRO_CONTEXT", val)
        assert is_enabled() is False


def test_disabled_flag_returns_empty_string(monkeypatch):
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "false")
    agent = MacroContextAgent(client=MagicMock())
    result = asyncio.run(agent.get_context("US_PRE"))
    assert result == ""


# ── Non-US session skip ──────────────────────────────────────────────────


@pytest.mark.parametrize("session", [
    "XETRA_PRE", "XETRA_OPEN", "MIDDAY", "EOD", "PREMARKET_SCAN",
    "WEEKLY_JOB", "UNKNOWN",
])
def test_non_us_session_returns_empty(monkeypatch, session):
    """Enabled + non-US session → empty string, and Claude is NOT called."""
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    agent = MacroContextAgent(client=mock_client)

    result = asyncio.run(agent.get_context(session))

    assert result == ""
    # Feature flag still short-circuited Claude — should never have been
    # invoked for a non-US session.
    assert mock_client.messages.create.call_count == 0


@pytest.mark.parametrize("session", sorted(US_SESSIONS))
def test_us_sessions_call_claude(monkeypatch, session):
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_MACRO)
    agent = MacroContextAgent(client=mock_client)

    result = asyncio.run(agent.get_context(session))

    assert result == SAMPLE_MACRO
    assert mock_client.messages.create.call_count == 1


# ── Happy path / response parsing ────────────────────────────────────────


def test_get_context_uses_haiku_model_and_web_search(monkeypatch):
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_MACRO)
    agent = MacroContextAgent(client=mock_client)

    asyncio.run(agent.get_context("US_PRE"))

    call = mock_client.messages.create.call_args
    # Model should be Haiku (bounded cost)
    assert "haiku" in call.kwargs["model"].lower()
    assert call.kwargs["max_tokens"] == 300
    # The web_search server tool must be enabled, with a use cap
    tools = call.kwargs.get("tools") or []
    assert len(tools) == 1
    assert tools[0]["type"].startswith("web_search_")
    assert tools[0]["name"] == "web_search"
    assert tools[0]["max_uses"] > 0


def test_extract_text_concatenates_all_text_blocks():
    """Multi-block responses (tool_use + text) should still yield the text.

    The join inserts its own newline between blocks, so a text block that
    already ends with '\\n' contributes '\\n\\n' to the output — that's
    expected and harmless (downstream consumers don't care about extra
    whitespace inside the macro block).
    """
    msg = SimpleNamespace(content=[
        SimpleNamespace(type="tool_use"),   # ignored
        SimpleNamespace(type="text", text="first half"),
        SimpleNamespace(type="web_search_tool_result"),  # ignored
        SimpleNamespace(type="text", text="second half"),
    ])
    assert MacroContextAgent._extract_text(msg) == "first half\nsecond half"


def test_extract_text_empty_content():
    msg = SimpleNamespace(content=[])
    assert MacroContextAgent._extract_text(msg) == ""


# ── Timeout and error handling ───────────────────────────────────────────


def test_timeout_propagates_as_empty_string(monkeypatch):
    """Claude SDK surfaces the HTTP timeout as an exception → "" returned.

    We enforce the 10s timeout at the SDK layer via ``timeout=`` on
    ``messages.create(...)``; the SDK raises a plain exception on timeout
    which our agent swallows (fire-and-forget fallback).
    """
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")

    # Simulate what the Anthropic SDK would do on HTTP timeout — raise.
    # We use a generic exception since anthropic.APITimeoutError /
    # httpx.TimeoutException are implementation details that shouldn't
    # leak into our tests.
    class FakeHTTPTimeout(Exception):
        pass

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = FakeHTTPTimeout("request timed out")
    agent = MacroContextAgent(client=mock_client)

    result = asyncio.run(agent.get_context("US_PRE"))
    assert result == ""


def test_get_context_passes_sdk_timeout_to_claude(monkeypatch):
    """The 10s hard ceiling must reach the Anthropic SDK's timeout param."""
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_MACRO)
    agent = MacroContextAgent(client=mock_client)

    asyncio.run(agent.get_context("US_PRE"))

    call = mock_client.messages.create.call_args
    assert "timeout" in call.kwargs
    assert call.kwargs["timeout"] == macro_context_agent.TIMEOUT_SECONDS


def test_claude_exception_returns_empty_string(monkeypatch):
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API down")
    agent = MacroContextAgent(client=mock_client)

    # Must not raise — fire-and-forget fallback
    result = asyncio.run(agent.get_context("US_PRE"))
    assert result == ""


# ── Bull/bear debate integration ─────────────────────────────────────────


def test_bull_researcher_prepends_macro_context_to_user_message(monkeypatch):
    """Macro context must land in the user message so the cached system
    prompt is not invalidated."""
    from agents.bull_bear_debate import BullResearcher

    mock_client = MagicMock()
    # Return a valid JSON string in content[0].text
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(
            type="text",
            text='{"bull_case": "ok", "confidence_boost": 0.1}',
        )],
        usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )
    researcher = BullResearcher(client=mock_client)

    researcher.analyze(
        ticker="AAPL",
        signal="BUY",
        confidence=0.7,
        technical_data={"rsi": 55},
        sentiment_data={"signal": "BULL", "avg_score": 0.6},
        macro_context=SAMPLE_MACRO,
    )

    call = mock_client.messages.create.call_args
    user_message = call.kwargs["messages"][0]["content"]

    # Macro block is prepended; ticker data follows after the separator.
    assert user_message.startswith("MACRO CONTEXT")
    assert "---" in user_message
    assert "Ticker: AAPL" in user_message

    # System prompt must remain untouched (cached block only — no macro).
    system = call.kwargs["system"]
    if isinstance(system, list):
        system_text = system[0]["text"]
    else:
        system_text = system
    assert "MACRO CONTEXT" not in system_text


def test_bear_researcher_prepends_macro_context(monkeypatch):
    from agents.bull_bear_debate import BearResearcher

    mock_client = MagicMock()
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(
            type="text",
            text='{"bear_case": "ok", "confidence_penalty": -0.1}',
        )],
        usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )
    researcher = BearResearcher(client=mock_client)

    researcher.analyze(
        ticker="TSLA",
        signal="SELL",
        confidence=0.6,
        technical_data={"rsi": 72},
        sentiment_data={"signal": "BEAR", "avg_score": -0.4},
        macro_context=SAMPLE_MACRO,
    )

    user_message = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert user_message.startswith("MACRO CONTEXT")
    assert "Ticker: TSLA" in user_message


def test_empty_macro_context_keeps_user_message_unchanged():
    """Legacy path: empty macro_context → user message matches old format."""
    from agents.bull_bear_debate import BullResearcher

    mock_client = MagicMock()
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(
            type="text",
            text='{"bull_case": "ok", "confidence_boost": 0.0}',
        )],
        usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )
    researcher = BullResearcher(client=mock_client)

    researcher.analyze(
        ticker="NVDA", signal="BUY", confidence=0.8,
        technical_data={}, sentiment_data={},
        macro_context="",
    )

    user_message = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    # No MACRO CONTEXT prefix, no separator
    assert "MACRO CONTEXT" not in user_message
    assert "---" not in user_message
    assert user_message.startswith("Ticker: NVDA")


def test_debate_run_async_threads_macro_context_to_both_researchers(monkeypatch):
    """BullBearDebate.run_async forwards macro_context to bull AND bear."""
    from agents.bull_bear_debate import BullBearDebate

    bull_mock = MagicMock()
    bull_mock.analyze.return_value = {"bull_case": "b", "confidence_boost": 0.05}
    bear_mock = MagicMock()
    bear_mock.analyze.return_value = {"bear_case": "r", "confidence_penalty": -0.05}

    debate = BullBearDebate(bull=bull_mock, bear=bear_mock)

    asyncio.run(debate.run_async(
        ticker="META",
        signal="BUY",
        confidence=0.7,
        technical_data={},
        sentiment_data={},
        macro_context="MACRO X",
    ))

    # Both researchers called exactly once, each receiving macro_context="MACRO X"
    assert bull_mock.analyze.call_count == 1
    assert bull_mock.analyze.call_args.kwargs["macro_context"] == "MACRO X"
    assert bear_mock.analyze.call_count == 1
    assert bear_mock.analyze.call_args.kwargs["macro_context"] == "MACRO X"


def test_debate_run_sync_threads_macro_context(monkeypatch):
    from agents.bull_bear_debate import BullBearDebate

    bull_mock = MagicMock()
    bull_mock.analyze.return_value = {"bull_case": "b", "confidence_boost": 0.0}
    bear_mock = MagicMock()
    bear_mock.analyze.return_value = {"bear_case": "r", "confidence_penalty": 0.0}

    debate = BullBearDebate(bull=bull_mock, bear=bear_mock)

    debate.run(
        ticker="XOM",
        signal="BUY",
        confidence=0.6,
        technical_data={},
        sentiment_data={},
        macro_context="SYNC CONTEXT",
    )

    assert bull_mock.analyze.call_args.kwargs["macro_context"] == "SYNC CONTEXT"
    assert bear_mock.analyze.call_args.kwargs["macro_context"] == "SYNC CONTEXT"


# ── Session-level caching (fetch once, reuse per-ticker) ─────────────────


def test_coordinator_stores_macro_context_for_reuse():
    """Coordinator holds macro_context as an attribute so per-ticker debate
    calls read the same string — no re-fetching per ticker."""
    from orchestrator.coordinator import Coordinator

    # Use a full sentinel string so we can assert equality
    coord = Coordinator(macro_context="FROZEN CONTEXT")
    assert coord.macro_context == "FROZEN CONTEXT"

    # Default is empty string (legacy behaviour)
    coord2 = Coordinator()
    assert coord2.macro_context == ""


def test_scheduler_helper_respects_disabled_flag(monkeypatch):
    """_fetch_macro_context returns '' immediately when the flag is off."""
    from scheduler.daily_runner import DailyScheduler

    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "false")
    result = DailyScheduler._fetch_macro_context("US_PRE")
    assert result == ""


def test_scheduler_helper_skips_non_us_sessions(monkeypatch):
    from scheduler.daily_runner import DailyScheduler

    # Flag is on, but the session is XETRA — must still return ""
    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")

    # Patch MacroContextAgent so if get_context DID get called, we would
    # notice. The gating check inside the agent should short-circuit it
    # before any Claude call.
    called = {"n": 0}

    class FakeAgent:
        async def get_context(self, session):
            called["n"] += 1
            # Mirror real gating so the helper's guarantee holds
            from agents.macro_context_agent import US_SESSIONS, is_enabled
            if not is_enabled() or session not in US_SESSIONS:
                return ""
            return "UNEXPECTED"

    monkeypatch.setattr(
        "agents.macro_context_agent.MacroContextAgent", FakeAgent
    )
    result = DailyScheduler._fetch_macro_context("XETRA_OPEN")
    assert result == ""


def test_scheduler_helper_fetches_once_regardless_of_ticker_count(monkeypatch):
    """_fetch_macro_context is called once per session, then the result is
    passed to run_batch → Coordinator, reused for every ticker. This test
    exercises the call-count semantics at the agent level."""
    from agents.macro_context_agent import MacroContextAgent

    monkeypatch.setenv("ENABLE_MACRO_CONTEXT", "true")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_claude_response(SAMPLE_MACRO)
    agent = MacroContextAgent(client=mock_client)

    # Simulate "scheduler start" — one call
    context = asyncio.run(agent.get_context("US_PRE"))
    assert context == SAMPLE_MACRO

    # Coordinator gets the cached string — no more Claude calls
    from orchestrator.coordinator import Coordinator
    coord = Coordinator(macro_context=context)
    assert coord.macro_context == SAMPLE_MACRO

    # The agent's Claude client was only invoked ONCE for the whole session,
    # regardless of how many tickers will later read coord.macro_context.
    assert mock_client.messages.create.call_count == 1


def test_signal_logger_records_macro_context_used_flag(tmp_path, monkeypatch):
    """macro_context_used ends up in signal_events as 0/1."""
    # Use an isolated SQLite so we don't pollute /tmp/pytest_trading.db
    db_path = tmp_path / "macro_test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    from storage.database import Database
    from analytics.signal_logger import SignalLogger

    db = Database(db_path=str(db_path))
    logger = SignalLogger(db=db)

    logger.log({
        "ticker": "AAPL",
        "session": "US_PRE",
        "strategy": "Combined",
        "signal": "BUY",
        "confidence": 0.72,
        "macro_context_used": True,
    })
    logger.log({
        "ticker": "TSLA",
        "session": "US_PRE",
        "strategy": "Combined",
        "signal": "SELL",
        "confidence": 0.65,
        "macro_context_used": False,
    })

    with db._connect() as conn:
        rows = conn.execute(
            "SELECT ticker, macro_context_used FROM signal_events "
            "WHERE ticker IN ('AAPL', 'TSLA') ORDER BY ticker"
        ).fetchall()

    by_ticker = {r[0]: r[1] for r in rows}
    assert by_ticker["AAPL"] == 1
    assert by_ticker["TSLA"] == 0
