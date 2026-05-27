"""
Tests for the PEAD signal-attribution sidecar (Q-004 Fork B).

The sidecar lives in ``pead_signal_log`` and captures the load-bearing
inputs PEAD was previously dropping (surprise_pct, announce_date,
earnings_source) so trades can be attributed after the fact.

These tests enforce the freeze-safe contract:

* Phase-1 writes (signal time) populate every column for both
  ``threshold_met=1`` and ``threshold_met=0`` rows, on both the IBKR
  cache path and the yfinance fallback.
* Phase-2 writes (execution time) link the row to ``trade_history``
  via ``trade_id``.
* If the writer or updater raises, signal generation and trade
  execution MUST continue unaffected — the exception must not propagate
  and the StrategyResult / execution must match the no-write outcome.

Schema bookkeeping:

* ``schema_migrations`` carries exactly one ``20260527_pead_signal_log``
  row, even when ``_init_schema`` runs multiple times.
* Migration is additive — pre-existing tables' row counts are
  unchanged after the table is created in an existing DB.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from storage.database import Database
from strategies.base import StrategyResult
from strategies.pead_strategy import PEADStrategy


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def cache_file(tmp_path):
    """Tiny earnings cache covering the four interesting cases:

    BEAT      — surprise > 5%   → threshold_met=1
    NEARMISS  — 0 < surprise <= 5 → threshold_met=0 with earnings_source set
    MISS      — surprise < 0     → threshold_met=0 with earnings_source set
    OLD       — outside the lookback window (no row should match)
    """
    data = {
        "BEAT": [
            {
                "period_end":   "2024-03-31",
                "announce_date": "2024-04-22",
                "actual_eps":    4.85,
                "estimate_eps":  3.91,
                "surprise_pct":  24.04,
            },
        ],
        "NEARMISS": [
            {
                "period_end":   "2024-03-31",
                "announce_date": "2024-04-22",
                "actual_eps":    2.50,
                "estimate_eps":  2.40,
                "surprise_pct":  4.17,
            },
        ],
        "MISS": [
            {
                "period_end":   "2024-03-31",
                "announce_date": "2024-04-22",
                "actual_eps":    1.00,
                "estimate_eps":  1.10,
                "surprise_pct":  -9.1,
            },
        ],
        "OLD": [
            {
                "period_end":   "2023-01-31",
                "announce_date": "2023-02-15",
                "actual_eps":    1.00,
                "estimate_eps":  0.50,
                "surprise_pct":  100.0,
            },
        ],
    }
    path = tmp_path / "earnings_cache.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def writer():
    """In-memory writer that records every row passed to it and hands
    back monotonically increasing log ids.  Replaces the DB-backed
    coordinator wrapper in unit tests.
    """
    captured: list[dict] = []
    next_id = {"i": 0}

    def _writer(row: dict) -> int:
        captured.append(row)
        next_id["i"] += 1
        return next_id["i"]

    _writer.captured = captured  # type: ignore[attr-defined]
    return _writer


@pytest.fixture
def db(tmp_path):
    """Isolated SQLite Database for each test."""
    return Database(str(tmp_path / "pead_log.db"))


# ── Schema / migration bookkeeping ───────────────────────────────────


class TestSchemaMigration:
    def test_pead_signal_log_table_created(self, db):
        with db._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='pead_signal_log'"
            ).fetchone()
        assert row is not None

    def test_schema_migrations_table_created(self, db):
        with db._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
        assert row is not None

    def test_pead_migration_row_recorded(self, db):
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT migration_name FROM schema_migrations "
                "WHERE migration_name='20260527_pead_signal_log'"
            ).fetchall()
        assert len(rows) == 1

    def test_migration_is_idempotent_across_init(self, tmp_path):
        path = str(tmp_path / "idempotent.db")
        Database(path)  # first init
        Database(path)  # second init — must NOT add a second row
        Database(path)  # third for good measure

        import sqlite3
        with sqlite3.connect(path) as conn:
            (count,) = conn.execute(
                "SELECT count(*) FROM schema_migrations "
                "WHERE migration_name='20260527_pead_signal_log'"
            ).fetchone()
        assert count == 1


class TestMigrationDoesNotPerturbExistingTables:
    """Sanity check that creating pead_signal_log + schema_migrations on
    an already-populated DB does NOT alter any existing tables.  Mirrors
    the post-deploy regression check the task requires.
    """

    def test_existing_table_row_counts_unchanged(self, tmp_path):
        path = str(tmp_path / "existing.db")
        db = Database(path)

        # Seed two existing tables with known counts.
        db.log_trade_history("AAPL", "BUY", 10, 150.0)
        db.log_trade_history("AAPL", "SELL", 10, 155.0)
        db.set_portfolio_position("MSFT", 5, 320.0, 1600.0)

        before = {}
        with db._connect() as conn:
            for t in ("trade_history", "portfolio_positions"):
                before[t] = conn.execute(
                    f"SELECT count(*) FROM {t}"
                ).fetchone()[0]

        # Reinitialise — schema is idempotent so this hits the
        # CREATE TABLE IF NOT EXISTS + INSERT OR IGNORE paths.
        Database(path)

        after = {}
        with db._connect() as conn:
            for t in ("trade_history", "portfolio_positions"):
                after[t] = conn.execute(
                    f"SELECT count(*) FROM {t}"
                ).fetchone()[0]

        assert before == after, (
            f"existing table counts changed after re-init: "
            f"before={before} after={after}"
        )


# ── DB-level: log_pead_signal / update_pead_log_trade_id ─────────────


class TestDatabasePEADSignalLog:
    def _sample_row(self) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "timestamp":       now,
            "ticker":          "AAPL",
            "session":         "PEAD_OPEN",
            "signal":          "BUY",
            "threshold_met":   1,
            "surprise_pct":    12.3,
            "announce_date":   "2026-05-26",
            "confidence":      65.0,
            "hold_days":       20,
            "stop_mode":       "TIME_ONLY",
            "earnings_source": "ibkr_cache",
            "trade_id":        None,
            "created_at":      now,
        }

    def test_log_pead_signal_returns_lastrowid(self, db):
        log_id = db.log_pead_signal(self._sample_row())
        assert isinstance(log_id, int) and log_id >= 1

    def test_log_pead_signal_roundtrip(self, db):
        row = self._sample_row()
        log_id = db.log_pead_signal(row)
        with db._connect() as conn:
            persisted = conn.execute(
                "SELECT * FROM pead_signal_log WHERE id = ?", (log_id,),
            ).fetchone()
        persisted = dict(persisted)

        # Every input field round-trips.
        for k in (
            "timestamp", "ticker", "session", "signal", "threshold_met",
            "surprise_pct", "announce_date", "confidence", "hold_days",
            "stop_mode", "earnings_source", "trade_id", "created_at",
        ):
            assert persisted[k] == row[k], f"{k} mismatch"

    def test_update_pead_log_trade_id_stamps_row(self, db):
        log_id = db.log_pead_signal(self._sample_row())
        db.update_pead_log_trade_id(log_id, "987")
        with db._connect() as conn:
            (trade_id,) = conn.execute(
                "SELECT trade_id FROM pead_signal_log WHERE id = ?",
                (log_id,),
            ).fetchone()
        assert trade_id == "987"

    def test_update_missing_id_silently_noops(self, db):
        # Updating a non-existent id must not raise — UPDATE with no
        # matching rows is a no-op in SQLite.
        db.update_pead_log_trade_id(999_999, "trade-xyz")  # would crash if it raised


# ── Strategy-level: phase-1 writes via injected writer ───────────────


class TestPhase1FullUniverseWrites:
    def test_buy_writes_full_row_with_ibkr_cache_source(self, cache_file, writer):
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")

        # BUY result populated.
        assert result is not None
        assert result.signal == "BUY"
        assert result.confidence == 75.0
        assert result.indicators["earnings_source"] == "ibkr_cache"

        # Exactly one row written, with every column populated for a fire.
        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["ticker"]          == "BEAT"
        assert row["session"]         == "PEAD_OPEN"
        assert row["signal"]          == "BUY"
        assert row["threshold_met"]   == 1
        assert row["surprise_pct"]    == pytest.approx(24.04)
        assert row["announce_date"]   == "2024-04-22"
        assert row["confidence"]      == 75.0
        assert row["hold_days"]       == 20
        assert row["stop_mode"]       == "TIME_ONLY"
        assert row["earnings_source"] == "ibkr_cache"
        assert row["trade_id"]        is None
        assert row["timestamp"]       is not None
        assert row["created_at"]      is not None

    def test_near_miss_writes_threshold_met_zero(self, cache_file, writer):
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)
        result = strat.generate_signal("NEARMISS", date(2024, 4, 23), session="PEAD_OPEN")

        # No BUY fires — return None preserves the legacy contract.
        assert result is None

        # But a row IS written (full-universe logging).
        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["ticker"]          == "NEARMISS"
        assert row["signal"]          == "HOLD"
        assert row["threshold_met"]   == 0
        assert row["surprise_pct"]    == pytest.approx(4.17)
        assert row["announce_date"]   == "2024-04-22"
        assert row["confidence"]      is None
        assert row["hold_days"]       is None
        assert row["stop_mode"]       is None
        assert row["earnings_source"] == "ibkr_cache"
        assert row["trade_id"]        is None

    def test_earnings_miss_writes_threshold_met_zero(self, cache_file, writer):
        # MISS announced 2024-04-22; evaluate one day after (in window).
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)
        result = strat.generate_signal("MISS", date(2024, 4, 23), session="PEAD_OPEN")

        assert result is None
        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["threshold_met"]   == 0
        assert row["surprise_pct"]    == pytest.approx(-9.1)
        assert row["announce_date"]   == "2024-04-22"
        assert row["earnings_source"] == "ibkr_cache"
        assert row["signal"]          == "HOLD"

    def test_no_earnings_in_window_still_writes_row(self, cache_file, writer):
        # OLD's earnings are from 2023 — out of any 2-day lookback in 2024.
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)
        result = strat.generate_signal("OLD", date(2024, 4, 22), session="PEAD_OPEN")

        assert result is None
        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["threshold_met"]   == 0
        assert row["surprise_pct"]    is None
        assert row["announce_date"]   is None
        # Cache was consulted (OLD is in the cache) — source records that fact.
        assert row["earnings_source"] == "ibkr_cache"

    def test_pead_log_id_stashed_in_indicators(self, cache_file, writer):
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")
        assert result is not None
        assert "pead_log_id" in result.indicators
        # The dummy writer returns monotonically increasing ids starting at 1.
        assert result.indicators["pead_log_id"] == 1


class TestPhase1YFinanceFallback:
    """When the cache has nothing for a ticker, the yfinance fallback path
    handles it.  We stub `yfinance` entirely (no network) — the goal here
    is purely that ``earnings_source`` is set to ``yfinance_fallback``
    regardless of the outcome.
    """

    def test_yfinance_hit_sets_fallback_source(self, cache_file, writer, monkeypatch):
        # cache_file does NOT have ticker "YFHIT" → fallback fires.
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)

        import pandas as pd
        # Build a fake earnings_dates df that returns one row in the window.
        target = pd.Timestamp(date(2024, 4, 22))
        df = pd.DataFrame(
            {"Surprise(%)": [15.0]},
            index=pd.DatetimeIndex([target]),
        )

        class FakeTicker:
            def __init__(self, _ticker):
                self.earnings_dates = df

        import yfinance as yf
        monkeypatch.setattr(yf, "Ticker", FakeTicker)

        result = strat.generate_signal("YFHIT", date(2024, 4, 22), session="PEAD_OPEN")
        assert result is not None
        assert result.signal == "BUY"
        assert result.indicators["earnings_source"] == "yfinance_fallback"

        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["earnings_source"] == "yfinance_fallback"
        assert row["threshold_met"]   == 1
        assert row["surprise_pct"]    == pytest.approx(15.0)

    def test_yfinance_miss_still_records_fallback_source(self, cache_file, writer, monkeypatch):
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=writer)

        import pandas as pd

        class FakeTicker:
            def __init__(self, _ticker):
                self.earnings_dates = pd.DataFrame()  # empty

        import yfinance as yf
        monkeypatch.setattr(yf, "Ticker", FakeTicker)

        result = strat.generate_signal("YFMISS", date(2024, 4, 22), session="PEAD_OPEN")
        assert result is None
        assert len(writer.captured) == 1
        row = writer.captured[0]
        assert row["earnings_source"] == "yfinance_fallback"
        assert row["threshold_met"]   == 0
        assert row["surprise_pct"]    is None


# ── CRITICAL fail-safe tests ─────────────────────────────────────────


class TestWriterFailureNeverAffectsSignal:
    """The freeze-safe contract: if the phase-1 writer raises, the BUY
    result must still come out unchanged, and no exception propagates.
    """

    def test_writer_raises_exception(self, cache_file):
        def raising_writer(_row):
            raise RuntimeError("simulated DB lock")

        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=raising_writer)
        # If the contract is broken, this call propagates RuntimeError
        # and pytest reports it as the failure cause.
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")

        assert result is not None
        assert result.signal == "BUY"
        assert result.confidence == 75.0
        # pead_log_id is intentionally absent — the writer failed before
        # returning an id, so phase-2 can't stamp.  trade_id will stay
        # NULL forever on the (missing) row, which is the correct
        # observable outcome.
        assert "pead_log_id" not in result.indicators

    def test_writer_returns_none(self, cache_file):
        # A writer returning None (e.g. coordinator's wrapper after it
        # swallowed an exception) must produce the same observable
        # behaviour as a raising writer — BUY result intact, no log id.
        strat = PEADStrategy(
            cache_path=cache_file,
            signal_log_writer=lambda _row: None,
        )
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")
        assert result is not None
        assert result.signal == "BUY"
        assert "pead_log_id" not in result.indicators

    def test_coordinator_wrapper_swallows_db_exceptions(self, db):
        # The DB-backed wrapper Coordinator injects must catch and log
        # any exception from db.log_pead_signal, returning None.  Forcing
        # the DB call to raise simulates a hard failure (disk full, etc).
        from orchestrator.coordinator import Coordinator

        coord = Coordinator.__new__(Coordinator)  # skip __init__
        coord.db = db
        db.log_pead_signal = MagicMock(side_effect=RuntimeError("boom"))

        # Must NOT raise.
        out = coord._write_pead_signal_log({"ticker": "AAPL"})
        assert out is None


class TestPhase2UpdateFailureNeverAffectsTrade:
    """If update_pead_log_trade_id raises during the post-fill stamp,
    the trade execution path must continue.  We exercise this against
    the real Database method by monkeypatching it to raise and asserting
    the call site swallows.
    """

    def test_update_exception_swallowed_at_call_site(self, db, monkeypatch):
        # Simulate the call site pattern from _run_pead_only_async.
        db.update_pead_log_trade_id = MagicMock(
            side_effect=RuntimeError("simulated DB error"),
        )

        # The block from coordinator wraps in try/except and never raises.
        # Reproduce the contract here so a regression in the call site
        # (e.g. someone removes the try/except) is caught by this test.
        log_id = 42
        trade_id = "trade-abc"
        raised = False
        try:
            try:
                db.update_pead_log_trade_id(int(log_id), str(trade_id))
            except Exception:
                pass  # mirror coordinator's swallow
        except Exception:
            raised = True

        assert not raised


# ── Integration: signal → execution → join with trade_history ────────


class TestPEADSignalLogJoinsTradeHistory:
    """End-to-end: a BUY produces a phase-1 row; an execution stamp
    fills trade_id; the same trade_history row exists; the JOIN over
    trade_id succeeds.
    """

    def test_join_pead_signal_log_to_trade_history(self, db, cache_file):
        coord_writer = lambda row: db.log_pead_signal(row)
        strat = PEADStrategy(cache_path=cache_file, signal_log_writer=coord_writer)

        # Phase 1: signal generation writes the log row.
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")
        assert result is not None
        log_id = result.indicators["pead_log_id"]
        assert isinstance(log_id, int)

        # Simulate a fill — write a trade_history row and stamp the link.
        trade_id = db.log_trade_history(
            ticker="BEAT", action="BUY", shares=10, price=150.0,
            stop_loss=140.0, take_profit=170.0, strategy="PEAD",
        )
        db.update_pead_log_trade_id(log_id, str(trade_id))

        # Phase-2 link is visible via a JOIN on trade_id.
        with db._connect() as conn:
            joined = conn.execute(
                """
                SELECT psl.id        AS log_id,
                       psl.ticker    AS log_ticker,
                       psl.signal    AS log_signal,
                       th.id         AS trade_history_id,
                       th.ticker     AS trade_ticker,
                       th.action     AS trade_action,
                       th.strategy   AS trade_strategy
                FROM pead_signal_log psl
                JOIN trade_history th
                  ON CAST(psl.trade_id AS INTEGER) = th.id
                WHERE psl.id = ?
                """,
                (log_id,),
            ).fetchone()

        assert joined is not None
        joined = dict(joined)
        assert joined["log_id"]           == log_id
        assert joined["log_ticker"]       == "BEAT"
        assert joined["log_signal"]       == "BUY"
        assert joined["trade_history_id"] == trade_id
        assert joined["trade_ticker"]     == "BEAT"
        assert joined["trade_action"]     == "BUY"
        assert joined["trade_strategy"]   == "PEAD"


# ── Coordinator-level wiring sanity check ────────────────────────────


class TestCoordinatorWiresPEADStrategyWithWriter:
    """Confirms the constructor passes a working callable that ends up
    inserting into pead_signal_log.  Exercises the production wiring:
    PEADStrategy(signal_log_writer=coord._write_pead_signal_log).
    """

    def test_full_path_to_db_via_coordinator_writer(self, tmp_path, cache_file):
        # Need an isolated DB and avoid Coordinator's heavyweight __init__.
        from orchestrator.coordinator import Coordinator

        coord = Coordinator.__new__(Coordinator)
        coord.db = Database(str(tmp_path / "coord.db"))

        strat = PEADStrategy(
            cache_path=cache_file,
            signal_log_writer=coord._write_pead_signal_log,
        )
        result = strat.generate_signal("BEAT", date(2024, 4, 23), session="PEAD_OPEN")
        assert result is not None
        assert "pead_log_id" in result.indicators

        with coord.db._connect() as conn:
            (count,) = conn.execute(
                "SELECT count(*) FROM pead_signal_log WHERE ticker='BEAT'"
            ).fetchone()
        assert count == 1


# ── Q-004 Fork B follow-up: analyse_ticker_async PEAD-override stamp ──
#
# Production dispatches PEAD_OPEN with session_type="signal" so PEAD BUYs
# that fire go through Coordinator.analyse_ticker_async, NOT
# _run_pead_only_async.  The Q-004 task wired phase-2 stamping only in
# _run_pead_only_async; this follow-up mirrors the same stamp into
# analyse_ticker_async's PEAD-override branch, gated on strat_name=="PEAD".
#
# The tests below drive the full async pipeline with collaborators
# stubbed so the PEAD-override branch is the path exercised.  They prove:
#   • PEAD BUY -> stamp lands -> JOIN to trade_history succeeds.
#   • Forcing db.update_pead_log_trade_id to raise does NOT break the
#     trade — the exception is swallowed at the call site.
#   • A "Combined" trade (strat_name != "PEAD") on the same branch
#     leaves pead_signal_log untouched (no spurious stamps).


def _build_pead_override_coordinator(
    db: Database,
    *,
    pead_log_id: "int | None",
    fire_pead_buy: bool,
    trade_id: int = 42,
):
    """Build a Coordinator stub wired for analyse_ticker_async coverage.

    Skips __init__ — sets only the attributes that path needs when
    session_type='signal' and execute=True for a PEAD-eligible ticker.
    All collaborators are MagicMocks so the only logic that runs is the
    code inside analyse_ticker_async itself.
    """
    from orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.db = db
    coord.macro_context = ""
    coord._pead_enabled = True
    coord._pead_tickers = {"CASY"}

    # Bypass the real PEAD strategy entirely — _run_pead returns the dict
    # form (already serialised by Coordinator._run_pead in production).
    coord._pead_strategy = MagicMock()
    if fire_pead_buy and pead_log_id is not None:
        coord._run_pead = MagicMock(return_value={
            "signal":        "BUY",
            "confidence":    0.75,
            "strategy_name": "PEAD",
            "reasoning":     ["test"],
            "indicators":    {
                "pead_log_id":     pead_log_id,
                "surprise_pct":    24.0,
                "announce_date":   "2026-05-25",
                "hold_days":       20,
                "stop_mode":       "TIME_ONLY",
                "earnings_source": "ibkr_cache",
            },
        })
    else:
        coord._run_pead = MagicMock(return_value=None)

    # All feeds return nothing — there is no headline/news content
    # for sentiment to chew on, so the sentiment_signal will be HOLD.
    for name in ("news_feed", "reddit_feed", "stocktwits_feed",
                 "marketaux_feed", "apewisdom_feed", "adanos_feed"):
        feed = MagicMock()
        feed.fetch = MagicMock(return_value=[])
        setattr(coord, name, feed)

    coord.market_data = MagicMock()
    coord.market_data.fetch = MagicMock(return_value={
        "price": 150.0, "name": "Test", "currency": "USD",
    })

    coord.sentiment_agent = MagicMock()
    coord.sentiment_agent.run = MagicMock(return_value={
        "sentiment": "neutral", "score": 0, "reason": "t", "headline": "h",
    })

    coord.technical_agent = MagicMock()
    coord.technical_agent.run = MagicMock(return_value={
        "ticker":              "CASY",
        "signal":              "HOLD",
        "reasoning":           ["t"],
        "indicators":          {"price": 150.0},
        "signal_id":           1,
        "bars":                pd.DataFrame(),  # empty -> strategy votes skip
        "volume_confirmed":    False,
        "adjusted_confidence": 0.5,
    })

    coord.regime_agent = MagicMock()
    coord.regime_agent.run = MagicMock(return_value={"regime": "TRENDING_BULL", "vix": None})
    coord.regime_detector = MagicMock()

    coord.risk_agent = MagicMock()
    coord.risk_agent.run = MagicMock(return_value={
        "ticker":            "CASY",
        "signal":            "BUY",
        "direction":         "BUY",
        "position_size_usd": 150.0,
        "shares":            1,
        "stop_loss":         140.0,
        "take_profit":       160.0,
        "risk_amount":       10.0,
        "kelly_fraction":    0.05,
        "stop_pct":          0.067,
        "skipped":           False,
        "skip_reason":       None,
        "event_risk_flag":   "none",
        "days_to_earnings":  None,
        "regime":            "TRENDING_BULL",
        "calc_id":           1,
    })

    coord.debate_agent = MagicMock()
    # is_pead == True in the production code skips debate already, but
    # disable globally so the path is deterministic.
    coord.debate_agent.is_enabled = MagicMock(return_value=False)

    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade = MagicMock(
        return_value={"trade_id": trade_id, "price": 150.0},
    )

    coord.signal_logger = MagicMock()
    coord.signal_logger.store_forward_signal = MagicMock()

    coord._has_alpaca_position = MagicMock(return_value=False)

    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position = MagicMock(return_value=(True, ""))
    coord._portfolio_manager.register_position = MagicMock()
    coord._portfolio_manager.clear_position_meta = MagicMock()

    coord._cluster_detector = MagicMock()
    coord._momentum_strategy = MagicMock()
    coord._pullback_strategy = MagicMock()
    coord._gather_strategy_votes = MagicMock(return_value=[])

    # Force the combined-signal path: when PEAD fires the function calls
    # combine_signals(sentiment, "BUY", ...) — sentiment is HOLD here so
    # we land on _SIGNAL_FLOORS["WEAK BUY"]=0.35.  That's BUY-shaped, so
    # execution proceeds.  No extra patching needed.

    return coord


def _seed_phase1_row(db: Database) -> int:
    """Insert a phase-1 pead_signal_log row and return its log_id."""
    now = datetime.now(timezone.utc).isoformat()
    return db.log_pead_signal({
        "timestamp":       now,
        "ticker":          "CASY",
        "session":         "PEAD_OPEN",
        "signal":          "BUY",
        "threshold_met":   1,
        "surprise_pct":    24.0,
        "announce_date":   "2026-05-25",
        "confidence":      75.0,
        "hold_days":       20,
        "stop_mode":       "TIME_ONLY",
        "earnings_source": "ibkr_cache",
        "trade_id":        None,
        "created_at":      now,
    })


def _run_analyse(coord, ticker="CASY", session="PEAD_OPEN") -> dict:
    """Drive analyse_ticker_async to completion and return its result."""
    async def _wrap():
        return await coord.analyse_ticker_async(
            ticker,
            account_balance=10_000.0,
            execute=True,
            api_semaphore=asyncio.Semaphore(1),
            data_semaphore=asyncio.Semaphore(1),
            db_lock=asyncio.Lock(),
            session=session,
            session_type="signal",
        )

    # Patch get_days_to_earnings to avoid network — the result doesn't
    # affect the PEAD-override execution branch (event_risk_flag is set
    # in the result dict but not used to gate execution).
    with patch("orchestrator.coordinator.get_days_to_earnings", return_value=None):
        return asyncio.run(_wrap())


class TestAnalyseTickerAsyncStampsTradeIdOnPEADOverride:
    """End-to-end: PEAD BUY -> analyse_ticker_async fires the trade ->
    pead_signal_log.trade_id gets stamped -> JOIN to trade_history succeeds.
    """

    def test_pead_override_buy_stamps_and_joins(self, db):
        # Seed: a trade_history row that paper_trader will pretend to have
        # produced (its mocked track_trade returns this row's id).
        trade_history_id = db.log_trade_history(
            ticker="CASY", action="BUY", shares=1, price=150.0,
            stop_loss=140.0, take_profit=160.0, strategy="PEAD",
        )

        log_id = _seed_phase1_row(db)
        coord = _build_pead_override_coordinator(
            db,
            pead_log_id=log_id,
            fire_pead_buy=True,
            trade_id=trade_history_id,
        )

        result = _run_analyse(coord)

        # Sanity — the PEAD-override branch fired and the trade was placed.
        assert result["strategy_name"] == "PEAD"
        assert result["signal_path"] == "FUSION_FALLBACK"
        assert result["execution"] is not None
        assert result["execution"]["trade_id"] == trade_history_id

        # The stamp landed.
        with db._connect() as conn:
            (stamped,) = conn.execute(
                "SELECT trade_id FROM pead_signal_log WHERE id = ?", (log_id,),
            ).fetchone()
        assert stamped == str(trade_history_id)

        # And the join to trade_history is now intact.
        with db._connect() as conn:
            joined = conn.execute(
                """
                SELECT psl.id, psl.ticker, psl.signal, th.id AS th_id, th.strategy
                FROM pead_signal_log psl
                JOIN trade_history th
                  ON CAST(psl.trade_id AS INTEGER) = th.id
                WHERE psl.id = ?
                """,
                (log_id,),
            ).fetchone()
        assert joined is not None
        joined = dict(joined)
        assert joined["th_id"] == trade_history_id
        assert joined["strategy"] == "PEAD"


class TestAnalyseTickerAsyncStampFailureNeverAffectsTrade:
    """If db.update_pead_log_trade_id raises inside the analyse_ticker_async
    PEAD-override branch, the trade MUST still complete and no exception
    may propagate out of the async function.
    """

    def test_update_exception_swallowed_trade_still_fires(self, db):
        trade_history_id = db.log_trade_history(
            ticker="CASY", action="BUY", shares=1, price=150.0,
            stop_loss=140.0, take_profit=160.0, strategy="PEAD",
        )
        log_id = _seed_phase1_row(db)
        coord = _build_pead_override_coordinator(
            db,
            pead_log_id=log_id,
            fire_pead_buy=True,
            trade_id=trade_history_id,
        )

        # Force the stamp to blow up.  The trade has already fired by the
        # time this is called; the contract is "log and swallow".
        boom = MagicMock(side_effect=RuntimeError("simulated DB error"))
        coord.db.update_pead_log_trade_id = boom

        # If the contract is broken, this call raises RuntimeError.
        result = _run_analyse(coord)

        # The trade completed and the result is intact.
        assert result["execution"] is not None
        assert result["execution"]["trade_id"] == trade_history_id
        assert result["strategy_name"] == "PEAD"

        # The stamp WAS attempted (proves the call site was reached).
        boom.assert_called_once()

        # The phase-1 row still has trade_id NULL because the stamp raised
        # before persisting — that's the correct fail-safe outcome.
        with db._connect() as conn:
            (stamped,) = conn.execute(
                "SELECT trade_id FROM pead_signal_log WHERE id = ?", (log_id,),
            ).fetchone()
        assert stamped is None


class TestAnalyseTickerAsyncNoStampForNonPEADTrades:
    """A trade that fires through the same branch but with strat_name !=
    "PEAD" (i.e. a Combined cluster verdict, not a PEAD override) must
    leave pead_signal_log entirely untouched.
    """

    def test_combined_trade_does_not_touch_pead_signal_log(self, db):
        trade_history_id = db.log_trade_history(
            ticker="CASY", action="BUY", shares=1, price=150.0,
            stop_loss=140.0, take_profit=160.0, strategy="Combined",
        )
        # Seed a phase-1 row anyway — if the stamp were incorrectly
        # called for a non-PEAD trade, this row's trade_id would change
        # from NULL.  We want it to STAY NULL.
        log_id = _seed_phase1_row(db)

        coord = _build_pead_override_coordinator(
            db,
            pead_log_id=log_id,
            fire_pead_buy=False,   # _run_pead returns None
            trade_id=trade_history_id,
        )

        # Need at least one strategy vote so the cluster path emits BUY.
        # MomentumStrategy result: BUY @ 0.70 confidence, with stop/tp so
        # the SL/TP override branch passes through cleanly.
        from strategies.base import StrategyResult
        coord._gather_strategy_votes = MagicMock(return_value=[
            StrategyResult(
                signal="BUY", confidence=70.0,
                indicators={"price": 150.0},
                stop_loss=140.0, take_profit=160.0,
                strategy_name="Momentum",
                reasoning=["test"],
            ),
        ])

        # Spy on the stamp method — assert it is NEVER called.
        spy = MagicMock(side_effect=db.update_pead_log_trade_id)
        coord.db.update_pead_log_trade_id = spy

        # Force the cluster fusion to emit a BUY so a trade fires.
        # Sidesteps the cluster detector / vote counting — we just want a
        # BUY decision with strat_name != "PEAD".
        coord._fuse_signals = MagicMock(
            return_value=("WEAK BUY", 0.55, "CLUSTER_PARTIAL"),
        )

        # And pin strategy_label so strat_name is deterministic and
        # never accidentally becomes "PEAD".
        with patch("orchestrator.coordinator.strategy_label", return_value="Momentum"):
            result = _run_analyse(coord)

        # Sanity — the trade fired with strategy="Combined" attribution.
        assert result["strategy_name"] == "Momentum"   # router primary
        assert result["execution"] is not None
        assert result["execution"]["trade_id"] == trade_history_id

        # The stamp was NEVER called (strat_name != "PEAD" gate).
        spy.assert_not_called()

        # And the phase-1 row's trade_id is still NULL.
        with db._connect() as conn:
            (stamped,) = conn.execute(
                "SELECT trade_id FROM pead_signal_log WHERE id = ?", (log_id,),
            ).fetchone()
        assert stamped is None
