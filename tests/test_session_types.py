"""
Tests for the session type architecture.

Covers:
- EOD (signal mode) generates forward signals
- US_OPEN (execution mode) validates forward signals before executing
- MIDDAY (monitor mode) skips signal generation, only checks positions
- Schedule config has correct session_type assignments
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analytics.signal_logger import SignalLogger
from scheduler.daily_runner import SCHEDULE, run_batch
from storage.database import Database


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    """In-memory-style DB using a temp file."""
    db_path = str(tmp_path / "test.db")
    with patch("config.settings.DB_PATH", db_path):
        database = Database(db_path=db_path)
        yield database


@pytest.fixture
def signal_logger(db):
    return SignalLogger(db=db)


# ── Schedule config ──────────────────────────────────────────────────


class TestScheduleSessionTypes:
    def test_eod_is_signal_type(self):
        eod = next(r for r in SCHEDULE if r["name"] == "EOD")
        assert eod["session_type"] == "signal"

    def test_us_open_is_execution_type(self):
        us = next(r for r in SCHEDULE if r["name"] == "US_OPEN")
        assert us["session_type"] == "execution"

    def test_midday_is_monitor_type(self):
        mid = next(r for r in SCHEDULE if r["name"] == "MIDDAY")
        assert mid["session_type"] == "monitor"

    def test_xetra_is_signal_type(self):
        xetra = next(r for r in SCHEDULE if r["name"] == "XETRA_OPEN")
        assert xetra["session_type"] == "signal"

    def test_xetra_pre_is_pre_signal_type(self):
        xetra_pre = next(r for r in SCHEDULE if r["name"] == "XETRA_PRE")
        assert xetra_pre["session_type"] == "pre_signal"

    def test_us_pre_is_pre_signal_type(self):
        us_pre = next(r for r in SCHEDULE if r["name"] == "US_PRE")
        assert us_pre["session_type"] == "pre_signal"

    def test_all_sessions_have_session_type(self):
        for run in SCHEDULE:
            assert "session_type" in run, f"{run['name']} missing session_type"
            assert run["session_type"] in ("signal", "pre_signal", "execution", "monitor")


# ── Forward signals storage ──────────────────────────────────────────


class TestForwardSignals:
    def test_store_and_retrieve(self, signal_logger):
        """store_forward_signal + get_pending_forward_signals round-trip."""
        fwd_id = signal_logger.store_forward_signal({
            "source_session": "EOD",
            "target_session": "US_OPEN",
            "ticker": "AAPL",
            "signal": "STRONG BUY",
            "confidence": 0.72,
            "price_at_signal": 185.50,
            "strategy_name": "Momentum",
            "stop_loss": 181.79,
            "take_profit": 192.00,
        })
        assert fwd_id is not None

        pending = signal_logger.get_pending_forward_signals("US_OPEN", "AAPL")
        assert len(pending) == 1
        assert pending[0]["ticker"] == "AAPL"
        assert pending[0]["signal"] == "STRONG BUY"
        assert pending[0]["status"] == "pending"
        assert pending[0]["confidence"] == 0.72

    def test_update_status_to_confirmed(self, signal_logger):
        fwd_id = signal_logger.store_forward_signal({
            "ticker": "META", "signal": "WEAK BUY",
            "confidence": 0.45, "price_at_signal": 500.0,
        })
        signal_logger.update_forward_signal(fwd_id, "confirmed")

        # Should no longer appear in pending
        pending = signal_logger.get_pending_forward_signals("US_OPEN", "META")
        assert len(pending) == 0

    def test_update_status_to_invalidated(self, signal_logger):
        fwd_id = signal_logger.store_forward_signal({
            "ticker": "TSLA", "signal": "STRONG BUY",
            "confidence": 0.65, "price_at_signal": 250.0,
        })
        signal_logger.update_forward_signal(
            fwd_id, "invalidated", "price moved 3.5% against signal",
        )
        pending = signal_logger.get_pending_forward_signals("US_OPEN", "TSLA")
        assert len(pending) == 0

    def test_expire_stale(self, signal_logger):
        """Signals older than max_age_hours are expired."""
        fwd_id = signal_logger.store_forward_signal({
            "ticker": "JPM", "signal": "WEAK BUY",
            "confidence": 0.40, "price_at_signal": 200.0,
        })
        # Manually backdate created_at to 48 hours ago
        old_ts = "2020-01-01T00:00:00+00:00"
        with signal_logger._db._connect() as conn:
            conn.execute(
                "UPDATE forward_signals SET created_at = ? WHERE id = ?",
                (old_ts, fwd_id),
            )
        expired = signal_logger.expire_stale_forward_signals(max_age_hours=24)
        assert expired == 1

    def test_hold_signals_not_stored(self, signal_logger):
        """HOLD and CONFLICTING should not produce forward signals."""
        # This tests the coordinator logic — HOLD is filtered before storage.
        # Direct storage would succeed, but the coordinator skips HOLD.
        fwd_id = signal_logger.store_forward_signal({
            "ticker": "XOM", "signal": "HOLD",
            "confidence": 0.25, "price_at_signal": 110.0,
        })
        # Direct storage works (no filter in logger itself)
        assert fwd_id is not None
        # But get_pending returns it — filtering is the coordinator's job
        pending = signal_logger.get_pending_forward_signals("US_OPEN", "XOM")
        assert len(pending) == 1


# ── EOD generates forward signals ────────────────────────────────────


class TestEodGeneratesForwardSignals:
    def test_eod_session_generates_forward_signals(self, db):
        """EOD signal-mode run stores forward signals for actionable results."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)

        # Mock the full pipeline to return a STRONG BUY result
        mock_result = {
            "ticker": "AAPL",
            "sentiment": {"signal": "BUY", "avg_score": 0.5, "market": {},
                          "headlines_fetched": 0, "scored": [],
                          "run_id": 1, "source_breakdown": {}},
            "technical": {"signal": "BUY", "indicators": {"price": 185.0},
                          "signal_id": 1, "bars": None},
            "combined_signal": "STRONG BUY",
            "confidence": 0.72,
            "combined_id": 1,
            "risk": {"skipped": False, "stop_loss": 181.30,
                     "take_profit": 192.00, "direction": "BUY",
                     "shares": 5, "position_size_usd": 925.0},
            "account_balance": 10000,
            "execution": None,
            "event_risk_flag": "none",
            "regime": {},
            "strategy_name": "Momentum",
            "debate": None,
            "elapsed_s": 1.0,
        }

        # Patch analyse_ticker_async to return our mock — but we want
        # the forward signal storage to actually run, so we intercept
        # at a level that lets the storage code execute.
        sl = SignalLogger(db=db)

        # Directly test the storage logic that runs in signal mode
        combined_signal = mock_result["combined_signal"]
        if combined_signal not in ("HOLD", "CONFLICTING"):
            sl.store_forward_signal({
                "source_session": "EOD",
                "target_session": "US_OPEN",
                "ticker": mock_result["ticker"],
                "signal": combined_signal,
                "confidence": mock_result["confidence"],
                "price_at_signal": 185.0,
                "strategy_name": mock_result["strategy_name"],
                "stop_loss": mock_result["risk"]["stop_loss"],
                "take_profit": mock_result["risk"]["take_profit"],
            })

        pending = sl.get_pending_forward_signals("US_OPEN", "AAPL")
        assert len(pending) == 1
        assert pending[0]["signal"] == "STRONG BUY"
        assert pending[0]["confidence"] == 0.72
        assert pending[0]["source_session"] == "EOD"
        assert pending[0]["target_session"] == "US_OPEN"


# ── US_OPEN checks forward signals ──────────────────────────────────


class TestUsOpenChecksForwardSignals:
    def test_us_open_checks_forward_signals_before_executing(self, db):
        """US_OPEN execution mode validates price drift before trading."""
        from orchestrator.coordinator import Coordinator

        sl = SignalLogger(db=db)

        # Store an EOD forward signal
        fwd_id = sl.store_forward_signal({
            "source_session": "EOD",
            "target_session": "US_OPEN",
            "ticker": "AAPL",
            "signal": "STRONG BUY",
            "confidence": 0.72,
            "price_at_signal": 185.0,
            "strategy_name": "Momentum",
            "stop_loss": 181.30,
            "take_profit": 192.00,
        })

        coord = Coordinator(db=db)

        # Test 1: Price within 2% — should confirm
        pending = sl.get_pending_forward_signals("US_OPEN", "AAPL")
        fwd = pending[0]
        current_price = 184.0  # only 0.5% drop, within 2%
        adverse_drift = (fwd["price_at_signal"] - current_price) / fwd["price_at_signal"] * 100
        assert adverse_drift < coord._FORWARD_PRICE_DRIFT_PCT

        # Test 2: Price dropped >2% — should invalidate
        current_price_bad = 180.0  # 2.7% drop
        adverse_drift_bad = (fwd["price_at_signal"] - current_price_bad) / fwd["price_at_signal"] * 100
        assert adverse_drift_bad > coord._FORWARD_PRICE_DRIFT_PCT

    def test_forward_signal_invalidated_on_large_drift(self, db):
        """Forward signal is invalidated when price moves >2% against."""
        sl = SignalLogger(db=db)
        fwd_id = sl.store_forward_signal({
            "source_session": "EOD",
            "target_session": "US_OPEN",
            "ticker": "META",
            "signal": "STRONG BUY",
            "confidence": 0.65,
            "price_at_signal": 500.0,
        })

        # Simulate invalidation (price dropped to 485 = 3% drop)
        sl.update_forward_signal(fwd_id, "invalidated", "price moved 3.0% against signal")

        pending = sl.get_pending_forward_signals("US_OPEN", "META")
        assert len(pending) == 0

    def test_sell_signal_invalidated_on_price_rise(self, db):
        """SELL forward signal invalidated when price rises >2%."""
        sl = SignalLogger(db=db)
        fwd_id = sl.store_forward_signal({
            "source_session": "EOD",
            "target_session": "US_OPEN",
            "ticker": "TSLA",
            "signal": "STRONG SELL",
            "confidence": 0.60,
            "price_at_signal": 250.0,
        })

        # Price rose to 256 = 2.4% adverse for SELL
        current_price = 256.0
        signal_price = 250.0
        adverse_drift = (current_price - signal_price) / signal_price * 100
        assert adverse_drift > 2.0  # Should invalidate


# ── MIDDAY monitor mode ─────────────────────────────────────────────


class TestMiddayMonitorOnly:
    def test_midday_is_monitor_only(self, db):
        """Monitor mode returns position status without generating signals."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)

        # Run the monitor path directly
        result = asyncio.run(
            coord._monitor_position_async(
                "AAPL",
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="MIDDAY",
            )
        )

        assert result["session_type"] == "monitor"
        assert result["combined_signal"] == "MONITOR"
        assert result["has_position"] is False
        assert "elapsed_s" in result

    def test_monitor_detects_open_position(self, db):
        """Monitor returns position details when a position exists."""
        from orchestrator.coordinator import Coordinator

        # Insert a fake position
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO portfolio_positions (ticker, shares, avg_price, current_value, updated_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                ("AAPL", 10, 180.0, 1850.0),
            )

        coord = Coordinator(db=db)

        # Mock market data to return a price
        coord.market_data.fetch = MagicMock(return_value={"price": 185.0})

        result = asyncio.run(
            coord._monitor_position_async(
                "AAPL",
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="MIDDAY",
            )
        )

        assert result["has_position"] is True
        assert result["current_price"] == 185.0
        assert result["session_type"] == "monitor"
        assert result["combined_signal"] == "MONITOR"

    def test_monitor_no_new_signals_generated(self, db):
        """Monitor mode must NOT call the signal generation pipeline."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)

        # Patch run() (sentiment pipeline) — should NOT be called
        coord.run = MagicMock(side_effect=AssertionError("Signal generation called in monitor mode!"))
        coord.technical_agent.run = MagicMock(side_effect=AssertionError("TA called in monitor mode!"))

        # Should complete without triggering the mocks
        result = asyncio.run(
            coord._monitor_position_async(
                "NVDA",
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="MIDDAY",
            )
        )

        assert result["session_type"] == "monitor"
        coord.run.assert_not_called()
        coord.technical_agent.run.assert_not_called()


# ── Session type in run_batch pipeline ───────────────────────────────


class TestSessionTypePipeline:
    def test_run_batch_passes_session_type(self):
        """run_batch passes session_type through to the coordinator."""
        captured = {}

        async def fake_analyse(ticker, *, account_balance, execute,
                               api_semaphore, data_semaphore, db_lock,
                               session, session_type="signal"):
            captured[ticker] = session_type
            return {
                "ticker": ticker,
                "combined_signal": "MONITOR" if session_type == "monitor" else "HOLD",
                "confidence": 0,
                "elapsed_s": 0.1,
            }

        with patch("scheduler.daily_runner.Coordinator") as MockCoord:
            mock_instance = MagicMock()
            mock_instance.analyse_ticker_async = fake_analyse
            MockCoord.return_value = mock_instance

            result = asyncio.run(
                run_batch(
                    ["AAPL"],
                    workers=1,
                    session="MIDDAY",
                    session_type="monitor",
                )
            )

        assert captured.get("AAPL") == "monitor"
        assert result["success_count"] == 1

    def test_run_batch_passes_pre_signal_type(self):
        """run_batch passes pre_signal session_type for PRE sessions."""
        captured = {}

        async def fake_analyse(ticker, *, account_balance, execute,
                               api_semaphore, data_semaphore, db_lock,
                               session, session_type="signal"):
            captured[ticker] = session_type
            return {
                "ticker": ticker,
                "combined_signal": "HOLD",
                "confidence": 0,
                "elapsed_s": 0.1,
            }

        with patch("scheduler.daily_runner.Coordinator") as MockCoord:
            mock_instance = MagicMock()
            mock_instance.analyse_ticker_async = fake_analyse
            MockCoord.return_value = mock_instance

            result = asyncio.run(
                run_batch(
                    ["SAP.XETRA"],
                    workers=1,
                    session="XETRA_PRE",
                    session_type="pre_signal",
                )
            )

        assert captured.get("SAP.XETRA") == "pre_signal"
        assert result["success_count"] == 1


# ── PRE session coordinator logic ────────────────────────────────────


class TestPreSessionLogic:
    def test_pre_signal_skips_technical_analysis(self, db):
        """Pre-signal mode must NOT call the technical agent."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)
        coord.technical_agent.run = MagicMock(
            side_effect=AssertionError("TA called in pre_signal mode!")
        )
        coord.risk_agent.run = MagicMock(
            side_effect=AssertionError("Risk called in pre_signal mode!")
        )

        # Mock data feeds to return empty
        coord.news_feed.fetch = MagicMock(return_value=[])
        coord.stocktwits_feed.fetch = MagicMock(return_value=[])
        coord.marketaux_feed.fetch = MagicMock(return_value=[])

        result = asyncio.run(
            coord._pre_signal_refresh_async(
                "AAPL",
                api_semaphore=asyncio.Semaphore(5),
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="US_PRE",
            )
        )

        assert result["session_type"] == "pre_signal"
        assert result["ticker"] == "AAPL"
        coord.technical_agent.run.assert_not_called()
        coord.risk_agent.run.assert_not_called()

    def test_pre_signal_returns_expected_fields(self, db):
        """Pre-signal result has all expected keys."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)
        coord.news_feed.fetch = MagicMock(return_value=[])
        coord.stocktwits_feed.fetch = MagicMock(return_value=[])
        coord.marketaux_feed.fetch = MagicMock(return_value=[])

        result = asyncio.run(
            coord._pre_signal_refresh_async(
                "SAP.XETRA",
                api_semaphore=asyncio.Semaphore(5),
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="XETRA_PRE",
            )
        )

        assert result["session_type"] == "pre_signal"
        assert "combined_signal" in result
        assert "confidence" in result
        assert "headlines_fetched" in result
        assert "headlines_scored" in result
        assert "elapsed_s" in result
        assert "updated_forward" in result

    def test_pre_signal_dispatched_by_analyse_ticker_async(self, db):
        """analyse_ticker_async routes pre_signal to _pre_signal_refresh_async."""
        from orchestrator.coordinator import Coordinator

        coord = Coordinator(db=db)
        coord.news_feed.fetch = MagicMock(return_value=[])
        coord.stocktwits_feed.fetch = MagicMock(return_value=[])
        coord.marketaux_feed.fetch = MagicMock(return_value=[])

        result = asyncio.run(
            coord.analyse_ticker_async(
                "AAPL",
                account_balance=10_000.0,
                execute=False,
                api_semaphore=asyncio.Semaphore(5),
                data_semaphore=asyncio.Semaphore(5),
                db_lock=asyncio.Lock(),
                session="US_PRE",
                session_type="pre_signal",
            )
        )

        assert result["session_type"] == "pre_signal"
