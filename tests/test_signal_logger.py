"""
Tests for the signal analytics layer.

Covers:
- SignalLogger.log() never raises, even on a broken DB
- All fields are stored correctly
- get_signals() filtering works (by ticker, by days)
- Outcome tracker backfill logic
- Report generation doesn't crash on empty or populated data
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from analytics.signal_logger import SignalLogger
from analytics.outcome_tracker import run_outcome_tracker, _is_directional
from analytics.report import generate_report
from storage.database import Database


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def signal_db(tmp_path):
    """Fresh DB with signal_events table."""
    db = Database(str(tmp_path / "signal_test.db"))
    return db


@pytest.fixture
def logger(signal_db):
    """SignalLogger backed by a temp DB."""
    return SignalLogger(db=signal_db)


def _make_signal(
    ticker: str = "AAPL",
    signal: str = "STRONG BUY",
    confidence: float = 0.72,
    **overrides,
) -> dict:
    """Build a minimal signal_data dict for logging."""
    data = {
        "ticker": ticker,
        "session": "US_OPEN",
        "strategy": "Combined",
        "signal": signal,
        "confidence": confidence,
        "rsi": 42.5,
        "sma_ratio": 1.02,
        "volume_ratio": 1.4,
        "sentiment_score": 0.65,
        "news_score": 0.7,
        "social_score": 0.6,
        "bull_case": "Strong momentum and earnings beat",
        "bear_case": "Valuation stretched at 30x P/E",
        "debate_outcome": "agree",
        "price_at_signal": 189.42,
        "trade_executed": 0,
        "trade_id": None,
    }
    data.update(overrides)
    return data


# ── SignalLogger.log() tests ─────────────────────────────────────────────

class TestSignalLoggerLog:
    """log() is fire-and-forget — must never raise."""

    def test_log_stores_all_fields(self, logger, signal_db):
        data = _make_signal()
        logger.log(data)

        rows = logger.get_signals("AAPL", days=1)
        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "AAPL"
        assert row["signal"] == "STRONG BUY"
        assert row["confidence"] == pytest.approx(0.72)
        assert row["rsi"] == pytest.approx(42.5)
        assert row["sma_ratio"] == pytest.approx(1.02)
        assert row["volume_ratio"] == pytest.approx(1.4)
        assert row["sentiment_score"] == pytest.approx(0.65)
        assert row["news_score"] == pytest.approx(0.7)
        assert row["social_score"] == pytest.approx(0.6)
        assert row["bull_case"] == "Strong momentum and earnings beat"
        assert row["bear_case"] == "Valuation stretched at 30x P/E"
        assert row["debate_outcome"] == "agree"
        assert row["price_at_signal"] == pytest.approx(189.42)
        assert row["trade_executed"] == 0
        assert row["trade_id"] is None

    def test_log_stores_trade_info(self, logger):
        data = _make_signal(trade_executed=1, trade_id="paper_12345")
        logger.log(data)

        rows = logger.get_signals("AAPL", days=1)
        assert rows[0]["trade_executed"] == 1
        assert rows[0]["trade_id"] == "paper_12345"

    def test_log_never_raises_on_broken_db(self, tmp_path):
        """Even if the DB is completely broken, log() must not raise."""
        db = Database(str(tmp_path / "broken.db"))
        lgr = SignalLogger(db=db)

        # Break the DB connection
        with patch.object(db, "_connect", side_effect=sqlite3.OperationalError("disk I/O error")):
            # This must NOT raise
            lgr.log(_make_signal())

    def test_log_never_raises_on_missing_fields(self, logger):
        """Partial data should not crash the logger."""
        logger.log({"ticker": "TSLA", "signal": "HOLD"})
        rows = logger.get_signals("TSLA", days=1)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "TSLA"
        assert rows[0]["confidence"] is None

    def test_log_never_raises_on_empty_dict(self, logger):
        """Even an empty dict must not crash."""
        logger.log({})

    def test_outcome_columns_initially_null(self, logger):
        logger.log(_make_signal())
        row = logger.get_signals("AAPL", days=1)[0]
        assert row["price_3d"] is None
        assert row["price_5d"] is None
        assert row["price_10d"] is None
        assert row["outcome_3d_pct"] is None
        assert row["outcome_5d_pct"] is None
        assert row["outcome_10d_pct"] is None
        assert row["outcome_correct"] is None


# ── SignalLogger.get_signals() tests ─────────────────────────────────────

class TestGetSignals:
    def test_filter_by_ticker(self, logger):
        logger.log(_make_signal(ticker="AAPL"))
        logger.log(_make_signal(ticker="MSFT"))
        logger.log(_make_signal(ticker="AAPL"))

        aapl = logger.get_signals("AAPL", days=1)
        assert len(aapl) == 2
        assert all(r["ticker"] == "AAPL" for r in aapl)

    def test_filter_by_days(self, logger, signal_db):
        # Insert one recent and one old signal
        logger.log(_make_signal(ticker="AAPL"))

        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        with signal_db._connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_events
                    (timestamp, ticker, signal, trade_executed)
                VALUES (?, 'AAPL', 'HOLD', 0)
                """,
                (old_ts,),
            )

        recent = logger.get_signals("AAPL", days=30)
        assert len(recent) == 1

        all_signals = logger.get_signals("AAPL", days=90)
        assert len(all_signals) == 2

    def test_get_all_tickers(self, logger):
        logger.log(_make_signal(ticker="AAPL"))
        logger.log(_make_signal(ticker="MSFT"))

        all_signals = logger.get_signals(ticker=None, days=1)
        assert len(all_signals) == 2

    def test_get_signals_never_raises_on_broken_db(self, tmp_path):
        db = Database(str(tmp_path / "broken2.db"))
        lgr = SignalLogger(db=db)

        with patch.object(db, "_connect", side_effect=sqlite3.OperationalError("disk I/O error")):
            result = lgr.get_signals()
            assert result == []


# ── Outcome tracker tests ────────────────────────────────────────────────

class TestOutcomeTracker:
    def test_is_directional(self):
        assert _is_directional("STRONG BUY") == 1
        assert _is_directional("WEAK BUY") == 1
        assert _is_directional("STRONG SELL") == -1
        assert _is_directional("WEAK SELL") == -1
        assert _is_directional("HOLD") == 0
        assert _is_directional("CONFLICTING") == 0
        assert _is_directional(None) == 0

    def test_backfill_skips_recent_signals(self, signal_db):
        """Signals less than 3 days old should not be backfilled."""
        lgr = SignalLogger(db=signal_db)
        lgr.log(_make_signal())

        result = run_outcome_tracker(db=signal_db)
        # All counts should be 0 — signal is too recent
        assert sum(result.values()) == 0

    def test_backfill_updates_old_signals(self, signal_db):
        """Signals >3 days old should be backfilled when price is available."""
        lgr = SignalLogger(db=signal_db)

        # Insert a signal from 5 days ago
        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        lgr.log(_make_signal(timestamp=old_ts, ticker="AAPL"))

        # Mock the price fetch to return a higher price (BUY was correct)
        with patch("analytics.outcome_tracker._fetch_price", return_value=200.0):
            result = run_outcome_tracker(db=signal_db)

        # price_3d should have been updated (signal is >3 days old)
        assert result.get("price_3d", 0) == 1

        rows = lgr.get_signals("AAPL", days=30)
        assert len(rows) == 1
        row = rows[0]
        assert row["price_3d"] == pytest.approx(200.0)
        assert row["outcome_3d_pct"] is not None
        # BUY signal + price went up → correct
        assert row["outcome_correct"] == 1


# ── Report tests ─────────────────────────────────────────────────────────

class TestReport:
    def test_report_empty_data(self, signal_db):
        """Report should not crash with no data."""
        with patch("analytics.report.SignalLogger", return_value=SignalLogger(db=signal_db)):
            output = generate_report(days=30)
            assert "SIGNAL QUALITY REPORT" in output
            assert "No signals found" in output

    def test_report_with_data(self, signal_db):
        """Report should include ticker and strategy tables."""
        lgr = SignalLogger(db=signal_db)
        for ticker in ["AAPL", "AAPL", "MSFT"]:
            lgr.log(_make_signal(ticker=ticker))

        with patch("analytics.report.SignalLogger", return_value=lgr):
            output = generate_report(days=30)
            assert "AAPL" in output
            assert "MSFT" in output
            assert "SIGNALS" in output
            assert "Combined" in output

    def test_report_ticker_filter(self, signal_db):
        lgr = SignalLogger(db=signal_db)
        lgr.log(_make_signal(ticker="AAPL"))
        lgr.log(_make_signal(ticker="MSFT"))

        with patch("analytics.report.SignalLogger", return_value=lgr):
            output = generate_report(days=30, ticker="AAPL")
            assert "AAPL" in output
            assert "Filter:" in output
