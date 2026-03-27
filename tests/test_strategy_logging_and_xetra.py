"""Tests for strategy result logging and XETRA bar fetch.

BUG 1: Every strategy result (including HOLD) must be logged to signal_events.
BUG 2: XETRA tickers must fetch bars from EODHD, not Alpaca.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.signal_logger import SignalLogger
from data.eodhd_feed import EODHDFeed
from strategies.base import StrategyResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Build synthetic daily OHLCV with title-case columns (Alpaca format)."""
    rng = np.random.default_rng(42)
    prices = base + rng.normal(0, 0.5, n).cumsum()
    dates = pd.bdate_range(end="2026-03-25", periods=n)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.full(n, 5_000_000, dtype=float),
    }, index=dates)


# ── BUG 1: Strategy result logging ──────────────────────────────────────────

class TestStrategyResultLogging:
    """Every strategy result should be logged to signal_events, including HOLD."""

    def test_hold_result_is_logged(self):
        """A HOLD strategy result (confidence 25%) must still be logged."""
        from orchestrator.coordinator import Coordinator

        # Build a HOLD strategy result
        hold_result = StrategyResult(
            signal="HOLD",
            confidence=25.0,
            strategy_name="NewsCatalyst",
            indicators={"price": 150.0, "rsi": 45.0},
            reasoning=["No news data available"],
        )

        # Create coordinator with mocked dependencies
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)

        # Call the method
        coordinator._log_strategy_result("AAPL", hold_result)

        # Verify the logger was called
        coordinator.signal_logger.log.assert_called_once()
        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["ticker"] == "AAPL"
        assert logged["strategy"] == "NewsCatalyst"
        assert logged["signal"] == "HOLD"
        assert logged["confidence"] == 0.25  # 25/100
        assert logged["price_at_signal"] == 150.0

    def test_buy_result_is_logged(self):
        """A BUY strategy result is logged with correct fields."""
        from orchestrator.coordinator import Coordinator

        buy_result = StrategyResult(
            signal="BUY",
            confidence=60.0,
            strategy_name="Momentum",
            indicators={"price": 200.0, "rsi": 57.0, "sma50": 195.0, "vol_ratio": 1.5},
        )

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)

        coordinator._log_strategy_result("META", buy_result)

        logged = coordinator.signal_logger.log.call_args[0][0]
        assert logged["ticker"] == "META"
        assert logged["strategy"] == "Momentum"
        assert logged["signal"] == "BUY"
        assert logged["confidence"] == 0.60
        assert logged["rsi"] == 57.0
        assert logged["trade_executed"] == 0

    def test_logging_exception_does_not_propagate(self):
        """If logging fails, the exception is swallowed (fire-and-forget)."""
        from orchestrator.coordinator import Coordinator

        hold_result = StrategyResult(
            signal="HOLD",
            confidence=25.0,
            strategy_name="Pullback",
        )

        coordinator = Coordinator.__new__(Coordinator)
        coordinator.signal_logger = MagicMock(spec=SignalLogger)
        coordinator.signal_logger.log.side_effect = RuntimeError("DB down")

        # Should not raise
        coordinator._log_strategy_result("AAPL", hold_result)


# ── BUG 2: XETRA bar fetch via EODHD ───────────────────────────────────────

class TestXetraBarFetch:
    """XETRA tickers must use EODHD, not Alpaca."""

    def test_eodhd_get_bars_returns_ohlcv(self):
        """EODHDFeed.get_bars() returns DataFrame with standard OHLCV columns."""
        mock_df = _make_ohlcv(100)
        feed = EODHDFeed(api_token="test")

        with patch.object(feed, "get_ohlcv_daily", return_value=mock_df):
            result = feed.get_bars("SAP.XETRA", limit=100)

        assert isinstance(result, pd.DataFrame)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns
        assert len(result) == 100

    def test_eodhd_get_bars_raises_on_empty(self):
        """EODHDFeed.get_bars() raises ValueError on empty data."""
        feed = EODHDFeed(api_token="test")
        with patch.object(feed, "get_ohlcv_daily", return_value=None):
            with pytest.raises(ValueError, match="no data"):
                feed.get_bars("SAP.XETRA")

    def test_eodhd_get_bars_raises_on_insufficient(self):
        """EODHDFeed.get_bars() raises ValueError when < 20 bars."""
        feed = EODHDFeed(api_token="test")
        small_df = _make_ohlcv(10)
        with patch.object(feed, "get_ohlcv_daily", return_value=small_df):
            with pytest.raises(ValueError, match="need >= 20"):
                feed.get_bars("SAP.XETRA")

    def test_xetra_ticker_uses_eodhd_not_alpaca(self):
        """Coordinator uses EODHD for .XETRA tickers in the bar-fetch block."""
        # Verify that the coordinator code dispatches based on .XETRA suffix.
        # We check this by importing and inspecting the source.
        import inspect
        from orchestrator.coordinator import Coordinator

        source = inspect.getsource(Coordinator)
        # Both sync and async paths should have the XETRA check
        assert 'ticker.endswith(".XETRA")' in source
        assert "EODHDFeed" in source
