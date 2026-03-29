"""Tests for IBKRFeed — IBKR bar data with yfinance fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ibkr_feed import IBKRFeed


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_df(n: int = 50) -> pd.DataFrame:
    """Minimal OHLCV DataFrame."""
    return pd.DataFrame({
        "Open": [100.0] * n,
        "High": [101.0] * n,
        "Low": [99.0] * n,
        "Close": [100.5] * n,
        "Volume": [1_000_000] * n,
    })


# ── Tests ────────────────────────────────────────────────────────────────────

class TestIBKRFeed:

    def test_get_bars_daily(self):
        """get_bars delegates to IBKRTrader.get_bars for daily."""
        mock_ib = MagicMock()
        mock_ib.get_bars.return_value = _sample_df(252)

        feed = IBKRFeed(ib=mock_ib)
        df = feed.get_bars("AAPL", "1Day", 252)

        mock_ib.get_bars.assert_called_once_with("AAPL", "1Day", 252)
        assert len(df) == 252
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_get_bars_hourly(self):
        """get_bars delegates to IBKRTrader.get_bars for 1Hour."""
        mock_ib = MagicMock()
        mock_ib.get_bars.return_value = _sample_df(100)

        feed = IBKRFeed(ib=mock_ib)
        df = feed.get_bars("AAPL", "1Hour", 100)

        mock_ib.get_bars.assert_called_once_with("AAPL", "1Hour", 100)
        assert len(df) == 100

    def test_fallback_to_yfinance(self):
        """Falls back to yfinance when IBKR raises."""
        mock_ib = MagicMock()
        mock_ib.get_bars.side_effect = ConnectionError("gateway down")

        mock_yf = MagicMock()
        mock_yf.get_bars.return_value = _sample_df(50)

        feed = IBKRFeed(ib=mock_ib)
        feed._yf = mock_yf  # inject mock yfinance

        df = feed.get_bars("AAPL", "1Hour", 50)

        mock_ib.get_bars.assert_called_once()
        mock_yf.get_bars.assert_called_once_with("AAPL", "1Hour", 50)
        assert len(df) == 50

    def test_fallback_when_no_ib_connection(self):
        """Falls back to yfinance when IBKRTrader can't connect."""
        mock_yf = MagicMock()
        mock_yf.get_bars.return_value = _sample_df(30)

        # ib=None and _get_ib will fail (no real gateway)
        feed = IBKRFeed(ib=None)
        feed._yf = mock_yf
        # Force _get_ib to return None (simulates connection failure)
        feed._ib = None
        feed._get_ib = lambda: None

        df = feed.get_bars("MSFT", "15Min", 30)
        mock_yf.get_bars.assert_called_once_with("MSFT", "15Min", 30)

    def test_xetra_ticker_conversion(self):
        """XETRA tickers are passed through — IBKRTrader.get_bars handles conversion."""
        mock_ib = MagicMock()
        mock_ib.get_bars.return_value = _sample_df(50)

        feed = IBKRFeed(ib=mock_ib)
        feed.get_bars("SAP.XETRA", "1Day", 50)

        # IBKRFeed passes the ticker as-is; IBKRTrader.get_bars strips the suffix
        mock_ib.get_bars.assert_called_once_with("SAP.XETRA", "1Day", 50)


class TestIBKRTraderGetBars:
    """Test the get_bars method on IBKRTrader directly."""

    def test_xetra_suffix_stripped(self):
        """SAP.XETRA → SAP when building the IBKR contract."""
        from execution.ibkr_trader import IBKRTrader

        mock_ib = MagicMock()
        mock_stock = MagicMock()

        bar = MagicMock()
        bar.date = "2026-03-28"
        bar.open = 100.0
        bar.high = 101.0
        bar.low = 99.0
        bar.close = 100.5
        bar.volume = 1_000_000
        mock_ib.reqHistoricalData.return_value = [bar] * 30

        trader = IBKRTrader(ib=mock_ib)
        trader._Stock = mock_stock

        df = trader.get_bars("SAP.XETRA", "1Day", 30)

        # Stock() should receive "SAP" not "SAP.XETRA"
        mock_stock.assert_called_once_with("SAP", "SMART", "USD")
        assert len(df) == 30
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_de_suffix_stripped(self):
        """SAP.DE → SAP when building the IBKR contract."""
        from execution.ibkr_trader import IBKRTrader

        mock_ib = MagicMock()
        mock_stock = MagicMock()

        bar = MagicMock()
        bar.date = "2026-03-28"
        bar.open = 200.0
        bar.high = 201.0
        bar.low = 199.0
        bar.close = 200.5
        bar.volume = 500_000
        mock_ib.reqHistoricalData.return_value = [bar] * 20

        trader = IBKRTrader(ib=mock_ib)
        trader._Stock = mock_stock

        df = trader.get_bars("SIE.DE", "1Hour", 20)

        mock_stock.assert_called_once_with("SIE", "SMART", "USD")

    def test_unsupported_timeframe_raises(self):
        """Unknown timeframe raises ValueError."""
        from execution.ibkr_trader import IBKRTrader

        mock_ib = MagicMock()
        trader = IBKRTrader(ib=mock_ib)

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            trader.get_bars("AAPL", "5Min", 100)

    def test_empty_bars_raises(self):
        """Empty result from IBKR raises ValueError."""
        from execution.ibkr_trader import IBKRTrader

        mock_ib = MagicMock()
        mock_stock = MagicMock()
        mock_ib.reqHistoricalData.return_value = []

        trader = IBKRTrader(ib=mock_ib)
        trader._Stock = mock_stock

        with pytest.raises(ValueError, match="no bars"):
            trader.get_bars("AAPL", "1Day", 50)
