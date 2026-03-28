"""Tests for YFinanceFeed — yfinance-based OHLCV data feed."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from data.yfinance_feed import YFinanceFeed, _yf_ticker


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 60, base: float = 150.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = base + rng.normal(0, 0.5, n).cumsum()
    dates = pd.bdate_range(end="2026-03-27", periods=n)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.full(n, 5_000_000, dtype=float),
    }, index=dates)


# ── Ticker conversion ───────────────────────────────────────────────────────

class TestXetraTickerConversion:
    def test_xetra_to_de(self):
        assert _yf_ticker("SAP.XETRA") == "SAP.DE"

    def test_sie_xetra_to_de(self):
        assert _yf_ticker("SIE.XETRA") == "SIE.DE"

    def test_us_ticker_unchanged(self):
        assert _yf_ticker("AAPL") == "AAPL"

    def test_case_insensitive(self):
        assert _yf_ticker("sap.xetra") == "SAP.DE"


# ── get_bars() ───────────────────────────────────────────────────────────────

class TestGetBars:
    def test_daily_bars(self):
        mock_df = _make_ohlcv(60)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df) as dl:
            result = feed.get_bars("AAPL", "1Day", limit=252)
            dl.assert_called_once()
            call_kw = dl.call_args
            assert call_kw.kwargs["interval"] == "1d"
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 60

    def test_hourly_bars(self):
        mock_df = _make_ohlcv(30)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df) as dl:
            result = feed.get_bars("MSFT", "1Hour", limit=100)
            call_kw = dl.call_args
            assert call_kw.kwargs["interval"] == "1h"
        assert len(result) == 30

    def test_15min_bars(self):
        mock_df = _make_ohlcv(50)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df) as dl:
            result = feed.get_bars("META", "15Min", limit=100)
            call_kw = dl.call_args
            assert call_kw.kwargs["interval"] == "15m"
        assert len(result) == 50

    def test_columns_are_titlecase(self):
        mock_df = _make_ohlcv(30)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df):
            result = feed.get_bars("AAPL", "1Day", limit=252)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_xetra_ticker_converted(self):
        mock_df = _make_ohlcv(60)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df) as dl:
            feed.get_bars("SAP.XETRA", "1Day", limit=252)
            assert dl.call_args.args[0] == "SAP.DE"

    def test_empty_data_raises(self):
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="no data"):
                feed.get_bars("AAPL", "1Day")

    def test_insufficient_bars_raises(self):
        small_df = _make_ohlcv(10)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=small_df):
            with pytest.raises(ValueError, match="need >= 20"):
                feed.get_bars("AAPL", "1Day")

    def test_unsupported_timeframe_raises(self):
        feed = YFinanceFeed()
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            feed.get_bars("AAPL", "5Min")

    def test_timezone_stripped(self):
        mock_df = _make_ohlcv(30)
        mock_df.index = mock_df.index.tz_localize("US/Eastern")
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=mock_df):
            result = feed.get_bars("AAPL", "1Day")
        assert result.index.tz is None

    def test_trimmed_to_limit(self):
        big_df = _make_ohlcv(300)
        feed = YFinanceFeed()
        with patch("data.yfinance_feed.yf.download", return_value=big_df):
            result = feed.get_bars("AAPL", "1Day", limit=100)
        assert len(result) == 100
