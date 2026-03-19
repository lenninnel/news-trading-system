"""
Tests for the Alpaca Data API client (data/alpaca_data.py).

All tests run offline with mocked Alpaca REST API responses.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np


class TestAlpacaDataClient(unittest.TestCase):
    """Unit tests for AlpacaDataClient."""

    def _make_client(self, mock_api=None):
        from data.alpaca_data import AlpacaDataClient
        return AlpacaDataClient(api=mock_api or MagicMock())

    # ------------------------------------------------------------------
    # get_current_price
    # ------------------------------------------------------------------

    def test_get_current_price_from_trade(self):
        mock_api = MagicMock()
        mock_trade = MagicMock()
        mock_trade.price = 254.50
        mock_api.get_latest_trade.return_value = mock_trade

        client = self._make_client(mock_api)
        price = client.get_current_price("AAPL")

        self.assertAlmostEqual(price, 254.50)
        mock_api.get_latest_trade.assert_called_once_with("AAPL")

    def test_get_current_price_fallback_to_quote(self):
        mock_api = MagicMock()
        mock_api.get_latest_trade.side_effect = Exception("no trade")
        mock_quote = MagicMock()
        mock_quote.bid_price = 253.0
        mock_quote.ask_price = 255.0
        mock_api.get_latest_quote.return_value = mock_quote

        client = self._make_client(mock_api)
        price = client.get_current_price("AAPL")

        # Mid-price of bid/ask
        self.assertAlmostEqual(price, 254.0)

    def test_get_current_price_raises_on_no_data(self):
        mock_api = MagicMock()
        mock_api.get_latest_trade.side_effect = Exception("no trade")
        mock_api.get_latest_quote.side_effect = Exception("no quote")

        client = self._make_client(mock_api)

        with self.assertRaises(ValueError, msg="Should raise when no data available"):
            client.get_current_price("FAKE")

    def test_get_current_price_zero_trade_falls_to_quote(self):
        mock_api = MagicMock()
        mock_trade = MagicMock()
        mock_trade.price = 0.0  # zero price
        mock_api.get_latest_trade.return_value = mock_trade
        mock_quote = MagicMock()
        mock_quote.bid_price = 100.0
        mock_quote.ask_price = 102.0
        mock_api.get_latest_quote.return_value = mock_quote

        client = self._make_client(mock_api)
        price = client.get_current_price("TEST")
        self.assertAlmostEqual(price, 101.0)

    # ------------------------------------------------------------------
    # get_bars
    # ------------------------------------------------------------------

    def test_get_bars_returns_dataframe(self):
        mock_api = MagicMock()
        dates = pd.date_range(end="2025-01-15", periods=25, freq="B")
        bars_df = pd.DataFrame({
            "open": [100.0] * 25,
            "high": [105.0] * 25,
            "low": [95.0] * 25,
            "close": [102.0] * 25,
            "volume": [1_000_000] * 25,
        }, index=dates)

        mock_bars = MagicMock()
        mock_bars.df = bars_df
        mock_api.get_bars.return_value = mock_bars

        client = self._make_client(mock_api)
        df = client.get_bars("AAPL", "1Day", limit=25)

        self.assertEqual(len(df), 25)
        # Column names should be title-case
        self.assertIn("Open", df.columns)
        self.assertIn("Close", df.columns)
        self.assertIn("Volume", df.columns)

    def test_get_bars_raises_on_empty(self):
        mock_api = MagicMock()
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame()
        mock_api.get_bars.return_value = mock_bars

        client = self._make_client(mock_api)

        with self.assertRaises(ValueError):
            client.get_bars("FAKE", "1Day", limit=10)

    # ------------------------------------------------------------------
    # get_snapshot
    # ------------------------------------------------------------------

    def test_get_snapshot_returns_dict(self):
        mock_api = MagicMock()
        mock_snap = MagicMock()
        mock_snap.latest_trade.price = 250.0
        mock_snap.daily_bar.volume = 5_000_000
        mock_snap.prev_daily_bar.close = 248.0
        mock_api.get_snapshot.return_value = mock_snap

        client = self._make_client(mock_api)
        snap = client.get_snapshot("AAPL")

        self.assertAlmostEqual(snap["price"], 250.0)
        self.assertEqual(snap["volume"], 5_000_000)
        self.assertAlmostEqual(snap["prev_close"], 248.0)
        self.assertIsNotNone(snap["change_pct"])
        self.assertEqual(snap["currency"], "USD")

    def test_get_snapshot_raises_on_no_price(self):
        mock_api = MagicMock()
        mock_snap = MagicMock()
        mock_snap.latest_trade = None
        mock_api.get_snapshot.return_value = mock_snap

        client = self._make_client(mock_api)
        with self.assertRaises(ValueError):
            client.get_snapshot("FAKE")


class TestGetIntradayBars(unittest.TestCase):
    """Unit tests for AlpacaDataClient.get_intraday_bars."""

    def _make_client(self, mock_api=None):
        from data.alpaca_data import AlpacaDataClient
        return AlpacaDataClient(api=mock_api or MagicMock())

    def _mock_bars_response(self, n=20):
        """Create a mock bars response with n rows of OHLCV data."""
        dates = pd.date_range(end="2026-03-18 16:00", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": [150.0 + i for i in range(n)],
            "high": [155.0 + i for i in range(n)],
            "low": [148.0 + i for i in range(n)],
            "close": [152.0 + i for i in range(n)],
            "volume": [100_000 + i * 1000 for i in range(n)],
        }, index=dates)
        mock_bars = MagicMock()
        mock_bars.df = df
        return mock_bars

    def test_returns_dataframe_5min(self):
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(20)
        client = self._make_client(mock_api)

        df = client.get_intraday_bars("AAPL", timeframe="5Min", limit=20)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 20)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, df.columns)
        mock_api.get_bars.assert_called_once_with("AAPL", "5Min", limit=20)

    def test_returns_dataframe_15min(self):
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(15)
        client = self._make_client(mock_api)

        df = client.get_intraday_bars("MSFT", timeframe="15Min", limit=15)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 15)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, df.columns)
        mock_api.get_bars.assert_called_once_with("MSFT", "15Min", limit=15)

    def test_returns_dataframe_1hour(self):
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(10)
        client = self._make_client(mock_api)

        df = client.get_intraday_bars("TSLA", timeframe="1Hour", limit=10)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, df.columns)

    def test_xetra_ticker_returns_none(self):
        mock_api = MagicMock()
        client = self._make_client(mock_api)

        result = client.get_intraday_bars("SAP.XETRA")
        self.assertIsNone(result)
        mock_api.get_bars.assert_not_called()

    def test_xetra_de_ticker_returns_none(self):
        mock_api = MagicMock()
        client = self._make_client(mock_api)

        result = client.get_intraday_bars("SIE.DE")
        self.assertIsNone(result)
        mock_api.get_bars.assert_not_called()

    def test_empty_dataframe_returns_none(self):
        mock_api = MagicMock()
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame()
        mock_api.get_bars.return_value = mock_bars
        client = self._make_client(mock_api)

        result = client.get_intraday_bars("AAPL", timeframe="5Min")
        self.assertIsNone(result)

    def test_api_exception_returns_none(self):
        mock_api = MagicMock()
        mock_api.get_bars.side_effect = Exception("market closed")
        client = self._make_client(mock_api)

        result = client.get_intraday_bars("AAPL", timeframe="5Min")
        self.assertIsNone(result)

    def test_invalid_timeframe_returns_none(self):
        mock_api = MagicMock()
        client = self._make_client(mock_api)

        result = client.get_intraday_bars("AAPL", timeframe="3Min")
        self.assertIsNone(result)
        mock_api.get_bars.assert_not_called()

    def test_ticker_is_uppercased(self):
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(5)
        client = self._make_client(mock_api)

        client.get_intraday_bars("aapl", timeframe="5Min")
        mock_api.get_bars.assert_called_once_with("AAPL", "5Min", limit=100)


class TestGetMultiTimeframeBars(unittest.TestCase):
    """Unit tests for AlpacaDataClient.get_multi_timeframe_bars."""

    def _make_client(self, mock_api=None):
        from data.alpaca_data import AlpacaDataClient
        return AlpacaDataClient(api=mock_api or MagicMock())

    def _mock_bars_response(self, n=20):
        """Create a mock bars response with n rows of OHLCV data."""
        dates = pd.date_range(end="2026-03-18 16:00", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": [150.0] * n,
            "high": [155.0] * n,
            "low": [148.0] * n,
            "close": [152.0] * n,
            "volume": [100_000] * n,
        }, index=dates)
        mock_bars = MagicMock()
        mock_bars.df = df
        return mock_bars

    def test_returns_dict_with_all_four_keys(self):
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(20)
        client = self._make_client(mock_api)

        result = client.get_multi_timeframe_bars("AAPL")

        self.assertIsInstance(result, dict)
        for key in ("5min", "15min", "1hour", "1day"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], pd.DataFrame)

    def test_partial_failure_some_none(self):
        """When some timeframes fail, those keys are None, others succeed."""
        mock_api = MagicMock()
        call_count = 0

        def side_effect(ticker, timeframe, **kwargs):
            nonlocal call_count
            call_count += 1
            # Let 5Min and 1Day succeed, fail on 15Min and 1Hour
            if timeframe in ("15Min", "1Hour"):
                raise Exception("no data available")
            return self._mock_bars_response(25)

        mock_api.get_bars.side_effect = side_effect
        client = self._make_client(mock_api)

        result = client.get_multi_timeframe_bars("AAPL")

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"5min", "15min", "1hour", "1day"})
        # 5min and 1day should have data
        self.assertIsInstance(result["5min"], pd.DataFrame)
        self.assertIsInstance(result["1day"], pd.DataFrame)
        # 15min and 1hour should be None
        self.assertIsNone(result["15min"])
        self.assertIsNone(result["1hour"])

    def test_all_timeframes_fail_returns_all_none(self):
        mock_api = MagicMock()
        mock_api.get_bars.side_effect = Exception("service unavailable")
        client = self._make_client(mock_api)

        result = client.get_multi_timeframe_bars("AAPL")

        self.assertIsInstance(result, dict)
        for key in ("5min", "15min", "1hour", "1day"):
            self.assertIsNone(result[key])

    def test_xetra_ticker_returns_all_none(self):
        mock_api = MagicMock()
        # Daily bars will also fail for XETRA since Alpaca has no coverage
        mock_api.get_bars.side_effect = Exception("no data")
        client = self._make_client(mock_api)

        result = client.get_multi_timeframe_bars("SAP.XETRA")

        self.assertIsInstance(result, dict)
        # Intraday bars short-circuit to None for XETRA
        self.assertIsNone(result["5min"])
        self.assertIsNone(result["15min"])
        self.assertIsNone(result["1hour"])
        # Daily bars will also be None (API error)
        self.assertIsNone(result["1day"])

    def test_correct_limits_passed(self):
        """Verify that the correct limits are passed for each timeframe."""
        mock_api = MagicMock()
        mock_api.get_bars.return_value = self._mock_bars_response(25)
        client = self._make_client(mock_api)

        client.get_multi_timeframe_bars("AAPL")

        # Check all 4 calls were made with correct params
        calls = mock_api.get_bars.call_args_list
        self.assertEqual(len(calls), 4)

        # 5Min with limit=100
        self.assertEqual(calls[0][0], ("AAPL", "5Min"))
        self.assertEqual(calls[0][1]["limit"], 100)

        # 15Min with limit=100
        self.assertEqual(calls[1][0], ("AAPL", "15Min"))
        self.assertEqual(calls[1][1]["limit"], 100)

        # 1Hour with limit=50
        self.assertEqual(calls[2][0], ("AAPL", "1Hour"))
        self.assertEqual(calls[2][1]["limit"], 50)

        # 1Day with limit=100
        self.assertEqual(calls[3][0], ("AAPL", "1Day"))
        self.assertEqual(calls[3][1]["limit"], 100)


class TestXetraFallback(unittest.TestCase):
    """Verify that XETRA tickers fall back to yfinance gracefully."""

    def test_price_fallback_xetra_uses_yfinance(self):
        """PriceFallback routes XETRA ticker to yfinance, not Alpaca."""
        from data.price_fallback import PriceFallback

        pf = PriceFallback()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 220.0,
            "currency": "EUR",
            "longName": "SAP SE",
            "marketCap": 200_000_000_000,
        }

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = pf.get_price("SAP.XETRA")

        self.assertEqual(result.level, 0)
        self.assertEqual(result.source, "yfinance")
        self.assertAlmostEqual(result.price, 220.0)
        self.assertEqual(result.currency, "EUR")


class TestErrorHandling(unittest.TestCase):
    """Verify error handling: no stale fallback prices."""

    def test_alpaca_no_data_raises_not_stale(self):
        """When Alpaca returns no data, raise an error -- never use stale prices."""
        from data.alpaca_data import AlpacaDataClient

        mock_api = MagicMock()
        mock_api.get_latest_trade.side_effect = Exception("no data")
        mock_api.get_latest_quote.side_effect = Exception("no data")

        client = AlpacaDataClient(api=mock_api)
        with self.assertRaises(ValueError):
            client.get_current_price("AAPL")

    def test_alpaca_bars_no_data_raises(self):
        from data.alpaca_data import AlpacaDataClient

        mock_api = MagicMock()
        mock_api.get_bars.side_effect = Exception("service unavailable")

        client = AlpacaDataClient(api=mock_api)
        with self.assertRaises(ValueError):
            client.get_bars("AAPL", "1Day", limit=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
