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
        dates = pd.date_range(end="2025-01-15", periods=10, freq="B")
        bars_df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1_000_000] * 10,
        }, index=dates)

        mock_bars = MagicMock()
        mock_bars.df = bars_df
        mock_api.get_bars.return_value = mock_bars

        client = self._make_client(mock_api)
        df = client.get_bars("AAPL", "1Day", limit=10)

        self.assertEqual(len(df), 10)
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
