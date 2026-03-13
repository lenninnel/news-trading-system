"""
Unit tests for the Binance crypto OHLCV feed module.

All tests run offline using mocked HTTP responses — no network access needed.

Run with:
    python3 -m pytest tests/test_binance_feed.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from data.binance_feed import BinanceFeed, clear_cache


# ===========================================================================
# Helper: build fake Binance kline data
# ===========================================================================

def _make_kline(open_time_ms, open_price, high, low, close, volume):
    """Return a single fake kline list (12 elements, matching Binance format)."""
    return [
        open_time_ms,           # 0  open time (ms)
        str(open_price),        # 1  open
        str(high),              # 2  high
        str(low),               # 3  low
        str(close),             # 4  close
        str(volume),            # 5  volume
        open_time_ms + 86400000,  # 6  close time (ms)
        "1000000.00",           # 7  quote asset volume
        150,                    # 8  number of trades
        "500.00",               # 9  taker buy base volume
        "500000.00",            # 10 taker buy quote volume
        "0",                    # 11 ignore
    ]


def _make_klines_response(num_candles=3, status_code=200):
    """Return a MagicMock mimicking ``requests.Response`` with fake klines."""
    base_time_ms = 1_700_000_000_000  # arbitrary start timestamp
    klines = []
    for i in range(num_candles):
        klines.append(
            _make_kline(
                open_time_ms=base_time_ms + i * 86_400_000,
                open_price=100.0 + i,
                high=110.0 + i,
                low=90.0 + i,
                close=105.0 + i,
                volume=1000.0 + i * 100,
            )
        )

    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = klines
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ===========================================================================
# Happy path: parsing and field mapping
# ===========================================================================

class TestBinanceFeedParsing:
    """Verify BinanceFeed correctly parses kline data into a DataFrame."""

    def setup_method(self):
        clear_cache()

    @patch("data.binance_feed.requests.get")
    def test_returns_dataframe_with_correct_columns(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=5)

        feed = BinanceFeed()
        df = feed.get_ohlcv("BTC")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 5

    @patch("data.binance_feed.requests.get")
    def test_index_is_datetimeindex(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=3)

        df = BinanceFeed().get_ohlcv("ETH")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    @patch("data.binance_feed.requests.get")
    def test_values_are_float(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=2)

        df = BinanceFeed().get_ohlcv("BTC")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert df[col].dtype == float

    @patch("data.binance_feed.requests.get")
    def test_correct_values_parsed(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=2)

        df = BinanceFeed().get_ohlcv("SOL")

        # First candle: open=100, high=110, low=90, close=105, volume=1000
        assert df.iloc[0]["Open"] == pytest.approx(100.0)
        assert df.iloc[0]["High"] == pytest.approx(110.0)
        assert df.iloc[0]["Low"] == pytest.approx(90.0)
        assert df.iloc[0]["Close"] == pytest.approx(105.0)
        assert df.iloc[0]["Volume"] == pytest.approx(1000.0)

        # Second candle: open=101, high=111, low=91, close=106, volume=1100
        assert df.iloc[1]["Open"] == pytest.approx(101.0)
        assert df.iloc[1]["Close"] == pytest.approx(106.0)

    @patch("data.binance_feed.requests.get")
    def test_symbol_gets_usdt_appended(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=1)

        BinanceFeed().get_ohlcv("BTC")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["symbol"] == "BTCUSDT"

    @patch("data.binance_feed.requests.get")
    def test_symbol_is_uppercased(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=1)

        BinanceFeed().get_ohlcv("btc")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["symbol"] == "BTCUSDT"

    @patch("data.binance_feed.requests.get")
    def test_limit_is_passed(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=1)

        BinanceFeed().get_ohlcv("BTC", limit=50)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 50


# ===========================================================================
# get_price
# ===========================================================================

class TestBinanceFeedGetPrice:
    """Verify get_price returns the latest close value."""

    def setup_method(self):
        clear_cache()

    @patch("data.binance_feed.requests.get")
    def test_get_price_returns_float(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=3)

        price = BinanceFeed().get_price("BTC")

        assert isinstance(price, float)
        # Last candle close: 105.0 + 2 = 107.0
        assert price == pytest.approx(107.0)

    @patch("data.binance_feed.requests.get")
    def test_get_price_returns_none_on_failure(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        price = BinanceFeed().get_price("BTC")

        assert price is None


# ===========================================================================
# Caching behaviour
# ===========================================================================

class TestBinanceFeedCaching:
    """Verify the 4-hour TTL cache avoids redundant API calls."""

    def setup_method(self):
        clear_cache()

    @patch("data.binance_feed.requests.get")
    def test_second_call_uses_cache(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=3)

        feed = BinanceFeed()
        first = feed.get_ohlcv("BTC")
        second = feed.get_ohlcv("BTC")

        assert first is second  # same object from cache
        assert mock_get.call_count == 1

    @patch("data.binance_feed.requests.get")
    def test_different_symbols_cached_separately(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=2)

        feed = BinanceFeed()
        feed.get_ohlcv("BTC")
        feed.get_ohlcv("ETH")

        assert mock_get.call_count == 2

    @patch("data.binance_feed.time.time")
    @patch("data.binance_feed.requests.get")
    def test_expired_cache_triggers_new_fetch(self, mock_get, mock_time):
        mock_get.return_value = _make_klines_response(num_candles=2)

        # First call at t=0
        mock_time.return_value = 0.0
        feed = BinanceFeed()
        feed.get_ohlcv("BTC")

        # Second call at t=5 hours (past 4-hour TTL)
        mock_time.return_value = 5 * 60 * 60
        feed.get_ohlcv("BTC")

        assert mock_get.call_count == 2

    @patch("data.binance_feed.requests.get")
    def test_clear_cache_forces_refetch(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=2)

        feed = BinanceFeed()
        feed.get_ohlcv("BTC")
        clear_cache()
        feed.get_ohlcv("BTC")

        assert mock_get.call_count == 2

    @patch("data.binance_feed.requests.get")
    def test_get_price_uses_ohlcv_cache(self, mock_get):
        mock_get.return_value = _make_klines_response(num_candles=3)

        feed = BinanceFeed()
        feed.get_ohlcv("BTC")
        feed.get_price("BTC")

        assert mock_get.call_count == 1


# ===========================================================================
# API error handling — never raise
# ===========================================================================

class TestBinanceFeedErrorHandling:
    """BinanceFeed should return None on any API failure, never raise."""

    def setup_method(self):
        clear_cache()

    @patch("data.binance_feed.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = _make_klines_response(status_code=500)

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_rate_limit_429_returns_none(self, mock_get):
        mock_get.return_value = _make_klines_response(status_code=429)

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_network_exception_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_timeout_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_malformed_json_returns_none(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_empty_klines_returns_none(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = []
        mock_get.return_value = resp

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_invalid_symbol_returns_none(self, mock_get):
        resp = MagicMock()
        resp.status_code = 400
        resp.raise_for_status.side_effect = Exception("HTTP 400: Invalid symbol")
        mock_get.return_value = resp

        result = BinanceFeed().get_ohlcv("ZZZZZ")

        assert result is None

    @patch("data.binance_feed.requests.get")
    def test_non_list_response_returns_none(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"error": "something"}
        mock_get.return_value = resp

        result = BinanceFeed().get_ohlcv("BTC")

        assert result is None
