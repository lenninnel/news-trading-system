"""
Unit tests for the EODHD data feed module.

All tests run offline using mocked HTTP responses — no network access needed.

Run with:
    python3 -m pytest tests/test_eodhd_feed.py -v
"""

import sys
import os
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from data.eodhd_feed import EODHDFeed, clear_cache


# ===========================================================================
# Helpers
# ===========================================================================

def _make_daily_response(num_bars=5, status_code=200):
    """Return a mock response mimicking EODHD /eod/ endpoint."""
    data = []
    for i in range(num_bars):
        data.append({
            "date": f"2026-03-{10 - i:02d}",
            "open": 100.0 + i,
            "high": 110.0 + i,
            "low": 90.0 + i,
            "close": 105.0 + i,
            "adjusted_close": 105.0 + i,
            "volume": 1_000_000 + i * 100_000,
        })
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _make_intraday_response(num_bars=30, status_code=200):
    """Return a mock response mimicking EODHD /intraday/ endpoint."""
    data = []
    for i in range(num_bars):
        data.append({
            "datetime": f"2026-03-10 09:{i:02d}:00",
            "open": 100.0 + i * 0.1,
            "high": 100.5 + i * 0.1,
            "low": 99.5 + i * 0.1,
            "close": 100.2 + i * 0.1,
            "volume": 50_000 + i * 1000,
        })
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _make_price_response(price=245.50, status_code=200):
    """Return a mock response for EODHD /real-time/ endpoint."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"close": price, "previousClose": price - 1.0}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _make_earnings_response(report_date="2026-04-20", status_code=200):
    """Return a mock response for EODHD /calendar/earnings endpoint."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "earnings": [{"report_date": report_date, "code": "SAP.XETRA"}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _make_news_response(num_articles=3, status_code=200):
    """Return a mock response for EODHD /news endpoint."""
    data = []
    for i in range(num_articles):
        data.append({
            "title": f"SAP beats expectations in Q{i+1}",
            "date": f"2026-03-{10 - i:02d}",
            "link": f"https://example.com/news/{i}",
        })
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ===========================================================================
# No-token graceful skip
# ===========================================================================

class TestEODHDNoToken:
    """When EODHD_API_TOKEN is not set, all methods return None / []."""

    def setup_method(self):
        clear_cache()

    def test_available_is_false(self):
        feed = EODHDFeed(api_token="")
        assert feed.available is False

    def test_daily_returns_none(self):
        assert EODHDFeed(api_token="").get_ohlcv_daily("SAP.XETRA") is None

    def test_intraday_returns_none(self):
        assert EODHDFeed(api_token="").get_ohlcv_intraday("SAP.XETRA") is None

    def test_price_returns_none(self):
        assert EODHDFeed(api_token="").get_price("SAP.XETRA") is None

    def test_earnings_returns_none(self):
        assert EODHDFeed(api_token="").get_earnings_calendar("SAP.XETRA") is None

    def test_news_returns_empty(self):
        assert EODHDFeed(api_token="").get_news("SAP.XETRA") == []


# ===========================================================================
# Daily OHLCV
# ===========================================================================

class TestEODHDDaily:
    """Verify get_ohlcv_daily parsing."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=5)
        feed = EODHDFeed(api_token="test-token")
        df = feed.get_ohlcv_daily("SAP.XETRA")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 5

    @patch("data.eodhd_feed.requests.get")
    def test_index_is_datetime(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=3)
        df = EODHDFeed(api_token="test").get_ohlcv_daily("SIE.XETRA")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    @patch("data.eodhd_feed.requests.get")
    def test_sorted_ascending(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=5)
        df = EODHDFeed(api_token="test").get_ohlcv_daily("SAP.XETRA")

        assert df.index[0] < df.index[-1]

    @patch("data.eodhd_feed.requests.get")
    def test_correct_api_params(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=1)
        EODHDFeed(api_token="mytoken").get_ohlcv_daily("BMW.XETRA", limit=100)

        args, kwargs = mock_get.call_args
        assert "eod/BMW.XETRA" in args[0]
        assert kwargs["params"]["api_token"] == "mytoken"
        assert kwargs["params"]["limit"] == 100

    @patch("data.eodhd_feed.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = _make_daily_response(status_code=500)
        result = EODHDFeed(api_token="test").get_ohlcv_daily("SAP.XETRA")
        assert result is None

    @patch("data.eodhd_feed.requests.get")
    def test_empty_response_returns_none(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = []
        mock_get.return_value = resp

        result = EODHDFeed(api_token="test").get_ohlcv_daily("SAP.XETRA")
        assert result is None


# ===========================================================================
# Intraday OHLCV
# ===========================================================================

class TestEODHDIntraday:
    """Verify get_ohlcv_intraday parsing."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = _make_intraday_response(num_bars=30)
        df = EODHDFeed(api_token="test").get_ohlcv_intraday("SAP.XETRA", interval="5m")

        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns
        assert len(df) == 30

    @patch("data.eodhd_feed.requests.get")
    def test_correct_interval_param(self, mock_get):
        mock_get.return_value = _make_intraday_response(num_bars=5)
        EODHDFeed(api_token="tok").get_ohlcv_intraday("SAP.XETRA", interval="1h")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["interval"] == "1h"


# ===========================================================================
# Real-time price
# ===========================================================================

class TestEODHDPrice:
    """Verify get_price."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_returns_float(self, mock_get):
        mock_get.return_value = _make_price_response(price=245.50)
        price = EODHDFeed(api_token="test").get_price("SAP.XETRA")

        assert isinstance(price, float)
        assert price == pytest.approx(245.50)

    @patch("data.eodhd_feed.requests.get")
    def test_error_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        assert EODHDFeed(api_token="test").get_price("SAP.XETRA") is None


# ===========================================================================
# Earnings calendar
# ===========================================================================

class TestEODHDEarnings:
    """Verify get_earnings_calendar."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_returns_date(self, mock_get):
        mock_get.return_value = _make_earnings_response(report_date="2026-04-20")
        result = EODHDFeed(api_token="test").get_earnings_calendar("SAP.XETRA")

        assert isinstance(result, date)
        assert result == date(2026, 4, 20)

    @patch("data.eodhd_feed.requests.get")
    def test_no_earnings_returns_none(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"earnings": []}
        mock_get.return_value = resp

        assert EODHDFeed(api_token="test").get_earnings_calendar("SAP.XETRA") is None


# ===========================================================================
# News
# ===========================================================================

class TestEODHDNews:
    """Verify get_news."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_returns_list_of_dicts(self, mock_get):
        mock_get.return_value = _make_news_response(num_articles=3)
        news = EODHDFeed(api_token="test").get_news("SAP.XETRA")

        assert isinstance(news, list)
        assert len(news) == 3
        assert "title" in news[0]
        assert "date" in news[0]
        assert "url" in news[0]

    @patch("data.eodhd_feed.requests.get")
    def test_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        assert EODHDFeed(api_token="test").get_news("SAP.XETRA") == []


# ===========================================================================
# Caching
# ===========================================================================

class TestEODHDCaching:
    """Verify cache behaviour."""

    def setup_method(self):
        clear_cache()

    @patch("data.eodhd_feed.requests.get")
    def test_daily_cache_hit(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=3)
        feed = EODHDFeed(api_token="test")

        first = feed.get_ohlcv_daily("SAP.XETRA")
        second = feed.get_ohlcv_daily("SAP.XETRA")

        assert first is second
        assert mock_get.call_count == 1

    @patch("data.eodhd_feed.requests.get")
    def test_clear_cache_forces_refetch(self, mock_get):
        mock_get.return_value = _make_daily_response(num_bars=3)
        feed = EODHDFeed(api_token="test")

        feed.get_ohlcv_daily("SAP.XETRA")
        clear_cache()
        feed.get_ohlcv_daily("SAP.XETRA")

        assert mock_get.call_count == 2

    @patch("data.eodhd_feed.time.time")
    @patch("data.eodhd_feed.requests.get")
    def test_expired_cache_refetches(self, mock_get, mock_time):
        mock_get.return_value = _make_daily_response(num_bars=3)

        mock_time.return_value = 0.0
        feed = EODHDFeed(api_token="test")
        feed.get_ohlcv_daily("SAP.XETRA")

        # Jump past 4h TTL
        mock_time.return_value = 5 * 3600
        feed.get_ohlcv_daily("SAP.XETRA")

        assert mock_get.call_count == 2


# ===========================================================================
# Fallback integration (is_german_ticker)
# ===========================================================================

class TestGermanTickerDetection:
    """Verify the is_german_ticker helper from settings."""

    def test_xetra_suffix(self):
        from config.settings import is_german_ticker
        assert is_german_ticker("SAP.XETRA") is True
        assert is_german_ticker("SIE.XETRA") is True

    def test_de_suffix(self):
        from config.settings import is_german_ticker
        assert is_german_ticker("SAP.DE") is True

    def test_us_ticker_not_german(self):
        from config.settings import is_german_ticker
        assert is_german_ticker("AAPL") is False
        assert is_german_ticker("MSFT") is False

    def test_crypto_not_german(self):
        from config.settings import is_german_ticker
        assert is_german_ticker("BTC") is False
