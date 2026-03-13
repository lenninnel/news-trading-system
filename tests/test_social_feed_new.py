"""
Unit tests for ApeWisdomFeed and AdanosFeed social feed sources.

All tests run offline using mocked HTTP responses -- no network access needed.

Run with:
    python3 -m pytest tests/test_social_feed_new.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import pytest

from data.social_feed import (
    ApeWisdomFeed,
    AdanosFeed,
    clear_apewisdom_cache,
    _apewisdom_cache,
)


# ===========================================================================
# ApeWisdom leaderboard feed
# ===========================================================================

class TestApeWisdomFeed:
    """Verify ApeWisdomFeed correctly fetches and caches the leaderboard."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_apewisdom_cache()

    def _mock_response(self, results, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"results": results}
        resp.raise_for_status.return_value = None
        return resp

    @patch("data.social_feed.requests.get")
    def test_happy_path_ticker_found(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"ticker": "AAPL", "mentions": 542, "upvotes": 12345, "rank": 3, "rank_24h_ago": 5},
            {"ticker": "TSLA", "mentions": 300, "upvotes": 8000, "rank": 1, "rank_24h_ago": 2},
        ])
        feed = ApeWisdomFeed()
        results = feed.fetch("AAPL")

        assert len(results) == 1
        assert results[0]["source"] == "apewisdom"
        assert results[0]["mentions"] == 542
        assert results[0]["rank"] == 3
        assert results[0]["rank_24h_ago"] == 5
        assert "542 Reddit mentions" in results[0]["text"]
        assert "rank #3" in results[0]["text"]
        assert "was #5" in results[0]["text"]

    @patch("data.social_feed.requests.get")
    def test_ticker_not_found(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"ticker": "TSLA", "mentions": 300, "upvotes": 8000, "rank": 1, "rank_24h_ago": 2},
        ])
        feed = ApeWisdomFeed()
        results = feed.fetch("AAPL")

        assert results == []

    @patch("data.social_feed.requests.get")
    def test_cache_prevents_second_request(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"ticker": "AAPL", "mentions": 542, "upvotes": 12345, "rank": 3, "rank_24h_ago": 5},
        ])
        feed = ApeWisdomFeed()

        # First call populates cache
        results1 = feed.fetch("AAPL")
        assert len(results1) == 1
        assert mock_get.call_count == 1

        # Second call should use cache
        results2 = feed.fetch("AAPL")
        assert len(results2) == 1
        assert mock_get.call_count == 1  # still 1

    @patch("data.social_feed.time.time")
    @patch("data.social_feed.requests.get")
    def test_cache_expires_after_ttl(self, mock_get, mock_time):
        mock_get.return_value = self._mock_response([
            {"ticker": "AAPL", "mentions": 542, "upvotes": 12345, "rank": 3, "rank_24h_ago": 5},
        ])
        feed = ApeWisdomFeed()

        # First call at t=1000
        mock_time.return_value = 1000.0
        feed.fetch("AAPL")
        assert mock_get.call_count == 1

        # Second call at t=1000+3601 (past 1 hour TTL)
        mock_time.return_value = 4601.0
        feed.fetch("AAPL")
        assert mock_get.call_count == 2  # cache expired, refetched

    @patch("data.social_feed.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        feed = ApeWisdomFeed()
        results = feed.fetch("AAPL")

        assert results == []

    @patch("data.social_feed.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = resp

        feed = ApeWisdomFeed()
        results = feed.fetch("AAPL")

        assert results == []

    @patch("data.social_feed.requests.get")
    def test_case_insensitive_ticker_lookup(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"ticker": "AAPL", "mentions": 542, "upvotes": 12345, "rank": 3, "rank_24h_ago": 5},
        ])
        feed = ApeWisdomFeed()
        results = feed.fetch("aapl")

        assert len(results) == 1
        assert results[0]["mentions"] == 542

    def test_clear_cache_resets_state(self):
        _apewisdom_cache["data"] = [{"ticker": "AAPL"}]
        _apewisdom_cache["fetched_at"] = 9999.0

        clear_apewisdom_cache()

        assert _apewisdom_cache["data"] == []
        assert _apewisdom_cache["fetched_at"] == 0.0


# ===========================================================================
# Adanos sentiment feed
# ===========================================================================

class TestAdanosFeed:
    """Verify AdanosFeed handles API key, quota, and responses correctly."""

    def setup_method(self):
        """Reset class-level state before each test."""
        AdanosFeed._request_count = 0
        AdanosFeed._quota_exhausted = False

    def _mock_response(self, data, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        return resp

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_happy_path(self, mock_get):
        mock_get.return_value = self._mock_response({
            "ticker": "AAPL",
            "buzz": 85,
            "bullish_pct": 0.65,
            "bearish_pct": 0.35,
        })
        feed = AdanosFeed()
        results = feed.fetch("AAPL")

        assert len(results) == 1
        assert results[0]["source"] == "adanos"
        assert results[0]["adanos_bullish"] == 0.65
        assert results[0]["adanos_buzz"] == 85
        assert "65% bullish" in results[0]["text"]
        assert "buzz score 85" in results[0]["text"]

        # Verify API key header was sent
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["X-API-Key"] == "test-key-123"

    @patch("data.social_feed.ADANOS_API_KEY", "")
    @patch("data.social_feed.requests.get")
    def test_no_api_key_skips(self, mock_get):
        feed = AdanosFeed()
        results = feed.fetch("AAPL")

        assert results == []
        mock_get.assert_not_called()

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_quota_exhausted_on_429(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        feed = AdanosFeed()
        results = feed.fetch("AAPL")

        assert results == []
        assert AdanosFeed._quota_exhausted is True
        assert AdanosFeed._request_count == 1

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_quota_exhausted_skips_subsequent_calls(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        feed = AdanosFeed()

        # First call triggers 429
        feed.fetch("AAPL")
        assert mock_get.call_count == 1

        # Subsequent calls should not hit the API
        feed.fetch("TSLA")
        feed.fetch("MSFT")
        assert mock_get.call_count == 1  # no additional calls

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection failed")
        feed = AdanosFeed()
        results = feed.fetch("AAPL")

        assert results == []

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = resp

        feed = AdanosFeed()
        results = feed.fetch("AAPL")

        assert results == []

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_request_count_increments(self, mock_get):
        mock_get.return_value = self._mock_response({
            "ticker": "AAPL",
            "buzz": 50,
            "bullish_pct": 0.5,
            "bearish_pct": 0.5,
        })
        feed = AdanosFeed()

        feed.fetch("AAPL")
        assert AdanosFeed._request_count == 1

        feed.fetch("TSLA")
        assert AdanosFeed._request_count == 2

    @patch("data.social_feed.ADANOS_API_KEY", "test-key-123")
    @patch("data.social_feed.requests.get")
    def test_429_logs_warning_once(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        feed = AdanosFeed()

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")  # triggers 429
            feed.fetch("TSLA")  # skipped (quota exhausted)
            feed.fetch("MSFT")  # skipped (quota exhausted)

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1
            assert "quota exhausted" in str(warning_calls[0]).lower()
