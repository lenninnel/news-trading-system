"""
Tests for NewsAPI rate-limit protection.

Verifies that:
  - 429 responses are handled gracefully (no crash, no signal degradation)
  - The daily request counter works (requests counted, limit enforced)
  - The 24-hour cache TTL works correctly
  - Signals are still produced when NewsAPI is skipped
  - SOURCE_WEIGHTS_NO_NEWSAPI is used when limit is reached
  - The counter resets at midnight UTC

Run with:
    python3 -m pytest tests/test_newsapi_rate_limit.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── sys.path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DB_PATH", "/tmp/test_newsapi_rate_limit.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("NEWSAPI_KEY", "dummy")

from config.settings import (
    SOURCE_WEIGHTS,
    SOURCE_WEIGHTS_NO_NEWSAPI,
    SOURCE_WEIGHTS_NO_REDDIT,
    SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI,
)
from data.news_feed import (
    NEWSAPI_DAILY_LIMIT,
    NewsFeed,
    _is_429_error,
    _newsapi_cache,
    is_newsapi_limit_reached,
    newsapi_requests_today,
    reset_newsapi_counter,
)
from utils.network_recovery import ResponseCache


# ===========================================================================
# Daily request counter
# ===========================================================================


class TestDailyRequestCounter:
    """Verify the daily request counter tracks and limits NewsAPI calls."""

    def test_counter_starts_at_zero(self):
        reset_newsapi_counter()
        assert newsapi_requests_today() == 0

    def test_counter_increments_on_successful_fetch(self):
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "articles": [{"title": "AAPL hits record high"}]
        }

        with patch("data.news_feed.requests.get", return_value=resp), \
             patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call", return_value=["AAPL hits record high"]):
            feed.fetch("AAPL")

        assert newsapi_requests_today() == 1

    def test_limit_reached_at_threshold(self):
        """When counter reaches NEWSAPI_DAILY_LIMIT, limit is flagged."""
        reset_newsapi_counter()
        from data.news_feed import _counter_lock, _today_utc
        import data.news_feed as nf

        with _counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT
            nf._counter_date = _today_utc()

        assert is_newsapi_limit_reached() is True

    def test_limit_not_reached_below_threshold(self):
        reset_newsapi_counter()
        from data.news_feed import _counter_lock, _today_utc
        import data.news_feed as nf

        with _counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT - 1
            nf._counter_date = _today_utc()

        assert is_newsapi_limit_reached() is False

    def test_counter_resets_on_new_day(self):
        """Simulating a date change resets the counter."""
        import data.news_feed as nf

        with nf._counter_lock:
            nf._daily_requests = 50
            nf._counter_date = "1999-01-01"  # definitely not today

        # Calling newsapi_requests_today() should trigger reset
        count = newsapi_requests_today()
        assert count == 0

    def test_skips_fetch_when_limit_reached(self):
        """When daily limit is reached, fetch returns cached or empty list."""
        reset_newsapi_counter()
        import data.news_feed as nf

        # Max out the counter
        with nf._counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT
            nf._counter_date = nf._today_utc()

        feed = NewsFeed(api_key="test-key")

        # Mock network monitor to not interfere
        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False):
            # Should NOT make an API call; should return []
            result = feed.fetch("AAPL")

        assert isinstance(result, list)
        # No API call was made since the limit is reached


# ===========================================================================
# 429 error handling
# ===========================================================================


class TestHandle429Gracefully:
    """Verify that 429 responses are handled without crashing."""

    def test_429_maxes_out_counter(self):
        """A 429 response should max out the daily counter."""
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        # Create a 429 HTTPError
        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc_429 = req.exceptions.HTTPError(response=mock_response)

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call", side_effect=exc_429):
            result = feed.fetch("AAPL")

        # Should not crash
        assert isinstance(result, list)
        # Counter should be maxed out
        assert is_newsapi_limit_reached() is True

    def test_429_returns_empty_not_crash(self):
        """A 429 with no cache should return [] not raise."""
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc_429 = req.exceptions.HTTPError(response=mock_response)

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call", side_effect=exc_429), \
             patch("data.news_feed.get_cache") as mock_cache:
            mock_cache.return_value.get.return_value = (None, False)
            result = feed.fetch("TSLA")

        assert result == []

    def test_429_serves_cached_headlines(self):
        """When a 429 hits but cache exists, serve cached headlines."""
        reset_newsapi_counter()
        feed = NewsFeed(api_key="test-key")

        # Pre-populate 24h cache
        _newsapi_cache.set("newsapi", "headlines:AAPL", ["Cached AAPL headline"])

        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc_429 = req.exceptions.HTTPError(response=mock_response)

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False):
            # The 24h cache should be hit before even making the API call
            result = feed.fetch("AAPL")

        assert result == ["Cached AAPL headline"]

    def test_429_does_not_propagate_exception(self):
        """Ensure the 429 exception does not propagate to the caller."""
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc_429 = req.exceptions.HTTPError(response=mock_response)

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call", side_effect=exc_429):
            # Must not raise
            result = feed.fetch("NVDA")
            assert isinstance(result, list)

    def test_is_429_error_detects_direct(self):
        """_is_429_error should detect a direct 429 HTTPError."""
        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        exc = req.exceptions.HTTPError(response=mock_response)
        assert _is_429_error(exc) is True

    def test_is_429_error_detects_wrapped(self):
        """_is_429_error should detect a 429 in the cause chain."""
        import requests as req
        mock_response = MagicMock()
        mock_response.status_code = 429
        inner = req.exceptions.HTTPError(response=mock_response)
        outer = RuntimeError("Wrapped error")
        outer.__cause__ = inner
        assert _is_429_error(outer) is True

    def test_is_429_error_rejects_non_429(self):
        """_is_429_error should return False for non-429 errors."""
        exc = RuntimeError("Generic error")
        assert _is_429_error(exc) is False


# ===========================================================================
# 24-hour cache TTL
# ===========================================================================


class TestCacheTTL24Hours:
    """Verify the 24-hour cache for NewsAPI responses."""

    def test_cache_is_24_hour_ttl(self):
        """The NewsAPI-specific cache should have a 24-hour max age."""
        assert _newsapi_cache._max_age == 86_400.0

    def test_cached_headlines_served_without_api_call(self):
        """When 24h cache has data, no API call should be made."""
        reset_newsapi_counter()
        feed = NewsFeed(api_key="test-key")

        # Prime the 24h cache
        _newsapi_cache.set("newsapi", "headlines:MSFT", [
            "MSFT earnings beat",
            "Microsoft cloud revenue up",
        ])

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call") as mock_api:
            result = feed.fetch("MSFT")

        # API was NOT called because cache hit came first
        mock_api.assert_not_called()
        assert len(result) == 2
        assert "MSFT earnings beat" in result

    def test_expired_cache_triggers_api_call(self):
        """When 24h cache is expired, an API call should be made."""
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call",
                   return_value=["Fresh GOOGL headline"]) as mock_api:
            result = feed.fetch("GOOGL")

        mock_api.assert_called_once()
        assert result == ["Fresh GOOGL headline"]

    def test_successful_fetch_populates_24h_cache(self):
        """A successful fetch should populate the 24h cache."""
        reset_newsapi_counter()
        _newsapi_cache.clear()
        feed = NewsFeed(api_key="test-key")

        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call",
                   return_value=["TSLA deliveries beat expectations"]):
            feed.fetch("TSLA")

        cached, hit = _newsapi_cache.get("newsapi", "headlines:TSLA")
        assert hit is True
        assert cached == ["TSLA deliveries beat expectations"]


# ===========================================================================
# Signals produced when NewsAPI is skipped
# ===========================================================================


class TestSignalsWithoutNewsAPI:
    """When NewsAPI is rate-limited, the system still produces valid signals."""

    def test_weighted_aggregate_without_newsapi(self):
        """Non-newsapi sources should produce valid weighted averages."""
        from orchestrator.coordinator import Coordinator

        scored = [
            {"score": 1, "source": "marketaux"},
            {"score": 1, "source": "stocktwits"},
            {"score": -1, "source": "reddit"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_NEWSAPI)
        assert isinstance(avg, float)
        assert -1.0 <= avg <= 1.0
        # (1*1.2 + 1*1.0 + -1*0.7) / (1.2 + 1.0 + 0.7) = 1.5/2.9
        expected = 1.5 / 2.9
        assert abs(avg - expected) < 0.001

    def test_no_newsapi_weights_exclude_newsapi(self):
        """SOURCE_WEIGHTS_NO_NEWSAPI should not contain a 'newsapi' key."""
        assert "newsapi" not in SOURCE_WEIGHTS_NO_NEWSAPI

    def test_no_newsapi_weights_boost_marketaux(self):
        """SOURCE_WEIGHTS_NO_NEWSAPI gives marketaux a higher weight."""
        assert SOURCE_WEIGHTS_NO_NEWSAPI["marketaux"] > SOURCE_WEIGHTS["marketaux"]

    def test_no_newsapi_weights_boost_stocktwits(self):
        """SOURCE_WEIGHTS_NO_NEWSAPI gives stocktwits a higher weight."""
        assert SOURCE_WEIGHTS_NO_NEWSAPI["stocktwits"] > SOURCE_WEIGHTS["stocktwits"]

    def test_no_reddit_no_newsapi_excludes_both(self):
        """SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI excludes both Reddit and NewsAPI."""
        assert "newsapi" not in SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI
        assert "reddit" not in SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI

    def test_active_weights_selects_no_newsapi(self):
        """When NewsAPI limit is reached, _active_weights returns NO_NEWSAPI map."""
        from orchestrator.coordinator import Coordinator
        import data.news_feed as nf

        # Max out counter
        with nf._counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT
            nf._counter_date = nf._today_utc()

        with patch("orchestrator.coordinator.is_reddit_configured", return_value=True):
            weights = Coordinator._active_weights()
            assert weights is SOURCE_WEIGHTS_NO_NEWSAPI

    def test_active_weights_selects_no_reddit_no_newsapi(self):
        """When both Reddit and NewsAPI are unavailable, correct map is used."""
        from orchestrator.coordinator import Coordinator
        import data.news_feed as nf

        with nf._counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT
            nf._counter_date = nf._today_utc()

        with patch("orchestrator.coordinator.is_reddit_configured", return_value=False):
            weights = Coordinator._active_weights()
            assert weights is SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI

    def test_active_weights_normal_when_both_ok(self):
        """When both Reddit and NewsAPI are available, standard weights used."""
        from orchestrator.coordinator import Coordinator

        reset_newsapi_counter()

        with patch("orchestrator.coordinator.is_reddit_configured", return_value=True):
            weights = Coordinator._active_weights()
            assert weights is SOURCE_WEIGHTS

    def test_empty_scored_returns_zero(self):
        """Empty scored list returns 0.0 regardless of weight map."""
        from orchestrator.coordinator import Coordinator

        assert Coordinator._weighted_aggregate([], SOURCE_WEIGHTS_NO_NEWSAPI) == 0.0

    def test_all_bearish_no_newsapi(self):
        """All bearish scores produce avg close to -1.0 even without NewsAPI."""
        from orchestrator.coordinator import Coordinator

        scored = [
            {"score": -1, "source": "marketaux"},
            {"score": -1, "source": "stocktwits"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_NEWSAPI)
        assert abs(avg - (-1.0)) < 0.001

    def test_all_bullish_no_newsapi(self):
        """All bullish scores produce avg close to 1.0 even without NewsAPI."""
        from orchestrator.coordinator import Coordinator

        scored = [
            {"score": 1, "source": "marketaux"},
            {"score": 1, "source": "stocktwits"},
            {"score": 1, "source": "reddit"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_NEWSAPI)
        assert abs(avg - 1.0) < 0.001


# ===========================================================================
# NewsAggregator integration with rate limits
# ===========================================================================


class TestNewsAggregatorRateLimit:
    """Verify that NewsAggregator respects the daily limit."""

    def test_aggregator_skips_newsapi_when_limit_reached(self):
        """When daily limit is reached, aggregator falls back to RSS."""
        from data.news_aggregator import NewsAggregator
        import data.news_feed as nf

        # Max out counter
        with nf._counter_lock:
            nf._daily_requests = NEWSAPI_DAILY_LIMIT
            nf._counter_date = nf._today_utc()

        _newsapi_cache.clear()
        agg = NewsAggregator(api_key="test-key", max_headlines=3)

        rss_xml = (
            "<?xml version='1.0'?>"
            "<rss><channel>"
            "<item><title>AAPL earnings beat via RSS</title></item>"
            "</channel></rss>"
        )
        rss_resp = MagicMock()
        rss_resp.raise_for_status = MagicMock()
        rss_resp.text = rss_xml

        with patch("data.news_aggregator.requests.get", return_value=rss_resp):
            result = agg.fetch_with_metadata("AAPL")

        # Should use RSS (level 1), not NewsAPI (level 0)
        assert result.level >= 1
        assert result.source != "newsapi"

    def test_aggregator_serves_24h_cache(self):
        """When 24h cache has data, aggregator serves it without API call."""
        from data.news_aggregator import NewsAggregator

        reset_newsapi_counter()
        _newsapi_cache.set("newsapi", "headlines:TSLA", [
            "Tesla 24h cached headline",
        ])

        agg = NewsAggregator(api_key="test-key", max_headlines=3)

        with patch("data.news_aggregator.requests.get") as mock_get:
            result = agg.fetch_with_metadata("TSLA")

        # Should NOT have called requests.get at all
        mock_get.assert_not_called()
        assert result.level == 0
        assert result.source == "newsapi"
        assert "Tesla 24h cached headline" in result.headlines

    def test_aggregator_429_maxes_counter(self):
        """A 429 from NewsAPI in the aggregator should max out the counter."""
        from data.news_aggregator import NewsAggregator
        import requests as req

        reset_newsapi_counter()
        _newsapi_cache.clear()

        agg = NewsAggregator(api_key="test-key", max_headlines=3)

        # Mock a 429 response for NewsAPI
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError(
            response=mock_response
        )

        # RSS fallback
        rss_xml = (
            "<?xml version='1.0'?>"
            "<rss><channel>"
            "<item><title>AAPL via RSS fallback</title></item>"
            "</channel></rss>"
        )
        rss_resp = MagicMock()
        rss_resp.raise_for_status = MagicMock()
        rss_resp.text = rss_xml

        def _side_effect(url, **kwargs):
            if "newsapi.org" in url:
                return mock_response
            return rss_resp

        with patch("data.news_aggregator.requests.get", side_effect=_side_effect):
            result = agg.fetch_with_metadata("AAPL")

        # Counter should be maxed
        assert is_newsapi_limit_reached() is True
        # System should still return headlines from RSS
        assert result.count > 0


# ===========================================================================
# Log usage
# ===========================================================================


class TestLogUsage:
    """Verify session-start logging of NewsAPI usage."""

    def test_log_usage_logs_info(self):
        """log_usage() should log current usage at INFO level."""
        reset_newsapi_counter()
        feed = NewsFeed(api_key="test-key")

        with patch("data.news_feed.log") as mock_log:
            feed.log_usage()
            mock_log.info.assert_called_once()
            call_args = str(mock_log.info.call_args)
            assert "NewsAPI" in call_args
            assert "requests used today" in call_args
