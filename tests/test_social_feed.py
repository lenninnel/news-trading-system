"""
Unit tests for social feed sources and weighted aggregation.

All tests run offline using mocked HTTP responses — no network access needed.

Run with:
    python3 -m pytest tests/test_social_feed.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from unittest.mock import MagicMock, patch

import pytest

from data.social_feed import RedditFeed, StockTwitsFeed, _reddit_warned
from orchestrator.coordinator import Coordinator


# ===========================================================================
# StockTwits response parsing
# ===========================================================================

class TestStockTwitsParsing:
    """Verify StockTwitsFeed correctly parses the public API response."""

    def setup_method(self):
        """Reset class-level state before each test."""
        StockTwitsFeed._disabled = False
        StockTwitsFeed._fail_count = 0

    def _mock_response(self, messages, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"messages": messages}
        return resp

    @patch("data.social_feed.requests.get")
    def test_parses_body_and_sentiment(self, mock_get):
        mock_get.return_value = self._mock_response([
            {
                "body": "AAPL to the moon!",
                "entities": {"sentiment": {"basic": "Bullish"}},
            },
            {
                "body": "Selling everything",
                "entities": {"sentiment": {"basic": "Bearish"}},
            },
        ])
        feed = StockTwitsFeed(max_messages=10)
        results = feed.fetch("AAPL")

        assert len(results) == 2
        assert results[0]["text"] == "AAPL to the moon!"
        assert results[0]["source"] == "stocktwits"
        assert results[0]["stocktwits_sentiment"] == "Bullish"
        assert results[1]["stocktwits_sentiment"] == "Bearish"

    @patch("data.social_feed.requests.get")
    def test_handles_missing_sentiment(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"body": "Just watching", "entities": {}},
        ])
        results = StockTwitsFeed().fetch("TSLA")

        assert len(results) == 1
        assert results[0]["stocktwits_sentiment"] is None

    @patch("data.social_feed.requests.get")
    def test_handles_no_entities(self, mock_get):
        mock_get.return_value = self._mock_response([
            {"body": "Hmm"},
        ])
        results = StockTwitsFeed().fetch("TSLA")

        assert len(results) == 1
        assert results[0]["stocktwits_sentiment"] is None

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_non_200_returns_empty(self, mock_get, mock_sleep):
        mock_get.return_value = self._mock_response([], status_code=429)
        results = StockTwitsFeed().fetch("AAPL")

        assert results == []

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_network_error_returns_empty(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("Connection failed")
        results = StockTwitsFeed().fetch("AAPL")

        assert results == []

    @patch("data.social_feed.requests.get")
    def test_respects_max_messages(self, mock_get):
        messages = [{"body": f"msg {i}"} for i in range(20)]
        mock_get.return_value = self._mock_response(messages)
        results = StockTwitsFeed(max_messages=5).fetch("AAPL")

        assert len(results) == 5

    @patch("data.social_feed.requests.get")
    def test_empty_messages_list(self, mock_get):
        mock_get.return_value = self._mock_response([])
        results = StockTwitsFeed().fetch("AAPL")

        assert results == []


# ===========================================================================
# Reddit fallback when credentials missing
# ===========================================================================

class TestRedditFallback:
    """RedditFeed should never crash — returns [] when creds are missing."""

    def setup_method(self):
        """Reset the module-level warn-once flags before each test."""
        _reddit_warned["credentials"] = False
        _reddit_warned["praw"] = False

    def test_no_client_id_returns_empty(self):
        feed = RedditFeed(client_id="", client_secret="secret")
        results = feed.fetch("AAPL")
        assert results == []

    def test_no_client_secret_returns_empty(self):
        feed = RedditFeed(client_id="id", client_secret="")
        results = feed.fetch("AAPL")
        assert results == []

    def test_both_empty_returns_empty(self):
        feed = RedditFeed(client_id="", client_secret="")
        results = feed.fetch("AAPL")
        assert results == []

    @patch.dict("sys.modules", {"praw": None})
    def test_praw_not_installed_returns_empty(self):
        """When praw can't be imported, return empty list."""
        feed = RedditFeed(client_id="id", client_secret="secret")
        results = feed.fetch("AAPL")
        assert results == []


# ===========================================================================
# Reddit graceful degradation — warn-once, then silence
# ===========================================================================

class TestRedditGracefulDegradation:
    """
    Verify that missing Reddit credentials degrade gracefully:
      - Returns [] on every call
      - Logs exactly ONE warning at the first call
      - Subsequent calls are completely silent (no warnings, no errors)
      - Exceptions from praw are caught and silenced
    """

    def setup_method(self):
        """Reset the module-level warn-once flags before each test."""
        _reddit_warned["credentials"] = False
        _reddit_warned["praw"] = False

    def test_no_creds_logs_warning_exactly_once(self):
        """First call emits one WARNING; subsequent calls are silent."""
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")
            feed.fetch("TSLA")
            feed.fetch("NVDA")

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1
            assert "REDDIT_CLIENT_ID" in str(warning_calls[0])

    def test_no_creds_returns_empty_every_time(self):
        """All fetches return [] regardless of how many times called."""
        feed = RedditFeed(client_id="", client_secret="")
        for ticker in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]:
            assert feed.fetch(ticker) == []

    def test_no_creds_no_info_logs(self):
        """No info-level logs should be emitted for missing credentials."""
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")
            feed.fetch("TSLA")
            # Reset warning so next calls are 'already warned'
            _reddit_warned["credentials"] = True
            feed.fetch("NVDA")
            feed.fetch("MSFT")

            # After the first warning, zero further calls of any kind
            info_calls = mock_logger.info.call_args_list
            assert info_calls == []

    @patch.dict("sys.modules", {"praw": None})
    def test_praw_missing_logs_warning_exactly_once(self):
        """praw ImportError emits one WARNING at first call, then silent."""
        feed = RedditFeed(client_id="id", client_secret="secret")

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")
            feed.fetch("TSLA")
            feed.fetch("NVDA")

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1
            assert "praw" in str(warning_calls[0]).lower()

    @patch.dict("sys.modules", {"praw": None})
    def test_praw_missing_returns_empty_every_time(self):
        """All fetches return [] when praw is not installed."""
        feed = RedditFeed(client_id="id", client_secret="secret")
        for ticker in ["AAPL", "TSLA"]:
            assert feed.fetch(ticker) == []

    def test_praw_exception_silenced_returns_empty(self):
        """Any exception thrown by praw is caught and silenced."""
        mock_praw = MagicMock()
        mock_reddit = MagicMock()
        mock_praw.Reddit.return_value = mock_reddit
        mock_reddit.subreddit.side_effect = RuntimeError("praw internal error")

        with patch.dict("sys.modules", {"praw": mock_praw}):
            feed = RedditFeed(client_id="id", client_secret="secret")

            with patch("data.social_feed.logger") as mock_logger:
                result = feed.fetch("AAPL")

            assert result == []
            # No warnings or errors logged for fetch exceptions
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()

    def test_warned_flag_prevents_repeated_warnings(self):
        """Once the flag is set, repeated missing-cred fetches are silent."""
        _reddit_warned["credentials"] = True  # simulate already warned
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            for _ in range(10):
                feed.fetch("AAPL")

            mock_logger.warning.assert_not_called()
            mock_logger.info.assert_not_called()


# ===========================================================================
# Weighted average score calculation
# ===========================================================================

class TestWeightedAggregate:
    """Verify weighted aggregation from Coordinator."""

    def test_single_source_newsapi(self):
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": -1, "source": "newsapi"},
            {"score": 1, "source": "newsapi"},
        ]
        # Unweighted: (1 + -1 + 1) / 3 = 0.333
        # Weighted: all weight=1.0, same result
        avg = Coordinator._weighted_aggregate(scored)
        assert abs(avg - (1 / 3)) < 0.001

    def test_mixed_sources_apply_weights(self):
        scored = [
            {"score": 1, "source": "newsapi"},     # weight 1.0
            {"score": 1, "source": "stocktwits"},   # weight 0.8
            {"score": -1, "source": "reddit"},       # weight 0.6
        ]
        # weighted_sum = 1*1.0 + 1*0.8 + (-1)*0.6 = 1.2
        # total_weight = 1.0 + 0.8 + 0.6 = 2.4
        # avg = 1.2 / 2.4 = 0.5
        avg = Coordinator._weighted_aggregate(scored)
        assert abs(avg - 0.5) < 0.001

    def test_reddit_counts_less(self):
        """Same scores, but reddit contribution should be diluted."""
        all_newsapi = [
            {"score": 1, "source": "newsapi"},
            {"score": -1, "source": "newsapi"},
        ]
        mixed = [
            {"score": 1, "source": "newsapi"},
            {"score": -1, "source": "reddit"},
        ]
        avg_news = Coordinator._weighted_aggregate(all_newsapi)   # (1 + -1) / 2 = 0.0
        avg_mixed = Coordinator._weighted_aggregate(mixed)         # (1*1.0 + -1*0.6) / 1.6 = 0.25
        # reddit negative is diluted → mixed should be more positive
        assert avg_mixed > avg_news

    def test_empty_scored(self):
        assert Coordinator._weighted_aggregate([]) == 0.0

    def test_source_breakdown_structure(self):
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": -1, "source": "newsapi"},
            {"score": 1, "source": "stocktwits"},
            {"score": 0, "source": "reddit"},
        ]
        bd = Coordinator._source_breakdown(scored)

        assert bd["newsapi"]["count"] == 2
        assert bd["newsapi"]["avg"] == 0.0
        assert bd["stocktwits"]["count"] == 1
        assert bd["stocktwits"]["avg"] == 1.0
        assert bd["reddit"]["count"] == 1
        assert bd["reddit"]["avg"] == 0.0

    def test_source_breakdown_missing_source_defaults_to_newsapi(self):
        scored = [{"score": 1}]  # no "source" key
        bd = Coordinator._source_breakdown(scored)
        assert "newsapi" in bd
        assert bd["newsapi"]["count"] == 1

    def test_all_same_sentiment_weighted(self):
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": 1, "source": "stocktwits"},
            {"score": 1, "source": "reddit"},
        ]
        avg = Coordinator._weighted_aggregate(scored)
        assert abs(avg - 1.0) < 0.001  # all bullish → 1.0 regardless of weights


# ===========================================================================
# StockTwits exponential backoff and session-level disable
# ===========================================================================

class TestStockTwitsBackoff:
    """Verify retry, backoff, and _disabled flag behaviour."""

    def setup_method(self):
        StockTwitsFeed._disabled = False
        StockTwitsFeed._fail_count = 0

    def _mock_response(self, messages, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"messages": messages}
        return resp

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_retries_on_403(self, mock_get, mock_sleep):
        """403 should trigger exponential backoff retries."""
        mock_get.return_value = self._mock_response([], status_code=403)
        feed = StockTwitsFeed()
        feed.fetch("AAPL")

        # 1 initial + 2 retries = 3 calls
        assert mock_get.call_count == 3
        # Backoff sleeps: 2^0=1s, 2^1=2s
        assert mock_sleep.call_count == 2

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_success_after_retry(self, mock_get, mock_sleep):
        """Should return data if retry succeeds."""
        fail_resp = self._mock_response([], status_code=403)
        ok_resp = self._mock_response([{"body": "Works now"}])
        mock_get.side_effect = [fail_resp, ok_resp]

        results = StockTwitsFeed().fetch("AAPL")
        assert len(results) == 1
        assert results[0]["text"] == "Works now"
        assert StockTwitsFeed._fail_count == 0

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_disables_after_3_failures(self, mock_get, mock_sleep):
        """After 3 failed fetches, _disabled should be True."""
        mock_get.return_value = self._mock_response([], status_code=403)
        feed = StockTwitsFeed()

        feed.fetch("AAPL")   # _fail_count → 1
        assert StockTwitsFeed._fail_count == 1
        assert not StockTwitsFeed._disabled

        feed.fetch("MSFT")   # _fail_count → 2
        assert not StockTwitsFeed._disabled

        feed.fetch("NVDA")   # _fail_count → 3 → disabled
        assert StockTwitsFeed._disabled

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_disabled_skips_http_call(self, mock_get, mock_sleep):
        """Once disabled, fetch should return [] without making HTTP calls."""
        StockTwitsFeed._disabled = True
        results = StockTwitsFeed().fetch("AAPL")

        assert results == []
        mock_get.assert_not_called()

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_success_resets_fail_count(self, mock_get, mock_sleep):
        """A successful fetch should reset _fail_count to 0."""
        fail_resp = self._mock_response([], status_code=403)
        ok_resp = self._mock_response([{"body": "ok"}])

        mock_get.return_value = fail_resp
        feed = StockTwitsFeed()
        feed.fetch("AAPL")  # fail 1
        feed.fetch("MSFT")  # fail 2
        assert StockTwitsFeed._fail_count == 2

        mock_get.return_value = ok_resp
        feed.fetch("NVDA")  # success → resets
        assert StockTwitsFeed._fail_count == 0
        assert not StockTwitsFeed._disabled

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_non_200_also_retries(self, mock_get, mock_sleep):
        """Any non-200 (e.g. 404) retries with backoff."""
        mock_get.return_value = self._mock_response([], status_code=404)
        StockTwitsFeed().fetch("BADTICKER")

        # 1 initial + 2 retries = 3 attempts
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("data.social_feed.time.sleep")
    @patch("data.social_feed.requests.get")
    def test_logs_warning_once_on_disable(self, mock_get, mock_sleep):
        """WARNING should be logged exactly once when disabling."""
        mock_get.return_value = self._mock_response([], status_code=403)
        feed = StockTwitsFeed()

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("A")
            feed.fetch("B")
            feed.fetch("C")  # triggers disable
            feed.fetch("D")  # already disabled — no HTTP, no log

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1
            assert "disabled for this session after 3 failures" in str(warning_calls[0])


# ===========================================================================
# yfinance crumb / 401 retry
# ===========================================================================

class TestYfinanceCrumbRetry:
    """Verify MarketData retries on crumb errors and clears cache."""

    @patch("data.market_data._clear_yf_cache")
    @patch("data.market_data.yf")
    def test_retries_on_401_error(self, mock_yf, mock_clear):
        from data.market_data import MarketData

        # First Ticker().info raises crumb error, second succeeds
        bad_ticker = MagicMock()
        bad_ticker.info.__getitem__ = MagicMock(side_effect=Exception("401 Unauthorized"))
        type(bad_ticker).info = property(
            lambda self: (_ for _ in ()).throw(Exception("401 Unauthorized"))
        )
        good_ticker = MagicMock()
        good_ticker.info = {"longName": "Apple", "currentPrice": 190.0, "currency": "USD"}
        mock_yf.Ticker.side_effect = [bad_ticker, good_ticker]

        md = MarketData()
        result = md.fetch("AAPL")

        assert result["price"] == 190.0
        mock_clear.assert_called_once()

    @patch("data.market_data._clear_yf_cache")
    @patch("data.market_data.yf")
    def test_fallback_when_retry_also_fails(self, mock_yf, mock_clear):
        from data.market_data import MarketData

        bad = MagicMock()
        type(bad).info = property(
            lambda self: (_ for _ in ()).throw(Exception("401 Unauthorized"))
        )
        mock_yf.Ticker.return_value = bad

        md = MarketData()
        result = md.fetch("AAPL")

        # Should return fallback instead of raising
        assert result["ticker"] == "AAPL"
        assert result["price"] is None
        assert result["name"] == "N/A"
        mock_clear.assert_called_once()

    @patch("data.market_data._clear_yf_cache")
    @patch("data.market_data.yf")
    def test_no_retry_on_non_crumb_error(self, mock_yf, mock_clear):
        from data.market_data import MarketData

        mock_yf.Ticker.side_effect = ValueError("Bad ticker")
        md = MarketData()

        with pytest.raises(ValueError, match="Bad ticker"):
            md.fetch("INVALID")

        mock_clear.assert_not_called()

    @patch("data.market_data._clear_yf_cache")
    @patch("data.market_data.yf")
    def test_succeeds_without_retry(self, mock_yf, mock_clear):
        from data.market_data import MarketData

        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Apple", "currentPrice": 190.0, "currency": "USD"}
        mock_yf.Ticker.return_value = mock_ticker

        md = MarketData()
        result = md.fetch("AAPL")

        assert result["price"] == 190.0
        mock_clear.assert_not_called()
