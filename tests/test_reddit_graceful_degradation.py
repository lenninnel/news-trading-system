"""
Tests for Reddit graceful degradation.

Verifies that:
  - Missing Reddit credentials cause no errors anywhere in the pipeline
  - The system still produces valid sentiment scores using NewsAPI + Marketaux
  - The warning "Reddit credentials not configured, skipping Reddit sentiment"
    is logged exactly once (not on subsequent calls)
  - NewsAPI and Marketaux weights are increased when Reddit is unavailable

Run with:
    python3 -m pytest tests/test_reddit_graceful_degradation.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import pytest

from config.settings import SOURCE_WEIGHTS, SOURCE_WEIGHTS_NO_REDDIT
from data.social_feed import RedditFeed, _reddit_warned, is_reddit_configured
from orchestrator.coordinator import Coordinator


# ===========================================================================
# is_reddit_configured() helper
# ===========================================================================

class TestIsRedditConfigured:
    """Verify the is_reddit_configured() helper reads env settings."""

    @patch("data.social_feed.REDDIT_CLIENT_ID", "my_id")
    @patch("data.social_feed.REDDIT_CLIENT_SECRET", "my_secret")
    def test_returns_true_when_both_set(self):
        assert is_reddit_configured() is True

    @patch("data.social_feed.REDDIT_CLIENT_ID", "")
    @patch("data.social_feed.REDDIT_CLIENT_SECRET", "my_secret")
    def test_returns_false_when_id_missing(self):
        assert is_reddit_configured() is False

    @patch("data.social_feed.REDDIT_CLIENT_ID", "my_id")
    @patch("data.social_feed.REDDIT_CLIENT_SECRET", "")
    def test_returns_false_when_secret_missing(self):
        assert is_reddit_configured() is False

    @patch("data.social_feed.REDDIT_CLIENT_ID", "")
    @patch("data.social_feed.REDDIT_CLIENT_SECRET", "")
    def test_returns_false_when_both_missing(self):
        assert is_reddit_configured() is False


# ===========================================================================
# No errors when Reddit credentials are missing
# ===========================================================================

class TestNoErrorsWithoutReddit:
    """When Reddit creds are missing, the entire pipeline must not raise."""

    def setup_method(self):
        _reddit_warned["credentials"] = False
        _reddit_warned["praw"] = False

    def test_reddit_feed_no_error_on_fetch(self):
        """RedditFeed.fetch() returns [] without raising when creds missing."""
        feed = RedditFeed(client_id="", client_secret="")
        result = feed.fetch("AAPL")
        assert result == []

    def test_reddit_feed_no_error_multiple_tickers(self):
        """Multiple tickers never raise, always return []."""
        feed = RedditFeed(client_id="", client_secret="")
        for ticker in ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL"]:
            result = feed.fetch(ticker)
            assert result == [], f"Expected [] for {ticker}, got {result}"

    def test_reddit_feed_no_error_no_client_secret(self):
        """Only client_id set, no client_secret: no error."""
        feed = RedditFeed(client_id="some_id", client_secret="")
        result = feed.fetch("AAPL")
        assert result == []

    def test_reddit_feed_no_error_no_client_id(self):
        """Only client_secret set, no client_id: no error."""
        feed = RedditFeed(client_id="", client_secret="some_secret")
        result = feed.fetch("AAPL")
        assert result == []


# ===========================================================================
# Warning logged exactly once
# ===========================================================================

class TestWarningLoggedOnce:
    """The startup warning must fire exactly once, then go silent."""

    def setup_method(self):
        _reddit_warned["credentials"] = False
        _reddit_warned["praw"] = False

    def test_warning_logged_exactly_once(self):
        """First call logs the warning; calls 2..N are completely silent."""
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")
            feed.fetch("TSLA")
            feed.fetch("NVDA")
            feed.fetch("MSFT")
            feed.fetch("AMZN")

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1, (
                f"Expected exactly 1 warning, got {len(warning_calls)}: {warning_calls}"
            )

    def test_warning_message_text(self):
        """The warning must contain the canonical message."""
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            feed.fetch("AAPL")

            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 1
            msg = str(warning_calls[0])
            assert "Reddit credentials not configured" in msg
            assert "skipping Reddit sentiment" in msg

    def test_no_error_or_info_logs(self):
        """No error or info logs should be emitted for missing creds."""
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            for _ in range(10):
                feed.fetch("AAPL")

            mock_logger.error.assert_not_called()
            # Only one warning, no info
            assert mock_logger.info.call_args_list == []

    def test_already_warned_flag_suppresses_all(self):
        """If the warned flag is already set, zero warnings are emitted."""
        _reddit_warned["credentials"] = True
        feed = RedditFeed(client_id="", client_secret="")

        with patch("data.social_feed.logger") as mock_logger:
            for _ in range(5):
                feed.fetch("AAPL")

            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()
            mock_logger.info.assert_not_called()


# ===========================================================================
# Valid sentiment scores without Reddit
# ===========================================================================

class TestSentimentWithoutReddit:
    """When Reddit is unavailable, NewsAPI + Marketaux alone produce valid scores."""

    def test_newsapi_only_produces_valid_score(self):
        """Scored items from NewsAPI only should produce a valid average."""
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": -1, "source": "newsapi"},
            {"score": 1, "source": "newsapi"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_REDDIT)
        # (1*1.2 + -1*1.2 + 1*1.2) / (1.2*3) = 1.2/3.6 = 0.333
        assert isinstance(avg, float)
        assert -1.0 <= avg <= 1.0
        assert abs(avg - (1 / 3)) < 0.001

    def test_newsapi_and_marketaux_produce_valid_score(self):
        """Mixed NewsAPI + Marketaux items produce a valid weighted average."""
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": 1, "source": "marketaux"},
            {"score": -1, "source": "marketaux"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_REDDIT)
        # (1*1.2 + 1*1.1 + -1*1.1) / (1.2 + 1.1 + 1.1) = 1.2 / 3.4
        expected = 1.2 / 3.4
        assert isinstance(avg, float)
        assert -1.0 <= avg <= 1.0
        assert abs(avg - expected) < 0.001

    def test_no_reddit_weights_boost_newsapi_and_marketaux(self):
        """SOURCE_WEIGHTS_NO_REDDIT gives newsapi and marketaux higher weights."""
        assert SOURCE_WEIGHTS_NO_REDDIT["newsapi"] > SOURCE_WEIGHTS["newsapi"]
        assert SOURCE_WEIGHTS_NO_REDDIT["marketaux"] > SOURCE_WEIGHTS["marketaux"]

    def test_no_reddit_weights_exclude_reddit(self):
        """SOURCE_WEIGHTS_NO_REDDIT should not contain a 'reddit' key."""
        assert "reddit" not in SOURCE_WEIGHTS_NO_REDDIT

    def test_empty_scored_returns_zero(self):
        """Empty scored list returns 0.0 regardless of weight map."""
        assert Coordinator._weighted_aggregate([], SOURCE_WEIGHTS_NO_REDDIT) == 0.0

    def test_all_bullish_returns_one(self):
        """All bullish scores should produce avg close to 1.0."""
        scored = [
            {"score": 1, "source": "newsapi"},
            {"score": 1, "source": "marketaux"},
            {"score": 1, "source": "stocktwits"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_REDDIT)
        assert abs(avg - 1.0) < 0.001

    def test_all_bearish_returns_neg_one(self):
        """All bearish scores should produce avg close to -1.0."""
        scored = [
            {"score": -1, "source": "newsapi"},
            {"score": -1, "source": "marketaux"},
            {"score": -1, "source": "stocktwits"},
        ]
        avg = Coordinator._weighted_aggregate(scored, SOURCE_WEIGHTS_NO_REDDIT)
        assert abs(avg - (-1.0)) < 0.001


# ===========================================================================
# Active weights selection
# ===========================================================================

class TestActiveWeights:
    """Coordinator._active_weights() selects the right weight map."""

    @patch("orchestrator.coordinator.is_reddit_configured", return_value=True)
    def test_returns_standard_weights_when_reddit_available(self, _mock):
        weights = Coordinator._active_weights()
        assert weights is SOURCE_WEIGHTS

    @patch("orchestrator.coordinator.is_reddit_configured", return_value=False)
    def test_returns_no_reddit_weights_when_reddit_missing(self, _mock):
        weights = Coordinator._active_weights()
        assert weights is SOURCE_WEIGHTS_NO_REDDIT

    @patch("orchestrator.coordinator.is_reddit_configured", return_value=False)
    def test_no_reddit_weights_newsapi_higher(self, _mock):
        weights = Coordinator._active_weights()
        assert weights["newsapi"] > SOURCE_WEIGHTS["newsapi"]

    @patch("orchestrator.coordinator.is_reddit_configured", return_value=False)
    def test_no_reddit_weights_marketaux_higher(self, _mock):
        weights = Coordinator._active_weights()
        assert weights["marketaux"] > SOURCE_WEIGHTS["marketaux"]
