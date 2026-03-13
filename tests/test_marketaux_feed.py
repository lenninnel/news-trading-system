"""
Unit tests for the Marketaux news feed module.

All tests run offline using mocked HTTP responses — no network access needed.

Run with:
    python3 -m pytest tests/test_marketaux_feed.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import pytest
import requests

from data.marketaux_feed import MarketauxFeed, clear_cache


# ===========================================================================
# Helper: build a mock Marketaux API response
# ===========================================================================

def _make_api_response(articles, status_code=200):
    """Return a MagicMock mimicking ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": articles}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ===========================================================================
# Happy path: parsing and field mapping
# ===========================================================================

class TestMarketauxParsing:
    """Verify MarketauxFeed correctly parses the API response."""

    def setup_method(self):
        clear_cache()

    @patch("data.marketaux_feed.requests.get")
    def test_parses_title_and_sentiment(self, mock_get):
        mock_get.return_value = _make_api_response([
            {
                "title": "Apple beats earnings expectations",
                "description": "Revenue surpassed analyst estimates.",
                "entities": [
                    {"symbol": "AAPL", "sentiment_score": 0.65},
                ],
            },
            {
                "title": "Tech stocks rally",
                "description": "Broad market move.",
                "entities": [
                    {"symbol": "AAPL", "sentiment_score": -0.3},
                    {"symbol": "MSFT", "sentiment_score": 0.4},
                ],
            },
        ])

        feed = MarketauxFeed(api_token="test-token")
        results = feed.fetch("AAPL")

        assert len(results) == 2
        assert results[0]["text"] == "Apple beats earnings expectations"
        assert results[0]["source"] == "marketaux"
        assert results[0]["marketaux_sentiment"] == 0.65
        assert results[1]["marketaux_sentiment"] == -0.3

    @patch("data.marketaux_feed.requests.get")
    def test_returns_correct_dict_keys(self, mock_get):
        mock_get.return_value = _make_api_response([
            {
                "title": "Headline",
                "entities": [{"symbol": "TSLA", "sentiment_score": 0.1}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("TSLA")

        assert len(results) == 1
        item = results[0]
        assert set(item.keys()) == {"text", "source", "marketaux_sentiment"}

    @patch("data.marketaux_feed.requests.get")
    def test_skips_articles_without_title(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "", "entities": []},
            {"title": None, "entities": []},
            {"description": "no title key"},
            {"title": "Valid headline", "entities": []},
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["text"] == "Valid headline"

    @patch("data.marketaux_feed.requests.get")
    def test_ticker_case_insensitive(self, mock_get):
        """fetch("aapl") should match entities with symbol "AAPL"."""
        mock_get.return_value = _make_api_response([
            {
                "title": "Apple news",
                "entities": [{"symbol": "AAPL", "sentiment_score": 0.5}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("aapl")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] == 0.5

    @patch("data.marketaux_feed.requests.get")
    def test_respects_max_headlines(self, mock_get):
        articles = [
            {"title": f"Article {i}", "entities": []}
            for i in range(20)
        ]
        mock_get.return_value = _make_api_response(articles)

        results = MarketauxFeed(api_token="tok", max_headlines=5).fetch("AAPL")

        assert len(results) == 5


# ===========================================================================
# Caching behaviour
# ===========================================================================

class TestMarketauxCaching:
    """Verify the 4-hour TTL cache avoids redundant API calls."""

    def setup_method(self):
        clear_cache()

    @patch("data.marketaux_feed.requests.get")
    def test_second_call_uses_cache(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "Cached headline", "entities": []},
        ])

        feed = MarketauxFeed(api_token="tok")
        first = feed.fetch("AAPL")
        second = feed.fetch("AAPL")

        assert first == second
        assert mock_get.call_count == 1  # only one HTTP call

    @patch("data.marketaux_feed.requests.get")
    def test_different_tickers_are_cached_separately(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "News", "entities": []},
        ])

        feed = MarketauxFeed(api_token="tok")
        feed.fetch("AAPL")
        feed.fetch("MSFT")

        assert mock_get.call_count == 2

    @patch("data.marketaux_feed.time.time")
    @patch("data.marketaux_feed.requests.get")
    def test_expired_cache_triggers_new_fetch(self, mock_get, mock_time):
        mock_get.return_value = _make_api_response([
            {"title": "Fresh data", "entities": []},
        ])

        # First call at t=0
        mock_time.return_value = 0.0
        feed = MarketauxFeed(api_token="tok")
        feed.fetch("AAPL")

        # Second call at t=5 hours (past 4-hour TTL)
        mock_time.return_value = 5 * 60 * 60
        feed.fetch("AAPL")

        assert mock_get.call_count == 2

    @patch("data.marketaux_feed.requests.get")
    def test_clear_cache_forces_refetch(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "Data", "entities": []},
        ])

        feed = MarketauxFeed(api_token="tok")
        feed.fetch("AAPL")
        clear_cache()
        feed.fetch("AAPL")

        assert mock_get.call_count == 2


# ===========================================================================
# Empty API token — silent skip
# ===========================================================================

class TestMarketauxNoToken:
    """MarketauxFeed should return [] silently when the token is empty."""

    def setup_method(self):
        clear_cache()

    @patch("data.marketaux_feed.requests.get")
    def test_empty_token_returns_empty(self, mock_get):
        feed = MarketauxFeed(api_token="")
        results = feed.fetch("AAPL")

        assert results == []
        mock_get.assert_not_called()

    @patch("data.marketaux_feed.requests.get")
    def test_none_token_returns_empty(self, mock_get):
        feed = MarketauxFeed(api_token=None)
        results = feed.fetch("AAPL")

        assert results == []
        mock_get.assert_not_called()


# ===========================================================================
# API error handling — never raise
# ===========================================================================

class TestMarketauxErrorHandling:
    """MarketauxFeed should return [] on any API failure, never raise."""

    def setup_method(self):
        clear_cache()

    @patch("data.marketaux_feed.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_get.return_value = _make_api_response([], status_code=500)

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []

    @patch("data.marketaux_feed.requests.get")
    def test_rate_limit_429_returns_empty(self, mock_get):
        mock_get.return_value = _make_api_response([], status_code=429)

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []

    @patch("data.marketaux_feed.requests.get")
    def test_network_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []

    @patch("data.marketaux_feed.requests.get")
    def test_timeout_returns_empty(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []

    @patch("data.marketaux_feed.requests.get")
    def test_malformed_json_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []

    @patch("data.marketaux_feed.requests.get")
    def test_missing_data_key_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {}  # no "data" key
        mock_get.return_value = resp

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert results == []


# ===========================================================================
# Graceful handling of missing / malformed entities
# ===========================================================================

class TestMarketauxEntities:
    """Sentiment extraction should be resilient to missing/malformed entities."""

    def setup_method(self):
        clear_cache()

    @patch("data.marketaux_feed.requests.get")
    def test_missing_entities_key(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "No entities field"},
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_entities_is_none(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "Null entities", "entities": None},
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_entities_is_empty_list(self, mock_get):
        mock_get.return_value = _make_api_response([
            {"title": "Empty entities", "entities": []},
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_entity_without_matching_symbol(self, mock_get):
        mock_get.return_value = _make_api_response([
            {
                "title": "Other ticker only",
                "entities": [{"symbol": "MSFT", "sentiment_score": 0.8}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_entity_without_sentiment_score(self, mock_get):
        mock_get.return_value = _make_api_response([
            {
                "title": "No score",
                "entities": [{"symbol": "AAPL"}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_entities_is_dict_not_list(self, mock_get):
        """API should return a list, but if it returns a dict, handle gracefully."""
        mock_get.return_value = _make_api_response([
            {
                "title": "Weird entities",
                "entities": {"symbol": "AAPL", "sentiment_score": 0.5},
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None

    @patch("data.marketaux_feed.requests.get")
    def test_sentiment_score_is_string_number(self, mock_get):
        """API might return sentiment_score as a string — should convert."""
        mock_get.return_value = _make_api_response([
            {
                "title": "String score",
                "entities": [{"symbol": "AAPL", "sentiment_score": "0.42"}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] == pytest.approx(0.42)

    @patch("data.marketaux_feed.requests.get")
    def test_sentiment_score_non_numeric_string(self, mock_get):
        """Non-numeric string sentiment_score should become None."""
        mock_get.return_value = _make_api_response([
            {
                "title": "Bad score",
                "entities": [{"symbol": "AAPL", "sentiment_score": "positive"}],
            },
        ])

        results = MarketauxFeed(api_token="tok").fetch("AAPL")

        assert len(results) == 1
        assert results[0]["marketaux_sentiment"] is None
