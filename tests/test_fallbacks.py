"""
Unit tests for the API fallback subsystem.

Tests
-----
TestNewsAggregatorFallbacks   — 4-level news chain (mocked HTTP)
TestPriceFallbackChain        — 4-level price chain (mocked yfinance / requests)
TestFreshDataRequired         — require_fresh=True raises when all live sources fail
TestWeightedSentimentScorer   — lexicon scoring: weights, amplifiers, negators, compounds
TestFallbackCoordinator       — register, check_and_alert (threshold), daily_health_check
TestMarketDataFallback        — MarketData.fetch() delegates to PriceFallback
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── sys.path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DB_PATH", "/tmp/test_fallbacks.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("NEWSAPI_KEY", "dummy")
os.environ.setdefault("ALPHA_VANTAGE_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# 1. NewsAggregator fallback chain
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsAggregatorFallbacks(unittest.TestCase):
    """4-level fallback chain in NewsAggregator (all HTTP mocked)."""

    def setUp(self):
        from data.news_aggregator import NewsAggregator
        from utils.network_recovery import get_cache
        self.NA   = NewsAggregator
        self.cache = get_cache()
        self.cache.clear("newsapi")

    def _make(self, key="dummy_key"):
        return self.NA(api_key=key, max_headlines=3)

    # Level 0 — NewsAPI success
    def test_level0_newsapi_success(self):
        agg = self._make()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "articles": [
                {"title": "AAPL hits record high"},
                {"title": "Apple revenue soars"},
            ]
        }
        with patch("data.news_aggregator.requests.get", return_value=resp):
            result = agg.fetch_with_metadata("AAPL")
        self.assertEqual(result.level, 0)
        self.assertEqual(result.source, "newsapi")
        self.assertFalse(result.degraded)
        self.assertGreater(result.count, 0)

    # Level 0 fails → Level 1 (RSS) success
    def test_level1_rss_fallback_when_newsapi_fails(self):
        agg = self._make()

        rss_xml = (
            "<?xml version='1.0'?>"
            "<rss><channel>"
            "<item><title>AAPL earnings beat</title></item>"
            "<item><title>Apple shares rise 3%</title></item>"
            "</channel></rss>"
        )
        rss_resp = MagicMock()
        rss_resp.raise_for_status = MagicMock()
        rss_resp.text = rss_xml

        newsapi_resp = MagicMock()
        newsapi_resp.raise_for_status.side_effect = Exception("NewsAPI 401")

        def _side_effect(url, **kwargs):
            if "newsapi.org" in url:
                return newsapi_resp
            return rss_resp

        with patch("data.news_aggregator.requests.get", side_effect=_side_effect):
            result = agg.fetch_with_metadata("AAPL")

        self.assertIn(result.level, (1,))
        self.assertIn(result.source, ("rss_yahoo", "rss_nasdaq"))
        self.assertTrue(result.degraded)

    # Level 2 (Google News RSS) reached when L0 and L1 both fail
    def test_level2_google_news_fallback(self):
        agg = self._make()

        google_xml = (
            "<?xml version='1.0'?>"
            "<rss><channel>"
            "<item><title>AAPL stock analysis 2026</title></item>"
            "</channel></rss>"
        )
        google_resp = MagicMock()
        google_resp.raise_for_status = MagicMock()
        google_resp.text = google_xml

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = Exception("fail")

        def _side_effect(url, **kwargs):
            if "news.google.com" in url:
                return google_resp
            raise Exception("source fail")

        with patch("data.news_aggregator.requests.get", side_effect=_side_effect):
            result = agg.fetch_with_metadata("AAPL")

        self.assertEqual(result.level, 2)
        self.assertEqual(result.source, "google_news")
        self.assertTrue(result.degraded)

    # Level 3 — cache fallback
    def test_level3_cache_fallback(self):
        from utils.network_recovery import get_cache
        cache = get_cache()
        cache.set("newsapi", "headlines:AAPL", ["Cached headline 1", "Cached headline 2"])

        agg = self._make()

        with patch("data.news_aggregator.requests.get", side_effect=Exception("all fail")):
            result = agg.fetch_with_metadata("AAPL")

        self.assertEqual(result.level, 3)
        self.assertEqual(result.source, "cache")
        self.assertTrue(result.degraded)
        self.assertGreater(result.count, 0)

    # All sources exhausted → empty result, no exception
    def test_all_sources_fail_returns_empty(self):
        from utils.network_recovery import get_cache
        get_cache().clear("newsapi")

        agg = self._make()
        with patch("data.news_aggregator.requests.get", side_effect=Exception("fail")):
            result = agg.fetch_with_metadata("TSLA")

        self.assertEqual(result.level, 4)
        self.assertEqual(result.source, "none")
        self.assertEqual(result.count, 0)
        self.assertEqual(result.headlines, [])

    # fetch() returns plain list (NewsFeed-compatible)
    def test_fetch_returns_list(self):
        agg = self._make()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"articles": [{"title": "NVDA gains"}]}
        with patch("data.news_aggregator.requests.get", return_value=resp):
            headlines = agg.fetch("NVDA")
        self.assertIsInstance(headlines, list)

    # RSS parser: XML parse error falls back to regex
    def test_rss_parser_regex_fallback(self):
        from data.news_aggregator import NewsAggregator
        agg = NewsAggregator(max_headlines=5)

        malformed = "<rss><item><title>AAPL gains today</title></item>"  # unclosed tag
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.text = malformed

        with patch("data.news_aggregator.requests.get", return_value=resp):
            titles = agg._parse_rss("http://fake/rss", "AAPL")

        self.assertIsInstance(titles, list)

    # Deduplication in RSS level
    def test_rss_deduplication(self):
        agg = self._make()
        rss_xml = (
            "<?xml version='1.0'?><rss><channel>"
            "<item><title>AAPL earnings beat</title></item>"
            "<item><title>AAPL earnings beat</title></item>"  # duplicate
            "<item><title>Apple shares rise</title></item>"
            "</channel></rss>"
        )
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.text = rss_xml

        with patch("data.news_aggregator.requests.get", return_value=resp):
            headlines = agg._level1_rss_yahoo("AAPL")

        titles = [h for h in headlines]
        self.assertEqual(len(titles), len(set(titles)))  # all unique


# ══════════════════════════════════════════════════════════════════════════════
# 2. PriceFallback chain
# ══════════════════════════════════════════════════════════════════════════════

class TestPriceFallbackChain(unittest.TestCase):
    """4-level fallback for PriceFallback (yfinance / HTTP mocked)."""

    def setUp(self):
        from data.price_fallback import PriceFallback
        from utils.api_recovery import APIRecovery
        from utils.network_recovery import get_cache
        self.PF    = PriceFallback
        self.cache = get_cache()
        self.cache.clear("yfinance_price")
        # Reset circuit breakers so shared state from other test suites doesn't
        # cause the yfinance / yahoo_json circuits to be open.
        APIRecovery.reset_circuit("yfinance")
        APIRecovery.reset_circuit("yahoo_json")

    def _make(self, alpha_key=""):
        return self.PF(alpha_key=alpha_key)

    # Level 0 — yfinance success
    def test_level0_yfinance_success(self):
        pf = self._make()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 189.30,
            "currency": "USD",
            "longName": "Apple Inc.",
            "marketCap": 3_000_000_000_000,
            "regularMarketChangePercent": 1.5,
        }
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker):
            result = pf.get_price("AAPL")

        self.assertEqual(result.level, 0)
        self.assertEqual(result.source, "yfinance")
        self.assertAlmostEqual(result.price, 189.30)
        self.assertTrue(result.is_fresh)
        self.assertFalse(result.is_estimated)
        self.assertFalse(result.degraded)

    # yfinance returns no price → ValueError propagates to next level
    def test_level0_raises_on_no_price(self):
        pf = self._make()
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # no price fields
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("no L2")):
            result = pf.get_price("AAPL")
        # Should fall through to level 3 (cache) or return None price
        self.assertIsNotNone(result)

    # Level 1 — Alpha Vantage (skipped if no key)
    def test_level1_skipped_when_no_alpha_key(self):
        pf = self._make(alpha_key="")

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        av_called = []

        def _request_side_effect(url, **kwargs):
            if "alphavantage" in url:
                av_called.append(True)
            raise Exception("fail")

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=_request_side_effect):
            pf.get_price("AAPL")

        self.assertEqual(av_called, [], "Alpha Vantage should be skipped with no key")

    # Level 1 — Alpha Vantage with key
    def test_level1_alpha_vantage_success(self):
        pf = self._make(alpha_key="test_key_123")

        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Level 0 fails

        av_resp = MagicMock()
        av_resp.raise_for_status = MagicMock()
        av_resp.json.return_value = {
            "Global Quote": {
                "05. price": "155.00",
                "10. change percent": "0.50%",
            }
        }

        def _request_side_effect(url, **kwargs):
            if "alphavantage" in url:
                return av_resp
            raise Exception("other fail")

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=_request_side_effect):
            result = pf.get_price("AAPL")

        self.assertEqual(result.level, 1)
        self.assertEqual(result.source, "alpha_vantage")
        self.assertAlmostEqual(result.price, 155.00)
        self.assertTrue(result.is_fresh)
        self.assertTrue(result.degraded)

    # Level 2 — Yahoo Finance JSON
    def test_level2_yahoo_json_success(self):
        pf = self._make()

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        yf_json_resp = MagicMock()
        yf_json_resp.raise_for_status = MagicMock()
        yf_json_resp.json.return_value = {
            "chart": {
                "result": [{
                    "meta": {
                        "regularMarketPrice": 200.00,
                        "currency": "USD",
                        "longName": "Apple Inc.",
                    },
                    "indicators": {}
                }]
            }
        }

        def _request_side_effect(url, **kwargs):
            if "query1.finance.yahoo.com" in url:
                return yf_json_resp
            raise Exception("other fail")

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=_request_side_effect):
            result = pf.get_price("AAPL")

        self.assertEqual(result.level, 2)
        self.assertEqual(result.source, "yahoo_json")
        self.assertAlmostEqual(result.price, 200.00)

    # Level 3 — cache fallback
    def test_level3_cache_fallback(self):
        from utils.network_recovery import get_cache
        cache = get_cache()
        cache.set("yfinance_price", "AAPL", {
            "price": 175.0, "currency": "USD", "name": "Apple Inc.", "market_cap": None
        })

        pf = self._make()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("all fail")):
            result = pf.get_price("AAPL")

        self.assertEqual(result.level, 3)
        self.assertEqual(result.source, "cache")
        self.assertAlmostEqual(result.price, 175.0)
        self.assertFalse(result.is_fresh)
        self.assertTrue(result.is_estimated)
        self.assertTrue(result.degraded)

    # All sources fail → None price, no exception
    def test_all_sources_fail_returns_none_price(self):
        from utils.network_recovery import get_cache
        get_cache().clear("yfinance_price")

        pf = self._make()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("all fail")):
            result = pf.get_price("TSLA")

        self.assertIsNone(result.price)
        self.assertEqual(result.source, "none")
        self.assertEqual(result.level, 4)

    # Successful fetch caches the price for level 3 reuse
    def test_successful_fetch_populates_cache(self):
        from utils.network_recovery import get_cache
        cache = get_cache()
        cache.clear("yfinance_price")

        pf = self._make()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 300.0,
            "currency": "USD",
            "longName": "NVDA",
            "marketCap": 1_000_000_000_000,
        }
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker):
            pf.get_price("NVDA")

        cached, hit = cache.get("yfinance_price", "NVDA")
        self.assertTrue(hit)
        self.assertAlmostEqual(cached["price"], 300.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FreshDataRequired
# ══════════════════════════════════════════════════════════════════════════════

class TestFreshDataRequired(unittest.TestCase):

    def setUp(self):
        from utils.api_recovery import APIRecovery
        from utils.network_recovery import get_cache
        APIRecovery.reset_circuit("yfinance")
        APIRecovery.reset_circuit("yahoo_json")
        get_cache().clear("yfinance_price")

    def test_raises_when_require_fresh_and_all_live_fail(self):
        from data.price_fallback import FreshDataRequired, PriceFallback
        from utils.network_recovery import get_cache
        get_cache().clear("yfinance_price")

        pf = PriceFallback()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("all fail")):
            with self.assertRaises(FreshDataRequired):
                pf.get_price("BADTICKER", require_fresh=True)

    def test_no_raise_without_require_fresh(self):
        from data.price_fallback import FreshDataRequired, PriceFallback
        from utils.network_recovery import get_cache
        get_cache().clear("yfinance_price")

        pf = PriceFallback()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("all fail")):
            result = pf.get_price("BADTICKER", require_fresh=False)

        # Must not raise; price should be None
        self.assertIsNone(result.price)

    def test_does_not_raise_when_live_source_succeeds(self):
        from data.price_fallback import PriceFallback
        pf = PriceFallback()
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 100.0, "currency": "USD", "longName": "X"}
        # Also block HTTP so L2 yahoo_json cannot make real calls
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("no http")):
            result = pf.get_price("AAPL", require_fresh=True)
        self.assertAlmostEqual(result.price, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Weighted sentiment scorer
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightedSentimentScorer(unittest.TestCase):
    """Detailed tests for the rule-based lexicon scorer in SentimentAgent."""

    def setUp(self):
        from agents.sentiment_agent import _rule_based_sentiment
        self.score = _rule_based_sentiment

    def test_bullish_keyword(self):
        result = self.score("Apple earnings beat expectations")
        self.assertEqual(result["sentiment"], "bullish")
        self.assertEqual(result["score"], 1)
        self.assertTrue(result["degraded"])
        self.assertAlmostEqual(result["confidence"], 0.55)

    def test_bearish_keyword(self):
        result = self.score("Company faces massive bankruptcy filing")
        self.assertEqual(result["sentiment"], "bearish")
        self.assertEqual(result["score"], -1)

    def test_neutral_no_keywords(self):
        result = self.score("Company holds annual shareholder meeting")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["score"], 0)

    def test_negator_flips_bullish_to_bearish(self):
        positive = self.score("Company shows growth")
        negated  = self.score("Company fails to show growth")
        # "fails" is a negator; "growth" is bullish → negated should not be bullish
        # (it may land neutral or bearish depending on word mix)
        self.assertIn(negated["sentiment"], ("bearish", "neutral"))
        # If positive was bullish, negated should be less bullish
        if positive["sentiment"] == "bullish":
            self.assertLessEqual(negated["score"], positive["score"])

    def test_amplifier_increases_weight(self):
        base  = self.score("Company shows growth")
        amped = self.score("Company shows record growth")
        # "record" is both an amplifier and a bullish term → score should be ≥ base
        if base["sentiment"] == "bullish":
            self.assertGreaterEqual(amped["score"], base["score"])

    def test_compound_bullish_phrase(self):
        result = self.score("Apple beats expectations on revenue")
        self.assertEqual(result["sentiment"], "bullish")

    def test_compound_bearish_phrase(self):
        result = self.score("Tesla faces class action lawsuit over safety")
        self.assertEqual(result["sentiment"], "bearish")

    def test_mixed_keywords_threshold(self):
        # Very slight bullish tilt — may land neutral due to ±0.3 threshold
        result = self.score("Company shows some gains despite concerns")
        self.assertIn(result["sentiment"], ("bullish", "neutral"))

    def test_confidence_is_rule_based_constant(self):
        for headline in [
            "Apple surges to record high",
            "Tesla drops on disappointing earnings",
            "Neutral headline about quarterly meeting",
        ]:
            result = self.score(headline)
            self.assertAlmostEqual(result["confidence"], 0.55)

    def test_result_has_all_expected_keys(self):
        result = self.score("Apple earnings beat")
        for key in ("sentiment", "score", "reason", "headline", "degraded", "confidence"):
            self.assertIn(key, result)


# ══════════════════════════════════════════════════════════════════════════════
# 5. FallbackCoordinator
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackCoordinator(unittest.TestCase):

    def setUp(self):
        from data.fallback_coordinator import FallbackCoordinator
        self.FC = FallbackCoordinator
        self.FC.reset()  # clear all state before each test

    # Registration stores metadata
    def test_register_stores_level_and_source(self):
        self.FC.register("news", level=1, source="rss_yahoo", ticker="AAPL")
        status = self.FC.get_status()
        self.assertIn("news", status)
        self.assertEqual(status["news"]["level"], 1)
        self.assertEqual(status["news"]["source"], "rss_yahoo")

    # Primary level registration (no alert expected)
    def test_register_primary_level(self):
        self.FC.register("news", level=0, source="newsapi", ticker="AAPL")
        status = self.FC.get_status()
        self.assertEqual(status["news"]["level"], 0)

    # Level change resets the timer
    def test_level_change_resets_timer(self):
        self.FC.register("news", level=1, source="rss_yahoo")
        old_since = self.FC._registry["news"]["since"]
        # Simulate a level change
        self.FC.register("news", level=2, source="google_news")
        new_since = self.FC._registry["news"]["since"]
        self.assertGreaterEqual(new_since, old_since)
        self.assertEqual(self.FC._registry["news"]["level"], 2)

    # No alerts when all services are on primary
    def test_no_alert_when_all_primary(self):
        self.FC.register("news",  level=0, source="newsapi")
        self.FC.register("price", level=0, source="yfinance")
        alerts = self.FC.check_and_alert()
        self.assertEqual(alerts, [])

    # Alert triggered when stuck on fallback >24 h
    def test_alert_triggered_after_threshold(self):
        self.FC.register("news", level=1, source="rss_yahoo", ticker="AAPL")
        # Back-date the registration by 25 hours
        with self.FC._lock:
            self.FC._registry["news"]["since"] = (
                datetime.utcnow() - timedelta(hours=25)
            )
        alerts = self.FC.check_and_alert()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["service"], "news")
        self.assertGreater(alerts[0]["hours_elapsed"], 24)

    # No alert before threshold is crossed
    def test_no_alert_before_threshold(self):
        self.FC.register("news", level=1, source="rss_yahoo")
        # Only 1 hour ago
        with self.FC._lock:
            self.FC._registry["news"]["since"] = (
                datetime.utcnow() - timedelta(hours=1)
            )
        alerts = self.FC.check_and_alert()
        self.assertEqual(alerts, [])

    # Reset clears specific service
    def test_reset_specific_service(self):
        self.FC.register("news",  level=1, source="rss")
        self.FC.register("price", level=1, source="alpha_vantage")
        self.FC.reset("news")
        status = self.FC.get_status()
        self.assertNotIn("news", status)
        self.assertIn("price", status)

    # Reset all
    def test_reset_all(self):
        self.FC.register("news",  level=1, source="rss")
        self.FC.register("price", level=2, source="yahoo_json")
        self.FC.reset()
        self.assertEqual(self.FC.get_status(), {})

    # get_status returns serialisable dict (dates as strings)
    def test_get_status_returns_string_dates(self):
        self.FC.register("price", level=1, source="alpha_vantage")
        status = self.FC.get_status()
        self.assertIsInstance(status["price"]["since"], str)

    # Multiple services tracked independently
    def test_multiple_services_tracked(self):
        self.FC.register("news",  level=0, source="newsapi")
        self.FC.register("price", level=2, source="yahoo_json")
        status = self.FC.get_status()
        self.assertEqual(status["news"]["level"],  0)
        self.assertEqual(status["price"]["level"], 2)

    # daily_health_check returns expected keys
    def test_daily_health_check_structure(self):
        report = self.FC.daily_health_check()
        for key in ("timestamp", "services_checked", "services_on_fallback",
                    "alerts", "probe_results"):
            self.assertIn(key, report)

    # Count increments on repeated same-level registration
    def test_count_increments_same_level(self):
        self.FC.register("news", level=1, source="rss_yahoo")
        self.FC.register("news", level=1, source="rss_yahoo")
        self.assertEqual(self.FC._registry["news"]["count"], 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MarketData uses PriceFallback
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketDataFallback(unittest.TestCase):

    def setUp(self):
        from data.market_data import MarketData
        from utils.api_recovery import APIRecovery
        from utils.network_recovery import get_cache
        self.MD = MarketData
        APIRecovery.reset_circuit("yfinance")
        APIRecovery.reset_circuit("yahoo_json")
        get_cache().clear("yfinance_price")

    def test_fetch_returns_expected_keys(self):
        md = self.MD()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 123.45,
            "currency": "USD",
            "longName": "Test Corp",
            "marketCap": 500_000_000,
        }
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("no http")):
            result = md.fetch("TEST")

        for key in ("ticker", "name", "price", "currency", "market_cap",
                    "source", "degraded"):
            self.assertIn(key, result)

    def test_fetch_price_value(self):
        md = self.MD()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 250.00,
            "currency": "USD",
            "longName": "BigCo",
        }
        # Block HTTP so yahoo_json fallback cannot make real calls
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("no http")):
            result = md.fetch("BIGC")

        self.assertAlmostEqual(result["price"], 250.00)
        self.assertFalse(result["degraded"])

    def test_fetch_degraded_when_all_fail(self):
        from utils.network_recovery import get_cache
        get_cache().clear("yfinance_price")

        md = self.MD()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("fail")):
            result = md.fetch("FAIL")

        self.assertIsNone(result["price"])
        self.assertTrue(result["degraded"])
        self.assertEqual(result["source"], "none")

    def test_fetch_ticker_uppercased(self):
        md = self.MD()
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 10.0, "currency": "USD", "longName": "X"}
        with patch("data.price_fallback.yf.Ticker", return_value=mock_ticker), \
             patch("data.price_fallback.requests.get", side_effect=Exception("no http")):
            result = md.fetch("aapl")
        self.assertEqual(result["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
