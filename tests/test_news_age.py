"""News age (2026-09-03): the publication timestamp of every processed
headline survives from provider → item → scored → DB, and the age at
decision time is persisted on the signal.

Before this change ``NewsFeed._live_fetch`` returned bare titles and
every downstream consumer implicitly treated each headline as "now".
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.timeparse import age_minutes, normalise_published, parse_utc, to_iso_utc  # noqa: E402


# ── utils.timeparse ──────────────────────────────────────────────────────

class TestParseUtc:

    @pytest.mark.parametrize("raw,expected", [
        ("2026-09-02T13:05:00Z", "2026-09-02T13:05:00+00:00"),            # NewsAPI
        ("2026-09-02T13:05:00.000000Z", "2026-09-02T13:05:00+00:00"),     # Marketaux
        ("2026-09-02T09:05:00-04:00", "2026-09-02T13:05:00+00:00"),       # EODHD offset
        ("2026-09-02T13:05:00", "2026-09-02T13:05:00+00:00"),             # naive → UTC
        (1788440700, "2026-09-03T13:05:00+00:00"),                        # Reddit epoch
        ("Wed, 02 Sep 2026 13:05:00 GMT", "2026-09-02T13:05:00+00:00"),   # RSS pubDate
    ])
    def test_parses_provider_formats_to_utc(self, raw, expected):
        assert normalise_published(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "yesterday", "N/A", True, 0, -5, [], {}])
    def test_absent_or_garbage_is_none_not_now(self, raw):
        assert parse_utc(raw) is None
        assert normalise_published(raw) is None

    def test_aware_datetime_passthrough(self):
        dt = datetime(2026, 9, 2, 15, 0, tzinfo=timezone(timedelta(hours=2)))
        assert to_iso_utc(parse_utc(dt)) == "2026-09-02T13:00:00+00:00"

    def test_age_minutes(self):
        now = datetime(2026, 9, 2, 14, 30, tzinfo=timezone.utc)
        assert age_minutes("2026-09-02T13:05:00Z", now) == 85.0
        assert age_minutes(None, now) is None
        assert age_minutes("garbage", now) is None
        # future-dated (provider clock skew) stays visible as negative
        assert age_minutes("2026-09-02T14:40:00Z", now) == -10.0


# ── NewsFeed keeps publishedAt ───────────────────────────────────────────

def _resp(articles):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"articles": articles}
    return resp


class TestNewsFeedTimestamps:

    def _feed(self):
        from data.news_feed import NewsFeed
        return NewsFeed(api_key="test-key", max_headlines=5)

    def test_live_fetch_keeps_published_at_and_flags_missing(self, caplog):
        import logging
        feed = self._feed()
        caplog.set_level(logging.WARNING, logger="data.news_feed")
        articles = [
            {"title": "AAPL beats", "publishedAt": "2026-09-02T13:05:00Z"},
            {"title": "AAPL no timestamp"},
            {"title": "AAPL bad timestamp", "publishedAt": "not-a-date"},
            {"title": "", "publishedAt": "2026-09-02T13:00:00Z"},  # dropped: no title
        ]
        with patch("data.news_feed.requests.get", return_value=_resp(articles)):
            out = feed._live_fetch("AAPL")
        assert [a["text"] for a in out] == ["AAPL beats", "AAPL no timestamp", "AAPL bad timestamp"]
        assert out[0]["published_at"] == "2026-09-02T13:05:00+00:00"
        assert out[1]["published_at"] is None
        assert out[2]["published_at"] is None
        assert all(a["source"] == "newsapi" for a in out)
        assert all(parse_utc(a["fetched_at"]) is not None for a in out)
        assert "2/3 article(s) for AAPL without a usable publishedAt" in caplog.text

    def test_fetch_articles_live_and_both_cache_forms(self):
        from data.news_feed import _newsapi_cache
        from utils.network_recovery import get_cache
        feed = self._feed()
        articles = [{"title": "AAPL beats", "publishedAt": "2026-09-02T13:05:00Z"}]
        with patch("data.news_feed.requests.get", return_value=_resp(articles)), \
             patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False):
            out = feed.fetch_articles("AAPL")
        assert out[0]["published_at"] == "2026-09-02T13:05:00+00:00"
        # structured + legacy entries in both caches
        for cache in (_newsapi_cache, get_cache()):
            structured, hit = cache.get("newsapi", "articles:AAPL")
            assert hit and structured[0]["published_at"] == "2026-09-02T13:05:00+00:00"
            legacy, hit = cache.get("newsapi", "headlines:AAPL")
            assert hit and legacy == ["AAPL beats"]
        # fetch() is titles-only over the same data
        with patch("data.news_feed.NetworkMonitor.check_and_update"):
            assert feed.fetch("AAPL") == ["AAPL beats"]

    def test_legacy_string_cache_yields_unknown_age(self):
        from data.news_feed import _newsapi_cache
        feed = self._feed()
        _newsapi_cache.set("newsapi", "headlines:AAPL", ["Cached AAPL headline"])
        with patch("data.news_feed.NetworkMonitor.check_and_update"):
            out = feed.fetch_articles("AAPL")
        assert out == [{"text": "Cached AAPL headline", "source": "newsapi",
                        "published_at": None, "fetched_at": None}]
        with patch("data.news_feed.NetworkMonitor.check_and_update"):
            assert feed.fetch("AAPL") == ["Cached AAPL headline"]

    def test_patched_call_returning_titles_still_works(self):
        feed = self._feed()
        with patch("data.news_feed.NetworkMonitor.check_and_update"), \
             patch("data.news_feed.NetworkMonitor.is_degraded", return_value=False), \
             patch("data.news_feed.APIRecovery.call", return_value=["AAPL hits record high"]):
            out = feed.fetch_articles("AAPL")
        assert out[0]["text"] == "AAPL hits record high"
        assert out[0]["published_at"] is None


# ── Other providers ──────────────────────────────────────────────────────

class TestOtherProviders:

    def test_marketaux_parse_keeps_published_at(self):
        from data.marketaux_feed import MarketauxFeed
        feed = MarketauxFeed(api_token="t", eodhd_feed=MagicMock())
        out = feed._parse_articles([
            {"title": "XOM up", "published_at": "2026-09-02T13:05:00.000000Z", "entities": []},
            {"title": "XOM flat", "entities": []},
        ], "XOM")
        assert out[0]["published_at"] == "2026-09-02T13:05:00+00:00"
        assert out[1]["published_at"] is None

    def test_eodhd_enrichment_keeps_date(self):
        from data.marketaux_feed import MarketauxFeed
        eodhd = MagicMock()
        eodhd.get_news.return_value = [
            {"title": "CVX news", "date": "2026-09-02T09:00:00+00:00", "url": ""},
            {"title": "CVX undated", "date": "", "url": ""},
        ]
        feed = MarketauxFeed(api_token="t", eodhd_feed=eodhd)
        out = feed._enrich_with_eodhd("CVX", [])
        assert out[0]["published_at"] == "2026-09-02T09:00:00+00:00"
        assert out[1]["published_at"] is None

    def test_stocktwits_parse_keeps_created_at(self):
        from data.social_feed import StockTwitsFeed
        resp = MagicMock()
        resp.json.return_value = {"messages": [
            {"body": "TSLA to the moon", "created_at": "2026-09-02T13:05:00Z", "entities": {}},
            {"body": "TSLA ???", "entities": {}},
        ]}
        out = StockTwitsFeed()._parse_response(resp)
        assert out[0]["published_at"] == "2026-09-02T13:05:00+00:00"
        assert out[1]["published_at"] is None


# ── Coordinator: items → scored → signal ─────────────────────────────────

class TestCoordinatorNewsTiming:

    def _coord(self):
        from orchestrator.coordinator import Coordinator
        return Coordinator.__new__(Coordinator)

    def test_news_timing_newest_and_missing(self):
        from orchestrator.coordinator import Coordinator
        now = datetime(2026, 9, 2, 14, 30, tzinfo=timezone.utc)
        scored = [
            {"headline": "a", "published_at": "2026-09-02T13:05:00+00:00"},
            {"headline": "b", "published_at": "2026-09-02T11:00:00+00:00"},
            {"headline": "c", "published_at": None},
            {"headline": "d"},
        ]
        t = Coordinator._news_timing(scored, now)
        assert t == {
            "news_newest_published_at": "2026-09-02T13:05:00+00:00",
            "news_age_minutes": 85.0,
            "news_ts_missing": 2,
        }

    def test_news_timing_all_missing_is_null_not_zero(self):
        from orchestrator.coordinator import Coordinator
        t = Coordinator._news_timing([{"headline": "a"}, {"headline": "b"}])
        assert t["news_newest_published_at"] is None
        assert t["news_age_minutes"] is None
        assert t["news_ts_missing"] == 2
        assert Coordinator._news_timing([]) == {
            "news_newest_published_at": None, "news_age_minutes": None, "news_ts_missing": 0,
        }

    def test_fetch_newsapi_items_uses_fetch_articles(self):
        c = self._coord()

        class _Feed:
            def fetch(self, t):
                raise AssertionError("fetch() must not be used when fetch_articles exists")

            def fetch_articles(self, t):
                return [{"text": "h1", "source": "newsapi",
                         "published_at": "2026-09-02T13:05:00+00:00", "fetched_at": "x"},
                        {"text": "", "published_at": None}]

        c.news_feed = _Feed()
        assert c._fetch_newsapi_items("AAPL") == [
            {"text": "h1", "source": "newsapi", "published_at": "2026-09-02T13:05:00+00:00"},
        ]

    def test_fetch_newsapi_items_legacy_double(self):
        c = self._coord()
        c.news_feed = MagicMock()
        c.news_feed.fetch.return_value = ["h1", "h2"]
        assert c._fetch_newsapi_items("AAPL") == [
            {"text": "h1", "source": "newsapi", "published_at": None},
            {"text": "h2", "source": "newsapi", "published_at": None},
        ]

    def test_build_news_data_carries_timing(self):
        from orchestrator.coordinator import Coordinator
        nd = Coordinator._build_news_data({
            "avg_score": 0.6, "signal": "BUY",
            "scored": [{"published_at": "2026-09-02T13:05:00+00:00"}, {"published_at": None}],
        })
        assert nd["news_score"] == pytest.approx(0.8)
        assert nd["headline_count"] == 2
        assert nd["news_newest_published_at"] == "2026-09-02T13:05:00+00:00"
        assert nd["news_ts_missing"] == 1
        assert isinstance(nd["news_age_minutes"], float)

    def test_news_catalyst_indicators_pass_through_without_logic_change(self):
        import pandas as pd
        from strategies.news_catalyst import NewsCatalystStrategy
        bars = pd.DataFrame({
            "Close": [100.0] * 24 + [101.0],
            "Volume": [1_000_000] * 25,
        })
        base = {"news_score": 0.9, "headline_count": 3, "sentiment_direction": "BUY"}
        plain = NewsCatalystStrategy().analyze("AAPL", bars, "BUY", news_data=dict(base))
        timed = NewsCatalystStrategy().analyze("AAPL", bars, "BUY", news_data={
            **base, "news_newest_published_at": "2026-09-02T13:05:00+00:00",
            "news_age_minutes": 85.0, "news_ts_missing": 1,
        })
        assert (timed.signal, timed.confidence) == (plain.signal, plain.confidence)
        assert timed.indicators["news_age_minutes"] == 85.0
        assert timed.indicators["news_newest_published_at"] == "2026-09-02T13:05:00+00:00"
        assert timed.indicators["news_ts_missing"] == 1
        assert plain.indicators["news_age_minutes"] is None


# ── Persistence ──────────────────────────────────────────────────────────

class TestPersistence:

    def test_headline_scores_published_at(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "n.db"))
        run_id = db.log_run("AAPL", 2, 2, 0.5, "BUY")
        db.log_headline_score(run_id, "h1", "bullish", 1, "r", "newsapi",
                              published_at="2026-09-02T13:05:00+00:00")
        db.log_headline_score(run_id, "h2", "neutral", 0, "r", "reddit")
        with sqlite3.connect(db.db_path) as c:
            rows = c.execute(
                "SELECT headline, published_at FROM headline_scores ORDER BY id"
            ).fetchall()
        assert rows == [("h1", "2026-09-02T13:05:00+00:00"), ("h2", None)]

    def test_signal_events_news_age_columns(self, tmp_path):
        from analytics.signal_logger import SignalLogger
        from storage.database import Database
        db = Database(str(tmp_path / "n.db"))
        logger = SignalLogger(db)
        SignalLogger(db)  # idempotent migration
        logger.log({
            "ticker": "AAPL", "signal": "BUY", "strategy": "NewsCatalyst",
            "news_newest_published_at": "2026-09-02T13:05:00+00:00",
            "news_age_minutes": 85.0, "news_ts_missing": 1,
        })
        logger.log({"ticker": "MSFT", "signal": "HOLD", "strategy": "Momentum"})
        with sqlite3.connect(db.db_path) as c:
            rows = c.execute(
                "SELECT ticker, news_newest_published_at, news_age_minutes, news_ts_missing "
                "FROM signal_events ORDER BY id"
            ).fetchall()
        assert rows == [
            ("AAPL", "2026-09-02T13:05:00+00:00", 85.0, 1),
            ("MSFT", None, None, None),
        ]

    def test_strategy_result_logging_writes_news_age(self, tmp_path):
        from analytics.signal_logger import SignalLogger
        from orchestrator.coordinator import Coordinator
        from storage.database import Database
        from strategies.base import StrategyResult
        db = Database(str(tmp_path / "n.db"))
        c = Coordinator.__new__(Coordinator)
        c.signal_logger = SignalLogger(db)
        res = StrategyResult(
            signal="WEAK BUY", confidence=40.0, strategy_name="NewsCatalyst",
            indicators={"price": 100.0, "news_score": 0.8,
                        "news_newest_published_at": "2026-09-02T13:05:00+00:00",
                        "news_age_minutes": 85.0, "news_ts_missing": 0},
        )
        c._log_strategy_result("AAPL", res, session="US_PRE")
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT strategy, news_age_minutes, news_newest_published_at, news_ts_missing "
                "FROM signal_events"
            ).fetchone()
        assert row == ("NewsCatalyst", 85.0, "2026-09-02T13:05:00+00:00", 0)

    def test_combined_signal_event_writes_news_age(self, tmp_path):
        from analytics.signal_logger import SignalLogger
        from orchestrator.coordinator import Coordinator
        from storage.database import Database
        db = Database(str(tmp_path / "n.db"))
        c = Coordinator.__new__(Coordinator)
        c.signal_logger = SignalLogger(db)
        c._log_signal_event({
            "ticker": "AAPL", "combined_signal": "BUY", "confidence": 0.7,
            "technical": {"indicators": {"price": 100.0}},
            "sentiment": {"avg_score": 0.4, "source_breakdown": {},
                          "scored": [{"published_at": "2026-09-02T13:05:00+00:00"},
                                     {"published_at": None}]},
        }, session="US_OPEN")
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT strategy, news_newest_published_at, news_ts_missing FROM signal_events"
            ).fetchone()
        assert row == ("Combined", "2026-09-02T13:05:00+00:00", 1)
