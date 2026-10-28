"""News input hygiene (2026-10-28, KW44 review).

Age window, company relevance, in-batch dedupe and the NewsAPI query.
Cases are taken from the 2026-10 production headlines that triggered
the fix (CVX "cvx-linalg", JPM stablecoin/GenScript, week-old repeats).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config.settings as settings
from data.news_feed import (
    NewsFeed,
    build_news_query,
    filter_articles,
    is_relevant_headline,
)

NOW = datetime(2026, 10, 27, 13, 15, tzinfo=timezone.utc)


def _art(text, hours_old=None):
    ts = None if hours_old is None else (NOW - timedelta(hours=hours_old)).isoformat()
    return {"text": text, "source": "newsapi", "published_at": ts, "fetched_at": None}


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_MAX_AGE_HOURS", 72.0)
    monkeypatch.setattr(settings, "NEWS_RELEVANCE_FILTER", True)


class TestRelevance:
    @pytest.mark.parametrize("ticker,text", [
        ("CVX", "Why Jim Cramer Calls Chevron (CVX) the King"),
        ("CVX", "Four Energy Deals in Four Days as Chevron eyes Brent"),
        ("XOM", "ExxonMobil Is Up 49% in 2026"),
        ("XOM", "29 ETFs Add to Exxon Mobil (XOM) Holdings"),
        ("JPM", "JPMorgan Puts a Number on Bitcoin for the First Time"),
        ("META", "Elon Musk compared Tesla's robot to a toddler. Meta had a different AI surprise."),
        ("TSLA", "$TSLA gaps up premarket"),
        ("BAC", "Bank of America beats on trading revenue"),
    ])
    def test_relevant(self, ticker, text):
        assert is_relevant_headline(ticker, text)

    @pytest.mark.parametrize("ticker,text", [
        ("CVX", "cvx-linalg 1.1.2"),
        ("JPM", "Lead Bank Rolls Out Stablecoin On-Ramps, Off-Ramps and 44-Country FX Payouts"),
        ("JPM", "GenScript Launches Life Sciences Startup PitchFest with NVIDIA"),
        ("META", "New metadata standard for photo archives"),
        ("XOM", "Four Energy Deals in Four Days as Brent Holds Above $100"),
        ("TSLA", "Musk Wants More Control Over Chips"),
    ])
    def test_irrelevant(self, ticker, text):
        assert not is_relevant_headline(ticker, text)


class TestFilter:
    def test_drops_old_keeps_fresh_and_untimestamped(self):
        arts = [_art("Chevron fresh", 5), _art("Chevron old", 100), _art("Chevron no ts")]
        out = filter_articles("CVX", arts, now=NOW)
        assert [a["text"] for a in out] == ["Chevron fresh", "Chevron no ts"]

    def test_dedupes_within_batch(self):
        arts = [_art("Chevron (CVX) rallies", 2), _art("chevron (CVX) rallies", 3)]
        assert len(filter_articles("CVX", arts, now=NOW)) == 1

    def test_switches_off(self, monkeypatch):
        monkeypatch.setattr(settings, "NEWS_MAX_AGE_HOURS", 0.0)
        monkeypatch.setattr(settings, "NEWS_RELEVANCE_FILTER", False)
        arts = [_art("cvx-linalg 1.1.2", 500)]
        assert filter_articles("CVX", arts, now=NOW) == arts


class TestQuery:
    def test_query_uses_names_and_ticker(self):
        assert build_news_query("CVX") == '"Chevron" OR CVX'

    def test_query_legacy_when_off(self, monkeypatch):
        monkeypatch.setattr(settings, "NEWS_RELEVANCE_FILTER", False)
        assert build_news_query("CVX") == "CVX"

    def test_live_fetch_sends_window_and_title_search(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"articles": []}
        with patch("data.news_feed.requests.get", return_value=resp) as get:
            NewsFeed(api_key="k", max_headlines=10)._live_fetch("CVX")
        params = get.call_args.kwargs["params"]
        assert params["q"] == '"Chevron" OR CVX'
        assert params["searchIn"] == "title"
        assert params["pageSize"] == 30
        assert "from" in params

    def test_fetch_articles_filters_cached_batch(self):
        """A cached batch is filtered on every read, so stale items age out."""
        feed = NewsFeed(api_key="k", max_headlines=10)
        now = datetime.now(timezone.utc)

        def at(text, hours):
            return {"text": text, "source": "newsapi",
                    "published_at": (now - timedelta(hours=hours)).isoformat(),
                    "fetched_at": None}

        cached = [at("Chevron (CVX) fresh", 1), at("cvx-linalg 1.1.2", 1),
                  at("Chevron week old", 160)]
        with patch.object(NewsFeed, "_fetch_articles_raw", return_value=cached):
            out = feed.fetch_articles("CVX")
        assert [a["text"] for a in out] == ["Chevron (CVX) fresh"]
