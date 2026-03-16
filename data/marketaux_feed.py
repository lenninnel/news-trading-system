"""
Marketaux news feed data source.

MarketauxFeed fetches recent headlines for a given ticker symbol from the
Marketaux ``/v1/news/all`` endpoint.  Each item includes the article title
and the per-entity sentiment score returned by the API (``-1`` to ``+1``).

Caching
-------
The free tier allows only 100 requests per day, so every successful response
is cached in a module-level dict with an 8-hour TTL.  Repeated calls for the
same ticker within that window are served from cache without hitting the API.

Rate limiting
-------------
A daily request counter tracks API usage.  When usage exceeds 80 requests,
all further calls are skipped for the rest of the UTC day and served from
cache (or return ``[]``).

Signal-based filtering
----------------------
Pass ``signal_hint="HOLD"`` to skip the API call for tickers that had a
HOLD signal on their last run.  This saves ~60% of daily quota.

Recovery behaviour
------------------
If ``MARKETAUX_API_TOKEN`` is empty the feed silently returns ``[]``.
Network errors and non-200 responses are logged at WARNING level and the
feed returns ``[]`` — it never raises.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from config.settings import MARKETAUX_API_TOKEN, MAX_HEADLINES, is_german_ticker
from data.eodhd_feed import EODHDFeed

logger = logging.getLogger(__name__)

_MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
_CACHE_TTL_SECONDS = 8 * 60 * 60  # 8 hours

_DAILY_LIMIT = 100
_DAILY_SKIP_THRESHOLD = 80

# {ticker_upper: {"items": list[dict], "fetched_at": float}}
_cache: dict[str, dict[str, Any]] = {}

# Daily request counter (resets at midnight UTC)
_daily_requests: int = 0
_daily_reset_date: str = ""


def _get_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _increment_daily_counter() -> int:
    """Increment and return the daily request count. Resets at midnight UTC."""
    global _daily_requests, _daily_reset_date
    today = _get_today()
    if _daily_reset_date != today:
        _daily_requests = 0
        _daily_reset_date = today
    _daily_requests += 1
    if _daily_requests == 75:
        logger.warning("Marketaux: 75/%d daily requests used", _DAILY_LIMIT)
    return _daily_requests


def get_daily_request_count() -> int:
    """Return current daily request count (for monitoring)."""
    global _daily_requests, _daily_reset_date
    if _daily_reset_date != _get_today():
        return 0
    return _daily_requests


def clear_cache() -> None:
    """Clear the module-level response cache (useful for testing)."""
    global _daily_requests, _daily_reset_date
    _cache.clear()
    _daily_requests = 0
    _daily_reset_date = ""


def _is_cache_valid(ticker: str) -> bool:
    """Return True if an unexpired cache entry exists for *ticker*."""
    entry = _cache.get(ticker)
    if entry is None:
        return False
    return (time.time() - entry["fetched_at"]) < _CACHE_TTL_SECONDS


class MarketauxFeed:
    """
    Fetches recent news headlines for a ticker from the Marketaux API.

    Returns a list of dicts with keys:
        - ``text``  (str): Article title.
        - ``source`` (str): Always ``"marketaux"``.
        - ``marketaux_sentiment`` (float | None): Per-entity sentiment score
          from the API (range -1 to +1), or None when unavailable.

    Args:
        api_token:     Marketaux API token.  Defaults to ``settings.MARKETAUX_API_TOKEN``.
        max_headlines: Maximum number of articles to return per call.

    Example::

        feed = MarketauxFeed()
        items = feed.fetch("AAPL")
        # [{"text": "Apple beats ...", "source": "marketaux", "marketaux_sentiment": 0.65}, ...]
    """

    def __init__(
        self,
        api_token: str = MARKETAUX_API_TOKEN,
        max_headlines: int = MAX_HEADLINES,
        eodhd_feed: EODHDFeed | None = None,
    ) -> None:
        self.api_token = api_token
        self.max_headlines = max_headlines
        self._eodhd = eodhd_feed or EODHDFeed()

    def fetch(self, ticker: str, signal_hint: str = "") -> list[dict]:
        """
        Fetch recent headlines mentioning *ticker* from Marketaux.

        Successful responses are cached for 8 hours.  Returns ``[]`` silently
        when the API token is empty, on network errors, or on non-200 responses.

        Args:
            ticker:      Stock ticker symbol (e.g. ``"AAPL"``).
            signal_hint: Last known combined signal (e.g. ``"HOLD"``).
                         When ``"HOLD"``, the API is skipped and only
                         cached results (or ``[]``) are returned.

        Returns:
            List of dicts with keys ``text``, ``source``, ``marketaux_sentiment``.
            Empty list on any failure.
        """
        if not self.api_token:
            logger.debug("MARKETAUX_API_TOKEN is empty — skipping Marketaux feed")
            return []

        ticker_upper = ticker.upper()

        # -- Return cached result if still valid ------------------------------
        if _is_cache_valid(ticker_upper):
            return _cache[ticker_upper]["items"]

        # -- Skip HOLD tickers (save API quota) -------------------------------
        if signal_hint.upper() == "HOLD":
            logger.debug("Marketaux: skipping %s (last signal was HOLD)", ticker_upper)
            return []

        # -- Skip if daily quota nearly exhausted -----------------------------
        if get_daily_request_count() >= _DAILY_SKIP_THRESHOLD:
            logger.warning(
                "Marketaux: skipping %s — daily limit approaching (%d/%d used)",
                ticker_upper, get_daily_request_count(), _DAILY_LIMIT,
            )
            return []

        # -- Live fetch -------------------------------------------------------
        _increment_daily_counter()
        try:
            params = {
                "symbols": ticker_upper,
                "api_token": self.api_token,
                "language": "en",
                "limit": self.max_headlines,
            }
            response = requests.get(_MARKETAUX_URL, params=params, timeout=15)
            response.raise_for_status()

            data = response.json().get("data", [])
            items = self._parse_articles(data, ticker_upper)

            # Cache the result
            _cache[ticker_upper] = {
                "items": items,
                "fetched_at": time.time(),
            }
            return items

        except Exception as exc:
            logger.warning("Marketaux fetch failed for %s: %s", ticker_upper, exc)
            items = []

        # Enrich with EODHD news for German tickers (Marketaux may have gaps)
        if is_german_ticker(ticker_upper) and self._eodhd.available:
            items = self._enrich_with_eodhd(ticker_upper, items)

        return items

    def _enrich_with_eodhd(
        self, ticker: str, existing: list[dict]
    ) -> list[dict]:
        """Add EODHD news items not already present in *existing*."""
        try:
            eodhd_news = self._eodhd.get_news(ticker, limit=5)
            existing_titles = {item["text"] for item in existing}
            for article in eodhd_news:
                title = article.get("title", "")
                if title and title not in existing_titles:
                    existing.append({
                        "text": title,
                        "source": "eodhd",
                        "marketaux_sentiment": None,
                    })
                    existing_titles.add(title)
        except Exception as exc:
            logger.debug("EODHD news enrichment failed for %s: %s", ticker, exc)
        return existing[:self.max_headlines]

    def _parse_articles(self, articles: list[dict], ticker: str) -> list[dict]:
        """Extract title and per-entity sentiment from the raw API response."""
        results: list[dict] = []

        for article in articles[: self.max_headlines]:
            title = article.get("title")
            if not title:
                continue

            sentiment = self._extract_sentiment(article, ticker)
            results.append({
                "text": title,
                "source": "marketaux",
                "marketaux_sentiment": sentiment,
            })

        return results

    @staticmethod
    def _extract_sentiment(article: dict, ticker: str) -> float | None:
        """
        Return the sentiment_score for *ticker* from the article's entities list.

        Falls back to None when the entities list is missing, empty, or does
        not contain a matching symbol.
        """
        entities = article.get("entities")
        if not isinstance(entities, list):
            return None

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("symbol", "").upper() == ticker.upper():
                score = entity.get("sentiment_score")
                if score is not None:
                    try:
                        return float(score)
                    except (TypeError, ValueError):
                        return None
        return None
