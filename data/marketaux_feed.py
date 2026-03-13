"""
Marketaux news feed data source.

MarketauxFeed fetches recent headlines for a given ticker symbol from the
Marketaux ``/v1/news/all`` endpoint.  Each item includes the article title
and the per-entity sentiment score returned by the API (``-1`` to ``+1``).

Caching
-------
The free tier allows only 100 requests per day, so every successful response
is cached in a module-level dict with a 4-hour TTL.  Repeated calls for the
same ticker within that window are served from cache without hitting the API.

Recovery behaviour
------------------
If ``MARKETAUX_API_TOKEN`` is empty the feed silently returns ``[]``.
Network errors and non-200 responses are logged at WARNING level and the
feed returns ``[]`` — it never raises.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import MARKETAUX_API_TOKEN, MAX_HEADLINES

logger = logging.getLogger(__name__)

_MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours

# {ticker_upper: {"items": list[dict], "fetched_at": float}}
_cache: dict[str, dict[str, Any]] = {}


def clear_cache() -> None:
    """Clear the module-level response cache (useful for testing)."""
    _cache.clear()


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
    ) -> None:
        self.api_token = api_token
        self.max_headlines = max_headlines

    def fetch(self, ticker: str) -> list[dict]:
        """
        Fetch recent headlines mentioning *ticker* from Marketaux.

        Successful responses are cached for 4 hours.  Returns ``[]`` silently
        when the API token is empty, on network errors, or on non-200 responses.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).

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

        # -- Live fetch -------------------------------------------------------
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
            return []

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
