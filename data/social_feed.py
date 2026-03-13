"""
Social media data sources for multi-source sentiment analysis.

RedditFeed
    Fetches top post titles from r/wallstreetbets and r/investing via PRAW.
    Gracefully returns an empty list when PRAW is not installed or Reddit
    credentials are not configured.

StockTwitsFeed
    Fetches recent messages from the public StockTwits API (no auth required).
    Returns an empty list on non-200 responses.
"""

from __future__ import annotations

import logging
import time

import requests

from config.settings import (
    ADANOS_API_KEY,
    CRYPTO_TICKERS,
    MAX_HEADLINES,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)

logger = logging.getLogger(__name__)

_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

_REDDIT_SUBREDDITS = ["wallstreetbets", "investing"]


class RedditFeed:
    """
    Fetches recent posts mentioning a ticker from Reddit.

    Returns a list of dicts with keys: text, source.
    Returns an empty list if PRAW is not installed or credentials are missing.
    """

    def __init__(
        self,
        client_id: str = REDDIT_CLIENT_ID,
        client_secret: str = REDDIT_CLIENT_SECRET,
        user_agent: str = REDDIT_USER_AGENT,
        max_posts: int = MAX_HEADLINES,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.max_posts = max_posts

    def fetch(self, ticker: str) -> list[dict]:
        """
        Search Reddit for posts mentioning *ticker*.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            List of dicts with keys: text (str), source ("reddit").
            Empty list if PRAW is unavailable or credentials are missing.
        """
        if not self.client_id or not self.client_secret:
            logger.info("Reddit credentials not set — skipping Reddit feed")
            return []

        try:
            import praw  # noqa: F811
        except ImportError:
            logger.warning("praw not installed — skipping Reddit feed")
            return []

        try:
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )

            results: list[dict] = []
            per_sub = max(1, self.max_posts // len(_REDDIT_SUBREDDITS))
            for sub_name in _REDDIT_SUBREDDITS:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.search(ticker, sort="new", limit=per_sub):
                    text = post.title
                    if post.selftext:
                        preview = post.selftext[:200]
                        text = f"{text} — {preview}"
                    results.append({"text": text, "source": "reddit"})
            return results[: self.max_posts]
        except Exception as exc:
            logger.warning("Reddit fetch failed: %s", exc)
            return []


class StockTwitsFeed:
    """
    Fetches recent messages for a ticker from the StockTwits public API.

    Returns a list of dicts with keys: text, source, stocktwits_sentiment.

    Resilience:
        - On 403 / non-200: retries twice with exponential backoff (1 s, 2 s).
        - Increments ``_fail_count`` after each failed fetch (retries exhausted).
        - When ``_fail_count`` reaches 3, sets ``_disabled = True``, logs a
          single WARNING, and returns ``[]`` silently for all future calls.
    """

    _disabled: bool = False
    _fail_count: int = 0
    _MAX_RETRIES: int = 2

    def __init__(self, max_messages: int = MAX_HEADLINES) -> None:
        self.max_messages = max_messages

    def fetch(self, ticker: str) -> list[dict]:
        """
        Fetch recent StockTwits messages for *ticker*.

        Returns an empty list when disabled, on non-200 responses, or on
        network errors.
        """
        if StockTwitsFeed._disabled:
            return []

        url = _STOCKTWITS_URL.format(ticker=ticker.upper())

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=10)

                if resp.status_code == 200:
                    StockTwitsFeed._fail_count = 0
                    return self._parse_response(resp)

                # Non-200 — retry with backoff on retryable codes
                if attempt < self._MAX_RETRIES:
                    time.sleep(2 ** attempt)  # 1 s, 2 s
                    continue

            except Exception:
                if attempt < self._MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    continue

        # All attempts exhausted — record failure
        StockTwitsFeed._fail_count += 1
        if StockTwitsFeed._fail_count >= 3:
            logger.warning(
                "StockTwits disabled for this session after 3 failures"
            )
            StockTwitsFeed._disabled = True

        return []

    def _parse_response(self, resp: requests.Response) -> list[dict]:
        """Extract messages from a successful StockTwits response."""
        data = resp.json()
        messages = data.get("messages", [])
        results: list[dict] = []
        for msg in messages[: self.max_messages]:
            body = msg.get("body", "")
            st_sentiment = None
            entities = msg.get("entities", {})
            if isinstance(entities, dict):
                sentiment_obj = entities.get("sentiment")
                if isinstance(sentiment_obj, dict):
                    st_sentiment = sentiment_obj.get("basic")
            results.append({
                "text": body,
                "source": "stocktwits",
                "stocktwits_sentiment": st_sentiment,
            })
        return results


# ===========================================================================
# ApeWisdom — Reddit mention leaderboard (no API key required)
# ===========================================================================

_APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"
_APEWISDOM_CACHE_TTL = 3600  # 1 hour in seconds

_apewisdom_cache: dict = {"data": [], "fetched_at": 0.0}


def clear_apewisdom_cache() -> None:
    """Reset the module-level ApeWisdom leaderboard cache."""
    _apewisdom_cache["data"] = []
    _apewisdom_cache["fetched_at"] = 0.0


class ApeWisdomFeed:
    """
    Fetches Reddit mention data for a ticker from the ApeWisdom leaderboard.

    Returns a list of dicts with keys: text, source, mentions, rank, rank_24h_ago.
    The full leaderboard is cached for 1 hour to avoid excessive requests.
    """

    def fetch(self, ticker: str) -> list[dict]:
        """
        Look up *ticker* on the ApeWisdom all-stocks leaderboard.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            Single-element list with mention data, or ``[]`` if the ticker
            is not on the leaderboard or an error occurs.
        """
        try:
            results = self._get_leaderboard()
        except Exception as exc:
            logger.warning("ApeWisdom fetch failed: %s", exc)
            return []

        ticker_upper = ticker.upper()
        for entry in results:
            if entry.get("ticker", "").upper() == ticker_upper:
                mentions = entry.get("mentions", 0)
                rank = entry.get("rank", 0)
                rank_24h_ago = entry.get("rank_24h_ago", 0)
                return [{
                    "text": (
                        f"{ticker_upper} has {mentions} Reddit mentions "
                        f"(rank #{rank}, was #{rank_24h_ago})"
                    ),
                    "source": "apewisdom",
                    "mentions": mentions,
                    "rank": rank,
                    "rank_24h_ago": rank_24h_ago,
                }]

        return []

    def _get_leaderboard(self) -> list[dict]:
        """Return the cached leaderboard, refreshing if stale."""
        now = time.time()
        if (
            _apewisdom_cache["data"]
            and (now - _apewisdom_cache["fetched_at"]) < _APEWISDOM_CACHE_TTL
        ):
            return _apewisdom_cache["data"]

        resp = requests.get(_APEWISDOM_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        _apewisdom_cache["data"] = results
        _apewisdom_cache["fetched_at"] = now
        return results


# ===========================================================================
# Adanos — Reddit + X stock sentiment (API key required)
# ===========================================================================

_ADANOS_STOCK_URLS = [
    ("Reddit", "https://api.adanos.org/reddit/stocks/v1/stock/{ticker}?days=1"),
    ("X", "https://api.adanos.org/x/stocks/v1/stock/{ticker}?days=1"),
]
_ADANOS_CRYPTO_URLS = [
    ("Reddit", "https://api.adanos.org/reddit/crypto/v1/token/{ticker}?days=1"),
]


class AdanosFeed:
    """
    Fetches social sentiment for a ticker from the Adanos API (Reddit + X).

    Returns a list of dicts with keys: text, source, adanos_bullish, adanos_buzz,
    adanos_sentiment.  Requires ``ADANOS_API_KEY`` to be set.  Once a 429 is
    received the feed disables itself for the rest of the session.
    """

    _request_count: int = 0
    _quota_exhausted: bool = False

    def fetch(self, ticker: str) -> list[dict]:
        """
        Fetch Adanos sentiment for *ticker* from Reddit and X endpoints.

        Returns an empty list when the API key is missing, the quota is
        exhausted, or an error occurs.
        """
        if not ADANOS_API_KEY:
            return []

        if AdanosFeed._quota_exhausted:
            return []

        ticker_upper = ticker.upper()
        headers = {
            "X-API-Key": ADANOS_API_KEY,
            "Accept": "application/json",
        }

        endpoints = (
            _ADANOS_CRYPTO_URLS
            if ticker_upper in CRYPTO_TICKERS
            else _ADANOS_STOCK_URLS
        )

        results: list[dict] = []
        for label, url_tpl in endpoints:
            try:
                url = url_tpl.format(ticker=ticker_upper)
                resp = requests.get(url, headers=headers, timeout=10)
                AdanosFeed._request_count += 1

                if resp.status_code == 429:
                    AdanosFeed._quota_exhausted = True
                    logger.warning(
                        "Adanos API quota exhausted (429) after %d requests",
                        AdanosFeed._request_count,
                    )
                    return results  # return whatever we collected so far

                resp.raise_for_status()
                data = resp.json()

                if not data.get("found", False):
                    continue

                bullish_pct = data.get("bullish_pct", 0)
                buzz = data.get("buzz_score", 0)
                sentiment = data.get("sentiment_score", 0.0)
                mentions = data.get("total_mentions", 0)

                results.append({
                    "text": (
                        f"{ticker_upper} {label} sentiment: "
                        f"{bullish_pct}% bullish, {mentions} mentions, "
                        f"buzz {buzz:.1f}"
                    ),
                    "source": "adanos",
                    "adanos_bullish": bullish_pct,
                    "adanos_buzz": buzz,
                    "adanos_sentiment": sentiment,
                })

            except Exception as exc:
                logger.warning("Adanos %s fetch failed for %s: %s", label, ticker_upper, exc)

        return results
