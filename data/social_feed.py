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
