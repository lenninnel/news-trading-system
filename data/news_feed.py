"""
News feed data source.

NewsFeed fetches recent English-language headlines that mention a given
ticker symbol from the NewsAPI /v2/everything endpoint and returns them
as a plain list of strings.

The class is intentionally thin — it owns only I/O concerns (HTTP request,
JSON parsing, error propagation) and has no dependency on agents or storage.
"""

from __future__ import annotations

import requests

from config.settings import MAX_HEADLINES, NEWSAPI_KEY, NEWSAPI_URL


class NewsFeed:
    """
    Retrieves news headlines for a ticker symbol from NewsAPI.

    Args:
        api_key:       NewsAPI key. Defaults to settings.NEWSAPI_KEY.
        max_headlines: Maximum number of articles to retrieve per call.

    Example::

        feed = NewsFeed()
        headlines = feed.fetch("TSLA")
        # ["Tesla delivers record ...", "Elon Musk says ...", ...]
    """

    def __init__(
        self,
        api_key: str = NEWSAPI_KEY,
        max_headlines: int = MAX_HEADLINES,
    ) -> None:
        self.api_key = api_key
        self.max_headlines = max_headlines

    def fetch(self, ticker: str) -> list[str]:
        """
        Fetch recent headlines mentioning *ticker*.

        Args:
            ticker: Stock ticker symbol to search for (e.g. "AAPL").

        Returns:
            List of headline strings, capped at max_headlines.

        Raises:
            requests.HTTPError: If the NewsAPI request fails (4xx / 5xx).
            requests.Timeout:   If the request exceeds the 15-second timeout.
        """
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self.max_headlines,
            "apiKey": self.api_key,
        }
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()

        articles = response.json().get("articles", [])
        return [article["title"] for article in articles if article.get("title")]
