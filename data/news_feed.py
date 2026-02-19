"""
News feed data source.

NewsFeed fetches recent English-language headlines that mention a given
ticker symbol from the NewsAPI /v2/everything endpoint and returns them
as a plain list of strings.

Recovery behaviour
------------------
Every successful fetch is cached in the module-level ResponseCache
(max age: 1 hour).  On any failure the recovery path is:

  1. APIRecovery.call("newsapi", …) wraps the HTTP request with
     per-service retry logic (max 3 attempts, 60 s backoff) and
     a circuit breaker (opens after 5 consecutive failures).

  2. If all retries fail (or the circuit is OPEN), the cache is
     checked.  A cache hit returns stale headlines and activates
     degraded mode; a miss returns an empty list.

  3. All fallback activations are logged to the recovery_log table
     when a Database instance has been attached via set_db().

Degraded-mode activations are also logged at WARNING level.
"""

from __future__ import annotations

import logging

import requests

from config.settings import MAX_HEADLINES, NEWSAPI_KEY, NEWSAPI_URL
from utils.api_recovery import APIRecovery, CircuitOpenError
from utils.network_recovery import NetworkMonitor, get_cache

log = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "headlines"


class NewsFeed:
    """
    Retrieves news headlines for a ticker symbol from NewsAPI.

    Args:
        api_key:       NewsAPI key. Defaults to settings.NEWSAPI_KEY.
        max_headlines: Maximum number of articles to retrieve per call.
        db:            Optional Database instance for recovery_log writes.

    Example::

        feed = NewsFeed()
        headlines = feed.fetch("TSLA")
        # ["Tesla delivers record ...", "Elon Musk says ...", ...]
    """

    def __init__(
        self,
        api_key:       str = NEWSAPI_KEY,
        max_headlines: int = MAX_HEADLINES,
        db:            "object | None" = None,
    ) -> None:
        self.api_key       = api_key
        self.max_headlines = max_headlines
        if db is not None:
            APIRecovery.set_db(db)
            NetworkMonitor.set_db(db)

    def fetch(self, ticker: str) -> list[str]:
        """
        Fetch recent headlines mentioning *ticker*.

        Successful responses are cached for up to 1 hour so that the system
        can operate in degraded mode (cached headlines) when NewsAPI is down.

        Args:
            ticker: Stock ticker symbol to search for (e.g. "AAPL").

        Returns:
            List of headline strings, capped at max_headlines.
            Returns cached headlines when NewsAPI is unavailable.
            Returns an empty list when neither live nor cached data exists.
        """
        # Check network before even trying (avoids pointless TCP handshakes)
        NetworkMonitor.check_and_update()

        cache     = get_cache()
        cache_key = f"{_CACHE_KEY_PREFIX}:{ticker.upper()}"

        # -- Attempt live fetch ------------------------------------------------
        if not NetworkMonitor.is_degraded():
            try:
                headlines = APIRecovery.call(
                    "newsapi",
                    self._live_fetch,
                    ticker,
                    ticker=ticker,
                )
                # Cache the fresh result
                cache.set("newsapi", cache_key, headlines)
                return headlines
            except CircuitOpenError as exc:
                log.warning(
                    "NewsAPI circuit OPEN for %s — using cached headlines: %s",
                    ticker, exc,
                )
                self._log_degraded(ticker, "circuit_open", str(exc))
            except Exception as exc:
                log.warning(
                    "NewsAPI unavailable for %s (%s) — checking cache", ticker, exc
                )
                self._log_degraded(ticker, "api_error", str(exc))
        else:
            log.info("Network degraded — skipping NewsAPI call for %s", ticker)

        # -- Fallback: cache ---------------------------------------------------
        cached, hit = cache.get("newsapi", cache_key)
        if hit:
            log.warning(
                "[DEGRADED MODE] %s: using %d cached headline(s) for %s",
                "newsapi", len(cached), ticker,
            )
            self._log_cache_hit(ticker, len(cached))
            return cached  # type: ignore[return-value]

        # -- Nothing available -------------------------------------------------
        log.warning(
            "[DEGRADED MODE] %s: no cached headlines for %s — returning empty list",
            "newsapi", ticker,
        )
        return []

    def _live_fetch(self, ticker: str) -> list[str]:
        """Raw HTTP call to NewsAPI; raises on any error."""
        params = {
            "q":        ticker,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": self.max_headlines,
            "apiKey":   self.api_key,
        }
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]

    # -- Recovery logging ------------------------------------------------------

    @staticmethod
    def _log_degraded(ticker: str, reason: str, error: str) -> None:
        db = APIRecovery._db
        if db is None:
            return
        try:
            db.log_recovery_event(
                service="newsapi",
                event_type="degraded_mode",
                ticker=ticker,
                error_msg=error,
                recovery_action="using_cache",
                success=False,
            )
        except Exception:
            pass

    @staticmethod
    def _log_cache_hit(ticker: str, count: int) -> None:
        db = APIRecovery._db
        if db is None:
            return
        try:
            db.log_recovery_event(
                service="newsapi",
                event_type="cache_hit",
                ticker=ticker,
                recovery_action=f"served_{count}_cached_headlines",
                success=True,
            )
        except Exception:
            pass
