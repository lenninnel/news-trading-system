"""
News feed data source.

NewsFeed fetches recent English-language headlines that mention a given
ticker symbol from the NewsAPI /v2/everything endpoint and returns them
as a plain list of strings.

Rate-limit protection
---------------------
NewsAPI free tier allows 100 requests/day.  To stay within budget:

  * Successful responses are cached for 24 hours (per ticker).
  * A module-level daily counter tracks how many requests have been made.
    When the counter hits ``NEWSAPI_DAILY_LIMIT`` (default 80), all
    further calls for the rest of the UTC day are skipped and served
    from cache instead.
  * A 429 (Too Many Requests) response from NewsAPI immediately maxes
    out the counter, preventing further calls for the day.

Recovery behaviour
------------------
Every successful fetch is cached in the module-level ``_newsapi_cache``
(max age: 24 hours).  On any failure the recovery path is:

  1. APIRecovery.call("newsapi", ...) wraps the HTTP request with
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
import threading
from datetime import datetime, timezone

import requests

from config.settings import MAX_HEADLINES, NEWSAPI_KEY, NEWSAPI_URL, get_search_term
from utils.api_recovery import APIRecovery, CircuitOpenError
from utils.network_recovery import NetworkMonitor, ResponseCache, get_cache

log = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "headlines"

# ---------------------------------------------------------------------------
# NewsAPI-specific 24-hour cache (separate from the 1-hour global cache)
# ---------------------------------------------------------------------------
_newsapi_cache = ResponseCache(max_age_seconds=86_400.0)  # 24 hours

# ---------------------------------------------------------------------------
# Daily request counter (resets at midnight UTC)
# ---------------------------------------------------------------------------
NEWSAPI_DAILY_LIMIT: int = 80  # warn & skip threshold (free tier = 100)

_counter_lock = threading.Lock()
_daily_requests: int = 0
_counter_date: str = ""  # ISO date string, e.g. "2026-03-18"


def _today_utc() -> str:
    """Return today's date as an ISO string in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_counter_reset() -> None:
    """Reset the counter if the UTC date has changed."""
    global _daily_requests, _counter_date
    today = _today_utc()
    if _counter_date != today:
        _daily_requests = 0
        _counter_date = today


def newsapi_requests_today() -> int:
    """Return the number of NewsAPI requests made so far today (UTC)."""
    with _counter_lock:
        _ensure_counter_reset()
        return _daily_requests


def _increment_counter() -> int:
    """Increment the daily counter and return the new value."""
    global _daily_requests
    with _counter_lock:
        _ensure_counter_reset()
        _daily_requests += 1
        return _daily_requests


def _max_out_counter() -> None:
    """Set the counter to the limit (e.g. after a 429 response)."""
    global _daily_requests
    with _counter_lock:
        _ensure_counter_reset()
        _daily_requests = NEWSAPI_DAILY_LIMIT
        log.warning(
            "NewsAPI daily counter maxed out at %d — no further requests today",
            NEWSAPI_DAILY_LIMIT,
        )


def is_newsapi_limit_reached() -> bool:
    """Return True when the daily request budget has been exhausted."""
    with _counter_lock:
        _ensure_counter_reset()
        return _daily_requests >= NEWSAPI_DAILY_LIMIT


def reset_newsapi_counter() -> None:
    """Reset the daily counter (for testing or manual override)."""
    global _daily_requests, _counter_date
    with _counter_lock:
        _daily_requests = 0
        _counter_date = _today_utc()


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

    # -- Public API -----------------------------------------------------------

    def fetch(self, ticker: str) -> list[str]:
        """
        Fetch recent headlines mentioning *ticker*.

        Successful responses are cached for 24 hours so that the system
        can operate in degraded mode (cached headlines) when NewsAPI is
        unavailable or rate-limited.

        The daily request counter is checked before each call.  When the
        limit is approaching (>=80/100), NewsAPI is skipped for the rest
        of the UTC day and cached headlines are served instead.

        Args:
            ticker: Stock ticker symbol to search for (e.g. "AAPL").

        Returns:
            List of headline strings, capped at max_headlines.
            Returns cached headlines when NewsAPI is unavailable.
            Returns an empty list when neither live nor cached data exists.
        """
        ticker = ticker.upper()

        # Check network before even trying (avoids pointless TCP handshakes)
        NetworkMonitor.check_and_update()

        cache_key = f"{_CACHE_KEY_PREFIX}:{ticker}"

        # -- Check 24h cache first (avoids unnecessary API calls) -------------
        cached_24h, hit_24h = _newsapi_cache.get("newsapi", cache_key)
        if hit_24h and cached_24h:
            log.debug(
                "NewsAPI 24h cache hit for %s (%d headlines)",
                ticker, len(cached_24h),
            )
            return list(cached_24h)

        # -- Check daily rate limit -------------------------------------------
        if is_newsapi_limit_reached():
            log.warning(
                "NewsAPI daily limit approaching (%d/%d), skipping for today",
                newsapi_requests_today(), NEWSAPI_DAILY_LIMIT + 20,
            )
            return self._serve_from_cache(ticker, cache_key)

        # -- Attempt live fetch -----------------------------------------------
        if not NetworkMonitor.is_degraded():
            try:
                headlines = APIRecovery.call(
                    "newsapi",
                    self._live_fetch,
                    ticker,
                    ticker=ticker,
                )
                # Track the request
                count = _increment_counter()
                log.info(
                    "NewsAPI request successful for %s (request %d/%d today)",
                    ticker, count, NEWSAPI_DAILY_LIMIT + 20,
                )

                # Cache in both the global 1h cache and the 24h NewsAPI cache
                get_cache().set("newsapi", cache_key, headlines)
                _newsapi_cache.set("newsapi", cache_key, headlines)
                return headlines
            except CircuitOpenError as exc:
                log.warning(
                    "NewsAPI circuit OPEN for %s — using cached headlines: %s",
                    ticker, exc,
                )
                self._log_degraded(ticker, "circuit_open", str(exc))
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code == 429:
                    log.error(
                        "NewsAPI returned 429 Too Many Requests for %s — "
                        "disabling NewsAPI for the rest of today",
                        ticker,
                    )
                    _max_out_counter()
                    self._log_degraded(ticker, "rate_limited_429", str(exc))
                else:
                    log.warning(
                        "NewsAPI HTTP error for %s (%s) — checking cache",
                        ticker, exc,
                    )
                    self._log_degraded(ticker, "api_error", str(exc))
            except Exception as exc:
                # Check if the wrapped exception contains a 429
                if _is_429_error(exc):
                    log.error(
                        "NewsAPI returned 429 Too Many Requests for %s — "
                        "disabling NewsAPI for the rest of today",
                        ticker,
                    )
                    _max_out_counter()
                    self._log_degraded(ticker, "rate_limited_429", str(exc))
                else:
                    log.warning(
                        "NewsAPI unavailable for %s (%s) — checking cache",
                        ticker, exc,
                    )
                    self._log_degraded(ticker, "api_error", str(exc))
        else:
            log.info("Network degraded — skipping NewsAPI call for %s", ticker)

        # -- Fallback: cache --------------------------------------------------
        return self._serve_from_cache(ticker, cache_key)

    def log_usage(self) -> None:
        """Log the current daily NewsAPI usage. Call at session start."""
        used = newsapi_requests_today()
        total = NEWSAPI_DAILY_LIMIT + 20  # free tier is ~100
        log.info("NewsAPI: %d/%d requests used today", used, total)

    # -- Internal helpers -----------------------------------------------------

    def _serve_from_cache(self, ticker: str, cache_key: str) -> list[str]:
        """Try to return headlines from the 24h cache, then the 1h cache."""
        # Try 24h cache first
        cached_24h, hit_24h = _newsapi_cache.get("newsapi", cache_key)
        if hit_24h and cached_24h:
            log.warning(
                "[DEGRADED MODE] newsapi: using %d cached headline(s) for %s (24h cache)",
                len(cached_24h), ticker,
            )
            self._log_cache_hit(ticker, len(cached_24h))
            return list(cached_24h)

        # Then try global 1h cache
        cached, hit = get_cache().get("newsapi", cache_key)
        if hit and cached:
            log.warning(
                "[DEGRADED MODE] newsapi: using %d cached headline(s) for %s",
                len(cached), ticker,
            )
            self._log_cache_hit(ticker, len(cached))
            return list(cached)

        # Nothing available
        log.warning(
            "[DEGRADED MODE] newsapi: no cached headlines for %s — returning empty list",
            ticker,
        )
        return []

    def _live_fetch(self, ticker: str) -> list[str]:
        """Raw HTTP call to NewsAPI; raises on any error."""
        search_term = get_search_term(ticker)
        params = {
            "q":        search_term,
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


def _is_429_error(exc: Exception) -> bool:
    """Check if an exception (possibly wrapped) indicates a 429 response."""
    # Direct HTTPError check
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code == 429:
            return True
    # Check cause chain
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return _is_429_error(cause)
    # Check string representation as last resort
    if "429" in str(exc) and "Too Many Requests" in str(exc):
        return True
    return False
