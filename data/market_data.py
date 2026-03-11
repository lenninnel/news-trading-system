"""
Market data source.

MarketData fetches current price and basic fundamental information for a
ticker symbol using the yfinance library, which wraps the Yahoo Finance API.

On a 401 "Invalid Crumb" error the yfinance JSON cache is cleared and the
request is retried once with a fresh Ticker object.  If the retry also
fails, a fallback dict with ``price=None`` is returned instead of raising.

Requires:
    pip install yfinance
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def _is_crumb_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a yfinance crumb / auth error."""
    msg = str(exc).lower()
    return "401" in msg or "invalid crumb" in msg


def _clear_yf_cache() -> None:
    """Best-effort cache clear across yfinance versions."""
    # Preferred: functools.lru_cache on get_json (yfinance ≥ 0.2.31)
    utils = getattr(yf, "utils", None)
    if utils is not None:
        get_json = getattr(utils, "get_json", None)
        if get_json is not None and hasattr(get_json, "cache_clear"):
            try:
                get_json.cache_clear()
            except Exception:
                pass

    # Fallback: clear session cookies on known internal locations
    for submod_name in ("data", "utils", "shared"):
        submod = getattr(yf, submod_name, None)
        if submod is None:
            continue
        for attr in ("_session", "session", "_REQUESTS_SESSION"):
            sess = getattr(submod, attr, None)
            if sess is not None and hasattr(sess, "cookies"):
                try:
                    sess.cookies.clear()
                except Exception:
                    pass


class MarketData:
    """
    Fetches current market data for a given ticker.

    Example::

        md = MarketData()
        data = md.fetch("AAPL")
        # {"ticker": "AAPL", "name": "Apple Inc.", "price": 189.30, ...}
    """

    def fetch(self, ticker: str) -> dict:
        """
        Retrieve current market data for *ticker*.

        On a 401 / crumb error the cache is cleared and the call is
        retried once with a fresh ``yf.Ticker`` object.  If the retry
        also fails, a fallback dict with ``price=None`` is returned.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys: ticker, name, price, currency, market_cap.
        """
        try:
            return self._fetch_once(ticker)
        except Exception as exc:
            if not _is_crumb_error(exc):
                raise

            logger.warning(
                "yfinance crumb error for %s — clearing cache and retrying",
                ticker,
            )
            _clear_yf_cache()

            try:
                return self._fetch_once(ticker)
            except Exception:
                logger.warning(
                    "yfinance retry failed for %s — returning fallback",
                    ticker,
                )
                return {
                    "ticker": ticker,
                    "name": "N/A",
                    "price": None,
                    "currency": "USD",
                    "market_cap": None,
                }

    @staticmethod
    def _fetch_once(ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "price": price,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
        }
