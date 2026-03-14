"""
Market data source.

MarketData fetches current price and basic fundamental information for a
ticker symbol.  Data sources are routed by ticker type:

  - Crypto (BTC, ETH, …) → Binance
  - German / EU (.XETRA, .DE) → EODHD (fallback: yfinance)
  - Everything else → yfinance

On a 401 "Invalid Crumb" error the yfinance JSON cache is cleared and the
request is retried once with a fresh Ticker object.  If the retry also
fails, a fallback dict with ``price=None`` is returned instead of raising.

Requires:
    pip install yfinance
"""

import logging

import pandas as pd
import yfinance as yf

from config.settings import CRYPTO_TICKERS, is_german_ticker
from data.binance_feed import BinanceFeed
from data.eodhd_feed import EODHDFeed

logger = logging.getLogger(__name__)


def _is_no_price(exc: Exception) -> bool:
    """Return True if *exc* indicates missing price data (not a real error)."""
    return isinstance(exc, ValueError) and "no price" in str(exc).lower()


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

    Routing:
      - Crypto tickers (BTC, ETH, …) → Binance API
      - German/EU tickers (.XETRA, .DE) → EODHD API (fallback: yfinance)
      - All others → yfinance

    Example::

        md = MarketData()
        data = md.fetch("AAPL")
        # {"ticker": "AAPL", "name": "Apple Inc.", "price": 189.30, ...}
    """

    def __init__(
        self,
        *,
        binance_feed: BinanceFeed | None = None,
        eodhd_feed: EODHDFeed | None = None,
    ) -> None:
        self._binance = binance_feed or BinanceFeed()
        self._eodhd = eodhd_feed or EODHDFeed()

    def fetch(self, ticker: str) -> dict:
        """
        Retrieve current market data for *ticker*.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL") or crypto (e.g. "BTC").

        Returns:
            dict with keys: ticker, name, price, currency, market_cap,
            source, degraded.
        """
        ticker_upper = ticker.upper()

        if ticker_upper in CRYPTO_TICKERS:
            return self._fetch_crypto(ticker_upper)

        if is_german_ticker(ticker_upper):
            return self._fetch_german(ticker_upper)

        try:
            result = self._fetch_once(ticker_upper)
            result["ticker"] = ticker_upper
            result.setdefault("source", "yfinance")
            result.setdefault("degraded", False)
            return result
        except Exception as exc:
            if not _is_crumb_error(exc) and not _is_no_price(exc):
                raise

            if _is_crumb_error(exc):
                logger.warning(
                    "yfinance crumb error for %s — clearing cache and retrying",
                    ticker_upper,
                )
                _clear_yf_cache()
                try:
                    result = self._fetch_once(ticker_upper)
                    result["ticker"] = ticker_upper
                    result.setdefault("source", "yfinance")
                    result.setdefault("degraded", False)
                    return result
                except Exception:
                    pass

            logger.warning(
                "yfinance failed for %s — returning fallback", ticker_upper,
            )
            return {
                "ticker": ticker_upper,
                "name": "N/A",
                "price": None,
                "currency": "USD",
                "market_cap": None,
                "source": "none",
                "degraded": True,
            }

    def get_intraday(
        self, ticker: str, interval: str = "5m"
    ) -> pd.DataFrame | None:
        """
        Fetch intraday OHLCV data.

        Routes to EODHD for stocks, Binance klines for crypto.

        Args:
            ticker:   Ticker symbol.
            interval: Bar interval — "1m", "5m", or "1h".

        Returns:
            DataFrame with OHLCV columns, or None.
        """
        if ticker.upper() in CRYPTO_TICKERS:
            # Binance klines: map interval to Binance format
            binance_interval_map = {"1m": "1m", "5m": "5m", "1h": "1h"}
            bi = binance_interval_map.get(interval, "5m")
            return self._binance.get_ohlcv(ticker.upper(), limit=100)

        if is_german_ticker(ticker) and self._eodhd.available:
            return self._eodhd.get_ohlcv_intraday(ticker, interval=interval)

        # For US stocks, try EODHD if available, otherwise skip
        if self._eodhd.available:
            return self._eodhd.get_ohlcv_intraday(ticker, interval=interval)

        return None

    def _fetch_german(self, ticker: str) -> dict:
        """Fetch German/EU stock data via EODHD, falling back to yfinance."""
        if self._eodhd.available:
            price = self._eodhd.get_price(ticker)
            if price is not None:
                return {
                    "ticker": ticker,
                    "name": ticker.split(".")[0],
                    "price": price,
                    "currency": "EUR",
                    "market_cap": None,
                }
            logger.warning(
                "EODHD price unavailable for %s — falling back to yfinance",
                ticker,
            )

        # Fallback: convert .XETRA to .DE for yfinance compatibility
        yf_ticker = ticker
        if ticker.upper().endswith(".XETRA"):
            yf_ticker = ticker.rsplit(".", 1)[0] + ".DE"

        try:
            return self._fetch_once(yf_ticker)
        except Exception:
            logger.warning(
                "yfinance fallback also failed for %s — returning fallback",
                ticker,
            )
            return {
                "ticker": ticker,
                "name": "N/A",
                "price": None,
                "currency": "EUR",
                "market_cap": None,
            }

    def _fetch_crypto(self, ticker: str) -> dict:
        """Fetch crypto market data from Binance."""
        price = self._binance.get_price(ticker)
        return {
            "ticker": ticker,
            "name": f"{ticker}/USDT",
            "price": price,
            "currency": "USDT",
            "market_cap": None,
        }

    @staticmethod
    def _fetch_once(ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            raise ValueError(f"No price data for {ticker}")
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "price": price,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
        }
