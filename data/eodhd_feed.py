"""
EODHD data feed for German/EU stocks and intraday data.

EODHDFeed wraps the EODHD API (https://eodhd.com/api/) for:
  - Daily OHLCV (e.g. SAP.XETRA, SIE.XETRA, BMW.XETRA)
  - Intraday OHLCV (1m, 5m, 1h intervals)
  - Real-time price
  - Earnings calendar
  - News headlines

If EODHD_API_TOKEN is not set, all methods skip gracefully
(return None / empty list) so callers can fall back to yfinance.

Results are cached per symbol with a 4-hour TTL.

Requires:
    pip install pandas requests
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

import pandas as pd
import requests

from config.settings import EODHD_API_TOKEN

logger = logging.getLogger(__name__)

_BASE_URL = "https://eodhd.com/api"

# Module-level caches with 4-hour TTL
_CACHE_TTL = 4 * 3600  # 4 hours
_ohlcv_cache: dict[str, dict] = {}
_intraday_cache: dict[str, dict] = {}
_price_cache: dict[str, dict] = {}


def clear_cache() -> None:
    """Clear all module-level caches."""
    _ohlcv_cache.clear()
    _intraday_cache.clear()
    _price_cache.clear()


class EODHDFeed:
    """
    Fetches market data from the EODHD API.

    German tickers use the XETRA exchange suffix: SAP.XETRA, SIE.XETRA, etc.

    Example::

        feed = EODHDFeed()
        df = feed.get_ohlcv_daily("SAP.XETRA")
        price = feed.get_price("SAP.XETRA")
    """

    def __init__(self, api_token: str = EODHD_API_TOKEN) -> None:
        self.api_token = api_token

    @property
    def available(self) -> bool:
        """Return True if the API token is configured."""
        return bool(self.api_token)

    # ------------------------------------------------------------------
    # Daily OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv_daily(self, ticker: str, limit: int = 200) -> pd.DataFrame | None:
        """
        Fetch daily OHLCV data for *ticker*.

        Args:
            ticker: EODHD ticker with exchange suffix (e.g. "SAP.XETRA").
            limit:  Number of trading days to fetch (default 200).

        Returns:
            DataFrame with columns Open, High, Low, Close, Volume and a
            DatetimeIndex, or None on failure / missing token.
        """
        if not self.available:
            return None

        cache_key = f"{ticker.upper()}:{limit}"
        cached = _ohlcv_cache.get(cache_key)
        if cached and (time.time() - cached["fetched_at"]) < _CACHE_TTL:
            return cached["df"]

        try:
            url = f"{_BASE_URL}/eod/{ticker}"
            resp = requests.get(
                url,
                params={
                    "api_token": self.api_token,
                    "fmt": "json",
                    "order": "d",
                    "limit": limit,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or len(data) == 0:
                logger.warning("EODHD: empty daily data for %s", ticker)
                return None

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Normalise column names to title case
            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjusted_close": "Adj_Close",
                "volume": "Volume",
            }
            df.rename(columns=col_map, inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index.name = "Date"

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            _ohlcv_cache[cache_key] = {"df": df, "fetched_at": time.time()}
            logger.info("EODHD: fetched %d daily bars for %s", len(df), ticker)
            return df

        except Exception as exc:
            logger.warning("EODHD daily fetch failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Intraday OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv_intraday(
        self, ticker: str, interval: str = "5m"
    ) -> pd.DataFrame | None:
        """
        Fetch intraday OHLCV data for *ticker*.

        Args:
            ticker:   EODHD ticker (e.g. "SAP.XETRA").
            interval: "1m", "5m", or "1h".

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex, or None.
        """
        if not self.available:
            return None

        cache_key = f"{ticker.upper()}:{interval}"
        cached = _intraday_cache.get(cache_key)
        if cached and (time.time() - cached["fetched_at"]) < _CACHE_TTL:
            return cached["df"]

        try:
            url = f"{_BASE_URL}/intraday/{ticker}"
            resp = requests.get(
                url,
                params={
                    "api_token": self.api_token,
                    "fmt": "json",
                    "interval": interval,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or len(data) == 0:
                logger.warning("EODHD: empty intraday data for %s", ticker)
                return None

            df = pd.DataFrame(data)

            # EODHD intraday uses "datetime" or "timestamp" key
            dt_col = "datetime" if "datetime" in df.columns else "timestamp"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df.set_index(dt_col, inplace=True)
            df.sort_index(inplace=True)

            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df.rename(columns=col_map, inplace=True)

            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep]
            df.index.name = "Date"

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            _intraday_cache[cache_key] = {"df": df, "fetched_at": time.time()}
            logger.info("EODHD: fetched %d intraday bars (%s) for %s", len(df), interval, ticker)
            return df

        except Exception as exc:
            logger.warning("EODHD intraday fetch failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Real-time price
    # ------------------------------------------------------------------

    def get_price(self, ticker: str) -> float | None:
        """
        Fetch the latest real-time price for *ticker*.

        Returns:
            Latest price as float, or None on failure.
        """
        if not self.available:
            return None

        cached = _price_cache.get(ticker.upper())
        if cached and (time.time() - cached["fetched_at"]) < 300:  # 5 min TTL
            return cached["price"]

        try:
            url = f"{_BASE_URL}/real-time/{ticker}"
            resp = requests.get(
                url,
                params={"api_token": self.api_token, "fmt": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            price = data.get("close") or data.get("previousClose")
            if price is not None:
                price = float(price)
                _price_cache[ticker.upper()] = {
                    "price": price,
                    "fetched_at": time.time(),
                }
                return price

            return None

        except Exception as exc:
            logger.warning("EODHD price fetch failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Earnings calendar
    # ------------------------------------------------------------------

    def get_earnings_calendar(self, ticker: str) -> date | None:
        """
        Fetch the next earnings date for *ticker*.

        Returns:
            datetime.date or None if unavailable.
        """
        if not self.available:
            return None

        try:
            url = f"{_BASE_URL}/calendar/earnings"
            resp = requests.get(
                url,
                params={
                    "api_token": self.api_token,
                    "symbols": ticker,
                    "fmt": "json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            earnings = data.get("earnings") if isinstance(data, dict) else data
            if not isinstance(earnings, list) or len(earnings) == 0:
                return None

            # Find the next future date
            today = date.today()
            for entry in earnings:
                raw_date = entry.get("report_date") or entry.get("date")
                if not raw_date:
                    continue
                try:
                    dt = datetime.strptime(str(raw_date), "%Y-%m-%d").date()
                    if dt >= today:
                        return dt
                except ValueError:
                    continue

            return None

        except Exception as exc:
            logger.warning("EODHD earnings fetch failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(self, ticker: str, limit: int = 10) -> list[dict]:
        """
        Fetch recent news headlines for *ticker*.

        Returns:
            List of dicts with keys: title, date, url.
            Empty list on failure.
        """
        if not self.available:
            return []

        try:
            url = f"{_BASE_URL}/news"
            resp = requests.get(
                url,
                params={
                    "s": ticker,
                    "api_token": self.api_token,
                    "limit": limit,
                    "fmt": "json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                return []

            results = []
            for article in data[:limit]:
                title = article.get("title")
                if not title:
                    continue
                results.append({
                    "title": title,
                    "date": article.get("date", ""),
                    "url": article.get("link", ""),
                })
            return results

        except Exception as exc:
            logger.warning("EODHD news fetch failed for %s: %s", ticker, exc)
            return []
