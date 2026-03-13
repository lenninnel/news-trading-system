"""
FRED macro data feed using the fredapi package.

Fetches VIX, yield curve (10Y-2Y spread), and S&P 500 level.
Results are cached for 24 hours.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import fredapi

logger = logging.getLogger(__name__)

_cache: dict | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 24 * 3600  # 24 hours


class FredFeed:
    """Fetches macro indicators from the FRED API."""

    def __init__(self, api_key: str = "") -> None:
        if not api_key:
            from config.settings import FRED_API_KEY

            api_key = FRED_API_KEY
        self._api_key = api_key

    def get_macro_regime(self) -> dict | None:
        """
        Fetch macro indicators: VIX, yield curve, S&P 500.

        Returns:
            dict with keys:
                vix: float | None (VIXCLS - VIX close)
                yield_curve: float | None (T10Y2Y - 10Y minus 2Y spread,
                    negative = recession signal)
                sp500: float | None (SP500 - S&P 500 level)
            Returns None if FRED_API_KEY is not set.
        """
        global _cache, _cache_ts

        if not self._api_key:
            logger.debug("FRED API key not set; skipping macro fetch")
            return None

        # Return cached result if still fresh
        if _cache is not None and (time.time() - _cache_ts) < _CACHE_TTL:
            logger.debug("FRED cache hit")
            return _cache

        try:
            fred = fredapi.Fred(api_key=self._api_key)
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

            vix = self._last_valid(fred, "VIXCLS", start_date)
            yield_curve = self._last_valid(fred, "T10Y2Y", start_date)
            sp500 = self._last_valid(fred, "SP500", start_date)

            result = {
                "vix": vix,
                "yield_curve": yield_curve,
                "sp500": sp500,
            }

            _cache = result
            _cache_ts = time.time()
            logger.info(
                "FRED macro: VIX=%.2f, yield_curve=%.2f, SP500=%.1f",
                vix or 0,
                yield_curve or 0,
                sp500 or 0,
            )
            return result

        except Exception as exc:
            logger.error("FRED fetch failed: %s", exc)
            return None

    @staticmethod
    def _last_valid(
        fred: Any, series_id: str, start_date: str
    ) -> float | None:
        """Get the last non-NaN value from a FRED series."""
        try:
            series = fred.get_series(series_id, observation_start=start_date)
            series = series.dropna()
            if series.empty:
                return None
            return float(series.iloc[-1])
        except Exception as exc:
            logger.warning("Could not fetch FRED series %s: %s", series_id, exc)
            return None


def clear_cache() -> None:
    """Clear the module-level cache."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
