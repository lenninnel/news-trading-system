"""IBKR bar data feed with yfinance fallback.

Thin wrapper around IBKRTrader.get_bars() that gracefully falls back
to YFinanceFeed when IB Gateway is unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class IBKRFeed:
    """Fetch OHLCV bars from IBKR, falling back to yfinance on failure."""

    def __init__(self, ib=None) -> None:
        self._ib = ib  # IBKRTrader instance (or mock for tests)
        self._yf = None  # lazy-loaded YFinanceFeed

    def _get_ib(self):
        """Lazily connect to IBKR if no instance was injected."""
        if self._ib is not None:
            return self._ib
        try:
            from execution.ibkr_trader import IBKRTrader
            self._ib = IBKRTrader()
            return self._ib
        except Exception as exc:
            log.warning("IBKR connection failed, will use yfinance: %s", exc)
            return None

    def _get_yf(self):
        """Lazily create YFinanceFeed."""
        if self._yf is None:
            from data.yfinance_feed import YFinanceFeed
            self._yf = YFinanceFeed()
        return self._yf

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 252,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from IBKR, falling back to yfinance.

        Args:
            ticker:    Stock symbol.
            timeframe: ``"1Day"``, ``"1Hour"``, or ``"15Min"``.
            limit:     Number of bars.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
        """
        ib = self._get_ib()
        if ib is not None:
            try:
                df = ib.get_bars(ticker, timeframe, limit)
                log.debug("IBKR feed: %s %s → %d bars", ticker, timeframe, len(df))
                return df
            except Exception as exc:
                log.warning(
                    "IBKR bars failed for %s (%s), falling back to yfinance: %s",
                    ticker, timeframe, exc,
                )

        # Fallback to yfinance
        log.debug("Fallback to yfinance for %s %s", ticker, timeframe)
        return self._get_yf().get_bars(ticker, timeframe, limit)
