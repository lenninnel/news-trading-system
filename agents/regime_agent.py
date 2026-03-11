"""
Market regime detection agent.

Downloads SPY and VIX data via yfinance and classifies the current market
environment into one of four regimes.  The result is cached for 4 hours so
that multiple per-ticker pipeline runs in a session avoid redundant
downloads.

Regimes
-------
HIGH_VOL       VIX > 30 *or* 20-day realised vol > 25% annualised.
TRENDING_BULL  SPY SMA-50 > SMA-200 (golden cross territory) and not HIGH_VOL.
TRENDING_BEAR  SPY SMA-50 < SMA-200 (death cross territory) and not HIGH_VOL.
RANGING        Everything else.

Requires: yfinance, numpy
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent

_CACHE_TTL = 4 * 3600  # 4 hours in seconds

_VIX_THRESHOLD = 30.0
_REALISED_VOL_THRESHOLD = 0.25  # annualised


class RegimeAgent(BaseAgent):
    """
    Determines the current broad-market regime.

    The result is cached in memory for 4 hours.  Pass ``_yf`` to inject a
    mock yfinance module in tests.
    """

    def __init__(self, *, _yf: Any = None) -> None:
        self._yf = _yf
        self._cache: dict | None = None
        self._cache_ts: float = 0.0

    @property
    def name(self) -> str:
        return "RegimeAgent"

    def run(self, **kwargs: Any) -> dict:
        """
        Return the current market regime.

        Returns:
            dict with keys:
                regime         (str):   TRENDING_BULL | TRENDING_BEAR |
                                        RANGING | HIGH_VOL
                vix            (float|None): Latest VIX close.
                realised_vol   (float): 20-day annualised realised volatility.
                sma50          (float): SPY 50-day SMA.
                sma200         (float): SPY 200-day SMA.
                cached         (bool):  True if result came from cache.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL:
            return {**self._cache, "cached": True}

        yf = self._yf
        if yf is None:
            import yfinance as yf  # noqa: N811 — lazy import

        result = self._detect(yf)
        self._cache = result
        self._cache_ts = now
        return {**result, "cached": False}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect(yf: Any) -> dict:
        """Download data and classify the regime."""
        # --- SPY data ---
        spy = yf.download("SPY", period="250d", progress=False)
        close = spy["Close"].squeeze()

        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        # 20-day realised volatility (annualised)
        returns = close.pct_change().dropna().tail(20)
        realised_vol = float(np.std(returns, ddof=1) * np.sqrt(252))

        # --- VIX ---
        vix: float | None = None
        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix_close = vix_data["Close"].squeeze()
            if len(vix_close) > 0:
                vix = float(vix_close.iloc[-1])
        except Exception:
            pass  # fall back to realised_vol only

        # --- Classification ---
        high_vol = (vix is not None and vix > _VIX_THRESHOLD) or \
                   realised_vol > _REALISED_VOL_THRESHOLD

        if high_vol:
            regime = "HIGH_VOL"
        elif sma50 > sma200:
            regime = "TRENDING_BULL"
        elif sma50 < sma200:
            regime = "TRENDING_BEAR"
        else:
            regime = "RANGING"

        return {
            "regime": regime,
            "vix": vix,
            "realised_vol": round(realised_vol, 4),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
        }
