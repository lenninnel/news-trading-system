"""
Market regime detection agent.

Downloads SPY and VIX data via the Alpaca Data API and classifies the
current market environment into one of four regimes.  The result is cached
for 4 hours so that multiple per-ticker pipeline runs in a session avoid
redundant downloads.

For VIX (^VIX), Alpaca does not provide index data, so we fall back to
yfinance or FRED.

Regimes
-------
HIGH_VOL       VIX > 30 *or* 20-day realised vol > 25% annualised.
TRENDING_BULL  SPY SMA-50 > SMA-200 (golden cross territory) and not HIGH_VOL.
TRENDING_BEAR  SPY SMA-50 < SMA-200 (death cross territory) and not HIGH_VOL.
RANGING        Everything else.

Requires: alpaca-trade-api, numpy
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
    mock yfinance module in tests (used for VIX which Alpaca does not cover).
    Pass ``_fear_greed_fn`` and/or ``_fred_feed`` to inject mock data sources
    for Fear & Greed and FRED.
    """

    def __init__(
        self,
        *,
        _yf: Any = None,
        _fear_greed_fn: Any = None,
        _fred_feed: Any = None,
    ) -> None:
        self._yf = _yf
        self._fear_greed_fn = _fear_greed_fn
        self._fred_feed = _fred_feed
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
                fear_greed     (int|None): Crypto Fear & Greed Index (0-100).
                fear_greed_label (str|None): Classification (e.g. "Fear").
                yield_curve    (float|None): 10Y-2Y spread from FRED.
                cached         (bool):  True if result came from cache.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL:
            return {**self._cache, "cached": True}

        yf = self._yf
        if yf is None:
            import yfinance as yf  # noqa: N811 — lazy import for VIX only

        # Fetch supplementary macro data (never fails the pipeline)
        fear_greed_data: dict | None = None
        try:
            fg_fn = self._fear_greed_fn
            if fg_fn is None:
                from data.fear_greed_feed import get_fear_greed as fg_fn
            fear_greed_data = fg_fn()
        except Exception:
            pass

        fred_data: dict | None = None
        try:
            fred = self._fred_feed
            if fred is None:
                from data.fred_feed import FredFeed
                fred = FredFeed()
            fred_data = fred.get_macro_regime()
        except Exception:
            pass

        result = self._detect(yf, fear_greed=fear_greed_data, fred_data=fred_data)
        self._cache = result
        self._cache_ts = now
        return {**result, "cached": False}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect(
        yf: Any,
        *,
        fear_greed: dict | None = None,
        fred_data: dict | None = None,
    ) -> dict:
        """Download data and classify the regime.

        SPY data is fetched via Alpaca (reliable).  VIX is fetched via
        yfinance (Alpaca does not cover indices like ^VIX).
        """
        import pandas as pd

        # --- SPY data via Alpaca ---
        try:
            from data.alpaca_data import AlpacaDataClient
            alpaca = AlpacaDataClient()
            spy = alpaca.get_bars("SPY", "1Day", limit=252)
        except Exception:
            # Fallback to yfinance if Alpaca fails (e.g. in tests with mock yf)
            spy = yf.download("SPY", period="250d", progress=False)

        close = spy["Close"].squeeze()

        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        # 20-day realised volatility (annualised)
        returns = close.pct_change().dropna().tail(20)
        realised_vol = float(np.std(returns, ddof=1) * np.sqrt(252))

        # --- VIX (Alpaca does not cover ^VIX — use yfinance) ---
        vix: float | None = None
        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix_close = vix_data["Close"].squeeze()
            if len(vix_close) > 0:
                vix = float(vix_close.iloc[-1])
        except Exception:
            pass  # fall back to realised_vol only

        # Fall back to FRED VIX if yfinance VIX failed
        if vix is None and fred_data and fred_data.get("vix") is not None:
            vix = fred_data["vix"]

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

        result: dict = {
            "regime": regime,
            "vix": vix,
            "realised_vol": round(realised_vol, 4),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
        }

        # Attach supplementary macro data when available
        if fear_greed:
            result["fear_greed"] = fear_greed.get("value")
            result["fear_greed_label"] = fear_greed.get("classification")

        if fred_data:
            result["yield_curve"] = fred_data.get("yield_curve")

        return result
