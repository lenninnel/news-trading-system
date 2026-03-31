"""
Per-ticker market regime detector.

Unlike RegimeAgent (broad-market SPY/VIX), this classifies the regime
for *each individual ticker* using its own ADX, ATR percentile, and
price vs SMA50 — combined with the broad-market VIX level.

The result controls:
  - Which strategies are active for this ticker/session
  - Position size multiplier (reduced in HIGH_VOLATILITY / TRANSITIONAL)
  - Conviction threshold adjustments

Regimes
-------
TRENDING_UP       ADX > 25, VIX < 25, price > SMA50 → Momentum full size
TRENDING_DOWN     ADX > 25, VIX < 25, price < SMA50 → Pullback active
RANGING           ADX < 20, VIX < 20 → Pullback active, Momentum reduced
HIGH_VOLATILITY   VIX > 25 or ATR > 80th pct → all strategies, 50% size
TRANSITIONAL      ADX 20-25 or VIX 20-25 → 75% size, higher conviction needed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────

ADX_TRENDING = 25.0
ADX_RANGING = 20.0
VIX_HIGH = 25.0
VIX_LOW = 20.0
ATR_HIGH_PERCENTILE = 80


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class RegimeResult:
    """Output of per-ticker regime detection."""

    regime: str
    adx: float
    vix: float
    atr_percentile: float
    price: float
    sma50: float
    size_multiplier: float
    allowed_strategies: list[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in ("TRENDING_UP", "TRENDING_DOWN")


# ── ADX computation ──────────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the Average Directional Index from OHLC data.

    Returns the most recent ADX value, or 0.0 if data is insufficient.
    """
    if len(df) < period * 2 + 1:
        return 0.0

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # True Range, +DM, -DM
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    plus_dm = np.where(
        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
        np.maximum(high[1:] - high[:-1], 0),
        0.0,
    )
    minus_dm = np.where(
        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
        np.maximum(low[:-1] - low[1:], 0),
        0.0,
    )

    # Wilder smoothing
    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    smooth_plus = np.zeros(len(plus_dm))
    smooth_plus[period - 1] = np.mean(plus_dm[:period])
    for i in range(period, len(plus_dm)):
        smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period

    smooth_minus = np.zeros(len(minus_dm))
    smooth_minus[period - 1] = np.mean(minus_dm[:period])
    for i in range(period, len(minus_dm)):
        smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI, -DI, DX
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr > 0, 100 * smooth_plus / atr, 0)
        minus_di = np.where(atr > 0, 100 * smooth_minus / atr, 0)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)

    # ADX = smoothed DX
    valid_dx = dx[period - 1:]
    if len(valid_dx) < period:
        return float(dx[-1]) if len(dx) > 0 else 0.0

    adx_vals = np.zeros(len(valid_dx))
    adx_vals[period - 1] = np.mean(valid_dx[:period])
    for i in range(period, len(valid_dx)):
        adx_vals[i] = (adx_vals[i - 1] * (period - 1) + valid_dx[i]) / period

    return float(adx_vals[-1])


def _compute_atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 100) -> float:
    """Compute current ATR as percentile of recent ATR values (0-100)."""
    if len(df) < period + lookback:
        return 50.0  # neutral default

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )

    # Rolling ATR
    atrs = []
    for i in range(period - 1, len(tr)):
        atrs.append(np.mean(tr[max(0, i - period + 1):i + 1]))

    if len(atrs) < lookback:
        return 50.0

    recent = atrs[-lookback:]
    current = atrs[-1]
    percentile = float(np.sum(np.array(recent) <= current) / len(recent) * 100)
    return percentile


# ── Detector class ───────────────────────────────────────────────────

class RegimeDetector:
    """Classify per-ticker market regime using ADX, VIX, ATR percentile."""

    REGIMES = [
        "TRENDING_UP",
        "TRENDING_DOWN",
        "RANGING",
        "HIGH_VOLATILITY",
        "TRANSITIONAL",
    ]

    ALL_STRATEGIES = ["Momentum", "Pullback", "NewsCatalyst", "Combined"]

    def detect(
        self,
        ticker: str,
        prices: pd.DataFrame,
        vix: float | None = None,
    ) -> RegimeResult:
        """Detect the market regime for a single ticker.

        Args:
            ticker: Stock symbol.
            prices: DataFrame with Open/High/Low/Close/Volume columns.
            vix: Current VIX level (broad market). None = unavailable.

        Returns:
            RegimeResult with regime classification and sizing parameters.
        """
        vix_val = vix if vix is not None else 18.0  # neutral default

        if prices is None or prices.empty or len(prices) < 50:
            return RegimeResult(
                regime="TRANSITIONAL",
                adx=0.0,
                vix=vix_val,
                atr_percentile=50.0,
                price=0.0,
                sma50=0.0,
                size_multiplier=0.75,
                allowed_strategies=self.ALL_STRATEGIES,
            )

        close = prices["Close"]
        current_price = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])

        adx = _compute_adx(prices)
        atr_pct = _compute_atr_percentile(prices)

        # ── Classification ────────────────────────────────────────
        # Priority: HIGH_VOLATILITY > TRANSITIONAL > TRENDING > RANGING

        if vix_val > VIX_HIGH or atr_pct > ATR_HIGH_PERCENTILE:
            regime = "HIGH_VOLATILITY"
            size_mult = 0.50
            allowed = self.ALL_STRATEGIES
        elif (ADX_RANGING <= adx <= ADX_TRENDING) or (VIX_LOW <= vix_val <= VIX_HIGH):
            regime = "TRANSITIONAL"
            size_mult = 0.75
            allowed = self.ALL_STRATEGIES
        elif adx > ADX_TRENDING and current_price > sma50:
            regime = "TRENDING_UP"
            size_mult = 1.0
            allowed = ["Momentum", "NewsCatalyst", "Combined"]
        elif adx > ADX_TRENDING and current_price <= sma50:
            regime = "TRENDING_DOWN"
            size_mult = 1.0
            allowed = ["Pullback", "NewsCatalyst", "Combined"]
        elif adx < ADX_RANGING and vix_val < VIX_LOW:
            regime = "RANGING"
            size_mult = 1.0
            allowed = ["Pullback", "NewsCatalyst", "Combined"]
        else:
            regime = "TRANSITIONAL"
            size_mult = 0.75
            allowed = self.ALL_STRATEGIES

        result = RegimeResult(
            regime=regime,
            adx=round(adx, 1),
            vix=round(vix_val, 1),
            atr_percentile=round(atr_pct, 1),
            price=round(current_price, 2),
            sma50=round(sma50, 2),
            size_multiplier=size_mult,
            allowed_strategies=allowed,
        )

        log.info(
            "[%s] Regime: %s (ADX=%.1f, VIX=%.1f, ATR_pct=%.0f, size=%.0f%%)",
            ticker, regime, adx, vix_val, atr_pct, size_mult * 100,
        )

        return result
