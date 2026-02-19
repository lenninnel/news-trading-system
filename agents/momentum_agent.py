"""
MomentumAgent — Trend-following and breakout strategy.

Entry conditions (any one set triggers the signal):

    BUY set A — confirmed breakout:
        Price closes above 20-day rolling high
        + ADX > 25 (strong trend)
        + Volume ratio > 1.2× (volume participation)
        + +DI > -DI (trend is bullish)

    BUY set B — uptrend continuation:
        EMA-20 > EMA-50 (short-term above mid-term trend)
        + Price above EMA-20
        + ROC-10 > 3% (positive momentum)

    SELL mirrors BUY with reversed conditions.

Confidence:
    Base 50 pts
    + 15 if ADX > 25   (trend established)
    + 10 if ADX > 35   (very strong trend)
    + 10 if vol_ratio > 1.5× (strong volume)
    + 15 if |ROC| > 5% (strong momentum)
    Clamped to [30, 90].

Timeframe: "position" (days to weeks)

Requires:
    pip install yfinance ta pandas
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import ta

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.strategy_base import StrategyAgent, StrategySignal
from storage.database import Database

log = logging.getLogger(__name__)


class MomentumAgent(StrategyAgent):
    """
    Momentum / breakout strategy agent.

    Detects strong trending moves and breakouts above multi-week highs,
    confirmed by ADX trend strength and above-average volume.
    """

    _DOWNLOAD_PERIOD = "6mo"

    def __init__(self, db: Database | None = None) -> None:
        super().__init__(db=db)

    @property
    def name(self) -> str:
        return "MomentumAgent"

    def run(self, ticker: str, **kwargs: Any) -> StrategySignal:
        ticker = ticker.upper()
        try:
            df     = self._fetch_history(ticker)
            ind    = self._calculate_indicators(df)
            signal, reasoning, confidence = self._apply_signal_rules(ind)
        except Exception as exc:
            log.error("MomentumAgent failed for %s: %s", ticker, exc)
            return StrategySignal(
                ticker=ticker,
                strategy="momentum",
                signal="HOLD",
                confidence=25.0,
                reasoning=[f"Error: {exc}"],
                indicators={},
                timeframe="position",
            )

        return StrategySignal(
            ticker=ticker,
            strategy="momentum",
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            indicators=ind,
            timeframe="position",
        )

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calculate_indicators(self, df) -> dict:
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # EMA-20 and EMA-50 (trend direction)
        ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

        # ADX (14-period) — trend strength + directional indicators
        adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        adx     = adx_obj.adx()
        adx_pos = adx_obj.adx_pos()   # +DI
        adx_neg = adx_obj.adx_neg()   # -DI

        # Volume ratio: today vs 20-day rolling average
        vol_avg   = volume.rolling(20).mean()
        vol_last  = float(volume.iloc[-1])
        vol_avg_v = float(vol_avg.iloc[-1]) if float(vol_avg.iloc[-1]) > 0 else 1.0
        vol_ratio = vol_last / vol_avg_v

        # 20-day rolling high/low (use iloc[-2] to avoid same-bar look-ahead)
        high_20 = float(high.rolling(20).max().iloc[-2]) if len(high) >= 21 else float(high.max())
        low_20  = float(low.rolling(20).min().iloc[-2])  if len(low)  >= 21 else float(low.min())

        # Rate of Change (10-period)
        roc = ta.momentum.ROCIndicator(close=close, window=10).roc()

        price = float(close.iloc[-1]) if not close.empty else None

        return {
            "price":     price,
            "ema20":     self._latest(ema20),
            "ema50":     self._latest(ema50),
            "adx":       self._latest(adx),
            "adx_pos":   self._latest(adx_pos),
            "adx_neg":   self._latest(adx_neg),
            "vol_ratio": round(vol_ratio, 3),
            "high_20":   round(high_20, 4),
            "low_20":    round(low_20, 4),
            "roc":       self._latest(roc),
        }

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_signal_rules(ind: dict) -> tuple[str, list[str], float]:
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []

        price     = ind.get("price")
        ema20     = ind.get("ema20")
        ema50     = ind.get("ema50")
        adx       = ind.get("adx")     or 0.0
        adx_pos   = ind.get("adx_pos") or 0.0
        adx_neg   = ind.get("adx_neg") or 0.0
        vol_ratio = ind.get("vol_ratio") or 1.0
        high_20   = ind.get("high_20")
        low_20    = ind.get("low_20")
        roc       = ind.get("roc") or 0.0

        # BUY set A — confirmed breakout above 20-day high
        if (price is not None and high_20 is not None
                and price > high_20
                and adx > 25
                and vol_ratio > 1.2
                and adx_pos > adx_neg):
            buy_reasons.append(
                f"Breakout above 20-day high {high_20:.2f}"
                f" (ADX {adx:.1f}, vol {vol_ratio:.1f}x, +DI > -DI)"
            )

        # BUY set B — uptrend continuation
        if (price is not None and ema20 is not None and ema50 is not None
                and ema20 > ema50
                and price > ema20
                and roc > 3.0):
            buy_reasons.append(
                f"Uptrend: price {price:.2f} > EMA20 {ema20:.2f} > EMA50 {ema50:.2f}"
                f", ROC {roc:+.1f}%"
            )

        # SELL set A — confirmed breakdown below 20-day low
        if (price is not None and low_20 is not None
                and price < low_20
                and adx > 25
                and vol_ratio > 1.2
                and adx_neg > adx_pos):
            sell_reasons.append(
                f"Breakdown below 20-day low {low_20:.2f}"
                f" (ADX {adx:.1f}, vol {vol_ratio:.1f}x, -DI > +DI)"
            )

        # SELL set B — downtrend continuation
        if (price is not None and ema20 is not None and ema50 is not None
                and ema20 < ema50
                and price < ema20
                and roc < -3.0):
            sell_reasons.append(
                f"Downtrend: price {price:.2f} < EMA20 {ema20:.2f} < EMA50 {ema50:.2f}"
                f", ROC {roc:+.1f}%"
            )

        # Confidence
        confidence = 50.0
        if adx > 25:          confidence += 15.0
        if adx > 35:          confidence += 10.0
        if vol_ratio > 1.5:   confidence += 10.0
        if abs(roc) > 5.0:    confidence += 15.0
        confidence = min(max(confidence, 30.0), 90.0)

        if buy_reasons:
            return "BUY",  buy_reasons,  confidence
        if sell_reasons:
            return "SELL", sell_reasons, confidence
        return "HOLD", ["No momentum breakout conditions triggered"], 25.0
