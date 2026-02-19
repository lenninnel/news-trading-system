"""
MeanReversionAgent — Oversold / overbought bounce strategy.

Entry conditions:

    BUY (high-confidence, all four required):
        RSI-14 < 30 (oversold)
        + Stochastic %K < 20 (stochastic oversold)
        + Bollinger %B < 0 (price below lower band)
        + Green candle (today's close > yesterday's — first sign of reversal)

    BUY (moderate, RSI + one oscillator):
        RSI-14 < 35
        + (Stochastic %K < 25  OR  Williams %R < -80)

    SELL mirrors BUY with reversed thresholds.

Confidence:
    High-confidence setup → 80%
    Moderate setup        → 60%
    HOLD                  → 25%

Timeframe: "swing" (2–10 days)

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


class MeanReversionAgent(StrategyAgent):
    """
    Mean-reversion / oversold-bounce strategy agent.

    Detects extreme oversold or overbought conditions using three
    oscillators (RSI, Stochastic, Williams %R) plus Bollinger Band %B.
    Requires a green candle for high-confidence BUY entries.
    """

    _DOWNLOAD_PERIOD = "3mo"

    def __init__(self, db: Database | None = None) -> None:
        super().__init__(db=db)

    @property
    def name(self) -> str:
        return "MeanReversionAgent"

    def run(self, ticker: str, **kwargs: Any) -> StrategySignal:
        ticker = ticker.upper()
        try:
            df     = self._fetch_history(ticker)
            ind    = self._calculate_indicators(df)
            signal, reasoning, confidence = self._apply_signal_rules(ind)
        except Exception as exc:
            log.error("MeanReversionAgent failed for %s: %s", ticker, exc)
            return StrategySignal(
                ticker=ticker,
                strategy="mean_reversion",
                signal="HOLD",
                confidence=25.0,
                reasoning=[f"Error: {exc}"],
                indicators={},
                timeframe="swing",
            )

        return StrategySignal(
            ticker=ticker,
            strategy="mean_reversion",
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            indicators=ind,
            timeframe="swing",
        )

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calculate_indicators(self, df) -> dict:
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()

        # RSI-14
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # Stochastic Oscillator (14, 3, 3)
        stoch_obj = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        stoch_k = stoch_obj.stoch()
        stoch_d = stoch_obj.stoch_signal()

        # Bollinger Bands — %B  (0 = lower band, 1 = upper band, <0 = below lower)
        bb_obj   = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_pct_b = bb_obj.bollinger_pband()
        bb_upper = bb_obj.bollinger_hband()
        bb_lower = bb_obj.bollinger_lband()
        bb_mid   = bb_obj.bollinger_mavg()

        # Williams %R (14-period)  — range [-100, 0];  < -80 oversold, > -20 overbought
        williams_r = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14
        ).williams_r()

        price      = float(close.iloc[-1]) if not close.empty else None
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
        is_green   = (price is not None and prev_close is not None and price > prev_close)

        return {
            "price":      price,
            "rsi":        self._latest(rsi),
            "stoch_k":    self._latest(stoch_k),
            "stoch_d":    self._latest(stoch_d),
            "bb_pct_b":   self._latest(bb_pct_b),
            "bb_upper":   self._latest(bb_upper),
            "bb_lower":   self._latest(bb_lower),
            "bb_mid":     self._latest(bb_mid),
            "williams_r": self._latest(williams_r),
            "is_green":   is_green,
        }

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_signal_rules(ind: dict) -> tuple[str, list[str], float]:
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []
        confidence = 25.0

        rsi        = ind.get("rsi")        or 50.0
        stoch_k    = ind.get("stoch_k")    or 50.0
        bb_pct_b   = ind.get("bb_pct_b")   or 0.5
        williams_r = ind.get("williams_r") or -50.0
        is_green   = ind.get("is_green",   False)
        bb_lower   = ind.get("bb_lower")
        bb_upper   = ind.get("bb_upper")
        price      = ind.get("price")

        bb_lower_str = f"{bb_lower:.2f}" if bb_lower is not None else "N/A"
        bb_upper_str = f"{bb_upper:.2f}" if bb_upper is not None else "N/A"
        price_str    = f"{price:.2f}"    if price    is not None else "N/A"

        # BUY — high-confidence: all four conditions
        if rsi < 30 and stoch_k < 20 and bb_pct_b < 0 and is_green:
            buy_reasons.append(
                f"Triple oversold + reversal candle: RSI {rsi:.1f} < 30, "
                f"Stoch %K {stoch_k:.1f} < 20, "
                f"price {price_str} below lower BB ({bb_lower_str}), "
                f"green candle"
            )
            confidence = 80.0

        # BUY — moderate: RSI + one oscillator
        elif rsi < 35 and (stoch_k < 25 or williams_r < -80):
            parts = [f"RSI {rsi:.1f} < 35"]
            if stoch_k < 25:    parts.append(f"Stoch %K {stoch_k:.1f} < 25")
            if williams_r < -80: parts.append(f"Williams %R {williams_r:.1f} < -80")
            buy_reasons.append("Oversold: " + ", ".join(parts))
            confidence = 60.0

        # SELL — high-confidence: all three
        elif rsi > 70 and stoch_k > 80 and bb_pct_b > 1:
            sell_reasons.append(
                f"Triple overbought: RSI {rsi:.1f} > 70, "
                f"Stoch %K {stoch_k:.1f} > 80, "
                f"price {price_str} above upper BB ({bb_upper_str})"
            )
            confidence = 80.0

        # SELL — moderate
        elif rsi > 65 and (stoch_k > 75 or williams_r > -20):
            parts = [f"RSI {rsi:.1f} > 65"]
            if stoch_k > 75:     parts.append(f"Stoch %K {stoch_k:.1f} > 75")
            if williams_r > -20: parts.append(f"Williams %R {williams_r:.1f} > -20")
            sell_reasons.append("Overbought: " + ", ".join(parts))
            confidence = 60.0

        if buy_reasons:
            return "BUY",  buy_reasons,  confidence
        if sell_reasons:
            return "SELL", sell_reasons, confidence
        return "HOLD", ["No extreme oversold/overbought conditions detected"], 25.0
