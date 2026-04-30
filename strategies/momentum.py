"""
MomentumStrategy — trend-following strategy with volume confirmation.

Indicators: RSI(14), Volume ratio (current / 20-day avg), SMA20, SMA50
Entry conditions (all four = strongest signal):
    1. Price > SMA20 > SMA50  (uptrend)
    2. RSI between 50–65       (momentum but not overbought)
    3. Volume > 1.3x average   (confirmation)
    4. Sentiment aligns         (BUY or WEAK BUY)

Confidence:  4/4 → 70-85%  |  3/4 → 45-60%  |  <=2 → HOLD
Stop-loss:   2.0% below entry
Take-profit: 3x ATR above entry
Best tickers: META, JPM (validated 2026-03-20 backtest)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import ta

from strategies.base import BaseStrategy, StrategyResult

log = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """Trend-following momentum strategy with volume confirmation."""

    PREFERRED_TICKERS = ["META", "JPM"]

    @property
    def name(self) -> str:
        return "Momentum"

    def analyze(
        self,
        ticker: str,
        bars: pd.DataFrame,
        sentiment_signal: str = "HOLD",
    ) -> StrategyResult:
        ticker = ticker.upper()
        try:
            ind = self._calculate_indicators(bars)
            signal, confidence, reasoning = self._apply_rules(ind, sentiment_signal)
            entry = ind["price"]
            stop = round(entry * 0.98, 4) if entry else None
            atr = ind.get("atr")
            tp = round(entry + 3 * atr, 4) if entry and atr else None
        except Exception as exc:
            log.error("MomentumStrategy failed for %s: %s", ticker, exc)
            return StrategyResult(
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=[f"Error: {exc}"],
            )

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            indicators=ind,
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            timeframe_alignment=True,
            strategy_name=self.name,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calculate_indicators(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # RSI-14
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # SMA-20, SMA-50, and SMA-200
        sma20_series = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50_series = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        sma200_series = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

        # Volume ratio: current bar vs 20-day average
        vol_avg = volume.rolling(20).mean()
        vol_last = float(volume.iloc[-1])
        vol_avg_val = float(vol_avg.iloc[-1]) if float(vol_avg.iloc[-1]) > 0 else 1.0
        vol_ratio = vol_last / vol_avg_val

        # ATR-14 (for take-profit calculation)
        atr_series = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14,
        ).average_true_range()

        price = float(close.iloc[-1])
        rsi = self._latest(rsi_series)
        sma20 = self._latest(sma20_series)
        sma50 = self._latest(sma50_series)
        sma200 = self._latest(sma200_series)
        atr = self._latest(atr_series)

        return {
            "price": price,
            "rsi": rsi,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "vol_ratio": round(vol_ratio, 3),
            "atr": atr,
        }

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rules(
        ind: dict[str, Any],
        sentiment_signal: str,
    ) -> tuple[str, float, list[str]]:
        conditions_met = 0
        reasoning: list[str] = []

        price = ind.get("price")
        rsi = ind.get("rsi")
        sma20 = ind.get("sma20")
        sma50 = ind.get("sma50")
        vol_ratio = ind.get("vol_ratio") or 0.0

        # Guard: need valid indicators
        if price is None or sma20 is None or sma50 is None or rsi is None:
            return "HOLD", 0.0, ["Insufficient indicator data"]

        # Condition 1: uptrend (price > SMA20 > SMA50)
        if price > sma20 > sma50:
            conditions_met += 1
            reasoning.append(
                f"Uptrend: price {price:.2f} > SMA20 {sma20:.2f} > SMA50 {sma50:.2f}"
            )

        # Condition 2: RSI in momentum sweet spot (50–65)
        if 50 <= rsi <= 65:
            conditions_met += 1
            reasoning.append(f"RSI {rsi:.1f} in momentum zone (50-65)")

        # Condition 3: volume confirmation (> 1.3x average)
        if vol_ratio > 1.3:
            conditions_met += 1
            reasoning.append(f"Volume {vol_ratio:.1f}x above average (>1.3x)")

        # Condition 4: sentiment alignment
        sentiment_upper = sentiment_signal.strip().upper()
        if sentiment_upper in (
            "BUY", "STRONG BUY", "WEAK BUY",
            "STRONG_BUY", "WEAK_BUY",
        ):
            conditions_met += 1
            reasoning.append(f"Sentiment aligned: {sentiment_signal}")

        log.debug(
            "Momentum conditions=%d price=%.2f sma20=%s sma50=%s rsi=%s vol=%.2f sent=%s",
            conditions_met, price,
            f"{sma20:.2f}" if sma20 else "N/A",
            f"{sma50:.2f}" if sma50 else "N/A",
            f"{rsi:.1f}" if rsi else "N/A",
            vol_ratio,
            sentiment_upper,
        )

        # Signal and confidence based on conditions met.
        # Confidence varies within each band based on indicator *strength*,
        # not just the branch count — otherwise every 3/4 BUY collapses to
        # the same constant and the logged value looks like a default.
        if conditions_met >= 4:
            # Full alignment: 70-85% confidence
            # Scale within range based on RSI proximity to sweet-spot center (57.5)
            rsi_bonus = max(0, 15 - abs(rsi - 57.5))  # 0-15 bonus
            confidence = 70.0 + rsi_bonus
            signal, confidence = "STRONG BUY", min(confidence, 85.0)
        elif conditions_met == 3:
            # Strong setup missing one piece: 45-60%, scaled by how
            # strongly the triggered indicators cleared their thresholds.
            rsi_bonus = max(0.0, 5.0 - abs(rsi - 57.5) * 0.33)       # 0-5
            vol_bonus = min(5.0, max(0.0, (vol_ratio - 1.3) * 2.5))  # 0-5
            trend_bonus = 0.0
            if sma50 and sma50 > 0 and price:
                trend_bonus = min(5.0, max(0.0, (price / sma50 - 1.0) * 50.0))  # 0-5
            confidence = 45.0 + rsi_bonus + vol_bonus + trend_bonus
            signal, confidence = "BUY", min(confidence, 60.0)
        else:
            # 2 or fewer conditions — no actionable signal
            if not reasoning:
                reasoning.append("No momentum conditions triggered")
            # HOLD confidence scales 10-40 with conditions_met / 4 so a
            # 0/4 HOLD reads weaker than a 2/4 near-miss. Total conditions = 4.
            signal = "HOLD"
            confidence = 10.0 + (conditions_met / 4.0) * 30.0

        # ── Downtrend filter: prevent catching falling knives ──
        sma200 = ind.get("sma200")
        if sma200 and price and sma200 > 0:
            sma_ratio = price / sma200
            if sma_ratio < 0.70:
                # Suppression of an otherwise-actionable setup is high-conviction
                # HOLD \u2014 keep conditions_met-based score so a 3/4 setup blocked
                # by the downtrend reads above a 0/4 HOLD.
                signal = "HOLD"
                confidence = 10.0 + (conditions_met / 4.0) * 30.0
                reasoning.append("Extreme downtrend \u2014 signal suppressed")
            elif sma_ratio < 0.80 and signal in ("BUY", "STRONG BUY"):
                signal = "WEAK BUY"
                confidence = min(confidence, 55.0)
                reasoning.append(
                    f"Severe downtrend filter applied (SMA ratio: {sma_ratio:.2f})"
                )

        return signal, confidence, reasoning
