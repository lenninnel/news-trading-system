"""
PullbackStrategy — mean-reversion bounce off SMA50 support.

Indicators: RSI(14), SMA50, distance from SMA50, Stochastic(14,3)
Entry conditions (all four = strongest signal):
    1. Price > SMA50              (uptrend confirmed)
    2. RSI dipped below 45 and now rising (bounce)
    3. Price within 3% of SMA50   (pullback not too deep)
    4. Stochastic crossing up from oversold (<30)

Confidence:  4/4 → 65-80%  |  3/4 → 40-55%  |  <=2 → HOLD
Stop-loss:   2% below entry
Take-profit: 2x risk (2:1 reward ratio)
Best tickers: AAPL, MSFT, CEG
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import ta

from strategies.base import BaseStrategy, StrategyResult

log = logging.getLogger(__name__)


class PullbackStrategy(BaseStrategy):
    """Mean-reversion pullback strategy using SMA50 as dynamic support."""

    PREFERRED_TICKERS = ["AAPL", "MSFT", "AMZN", "XOM", "CVX", "BAC", "PFE", "TSLA"]

    @property
    def name(self) -> str:
        return "Pullback"

    def analyze(
        self,
        ticker: str,
        bars: pd.DataFrame,
        sentiment_signal: str = "HOLD",
    ) -> StrategyResult:
        ticker = ticker.upper()
        try:
            ind = self._calculate_indicators(bars)
            signal, confidence, reasoning = self._apply_rules(ind)
            entry = ind["price"]
            stop = round(entry * 0.98, 4) if entry else None
            # Take-profit = 2x risk (2:1 reward ratio)
            risk = entry - stop if entry and stop else None
            tp = round(entry + 2 * risk, 4) if risk else None
        except Exception as exc:
            log.error("PullbackStrategy failed for %s: %s", ticker, exc)
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

        # RSI-14
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # SMA-50 and SMA-200
        sma50_series = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        sma200_series = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

        # Stochastic oscillator (14,3)
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=14, smooth_window=3,
        )
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()

        price = float(close.iloc[-1])
        sma50 = self._latest(sma50_series)
        rsi = self._latest(rsi_series)
        rsi_prev = self._prev(rsi_series)
        stoch_k_val = self._latest(stoch_k)
        stoch_k_prev = self._prev(stoch_k)
        stoch_d_val = self._latest(stoch_d)

        # Minimum RSI / stoch_k over last 5 bars (catches bounces that
        # started a few bars ago, not just the immediately previous bar)
        rsi_clean = rsi_series.dropna()
        rsi_min_5 = float(rsi_clean.iloc[-5:].min()) if len(rsi_clean) >= 5 else rsi_prev

        stoch_k_clean = stoch_k.dropna()
        stoch_k_min_5 = float(stoch_k_clean.iloc[-5:].min()) if len(stoch_k_clean) >= 5 else stoch_k_prev

        # Distance from SMA50 as percentage
        sma50_dist_pct = ((price - sma50) / sma50 * 100) if sma50 else None

        sma200 = self._latest(sma200_series)

        return {
            "price": price,
            "rsi": rsi,
            "rsi_prev": rsi_prev,
            "rsi_min_5": rsi_min_5,
            "sma50": sma50,
            "sma200": sma200,
            "sma50_dist_pct": round(sma50_dist_pct, 3) if sma50_dist_pct is not None else None,
            "stoch_k": stoch_k_val,
            "stoch_k_prev": stoch_k_prev,
            "stoch_k_min_5": stoch_k_min_5,
            "stoch_d": stoch_d_val,
        }

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rules(
        ind: dict[str, Any],
    ) -> tuple[str, float, list[str]]:
        conditions_met = 0
        reasoning: list[str] = []

        price = ind.get("price")
        rsi = ind.get("rsi")
        sma50 = ind.get("sma50")
        sma50_dist_pct = ind.get("sma50_dist_pct")
        stoch_k = ind.get("stoch_k")

        # 5-bar window minimums (fall back to prev-bar values for
        # backward compat with direct _apply_rules tests)
        rsi_min_5 = ind.get("rsi_min_5")
        if rsi_min_5 is None:
            rsi_min_5 = ind.get("rsi_prev")
        stoch_k_min_5 = ind.get("stoch_k_min_5")
        if stoch_k_min_5 is None:
            stoch_k_min_5 = ind.get("stoch_k_prev")

        # Guard: need valid indicators
        if price is None or sma50 is None or rsi is None or stoch_k is None:
            return "HOLD", 0.0, ["Insufficient indicator data"]

        # Condition 1: uptrend confirmed (price > SMA50)
        if price > sma50:
            conditions_met += 1
            reasoning.append(
                f"Uptrend: price {price:.2f} > SMA50 {sma50:.2f}"
            )

        # Condition 2: RSI dipped below 45 within last 5 bars and is
        # now above that low (bounce signal)
        if rsi_min_5 is not None and rsi_min_5 < 45 and rsi > rsi_min_5:
            conditions_met += 1
            reasoning.append(
                f"RSI bounce: min5 {rsi_min_5:.1f} -> {rsi:.1f} (dipped below 45, now rising)"
            )

        # Condition 3: price within 3% of SMA50 (pullback not too deep)
        if sma50_dist_pct is not None and 0 <= sma50_dist_pct <= 3.0:
            conditions_met += 1
            reasoning.append(
                f"Near SMA50: {sma50_dist_pct:.1f}% above (within 3%)"
            )

        # Condition 4: stochastic was oversold (<30) within last 5 bars
        # and is now above that low (crossing up)
        if stoch_k_min_5 is not None and stoch_k_min_5 < 30 and stoch_k > stoch_k_min_5:
            conditions_met += 1
            reasoning.append(
                f"Stochastic crossup: min5 %K {stoch_k_min_5:.1f} -> {stoch_k:.1f} (from oversold)"
            )

        log.debug(
            "Pullback conditions=%d rsi=%.1f rsi_min5=%s stoch_k=%.1f stoch_min5=%s sma50_dist=%.1f%%",
            conditions_met, rsi,
            f"{rsi_min_5:.1f}" if rsi_min_5 is not None else "N/A",
            stoch_k,
            f"{stoch_k_min_5:.1f}" if stoch_k_min_5 is not None else "N/A",
            sma50_dist_pct if sma50_dist_pct is not None else 0.0,
        )

        # Signal and confidence. Bonuses scale with indicator strength
        # (bounce magnitude, stochastic recovery, SMA50 proximity) so two
        # 3/4 setups never produce the identical confidence — otherwise
        # the logged value looks like a hard-coded default.
        rsi_recovery = 0.0
        if rsi_min_5 is not None:
            rsi_recovery = min(5.0, max(0.0, (rsi - rsi_min_5) * 0.5))
        stoch_recovery = 0.0
        if stoch_k_min_5 is not None:
            stoch_recovery = min(5.0, max(0.0, (stoch_k - stoch_k_min_5) * 0.2))
        proximity_bonus = 0.0
        if sma50_dist_pct is not None:
            proximity_bonus = min(5.0, max(0.0, (3.0 - abs(sma50_dist_pct)) * 1.67))

        if conditions_met >= 4:
            # Perfect pullback setup: 65-80%
            confidence = 65.0 + rsi_recovery + stoch_recovery + proximity_bonus
            signal, confidence = "BUY", min(confidence, 80.0)
        elif conditions_met == 3:
            # Good setup missing one piece: 40-55%
            confidence = 40.0 + rsi_recovery + stoch_recovery + proximity_bonus
            signal, confidence = "WEAK BUY", min(confidence, 55.0)
        else:
            # 2 or fewer — no actionable signal
            if not reasoning:
                reasoning.append("No pullback conditions triggered")
            signal, confidence = "HOLD", 20.0

        # ── Downtrend filter: prevent catching falling knives ──
        sma200 = ind.get("sma200")
        if sma200 and price and sma200 > 0:
            sma_ratio = price / sma200
            if sma_ratio < 0.70:
                signal = "HOLD"
                confidence = 20.0
                reasoning.append("Extreme downtrend \u2014 signal suppressed")
            elif sma_ratio < 0.80 and signal in ("BUY", "STRONG BUY"):
                signal = "WEAK BUY"
                confidence = min(confidence, 55.0)
                reasoning.append(
                    f"Severe downtrend filter applied (SMA ratio: {sma_ratio:.2f})"
                )

        return signal, confidence, reasoning
