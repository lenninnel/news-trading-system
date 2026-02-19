"""
SwingAgent — Multi-day swing trading strategy.

Entry conditions (any one set triggers the signal):

    BUY set A — short-term uptrend with MACD momentum + breakout:
        Price > SMA-20 > SMA-50  (in short-term uptrend)
        + MACD histogram turning positive (hist_cur > hist_prev AND hist_cur > 0)
        + Price breaks above 5-day pivot high

    BUY set B — long-term trend + SMA-20 reclaim:
        Price > SMA-200  (above long-term trend)
        + MACD histogram turning up
        + Price crossed above SMA-20 (yesterday below, today above)

    SELL mirrors BUY with reversed conditions.

Confidence:
    Base 55 pts
    + 20 if price is on the correct side of all three SMAs (20, 50, 200)
    -  10 if ATR% > 3% (high volatility — setup less predictable)
    +  10 if MACD line aligns with direction
    Clamped to [30, 85].

Timeframe: "swing" (3–15 days)

Note: Uses a 1-year download period to warm up SMA-200
      (~252 trading days needed for a valid reading).

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


class SwingAgent(StrategyAgent):
    """
    Multi-day swing trading strategy agent.

    Combines SMA trend alignment, MACD histogram direction change, and
    short-term pivot breakouts to identify multi-day swing entries.
    """

    # Needs 1y for SMA-200 to warm up (≈252 trading days)
    _DOWNLOAD_PERIOD = "1y"

    def __init__(self, db: Database | None = None) -> None:
        super().__init__(db=db)

    @property
    def name(self) -> str:
        return "SwingAgent"

    def run(self, ticker: str, **kwargs: Any) -> StrategySignal:
        ticker = ticker.upper()
        try:
            df     = self._fetch_history(ticker)
            ind    = self._calculate_indicators(df)
            signal, reasoning, confidence = self._apply_signal_rules(ind)
        except Exception as exc:
            log.error("SwingAgent failed for %s: %s", ticker, exc)
            return StrategySignal(
                ticker=ticker,
                strategy="swing",
                signal="HOLD",
                confidence=25.0,
                reasoning=[f"Error: {exc}"],
                indicators={},
                timeframe="swing",
            )

        return StrategySignal(
            ticker=ticker,
            strategy="swing",
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

        # SMA 20, 50, 200
        sma20  = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

        # MACD (12, 26, 9)
        macd_obj  = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        macd_line = macd_obj.macd()
        macd_sig  = macd_obj.macd_signal()
        macd_hist = macd_obj.macd_diff()

        hist_cur  = self._latest(macd_hist)
        hist_prev = self._prev(macd_hist)
        hist_turning_up   = (
            hist_cur is not None and hist_prev is not None
            and hist_cur > hist_prev and hist_cur > 0
        )
        hist_turning_down = (
            hist_cur is not None and hist_prev is not None
            and hist_cur < hist_prev and hist_cur < 0
        )

        # 5-day pivot high/low (iloc[-2] avoids same-bar look-ahead)
        pivot_high = float(high.rolling(5).max().iloc[-2]) if len(high) >= 6 else float(high.max())
        pivot_low  = float(low.rolling(5).min().iloc[-2])  if len(low)  >= 6 else float(low.min())

        # ATR-14 as volatility proxy
        atr     = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        atr_val = self._latest(atr) or 0.0
        price   = float(close.iloc[-1]) if not close.empty else None
        atr_pct = (atr_val / price) if (price and price > 0) else 0.0

        # SMA-20 crossover detection
        sma20_cur  = self._latest(sma20)
        sma20_prev = self._prev(sma20)
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None

        crossed_above_sma20 = (
            price is not None and prev_close is not None
            and sma20_cur is not None and sma20_prev is not None
            and prev_close < sma20_prev and price > sma20_cur
        )
        crossed_below_sma20 = (
            price is not None and prev_close is not None
            and sma20_cur is not None and sma20_prev is not None
            and prev_close > sma20_prev and price < sma20_cur
        )

        return {
            "price":               price,
            "sma20":               sma20_cur,
            "sma50":               self._latest(sma50),
            "sma200":              self._latest(sma200),
            "macd_line":           self._latest(macd_line),
            "macd_sig":            self._latest(macd_sig),
            "macd_hist":           hist_cur,
            "macd_hist_prev":      hist_prev,
            "hist_turning_up":     hist_turning_up,
            "hist_turning_down":   hist_turning_down,
            "pivot_high":          round(pivot_high, 4),
            "pivot_low":           round(pivot_low, 4),
            "atr_pct":             round(atr_pct, 4),
            "crossed_above_sma20": crossed_above_sma20,
            "crossed_below_sma20": crossed_below_sma20,
        }

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_signal_rules(ind: dict) -> tuple[str, list[str], float]:
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []

        price      = ind.get("price")
        sma20      = ind.get("sma20")
        sma50      = ind.get("sma50")
        sma200     = ind.get("sma200")
        macd_line  = ind.get("macd_line") or 0.0
        macd_sig   = ind.get("macd_sig")  or 0.0
        hist_up    = ind.get("hist_turning_up",    False)
        hist_down  = ind.get("hist_turning_down",  False)
        pivot_high = ind.get("pivot_high")
        pivot_low  = ind.get("pivot_low")
        atr_pct    = ind.get("atr_pct",   0.0)
        cross_up   = ind.get("crossed_above_sma20", False)
        cross_down = ind.get("crossed_below_sma20", False)

        # BUY set A — uptrend alignment + MACD momentum + pivot breakout
        if (price is not None and sma20 is not None and sma50 is not None
                and price > sma20 > sma50
                and hist_up
                and pivot_high is not None and price > pivot_high):
            buy_reasons.append(
                f"Swing BUY: price {price:.2f} > SMA20 {sma20:.2f} > SMA50 {sma50:.2f}, "
                f"MACD hist accelerating up, above 5-day pivot {pivot_high:.2f}"
            )

        # BUY set B — long-term trend + SMA-20 reclaim
        if (sma200 is not None and price is not None
                and price > sma200
                and hist_up
                and cross_up):
            buy_reasons.append(
                f"SMA-20 reclaim: price {price:.2f} crossed above SMA20, "
                f"above SMA200 {sma200:.2f}, MACD hist turning positive"
            )

        # SELL set A — downtrend alignment + MACD momentum + pivot breakdown
        if (price is not None and sma20 is not None and sma50 is not None
                and price < sma20 < sma50
                and hist_down
                and pivot_low is not None and price < pivot_low):
            sell_reasons.append(
                f"Swing SELL: price {price:.2f} < SMA20 {sma20:.2f} < SMA50 {sma50:.2f}, "
                f"MACD hist accelerating down, below 5-day pivot {pivot_low:.2f}"
            )

        # SELL set B — long-term trend break + SMA-20 loss
        if (sma200 is not None and price is not None
                and price < sma200
                and hist_down
                and cross_down):
            sell_reasons.append(
                f"SMA-20 lost: price {price:.2f} crossed below SMA20, "
                f"below SMA200 {sma200:.2f}, MACD hist turning negative"
            )

        # Confidence
        confidence = 55.0

        # Bonus: all three SMAs aligned
        if (price is not None and sma20 is not None
                and sma50 is not None and sma200 is not None):
            if buy_reasons  and price > sma20 > sma50 and price > sma200:
                confidence += 20.0
            elif sell_reasons and price < sma20 < sma50 and price < sma200:
                confidence += 20.0

        # Penalty: high volatility
        if atr_pct > 0.03:
            confidence -= 10.0

        # Bonus: MACD line confirms direction
        if buy_reasons  and macd_line > macd_sig:
            confidence += 10.0
        elif sell_reasons and macd_line < macd_sig:
            confidence += 10.0

        confidence = min(max(confidence, 30.0), 85.0)

        if buy_reasons:
            return "BUY",  buy_reasons,  confidence
        if sell_reasons:
            return "SELL", sell_reasons, confidence
        return "HOLD", ["No swing setup conditions triggered"], 25.0
