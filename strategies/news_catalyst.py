"""
News Catalyst Strategy — fires when significant news + price/volume confirms.

Unlike Momentum/Pullback which are pure TA, this strategy triggers on
breaking news events where sentiment is strong and the market is reacting.
"""

from __future__ import annotations

import logging

import pandas as pd

from strategies.base import BaseStrategy, StrategyResult

log = logging.getLogger(__name__)


class NewsCatalystStrategy(BaseStrategy):
    """Fire when significant news + sentiment confirms + price/volume reacts."""

    @property
    def name(self) -> str:
        return "NewsCatalyst"

    def analyze(
        self,
        ticker: str,
        bars: pd.DataFrame,
        sentiment_signal: str = "HOLD",
        news_data: dict | None = None,
    ) -> StrategyResult:
        """
        Evaluate news catalyst conditions.

        Args:
            ticker: Stock ticker.
            bars: Daily OHLCV DataFrame.
            sentiment_signal: Aggregated sentiment signal.
            news_data: Optional dict with keys:
                - news_score: float (0-1 scale, higher = more positive)
                - headline_count: int
                - sentiment_direction: "BUY" | "SELL" | "HOLD"
        """
        reasoning: list[str] = []
        conditions_met = 0

        if bars is None or bars.empty or len(bars) < 21:
            # 0-headline / no-data HOLD baseline (see fallthrough HOLD below)
            return StrategyResult(
                signal="HOLD",
                confidence=10.0,
                strategy_name=self.name,
                reasoning=["Insufficient bar data"],
            )

        # Extract latest data
        close = bars["Close"].iloc[-1]
        prev_close = bars["Close"].iloc[-2]
        volume = bars["Volume"].iloc[-1]
        avg_volume = bars["Volume"].iloc[-21:-1].mean()

        # Default news_data if not provided
        if not news_data:
            # 0-headline / no-data HOLD baseline (see fallthrough HOLD below)
            return StrategyResult(
                signal="HOLD",
                confidence=10.0,
                strategy_name=self.name,
                indicators={"price": close},
                reasoning=["No news data available"],
            )

        news_score = news_data.get("news_score", 0.0)
        headline_count = news_data.get("headline_count", 0)
        sent_direction = news_data.get("sentiment_direction", "HOLD")

        # Condition 1: News significance
        has_news = news_score >= 0.70 and headline_count >= 1
        if has_news:
            conditions_met += 1
            reasoning.append(f"Strong news signal ({news_score:.2f}, {headline_count} headlines)")
        else:
            reasoning.append(f"News below threshold (score={news_score:.2f}, headlines={headline_count})")

        # Condition 2: Price/volume reaction
        rvol = volume / avg_volume if avg_volume > 0 else 0
        pct_change = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0
        has_reaction = rvol > 1.3 and abs(pct_change) > 0.5
        if has_reaction:
            conditions_met += 1
            reasoning.append(f"Price reaction confirmed (RVOL={rvol:.1f}x, move={pct_change:+.1f}%)")
        else:
            reasoning.append(f"No price reaction (RVOL={rvol:.1f}x, move={pct_change:+.1f}%)")

        # Condition 3: Direction alignment
        price_direction = "BUY" if pct_change > 0 else "SELL" if pct_change < 0 else "HOLD"
        aligned = sent_direction == price_direction and sent_direction != "HOLD"
        if aligned:
            conditions_met += 1
            reasoning.append(f"Direction aligned (sentiment={sent_direction}, price={price_direction})")
        else:
            reasoning.append(f"Direction misaligned (sentiment={sent_direction}, price={price_direction})")

        # Determine direction for signal
        direction = sent_direction if sent_direction != "HOLD" else price_direction

        # Build indicators
        indicators = {
            "price": close,
            "rvol": rvol,
            "pct_change": pct_change,
            "news_score": news_score,
            "headline_count": headline_count,
        }

        # Confidence bonuses scale with the actual strength of each
        # input (news_score magnitude, headline volume, price/volume
        # reaction) instead of collapsing to a single constant per
        # branch — otherwise every catalyst logs the same number.
        news_bonus = min(10.0, max(0.0, (news_score - 0.70) * 33.0))   # 0-10
        headline_bonus = min(5.0, max(0.0, (headline_count - 1) * 2.5))  # 0-5
        rvol_bonus = min(5.0, max(0.0, (rvol - 1.3) * 5.0))             # 0-5

        # Signal output
        if conditions_met == 3:
            # All conditions met: 55-75% (scaled by input strength)
            if direction == "BUY":
                signal = "BUY"
            elif direction == "SELL":
                signal = "SELL"
            else:
                signal = "HOLD"
            confidence = min(75.0, 55.0 + news_bonus + headline_bonus + rvol_bonus)
            return StrategyResult(
                signal=signal,
                confidence=confidence,
                strategy_name=self.name,
                indicators=indicators,
                entry_price=close,
                reasoning=reasoning,
            )

        if conditions_met >= 1 and has_news:
            # News present but no full confirmation: 35-55%
            if direction == "BUY":
                signal = "WEAK BUY"
            elif direction == "SELL":
                signal = "WEAK SELL"
            else:
                signal = "HOLD"
            rvol_soft_bonus = min(5.0, max(0.0, (rvol - 1.0) * 5.0))
            confidence = min(55.0, 35.0 + news_bonus + headline_bonus + rvol_soft_bonus)
            return StrategyResult(
                signal=signal,
                confidence=confidence,
                strategy_name=self.name,
                indicators=indicators,
                entry_price=close,
                reasoning=reasoning,
            )

        # Data-driven HOLD: scale by sentiment magnitude, headline volume,
        # and price/volume reaction. A 0-headline neutral HOLD floors at 10
        # ("nothing happening"), a high-volume strong-news neutral HOLD caps
        # at 30 (real conviction in inaction).
        hold_conf = min(
            30.0,
            10.0
            + min(10.0, headline_count * 2.0)
            + news_score * 10.0
            + max(0.0, (rvol - 1.0)) * 5.0,
        )
        return StrategyResult(
            signal="HOLD",
            confidence=hold_conf,
            strategy_name=self.name,
            indicators=indicators,
            reasoning=reasoning,
        )
