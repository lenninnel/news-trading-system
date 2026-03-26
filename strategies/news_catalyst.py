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
            return StrategyResult(
                signal="HOLD",
                confidence=25.0,
                strategy_name=self.name,
                reasoning=["Insufficient bar data"],
            )

        # Extract latest data
        close = bars["close"].iloc[-1]
        prev_close = bars["close"].iloc[-2]
        volume = bars["volume"].iloc[-1]
        avg_volume = bars["volume"].iloc[-21:-1].mean()

        # Default news_data if not provided
        if not news_data:
            return StrategyResult(
                signal="HOLD",
                confidence=25.0,
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

        # Signal output
        if conditions_met == 3:
            # All conditions met
            if direction == "BUY":
                signal = "BUY"
            elif direction == "SELL":
                signal = "SELL"
            else:
                signal = "HOLD"
            return StrategyResult(
                signal=signal,
                confidence=65.0,
                strategy_name=self.name,
                indicators=indicators,
                entry_price=close,
                reasoning=reasoning,
            )

        if conditions_met >= 1 and has_news:
            # News present but no full confirmation
            if direction == "BUY":
                signal = "WEAK BUY"
            elif direction == "SELL":
                signal = "WEAK SELL"
            else:
                signal = "HOLD"
            return StrategyResult(
                signal=signal,
                confidence=45.0,
                strategy_name=self.name,
                indicators=indicators,
                entry_price=close,
                reasoning=reasoning,
            )

        return StrategyResult(
            signal="HOLD",
            confidence=25.0,
            strategy_name=self.name,
            indicators=indicators,
            reasoning=reasoning,
        )
