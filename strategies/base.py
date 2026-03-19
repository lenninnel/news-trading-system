"""
Abstract base class and result dataclass for strategy-specific TA.

Every concrete strategy implements ``analyze(ticker, bars, sentiment_signal)``
and returns a ``StrategyResult``.  Strategies receive pre-fetched bar data
so they stay pure analysers with no I/O dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    """Immutable result produced by a strategy's ``analyze()`` method.

    Attributes:
        signal:              "STRONG BUY" | "BUY" | "WEAK BUY" | "HOLD" | "SELL"
        confidence:          0.0 – 100.0
        indicators:          Strategy-specific numeric values.
        entry_price:         Suggested entry price (latest close).
        stop_loss:           Suggested stop-loss price.
        take_profit:         Suggested take-profit price.
        timeframe_alignment: True when multi-timeframe signals agree.
        strategy_name:       Identifier of the strategy that produced this result.
        reasoning:           Human-readable list of triggered conditions.
    """

    signal:              str
    confidence:          float
    indicators:          dict[str, Any]   = field(default_factory=dict)
    entry_price:         float | None     = None
    stop_loss:           float | None     = None
    take_profit:         float | None     = None
    timeframe_alignment: bool             = True
    strategy_name:       str              = ""
    reasoning:           list[str]        = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base for all strategy-specific TA modules.

    Subclasses must implement:
        name      — human-readable strategy identifier (property)
        analyze() — run indicators and return a StrategyResult

    Shared helpers are provided for common operations on price series.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def analyze(
        self,
        ticker: str,
        bars: pd.DataFrame,
        sentiment_signal: str = "HOLD",
    ) -> StrategyResult:
        """Run the strategy against *bars* and return a result.

        Args:
            ticker:           Stock symbol (e.g. "NVDA").
            bars:             OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
            sentiment_signal: Upstream sentiment: "BUY" | "WEAK BUY" | "HOLD" | "SELL".
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latest(series: pd.Series) -> float | None:
        """Return the last non-NaN value, or None."""
        clean = series.dropna()
        return float(clean.iloc[-1]) if not clean.empty else None

    @staticmethod
    def _prev(series: pd.Series) -> float | None:
        """Return the second-to-last non-NaN value, or None."""
        clean = series.dropna()
        return float(clean.iloc[-2]) if len(clean) >= 2 else None

    # Ticker universe — subclasses override to declare best-fit tickers
    PREFERRED_TICKERS: list[str] = []
