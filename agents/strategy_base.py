"""
Base class and signal dataclass for the multi-strategy framework.

All concrete strategy agents (MomentumAgent, MeanReversionAgent, SwingAgent)
inherit from StrategyAgent, which extends BaseAgent with:
    - A shared _fetch_history() helper (factored from TechnicalAgent)
    - Common _latest() / _prev() series accessors
    - The StrategySignal dataclass as the canonical return type

StrategySignal is a plain dataclass; it carries no business logic.
The StrategyCoordinator is responsible for comparing signals across agents
and calling RiskAgent with the ensemble result.
"""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# Path bootstrap (needed when a strategy file is run directly)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.base_agent import BaseAgent
from storage.database import Database

log = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """
    Standardised signal produced by every strategy agent.

    Attributes:
        ticker:     Stock ticker symbol (e.g. "AAPL").
        strategy:   Agent identifier: "momentum" | "mean_reversion" | "swing".
        signal:     "BUY" | "SELL" | "HOLD".
        confidence: 0.0 – 100.0 (same scale as RiskAgent input).
        reasoning:  Human-readable list of triggered conditions.
        indicators: Strategy-specific numeric values (for DB persistence and display).
        timeframe:  Intended holding period: "position" | "swing" | "intraday".
        signal_id:  DB primary key — None before persistence; set by StrategyCoordinator.
    """
    ticker:     str
    strategy:   str
    signal:     str
    confidence: float
    reasoning:  list[str]    = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)
    timeframe:  str           = "swing"
    signal_id:  int | None    = None


class StrategyAgent(BaseAgent):
    """
    Abstract base class for all strategy agents.

    Extends BaseAgent with shared data-fetching helpers so concrete agents
    can focus purely on their indicator and signal logic.

    Subclasses must implement:
        name  (property) — human-readable identifier
        run()            — accept ticker + optional kwargs, return StrategySignal
    """

    # Override in subclasses to control yfinance download window
    _DOWNLOAD_PERIOD: str = "6mo"

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or Database()

    @abstractmethod
    def run(self, ticker: str, **kwargs: Any) -> StrategySignal:  # type: ignore[override]
        """Execute the strategy analysis and return a StrategySignal."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _fetch_history(self, ticker: str, period: str | None = None) -> pd.DataFrame:
        """
        Download daily OHLCV data and return a clean DataFrame.

        Uses self._DOWNLOAD_PERIOD unless *period* is supplied explicitly.
        Flattens any MultiIndex columns produced by yfinance batch downloads.

        Raises:
            ValueError: If yfinance returns an empty DataFrame.
        """
        raw: pd.DataFrame = yf.download(
            ticker,
            period=period or self._DOWNLOAD_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(f"No price data returned for '{ticker}'")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    @staticmethod
    def _latest(series: pd.Series) -> float | None:
        """Return the last non-NaN value of *series*, or None."""
        clean = series.dropna()
        return float(clean.iloc[-1]) if not clean.empty else None

    @staticmethod
    def _prev(series: pd.Series) -> float | None:
        """Return the second-to-last non-NaN value of *series*, or None."""
        clean = series.dropna()
        return float(clean.iloc[-2]) if len(clean) >= 2 else None
