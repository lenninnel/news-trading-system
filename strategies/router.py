"""
Strategy router — maps tickers to their best-fit strategy module.

Based on walk-forward backtest results (2026-03-20):
    MOMENTUM tickers: META, JPM, VRT  (Sharpe > 0.5, win rate > 45%)
    PULLBACK tickers: AAPL, MSFT, AMZN, XOM, CVX, BAC, PFE, TSLA

Tickers not in either list fall back to the default strategy.
"""

from __future__ import annotations

from strategies.momentum import MomentumStrategy
from strategies.pullback import PullbackStrategy
from strategies.base import BaseStrategy

# Ticker → strategy mapping (from backtest validation 2026-03-20)
MOMENTUM_TICKERS: set[str] = {"META", "JPM", "VRT"}
PULLBACK_TICKERS: set[str] = {"AAPL", "MSFT", "AMZN", "XOM", "CVX", "BAC", "PFE", "TSLA"}

# Singletons — strategies are stateless so one instance per type is fine
_momentum = MomentumStrategy()
_pullback = PullbackStrategy()


def get_strategy(ticker: str) -> BaseStrategy:
    """Return the best-fit strategy instance for *ticker*.

    Falls back to MomentumStrategy for unknown tickers.
    """
    t = ticker.upper()
    if t in PULLBACK_TICKERS:
        return _pullback
    if t in MOMENTUM_TICKERS:
        return _momentum
    # Default: momentum (original system behaviour)
    return _momentum


def get_strategy_name(ticker: str) -> str:
    """Return the strategy name string for *ticker*."""
    return get_strategy(ticker).name


# Backward-compatible alias used by orchestrator/coordinator.py on main
strategy_label = get_strategy_name
