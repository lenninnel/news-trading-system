"""
Strategy router — maps tickers to their dedicated strategy module.

Ticker assignments
------------------
Momentum:  MSFT, NVDA, DELL, GOOGL, META
Pullback:  AAPL, CEG, VST
Generic:   Everything else (SAP.XETRA, SIE.XETRA, etc.) — uses TechnicalAgent

Usage::

    from strategies.router import get_strategy, strategy_label

    strategy = get_strategy("NVDA")   # MomentumStrategy instance
    strategy = get_strategy("AAPL")   # PullbackStrategy instance
    strategy = get_strategy("SAP.XETRA")  # None (use generic TA)

    strategy_label("NVDA")    # "momentum"
    strategy_label("SAP.XETRA")  # "generic"
"""

from __future__ import annotations

from strategies.momentum import MomentumStrategy
from strategies.pullback import PullbackStrategy
from strategies.base import BaseStrategy

_MOMENTUM_TICKERS: set[str] = {"MSFT", "NVDA", "DELL", "GOOGL", "META"}
_PULLBACK_TICKERS: set[str] = {"AAPL", "CEG", "VST"}

# Singletons — strategies are stateless so one instance per type is fine.
_momentum = MomentumStrategy()
_pullback = PullbackStrategy()


def get_strategy(ticker: str) -> BaseStrategy | None:
    """Return the strategy instance assigned to *ticker*, or None for generic TA."""
    ticker = ticker.upper()
    if ticker in _MOMENTUM_TICKERS:
        return _momentum
    if ticker in _PULLBACK_TICKERS:
        return _pullback
    return None


def strategy_label(ticker: str) -> str:
    """Return a short label for display: 'momentum', 'pullback', or 'generic'."""
    ticker = ticker.upper()
    if ticker in _MOMENTUM_TICKERS:
        return "momentum"
    if ticker in _PULLBACK_TICKERS:
        return "pullback"
    return "generic"
