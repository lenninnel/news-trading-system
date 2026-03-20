"""
Strategy-specific technical analysis modules.

Each strategy owns its own indicator calculations and entry/exit logic,
replacing the shared generic TA score with purpose-built analysis.

Available strategies:
    MomentumStrategy  — trend-following with volume confirmation
    PullbackStrategy  — mean-reversion bounce off SMA50 support
"""

from strategies.base import BaseStrategy, StrategyResult
from strategies.momentum import MomentumStrategy
from strategies.pullback import PullbackStrategy
from strategies.router import get_strategy, get_strategy_name

__all__ = [
    "BaseStrategy",
    "StrategyResult",
    "MomentumStrategy",
    "PullbackStrategy",
    "get_strategy",
    "get_strategy_name",
]
