"""
Broker factory — returns the correct trader based on TRADING_MODE.

Supported modes
---------------
paper_local   (default)  PaperTrader with local SQLite simulation.
alpaca_paper             AlpacaTrader pointed at Alpaca's paper environment.
alpaca_live              AlpacaTrader pointed at the real-money endpoint.
                         Raises RuntimeError unless LIVE_TRADING_CONFIRMED=true.

Usage::

    from execution.broker_factory import create_trader
    trader = create_trader()          # reads TRADING_MODE from env
    trader.track_trade("AAPL", "BUY", 10, 150.0)
"""

from __future__ import annotations

import os

from storage.database import Database

_VALID_MODES = ("paper_local", "alpaca_paper", "alpaca_live")


def get_trading_mode() -> str:
    """Return the current TRADING_MODE, validated."""
    mode = os.environ.get("TRADING_MODE", "paper_local").lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"TRADING_MODE must be one of {_VALID_MODES}, got '{mode}'"
        )
    return mode


def create_trader(db: Database | None = None):
    """
    Instantiate and return the correct trader for the current TRADING_MODE.

    Args:
        db: Optional Database instance for dependency injection.

    Returns:
        PaperTrader or AlpacaTrader instance.

    Raises:
        RuntimeError: If mode is ``alpaca_live`` and
            ``LIVE_TRADING_CONFIRMED`` is not ``"true"``.
        ValueError: If TRADING_MODE is not recognised.
    """
    mode = get_trading_mode()

    if mode == "paper_local":
        from execution.paper_trader import PaperTrader
        return PaperTrader(db=db)

    # --- Alpaca modes ---
    if mode == "alpaca_live":
        confirmed = os.environ.get("LIVE_TRADING_CONFIRMED", "").lower()
        if confirmed != "true":
            raise RuntimeError(
                "LIVE TRADING BLOCKED: Set LIVE_TRADING_CONFIRMED=true to "
                "enable real-money trading.  This is your last safety gate."
            )
        # Force the Alpaca SDK to hit the live URL
        os.environ["ALPACA_MODE"] = "live"

    elif mode == "alpaca_paper":
        os.environ["ALPACA_MODE"] = "paper"

    from execution.alpaca_trader import AlpacaTrader
    return AlpacaTrader(db=db)
