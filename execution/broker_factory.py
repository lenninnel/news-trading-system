"""
Broker factory — returns the correct trader based on TRADING_MODE.

Supported modes
---------------
paper_local   (default)  PaperTrader with local SQLite simulation.
alpaca_paper             AlpacaTrader pointed at Alpaca's paper environment.
alpaca_live              AlpacaTrader pointed at the real-money endpoint.
                         Raises RuntimeError unless LIVE_TRADING_CONFIRMED=true.
ibkr_paper               IBKRTrader connected to IB Gateway paper port (7497).
ibkr_live                IBKRTrader connected to IB Gateway live port (7496).
                         Raises RuntimeError unless LIVE_TRADING_CONFIRMED=true.

Usage::

    from execution.broker_factory import create_trader
    trader = create_trader()          # reads TRADING_MODE from env
    trader.track_trade("AAPL", "BUY", 10, 150.0)
"""

from __future__ import annotations

import os

from storage.database import Database

_VALID_MODES = (
    "paper_local",
    "alpaca_paper", "alpaca_live",
    "ibkr_paper", "ibkr_live",
)


def get_trading_mode() -> str:
    """Return the current TRADING_MODE, validated."""
    mode = os.environ.get("TRADING_MODE", "paper_local").lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"TRADING_MODE must be one of {_VALID_MODES}, got '{mode}'"
        )
    return mode


def create_trader(db: Database | None = None, client_id: int | None = None):
    """
    Instantiate and return the correct trader for the current TRADING_MODE.

    Args:
        db: Optional Database instance for dependency injection.
        client_id: Optional IBKR clientId override. Only applies to
            ``ibkr_paper`` / ``ibkr_live`` modes; ignored otherwise.
            Used to segregate concurrent IBKR connections (e.g. the
            PositionManager monitor vs. trading sessions).

    Returns:
        PaperTrader, AlpacaTrader, or IBKRTrader instance.

    Raises:
        RuntimeError: If a live mode is selected and
            ``LIVE_TRADING_CONFIRMED`` is not ``"true"``.
        ValueError: If TRADING_MODE is not recognised.
    """
    mode = get_trading_mode()

    if mode == "paper_local":
        from execution.paper_trader import PaperTrader
        return PaperTrader(db=db)

    # --- Alpaca modes ---
    if mode in ("alpaca_paper", "alpaca_live"):
        if mode == "alpaca_live":
            confirmed = os.environ.get("LIVE_TRADING_CONFIRMED", "").lower()
            if confirmed != "true":
                raise RuntimeError(
                    "LIVE TRADING BLOCKED: Set LIVE_TRADING_CONFIRMED=true to "
                    "enable real-money trading.  This is your last safety gate."
                )
            os.environ["ALPACA_MODE"] = "live"
        else:
            os.environ["ALPACA_MODE"] = "paper"

        from execution.alpaca_trader import AlpacaTrader
        return AlpacaTrader(db=db)

    # --- IBKR modes ---
    if mode in ("ibkr_paper", "ibkr_live"):
        if mode == "ibkr_live":
            confirmed = os.environ.get("LIVE_TRADING_CONFIRMED", "").lower()
            if confirmed != "true":
                raise RuntimeError(
                    "LIVE TRADING BLOCKED: Set LIVE_TRADING_CONFIRMED=true to "
                    "enable real-money trading.  This is your last safety gate."
                )
            os.environ["IBKR_PAPER"] = "false"
        else:
            os.environ["IBKR_PAPER"] = "true"

        from execution.ibkr_trader import IBKRTrader
        return IBKRTrader(db=db, client_id=client_id)

    raise ValueError(f"Unhandled TRADING_MODE: {mode}")
