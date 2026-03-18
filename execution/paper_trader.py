"""
Paper trading execution layer.

PaperTrader records simulated trades and maintains a virtual portfolio in
SQLite.  No real broker connection is made — this is purely for tracking
signal performance and building an audit trail before going live.

Portfolio logic
---------------
BUY
    Increases share count.  Average cost basis is recalculated using a
    weighted average of the existing position and the new fill price.

SELL
    Reduces share count.  Realised PnL = (fill_price − avg_cost) × shares.
    Positions reduced to zero shares are removed from the portfolio table.

Tables managed
--------------
portfolio
    One row per open position.  Updated atomically on every trade.

trade_history
    Immutable append-only record of every executed trade, including the
    realised PnL for sells.
"""

from __future__ import annotations

from typing import Any

from emergency_stop import KillSwitch
from storage.database import Database


class PaperTrader:
    """
    Simulated trade execution and portfolio tracking.

    Args:
        db: Optional Database instance for dependency injection.
            A new instance is created automatically when omitted.

    Example::

        trader = PaperTrader()

        # Open a position
        trade = trader.track_trade("AAPL", "BUY", 3, 263.88,
                                   stop_loss=258.60, take_profit=274.44)

        # Inspect portfolio
        positions = trader.get_portfolio()
        # [{"ticker": "AAPL", "shares": 3, "avg_price": 263.88, ...}]

        # Close the position
        trade = trader.track_trade("AAPL", "SELL", 3, 270.00)
        # trade["pnl"] == (270.00 - 263.88) * 3  →  $18.36
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or Database()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
        **kwargs: Any,
    ) -> dict:
        """
        Record a simulated trade and update the portfolio.

        Args:
            ticker:      Stock ticker symbol.
            action:      "BUY" or "SELL".
            shares:      Number of whole shares.
            price:       Fill price per share.
            stop_loss:   Stop-loss price (stored for reference).
            take_profit: Take-profit price (stored for reference).

        Returns:
            dict with keys:
                trade_id    (int):   DB primary key of the trade_history row.
                ticker      (str):   Ticker symbol.
                action      (str):   "BUY" or "SELL".
                shares      (int):   Shares traded.
                price       (float): Fill price.
                stop_loss   (float|None)
                take_profit (float|None)
                pnl         (float): Realised PnL (0.0 for buys).
                total_value (float): Gross trade value (shares × price).

        Raises:
            ValueError: If action is not "BUY" or "SELL", or shares < 1.
        """
        KillSwitch.assert_trading_allowed()
        ticker = ticker.upper()
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got '{action}'")
        if shares < 1:
            raise ValueError(f"shares must be ≥ 1, got {shares}")
        if not price or price <= 0:
            raise ValueError(f"Invalid price ${price} for {ticker} — trade aborted")
        if price < 50 or price > 10_000:
            raise ValueError(
                f"Price ${price:.2f} for {ticker} outside valid range [$50, $10,000] "
                f"— ghost price detected, trade aborted"
            )
        if action == "BUY" and (not stop_loss or stop_loss <= 0
                                or not take_profit or take_profit <= 0):
            raise ValueError(
                f"BUY requires valid stop_loss/take_profit for {ticker} "
                f"(got SL={stop_loss}, TP={take_profit}) — trade aborted"
            )

        pnl = 0.0
        existing = self._db.get_portfolio_position(ticker)

        if action == "BUY":
            if existing:
                total_shares = existing["shares"] + shares
                new_avg = (
                    existing["shares"] * existing["avg_price"] + shares * price
                ) / total_shares
            else:
                total_shares = shares
                new_avg = price

            self._db.set_portfolio_position(
                ticker=ticker,
                shares=total_shares,
                avg_price=new_avg,
                current_value=round(total_shares * price, 2),
            )

        else:  # SELL
            if existing:
                pnl = round((price - existing["avg_price"]) * shares, 2)
                remaining = existing["shares"] - shares
                if remaining <= 0:
                    self._db.delete_portfolio_position(ticker)
                else:
                    self._db.set_portfolio_position(
                        ticker=ticker,
                        shares=remaining,
                        avg_price=existing["avg_price"],
                        current_value=round(remaining * price, 2),
                    )

        trade_id = self._db.log_trade_history(
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
        )

        return {
            "trade_id":    trade_id,
            "ticker":      ticker,
            "action":      action,
            "shares":      shares,
            "price":       price,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "pnl":         pnl,
            "total_value": round(shares * price, 2),
        }

    def get_portfolio(self) -> list[dict]:
        """
        Return all current open positions.

        Returns:
            List of dicts, each with: ticker, shares, avg_price,
            current_value, updated_at.
        """
        return self._db.get_portfolio()

    def get_trade_history(
        self,
        ticker: "str | None" = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return past trades, newest first.

        Args:
            ticker: Filter by ticker symbol (None = all tickers).
            limit:  Maximum number of rows to return.

        Returns:
            List of dicts, each representing one trade_history row.
        """
        return self._db.get_trade_history(ticker=ticker, limit=limit)
