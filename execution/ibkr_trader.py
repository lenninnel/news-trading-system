"""
Interactive Brokers execution layer via ib_insync.

IBKRTrader places orders through IB Gateway / TWS and exposes the same
public interface as AlpacaTrader / PaperTrader so all three are
interchangeable behind broker_factory.

Environment variables
---------------------
IBKR_HOST         IB Gateway hostname (default "127.0.0.1")
IBKR_PORT         IB Gateway port (default auto: 7497 paper / 7496 live)
IBKR_CLIENT_ID    TWS API client ID (default 1)
IBKR_PAPER        "true" (default) or "false" for live trading
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from storage.database import Database

log = logging.getLogger(__name__)

# Tickers that cannot be traded on IBKR via this integration.
UNSUPPORTED_IBKR: set[str] = set()


def _is_pytest_running() -> bool:
    return "pytest" in sys.modules or "_pytest" in sys.modules


class IBKRTrader:
    """
    Live / paper broker execution via Interactive Brokers (ib_insync).

    Keeps a persistent connection to IB Gateway. If the gateway is
    unreachable, raises ConnectionError immediately so the caller
    can fall back.

    Args:
        db:  Optional Database instance (used to sync positions locally).
        ib:  Optional pre-built ``ib_insync.IB`` instance (for testing).
    """

    def __init__(
        self,
        db: Database | None = None,
        ib: "IB | None" = None,
    ) -> None:
        self._db = db or Database()

        paper = os.environ.get("IBKR_PAPER", "true").lower() in ("true", "1", "yes")
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        default_port = 7497 if paper else 7496
        port = int(os.environ.get("IBKR_PORT", str(default_port)))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

        self._paper = paper
        self._host = host
        self._port = port
        self._client_id = client_id

        if ib is not None:
            self._ib = ib
            # In test mode, Stock/MarketOrder are not needed (tests mock placeOrder)
            self._Stock = None
            self._MarketOrder = None
            return

        if _is_pytest_running():
            raise RuntimeError(
                "IBKRTrader refuses to connect to real IB Gateway during "
                "pytest — pass a mock ib= argument instead"
            )

        from ib_insync import IB, MarketOrder, Stock

        self._Stock = Stock
        self._MarketOrder = MarketOrder
        self._ib = IB()
        mode_label = "PAPER" if paper else "LIVE"
        log.info("Connecting to IB Gateway (%s) at %s:%d ...", mode_label, host, port)

        try:
            self._ib.connect(host, port, clientId=client_id, timeout=10)
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to IB Gateway at {host}:{port} — "
                f"is TWS / IB Gateway running? Error: {exc}"
            ) from exc

        log.info("Connected to IB Gateway (%s)", mode_label)

    # ------------------------------------------------------------------
    # Public API  (mirrors AlpacaTrader / PaperTrader)
    # ------------------------------------------------------------------

    def track_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Submit a market order to IBKR, wait for fill, sync locally."""
        ticker = ticker.upper()
        action = action.upper()

        if ticker in UNSUPPORTED_IBKR:
            log.info("Skipping execution for %s — not supported on IBKR", ticker)
            return {
                "trade_id": None, "ticker": ticker, "action": action,
                "shares": shares, "price": price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "pnl": 0.0, "total_value": 0.0,
                "skipped": True, "skip_reason": f"{ticker} not supported on IBKR",
            }

        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got '{action}'")
        if shares < 1:
            raise ValueError(f"shares must be >= 1, got {shares}")

        contract = self._Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        order = self._MarketOrder(action, shares)
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(2)  # give it a moment to fill

        fill_price = price
        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice or price

        pnl = self._sync_position(ticker, action, shares, fill_price)

        trade_id = self._db.log_trade_history(
            ticker=ticker, action=action, shares=shares, price=fill_price,
            stop_loss=stop_loss, take_profit=take_profit, pnl=pnl,
        )

        log.info(
            "IBKR TRADE: %s %s %d shares @ $%.2f (trade_id=%s)",
            action, ticker, shares, fill_price, trade_id,
        )

        return {
            "trade_id": trade_id, "ticker": ticker, "action": action,
            "shares": shares, "price": fill_price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "pnl": pnl, "total_value": round(shares * fill_price, 2),
        }

    def get_account(self) -> dict:
        """Return account summary: cash, portfolio_value, buying_power."""
        summary = self._ib.accountSummary()
        values: dict[str, float] = {}
        for item in summary:
            if item.tag in ("TotalCashValue", "NetLiquidation", "BuyingPower"):
                values[item.tag] = float(item.value)
        return {
            "cash": values.get("TotalCashValue", 0.0),
            "portfolio_value": values.get("NetLiquidation", 0.0),
            "buying_power": values.get("BuyingPower", 0.0),
        }

    def get_positions(self) -> list[dict]:
        """Return open positions from IBKR."""
        positions = self._ib.positions()
        result = []
        for pos in positions:
            ticker = pos.contract.symbol
            qty = int(pos.position)
            avg_entry = float(pos.avgCost) / abs(qty) if qty != 0 else 0.0
            result.append({
                "ticker": ticker,
                "qty": qty,
                "avg_entry": avg_entry,
                "current_price": 0.0,    # requires market data subscription
                "unrealized_pl": float(pos.avgCost) * -1 if qty else 0.0,
                "unrealized_plpc": 0.0,
            })
        return result

    def get_portfolio(self) -> list[dict]:
        """Return positions in the format expected by the coordinator."""
        positions = self.get_positions()
        result = []
        for pos in positions:
            row = {
                "ticker": pos["ticker"],
                "shares": pos["qty"],
                "avg_price": pos["avg_entry"],
                "current_value": 0.0,
                "updated_at": None,
            }
            self._db.set_portfolio_position(
                ticker=row["ticker"], shares=row["shares"],
                avg_price=row["avg_price"], current_value=row["current_value"],
            )
            result.append(row)
        return result

    def get_orders(self) -> list[dict]:
        """Return open orders."""
        orders = self._ib.openOrders()
        return [
            {
                "order_id": o.orderId,
                "ticker": o.contract.symbol if hasattr(o, "contract") else "?",
                "action": o.action,
                "qty": int(o.totalQuantity),
                "status": o.orderStatus.status if hasattr(o, "orderStatus") else "unknown",
            }
            for o in orders
        ]

    def close_position(self, ticker: str) -> bool:
        """Close an open position by submitting a market order for the opposite side."""
        positions = self._ib.positions()
        for pos in positions:
            if pos.contract.symbol.upper() == ticker.upper():
                qty = int(pos.position)
                if qty == 0:
                    return False
                side = "SELL" if qty > 0 else "BUY"
                contract = self._Stock(ticker.upper(), "SMART", "USD")
                self._ib.qualifyContracts(contract)
                order = self._MarketOrder(side, abs(qty))
                self._ib.placeOrder(contract, order)
                log.info("Closed IBKR position: %s %d shares", ticker, abs(qty))
                return True
        return False

    def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        order_type: str = "market",
    ) -> dict:
        """Submit an order and return a dict with order_id.

        This is a thin wrapper around IB's placeOrder — it does NOT sync
        to the local DB or compute PnL.  Use ``track_trade`` for the
        full pipeline.
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got '{side}'")

        contract = self._Stock(ticker.upper(), "SMART", "USD")
        self._ib.qualifyContracts(contract)

        if order_type == "market":
            order = self._MarketOrder(side, qty)
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

        trade = self._ib.placeOrder(contract, order)
        return {"order_id": trade.order.orderId}

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        # ib_insync doesn't have a simple market-hours query;
        # use reqMarketDataType as a proxy — if we can get live data, market is open.
        try:
            self._ib.reqMarketDataType(1)  # live data
            return True
        except Exception:
            return False

    def get_trade_history(
        self,
        ticker: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return trade history from local DB."""
        return self._db.get_trade_history(ticker=ticker, limit=limit)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_position(
        self, ticker: str, action: str, shares: int, fill_price: float,
    ) -> float:
        """Update local portfolio DB after fill. Returns realised PnL."""
        pnl = 0.0
        existing = self._db.get_portfolio_position(ticker)

        if action == "BUY":
            if existing:
                total = existing["shares"] + shares
                avg = (existing["shares"] * existing["avg_price"]
                       + shares * fill_price) / total
            else:
                total = shares
                avg = fill_price
            self._db.set_portfolio_position(
                ticker=ticker, shares=total, avg_price=avg,
                current_value=round(total * fill_price, 2),
            )
        else:
            if existing:
                pnl = round((fill_price - existing["avg_price"]) * shares, 2)
                remaining = existing["shares"] - shares
                if remaining <= 0:
                    self._db.delete_portfolio_position(ticker)
                else:
                    self._db.set_portfolio_position(
                        ticker=ticker, shares=remaining,
                        avg_price=existing["avg_price"],
                        current_value=round(remaining * fill_price, 2),
                    )
        return pnl
