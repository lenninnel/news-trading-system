"""
Interactive Brokers execution layer via ib_insync.

IBKRTrader places orders through IB Gateway / TWS and exposes the same
public interface as AlpacaTrader / PaperTrader so all three are
interchangeable behind broker_factory.

Environment variables
---------------------
IBKR_HOST         IB Gateway hostname (default "127.0.0.1")
IBKR_PORT         IB Gateway port (default auto: 4002 paper / 4001 live)
IBKR_CLIENT_ID    TWS API client ID (default 1)
IBKR_PAPER        "true" (default) or "false" for live trading
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import nest_asyncio
import pandas as pd

from storage.database import Database

# ib_insync's sync API (IB.connect, IB.placeOrder, ib.sleep) internally calls
# util.run() which grabs the running event loop. When IBKRTrader is instantiated
# inside asyncio.run(run_batch(...)) — as happens in every scheduled session —
# there is already a running loop and the connect fails with
# "This event loop is already running". nest_asyncio patches asyncio to allow
# the nested use; it's a no-op when no loop is running (daemon startup path).
# Applied at module level so reconnect() also works inside async contexts.
nest_asyncio.apply()

log = logging.getLogger(__name__)

# Tickers that cannot be traded on IBKR via this integration.
UNSUPPORTED_IBKR: set[str] = set()

# Order outcome polling — wait this long for placeOrder to reach a terminal
# orderStatus before we give up and cancel. Tuned to cover normal Gateway
# round-trip plus a few seconds of slack for slow paper sessions.
ORDER_FILL_TIMEOUT = 30.0
ORDER_POLL_INTERVAL = 0.5
_TERMINAL_FILLED = {"Filled"}
_TERMINAL_CANCELLED = {"Cancelled", "Inactive", "ApiCancelled"}


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
        client_id: int | None = None,
    ) -> None:
        self._db = db or Database()

        paper = os.environ.get("IBKR_PAPER", "true").lower() in ("true", "1", "yes")
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        default_port = 4002 if paper else 4001
        port = int(os.environ.get("IBKR_PORT", str(default_port)))
        if client_id is None:
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
    # Connection management
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Check if IBKR connection is alive."""
        if self._ib is None:
            return False
        try:
            return self._ib.isConnected()
        except Exception:
            return False

    def _new_ib_client(self):
        """Build a fresh ib_insync.IB() instance.

        Extracted so the fresh-client path has a single seam that
        tests can monkey-patch.
        """
        from ib_insync import IB
        return IB()

    def reconnect(self) -> bool:
        """Create a fresh ib_async client and connect to IB Gateway.

        After ``disconnect()`` the underlying ib_async client is in a
        destroyed state that cannot be revived — calling ``connect()``
        on it silently fails with "Socket disconnect". We therefore
        discard the old client and build a new ``IB()`` instance on
        every reconnect.

        clientId retry: IB Gateway holds a clientId briefly after the
        socket drops. A fresh connect with the same id can fail with
        Error 326 "client id is already in use". We try up to 3 ids
        (self._client_id, +1, +2) with a 10s wait between 326 failures.
        The successfully-connected id is stored back on ``self._client_id``.
        """
        mode_label = "PAPER" if self._paper else "LIVE"
        # Tear down any existing client so its socket / event loop
        # state is released before we replace it.
        try:
            if self._ib is not None:
                self._ib.disconnect()
        except Exception:
            pass

        base_id = self._client_id
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            client_id_try = base_id + attempt
            try:
                self._ib = self._new_ib_client()
            except Exception as exc:
                log.error("Could not create fresh IB client: %s", exc)
                return False
            try:
                log.info(
                    "Reconnecting to IB Gateway (%s) at %s:%d "
                    "(clientId=%d, attempt %d/%d) ...",
                    mode_label, self._host, self._port,
                    client_id_try, attempt + 1, MAX_ATTEMPTS,
                )
                self._ib.connect(
                    self._host, self._port,
                    clientId=client_id_try, timeout=10,
                )
                log.info(
                    "Reconnected to IB Gateway (%s) using clientId=%d",
                    mode_label, client_id_try,
                )
                # Remember the id that worked so disconnect() logs it and
                # a subsequent reconnect starts from a known-good base.
                self._client_id = client_id_try
                return True
            except Exception as exc:
                err_str = str(exc)
                is_326 = (
                    "326" in err_str
                    or "already in use" in err_str.lower()
                )
                # Best-effort teardown of the failed client before next try.
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                if not is_326:
                    log.error("IBKR reconnect failed (non-326): %s", exc)
                    return False
                log.warning(
                    "clientId %d already in use (Error 326) — "
                    "waiting 10s before retry",
                    client_id_try,
                )
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(10)
        log.error(
            "IBKR reconnect failed after %d clientId attempts (base=%d)",
            MAX_ATTEMPTS, base_id,
        )
        return False

    def disconnect(self) -> None:
        """Cleanly sever the ib_async connection if one exists.

        Called at the end of each trading session so the next session
        starts with a fresh socket instead of inheriting a stale one.

        Sets ``self._ib = None`` so ``ensure_connected()`` knows to
        build a fresh ``IB()`` instance on the next call. Reusing the
        same client object after disconnect leaves it in a destroyed
        state that silently fails on subsequent connect attempts.

        Sleeps 5s before returning to give IB Gateway time to release
        the clientId server-side. Without this pause, the next
        ``reconnect()`` routinely hits Error 326 "client id already in
        use" on the first attempt.
        """
        try:
            if self._ib is not None:
                self._ib.disconnect()
                log.info(
                    "IBKR disconnected cleanly (clientId=%d)",
                    self._client_id,
                )
        except Exception:
            pass
        self._ib = None
        time.sleep(5)

    def ensure_connected(self) -> bool:
        """Ensure a live connection — build a fresh client if needed.

        If ``self._ib`` is None (after ``disconnect()``) or the
        existing client is in an unhealthy state, a new ``IB()`` is
        created before connecting. Never tries to revive a dead
        client.
        """
        if self._ib is None:
            log.info("IBKR client is None — creating fresh instance")
            return self.reconnect()
        if self.is_connected():
            return True
        log.warning("IBKR connection lost — attempting reconnect")
        return self.reconnect()

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
        outcome, detail = self._wait_for_order_terminal(trade)

        if outcome != "filled":
            self._handle_unfilled(trade, ticker, outcome, detail)
            return {
                "trade_id": None, "ticker": ticker, "action": action,
                "shares": shares, "price": price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "pnl": 0.0, "total_value": 0.0,
                "skipped": True,
                "skip_reason": f"IBKR order {outcome}: {detail}",
            }

        fill_price = trade.orderStatus.avgFillPrice or price
        pnl = self._sync_position(ticker, action, shares, fill_price)

        trade_id = self._db.log_trade_history(
            ticker=ticker, action=action, shares=shares, price=fill_price,
            stop_loss=stop_loss, take_profit=take_profit, pnl=pnl,
        )

        log.info(
            "IBKR ORDER FILLED: %s %d shares @ $%.2f (trade_id=%s)",
            action, shares, fill_price, trade_id,
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
        """Return open *equity* positions from IBKR.

        FX cash positions (secType="CASH", e.g. EUR/USD/GBP balances after
        a currency conversion) and other non-stock contract types are
        filtered out: they share the namespace with the equity ticker
        column in portfolio_positions and confuse the PositionManager,
        which would try to evaluate stop-losses against yfinance prices
        for the symbol — e.g. a `pos.contract.symbol == "EUR"` from an
        FX balance hits yfinance's delisted "EUR" ETF and spams skip
        warnings every 60s. Only secType="STK" enters the local table.
        """
        positions = self._ib.positions()
        result = []
        for pos in positions:
            sec_type = getattr(pos.contract, "secType", "STK")
            if sec_type != "STK":
                log.debug(
                    "Skipping non-stock position: %s (secType=%s)",
                    pos.contract.symbol, sec_type,
                )
                continue
            ticker = pos.contract.symbol
            qty = int(pos.position)
            # IBKR Position.avgCost is already average cost per share for STK
            # (verified empirically 2026-05-04: dividing by qty produced
            # fill_price/qty noise — e.g. TRGP $254.49 fill recorded as $4.46).
            avg_entry = float(pos.avgCost) if qty != 0 else 0.0
            result.append({
                "ticker": ticker,
                "qty": qty,
                "avg_entry": avg_entry,
                "current_price": 0.0,    # requires market data subscription
                "unrealized_pl": 0.0,
                "unrealized_plpc": 0.0,
            })
        return result

    def get_portfolio(self) -> list[dict]:
        """Return positions in the format expected by the coordinator."""
        positions = self.get_positions()
        result = []
        for pos in positions:
            qty = int(pos["qty"])
            avg = float(pos["avg_entry"])
            # Entry-value approximation. Live mark-to-market is refreshed by
            # monitoring.position_manager._mark_to_market every 60s during
            # market hours; this write is a sane default between cycles.
            # Why not 0.0: PortfolioManager.can_add_position reads
            # sum(current_value) for the deployment cap. A zero clobbered
            # the numerator and silently disabled the 60 % cap (CASY breach
            # 2026-05-05).
            row = {
                "ticker": pos["ticker"],
                "shares": qty,
                "avg_price": avg,
                "current_value": round(qty * avg, 2),
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
                trade = self._ib.placeOrder(contract, order)
                outcome, detail = self._wait_for_order_terminal(trade)
                if outcome == "filled":
                    log.info(
                        "IBKR ORDER FILLED: %s %d shares of %s",
                        side, abs(qty), ticker,
                    )
                    return True
                self._handle_unfilled(trade, ticker, outcome, detail)
                return False
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
        outcome, detail = self._wait_for_order_terminal(trade)
        if outcome == "filled":
            log.info(
                "IBKR ORDER FILLED: %s %d shares of %s",
                side, qty, ticker,
            )
            return {"order_id": trade.order.orderId, "status": "Filled"}
        self._handle_unfilled(trade, ticker, outcome, detail)
        return {
            "order_id": trade.order.orderId,
            "status": "Cancelled" if outcome == "cancelled" else "Timeout",
            "reason": detail,
        }

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        # ib_insync doesn't have a simple market-hours query;
        # use reqMarketDataType as a proxy — if we can get live data, market is open.
        try:
            self._ib.reqMarketDataType(1)  # live data
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Bar data
    # ------------------------------------------------------------------

    # Timeframe mapping: our convention → IBKR (barSizeSetting, durationStr)
    _BAR_MAP: dict[str, tuple[str, str]] = {
        "1Day":  ("1 day",   "1 Y"),
        "1Hour": ("1 hour",  "5 D"),
        "15Min": ("15 mins", "2 D"),
    }

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 252,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from IB Gateway.

        Args:
            ticker:    Stock symbol (XETRA suffixes are stripped).
            timeframe: ``"1Day"``, ``"1Hour"``, or ``"15Min"``.
            limit:     Number of bars to return (most recent).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
        """
        if timeframe not in self._BAR_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {', '.join(self._BAR_MAP)}"
            )

        bar_size, duration = self._BAR_MAP[timeframe]

        # XETRA tickers: SAP.XETRA → SAP (IBKR uses exchange routing)
        symbol = ticker.upper()
        if symbol.endswith(".XETRA"):
            symbol = symbol.rsplit(".", 1)[0]
        elif symbol.endswith(".DE"):
            symbol = symbol.rsplit(".", 1)[0]

        contract = self._Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            raise ValueError(f"IBKR returned no bars for {ticker} ({timeframe})")

        rows = [
            {
                "date": b.date,
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": int(b.volume),
            }
            for b in bars
        ]
        df = pd.DataFrame(rows).set_index("date")

        # Trim to requested limit (keep most recent)
        if len(df) > limit:
            df = df.iloc[-limit:]

        log.debug("IBKR bars: %s → %d bars (%s)", ticker, len(df), timeframe)
        return df

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

    def _wait_for_order_terminal(
        self, trade, timeout: float | None = None,
        poll: float | None = None,
    ) -> tuple[str, str]:
        """Poll ``trade.orderStatus.status`` until terminal or timeout.

        Returns (outcome, detail):
            ("filled",    last_status)
            ("cancelled", reason from trade.log[-1].message, or status)
            ("timeout",   last_status)

        ``timeout`` / ``poll`` resolve to the module-level constants when
        ``None`` so test patches of those constants take effect.
        """
        eff_timeout = ORDER_FILL_TIMEOUT if timeout is None else timeout
        eff_poll = ORDER_POLL_INTERVAL if poll is None else poll
        deadline = time.monotonic() + eff_timeout
        while True:
            status = getattr(trade.orderStatus, "status", "") or ""
            if status in _TERMINAL_FILLED:
                return ("filled", status)
            if status in _TERMINAL_CANCELLED:
                reason = ""
                try:
                    log_entries = getattr(trade, "log", None) or []
                    if log_entries:
                        reason = getattr(log_entries[-1], "message", "") or ""
                except Exception:
                    reason = ""
                return ("cancelled", reason or status)
            if time.monotonic() >= deadline:
                return ("timeout", status)
            try:
                self._ib.sleep(eff_poll)
            except Exception:
                time.sleep(eff_poll)

    def _handle_unfilled(
        self, trade, ticker: str, outcome: str, detail: str,
    ) -> None:
        """Log + (on timeout) cancel an order that didn't fill."""
        if outcome == "cancelled":
            log.warning("IBKR ORDER CANCELLED: %s — %s", ticker, detail)
            return
        # timeout
        try:
            self._ib.cancelOrder(trade.order)
        except Exception as exc:
            log.warning(
                "IBKR ORDER TIMEOUT: %s — cancelOrder failed: %s",
                ticker, exc,
            )
        else:
            log.warning("IBKR ORDER TIMEOUT: %s — cancelled", ticker)

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
