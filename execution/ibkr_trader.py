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

import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time
from typing import Any, Callable

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


def _is_us_symbol(ticker: str) -> bool:
    """True if *ticker* is a US-listed symbol tradable via SMART/USD.

    Exchange-suffixed tickers contain a "." (VNA.DE, SAP.XETRA, COFA.PA).
    US class shares are stored dash-style in this system (BRK-B, never
    BRK.B — the Wikipedia scrape replaces "." with "-"), so a "." is a
    reliable non-US indicator.
    """
    return "." not in ticker

# Order outcome polling — wait this long for placeOrder to reach a terminal
# orderStatus before we give up and cancel. Tuned to cover normal Gateway
# round-trip plus a few seconds of slack for slow paper sessions.
ORDER_FILL_TIMEOUT = 30.0
ORDER_POLL_INTERVAL = 0.5

# Stop-loss / take-profit SELLs use an extended wait. IBKR fill confirmation
# for closing orders can lag 30-60s in volatile markets; cancelling at 30s
# and retrying cost $471 of slippage on VRT (2026-05-15) when the original
# would have filled at $364.30 but the retry hit $358.30. Strategy: fast-poll
# until STOP_EXTENDED_TIMEOUT, then slow-poll at STOP_SLOW_POLL_INTERVAL up to
# STOP_MAX_WAIT total. If still not filled, return ("stuck", …) so the caller
# alerts the user instead of auto-cancelling.
STOP_EXTENDED_TIMEOUT = 120.0
STOP_MAX_WAIT = 300.0
STOP_SLOW_POLL_INTERVAL = 10.0

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

        # Dedicated event-loop thread for all IBKR I/O. ib_insync's sync
        # API (placeOrder, qualifyContracts, ib.sleep) calls util.run()
        # which uses the *calling thread's* asyncio loop. When PositionManager
        # (a background thread) invoked placeOrder against an IB() built
        # on the main thread, the call hung forever because that thread's
        # loop knew nothing about the IB socket reader (verified
        # 2026-05-11: XOM SELL stalled inside _evaluate_position the
        # moment a position breached its stop). Pinning all IBKR I/O to
        # a single always-running loop on a dedicated thread eliminates
        # the race; cross-thread callers (PM, scheduler) dispatch via
        # asyncio.run_coroutine_threadsafe through _run_in_ib_loop with
        # a 60s timeout so a stuck broker call surfaces as a TimeoutError
        # instead of freezing the monitor.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_thread_ident: int | None = None

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

        from ib_insync import MarketOrder, Stock

        self._Stock = Stock
        self._MarketOrder = MarketOrder

        # Start the dedicated I/O loop thread before constructing IB().
        self._start_loop_thread()

        mode_label = "PAPER" if paper else "LIVE"
        # Dispatch timeout = (10s connect × 3) + (5s wait × 2) + slack = ~50s.
        try:
            self._run_in_ib_loop(self._build_and_connect_ib, timeout=60.0)
        except TimeoutError as exc:
            raise ConnectionError(
                f"Cannot connect to IB Gateway at {host}:{port} — "
                f"timed out after 60s across clientId retries: {exc}"
            ) from exc
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to IB Gateway at {host}:{port} — "
                f"is TWS / IB Gateway running? Error: {exc}"
            ) from exc

        log.info("Connected to IB Gateway (%s)", mode_label)

    # ------------------------------------------------------------------
    # Event-loop thread (IBKR I/O isolation)
    # ------------------------------------------------------------------

    def _start_loop_thread(self) -> None:
        """Spawn a daemon thread that runs an asyncio loop forever.

        The loop owned by this thread is the *only* loop that ever
        sees the IB() socket. All public methods that touch
        ``self._ib`` dispatch onto this thread via :meth:`_run_in_ib_loop`.
        """
        ready = threading.Event()
        startup_err: list[BaseException] = []

        def _loop_main() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._loop_thread_ident = threading.get_ident()
                ready.set()
                loop.run_forever()
            except BaseException as exc:  # pragma: no cover — daemon thread
                startup_err.append(exc)
                ready.set()

        self._loop_thread = threading.Thread(
            target=_loop_main,
            name=f"IBKRTrader-loop-cid{self._client_id}",
            daemon=True,
        )
        self._loop_thread.start()
        if not ready.wait(timeout=10):
            raise ConnectionError(
                "IBKRTrader event-loop thread failed to start within 10s"
            )
        if startup_err:
            raise ConnectionError(
                f"IBKRTrader loop thread crashed during startup: {startup_err[0]}"
            )

    def _build_and_connect_ib(self) -> None:
        """Construct IB() and connect with clientId retry — on the loop thread.

        Mirrors :meth:`reconnect`'s 3-attempt incrementing-clientId pattern.
        Necessary because IB Gateway holds a clientId briefly after the
        previous day's session disconnects — the first US_PRE of every
        morning hits "Peer closed connection" / "Socket disconnect" on
        clientId=1 (observed 4 days in a row, May 6–11 2026). Retrying
        on the same id never recovers; bumping to id+1 connects cleanly.
        """
        from ib_insync import IB
        mode_label = "PAPER" if self._paper else "LIVE"
        base_id = self._client_id
        MAX_ATTEMPTS = 3
        last_exc: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            client_id_try = base_id + attempt
            self._ib = IB()
            try:
                log.info(
                    "Connecting to IB Gateway (%s) at %s:%d "
                    "(clientId=%d, attempt %d/%d) ...",
                    mode_label, self._host, self._port,
                    client_id_try, attempt + 1, MAX_ATTEMPTS,
                )
                self._ib.connect(
                    self._host, self._port,
                    clientId=client_id_try, timeout=10,
                )
                # Remember the id that worked so disconnect() logs it and
                # a subsequent reconnect starts from a known-good base.
                self._client_id = client_id_try
                return
            except Exception as exc:
                last_exc = exc
                # Best-effort teardown of the failed client before next try.
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                if attempt < MAX_ATTEMPTS - 1:
                    log.warning(
                        "Initial connect failed on clientId=%d (%s) — "
                        "waiting 5s and retrying with clientId=%d",
                        client_id_try, str(exc)[:120], client_id_try + 1,
                    )
                    time.sleep(5)
        # All attempts exhausted — re-raise the last error so __init__
        # turns it into the existing ConnectionError contract.
        raise last_exc if last_exc is not None else ConnectionError(
            f"IBKR connect failed after {MAX_ATTEMPTS} clientId attempts "
            f"(base={base_id})"
        )

    def _run_in_ib_loop(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> Any:
        """Execute ``func`` on the IB event-loop thread.

        Same-thread callers run ``func`` directly. Cross-thread callers
        submit a coroutine wrapper via ``asyncio.run_coroutine_threadsafe``
        and block on the future for up to ``timeout`` seconds. On timeout
        the future is cancelled and a TimeoutError is raised so the
        caller can log loudly and alert — the alternative (no timeout)
        is the indefinite hang that wedged PositionManager for four
        days in May 2026.
        """
        # Test mode: a mock IB was injected, no loop thread exists.
        if self._loop is None:
            return func(*args, **kwargs)
        # Same-thread call: avoid the dispatch overhead.
        if threading.get_ident() == self._loop_thread_ident:
            return func(*args, **kwargs)

        async def _wrap() -> Any:
            return func(*args, **kwargs)

        future = asyncio.run_coroutine_threadsafe(_wrap(), self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            log.error(
                "IBKR call timed out after %.1fs in cross-thread dispatch "
                "(func=%s)",
                timeout, getattr(func, "__name__", repr(func)),
            )
            future.cancel()
            raise TimeoutError(
                f"IBKR call did not return within {timeout}s "
                f"(func={getattr(func, '__name__', repr(func))})"
            ) from exc

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Check if IBKR connection is alive."""
        if self._ib is None:
            return False
        def _impl() -> bool:
            try:
                return self._ib.isConnected()
            except Exception:
                return False
        try:
            return self._run_in_ib_loop(_impl, timeout=5.0)
        except TimeoutError:
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

        Runs on the dedicated IB loop thread so the new IB() instance
        is bound to the same loop that owns the rest of the I/O.
        """
        def _impl() -> bool:
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
                    self._client_id = client_id_try
                    return True
                except Exception as exc:
                    err_str = str(exc)
                    is_326 = (
                        "326" in err_str
                        or "already in use" in err_str.lower()
                    )
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

        # reconnect itself can take up to 3 × (10s connect + 10s wait) = 60s.
        try:
            return self._run_in_ib_loop(_impl, timeout=90.0)
        except TimeoutError as exc:
            log.error("reconnect() dispatch timed out: %s", exc)
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
        def _impl() -> None:
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
        try:
            self._run_in_ib_loop(_impl, timeout=15.0)
        except TimeoutError as exc:
            log.warning("disconnect() dispatch timed out: %s", exc)
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
        strategy: str | None = None,
        intended_price: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Submit a market order to IBKR, wait for fill, sync locally."""
        ticker = ticker.upper()
        action = action.upper()

        if not _is_us_symbol(ticker):
            log.warning("skipping non-US symbol %s — no US order path", ticker)
            return {
                "trade_id": None, "ticker": ticker, "action": action,
                "shares": shares, "price": price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "pnl": 0.0, "total_value": 0.0,
                "skipped": True,
                "skip_reason": f"{ticker} is not a US symbol — no US order path",
                "status": "skipped_non_us",
            }

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

        # SELL = stop/TP/exit, which IBKR can take 30-60s to confirm.
        # We give it STOP_MAX_WAIT before declaring it stuck; we never
        # auto-cancel a stuck SELL because doing so causes us to retry
        # against a worse price (VRT, 2026-05-15: $471 of slippage).
        # BUY = fresh entry; cancel + skip after 30s is fine, the next
        # session can re-evaluate.
        is_sell = action == "SELL"
        if is_sell:
            wait_kwargs = {
                "timeout": STOP_MAX_WAIT,
                "slow_poll_after": STOP_EXTENDED_TIMEOUT,
            }
            dispatch_timeout = STOP_MAX_WAIT + 20.0
        else:
            wait_kwargs = {}
            dispatch_timeout = ORDER_FILL_TIMEOUT + 20.0

        def _impl() -> dict:
            contract = self._Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            order = self._MarketOrder(action, shares)
            trade = self._ib.placeOrder(contract, order)
            outcome, detail = self._wait_for_order_terminal(trade, **wait_kwargs)

            if outcome != "filled":
                self._handle_unfilled(trade, ticker, outcome, detail)
                return {
                    "trade_id": None, "ticker": ticker, "action": action,
                    "shares": shares, "price": price,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                    "pnl": 0.0, "total_value": 0.0,
                    "skipped": True,
                    "skip_reason": f"IBKR order {outcome}: {detail}",
                    "outcome": outcome,
                }

            fill_price = trade.orderStatus.avgFillPrice or price
            pnl = self._sync_position(ticker, action, shares, fill_price)

            # Telemetry capture for execution-quality split.  Each value is
            # isolated so a single failure (e.g. malformed CommissionReport
            # on an exotic order) cannot block the trade record itself.
            _intended_val = None
            try:
                _intended_val = intended_price if intended_price is not None else price
            except Exception as exc:
                log.warning("[%s] intended_price capture failed (non-fatal): %s", ticker, exc)
            _executed_val = None
            try:
                _executed_val = fill_price
            except Exception as exc:
                log.warning("[%s] executed_price capture failed (non-fatal): %s", ticker, exc)
            _commission_val: float | None = None
            try:
                fills = getattr(trade, "fills", None) or []
                total = 0.0
                seen_any = False
                for f in fills:
                    cr = getattr(f, "commissionReport", None)
                    if cr is None:
                        continue
                    c = getattr(cr, "commission", None)
                    if c is None:
                        continue
                    total += float(c)
                    seen_any = True
                _commission_val = total if seen_any else None
            except Exception as exc:
                log.warning("[%s] commission parse failed (non-fatal): %s", ticker, exc)
            _strategy_val = None
            try:
                _strategy_val = strategy
            except Exception as exc:
                log.warning("[%s] strategy capture failed (non-fatal): %s", ticker, exc)

            trade_id = self._db.log_trade_history(
                ticker=ticker, action=action, shares=shares, price=fill_price,
                stop_loss=stop_loss, take_profit=take_profit, pnl=pnl,
                strategy=_strategy_val,
                commission=_commission_val,
                intended_price=_intended_val,
                executed_price=_executed_val,
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

        return self._run_in_ib_loop(_impl, timeout=dispatch_timeout)

    def get_account(self) -> dict:
        """Return account summary: cash, portfolio_value, buying_power."""
        def _impl() -> dict:
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
        return self._run_in_ib_loop(_impl, timeout=15.0)

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
        def _impl() -> list[dict]:
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
        return self._run_in_ib_loop(_impl, timeout=15.0)

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
        def _impl() -> list[dict]:
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
        return self._run_in_ib_loop(_impl, timeout=15.0)

    def close_position(self, ticker: str) -> bool:
        """Close an open position by submitting a market order for the opposite side.

        For long positions (the common case) this submits a SELL, which
        uses the extended wait semantics — a stop/exit can take up to 60s
        to confirm in volatile markets and auto-cancelling triggers a
        worse-priced retry (see ``STOP_EXTENDED_TIMEOUT``).
        """
        if not _is_us_symbol(ticker.upper()):
            log.warning("skipping non-US symbol %s — no US order path", ticker)
            return False

        def _impl() -> bool:
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
                    if side == "SELL":
                        outcome, detail = self._wait_for_order_terminal(
                            trade,
                            timeout=STOP_MAX_WAIT,
                            slow_poll_after=STOP_EXTENDED_TIMEOUT,
                        )
                    else:
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
        # SELL path may wait up to STOP_MAX_WAIT; BUY path 30s. Use the
        # larger window so the dispatch never strands a slow SELL.
        return self._run_in_ib_loop(_impl, timeout=STOP_MAX_WAIT + 20.0)

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
        if order_type != "market":
            raise ValueError(f"Unsupported order_type: {order_type}")
        if not _is_us_symbol(ticker.upper()):
            log.warning("skipping non-US symbol %s — no US order path", ticker)
            raise ValueError(f"non-US symbol {ticker}: no US order path")

        def _impl() -> dict:
            contract = self._Stock(ticker.upper(), "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = self._MarketOrder(side, qty)
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
        return self._run_in_ib_loop(_impl, timeout=ORDER_FILL_TIMEOUT + 20.0)

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        # ib_insync doesn't have a simple market-hours query;
        # use reqMarketDataType as a proxy — if we can get live data, market is open.
        def _impl() -> bool:
            try:
                self._ib.reqMarketDataType(1)  # live data
                return True
            except Exception:
                return False
        try:
            return self._run_in_ib_loop(_impl, timeout=10.0)
        except TimeoutError:
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

        def _impl() -> pd.DataFrame:
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
            df_ = pd.DataFrame(rows).set_index("date")
            if len(df_) > limit:
                df_ = df_.iloc[-limit:]
            log.debug("IBKR bars: %s → %d bars (%s)", ticker, len(df_), timeframe)
            return df_

        return self._run_in_ib_loop(_impl, timeout=30.0)

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
        slow_poll_after: float | None = None,
        slow_poll: float | None = None,
    ) -> tuple[str, str]:
        """Poll ``trade.orderStatus.status`` until terminal or timeout.

        Returns (outcome, detail):
            ("filled",    last_status)
            ("cancelled", reason from trade.log[-1].message, or status)
            ("timeout",   last_status)   — fast-poll ceiling (BUY semantics)
            ("stuck",     last_status)   — slow-poll ceiling (SELL semantics)

        When ``slow_poll_after`` is set, polling switches from ``poll`` to
        ``slow_poll`` once that many seconds have elapsed, and the final
        ceiling outcome is ``stuck`` rather than ``timeout`` so the caller
        knows not to auto-cancel — the order may yet fill.

        ``timeout`` / ``poll`` resolve to the module-level constants when
        ``None`` so test patches of those constants take effect.
        """
        eff_timeout = ORDER_FILL_TIMEOUT if timeout is None else timeout
        eff_poll = ORDER_POLL_INTERVAL if poll is None else poll
        eff_slow_poll = STOP_SLOW_POLL_INTERVAL if slow_poll is None else slow_poll
        start = time.monotonic()
        deadline = start + eff_timeout
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
                return (
                    "stuck" if slow_poll_after is not None else "timeout",
                    status,
                )
            if (slow_poll_after is not None
                    and (time.monotonic() - start) >= slow_poll_after):
                interval = eff_slow_poll
            else:
                interval = eff_poll
            try:
                self._ib.sleep(interval)
            except Exception:
                time.sleep(interval)

    def _handle_unfilled(
        self, trade, ticker: str, outcome: str, detail: str,
    ) -> None:
        """Log + (on BUY timeout) cancel an order that didn't fill.

        Outcomes:
            cancelled — IBKR already cancelled; log only.
            stuck     — SELL still PreSubmitted past STOP_MAX_WAIT.
                        DO NOT cancel; the in-flight order is likely a
                        slow fill and cancelling it triggers a worse-priced
                        retry (VRT 2026-05-15 lost $471 this way).
            timeout   — fast-poll ceiling (BUY only). Cancel + move on.
        """
        if outcome == "cancelled":
            log.warning("IBKR ORDER CANCELLED: %s — %s", ticker, detail)
            return
        if outcome == "stuck":
            log.warning(
                "IBKR ORDER STUCK: %s — status=%s after %.0fs, NOT cancelling. "
                "Manual intervention required.",
                ticker, detail, STOP_MAX_WAIT,
            )
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
