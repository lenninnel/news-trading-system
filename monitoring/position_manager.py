"""
PositionManager — continuous intraday stop-loss and trailing-stop monitor.

Runs as a background thread during US market hours (14:30–21:00 UTC).
Every 60 seconds it checks each open position against its stop-loss,
take-profit, and a trailing-stop rule that locks in gains once a
position is profitable by >2%.

Decisions are logged to ``signal_events`` via :class:`SignalLogger` and
Telegram alerts are sent for every triggered action.

Integration
-----------
Started by :class:`DailyScheduler` in ``scheduler/daily_runner.py``::

    pm = PositionManager(trader=trader, notifier=tg)
    pm.start()   # non-blocking — runs in a daemon thread
    ...
    pm.stop()    # graceful shutdown
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

# US market hours expressed in exchange-local time so DST is handled
# automatically. Previously the constants were hardcoded UTC (14:30-21:00),
# which is correct only during EST (winter); during EDT (March-November)
# the NYSE opens at 13:30 UTC, so the monitor was skipping the first
# hour of every summer trading day.
_NY_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN_LOCAL = (9, 30)    # 9:30 AM ET
_MARKET_CLOSE_LOCAL = (16, 0)   # 4:00 PM ET

# Trailing-stop parameters
_TRAILING_ACTIVATION_PCT = 2.0   # position must be up >2% to activate
_TRAILING_LOCK_PCT = 1.0         # trail locks in at least 1% gain

# Stuck-order cooldown: after a stop/TP SELL hits IBKR's STOP_MAX_WAIT and
# is left in flight, we don't want to evaluate the same position again on
# the next 60s cycle (which would submit a duplicate order). Wait this
# many seconds before re-evaluating so the original has a chance to fill
# or be cleared manually.
_STUCK_ORDER_COOLDOWN_S = 600.0


class PositionManager:
    """
    Background thread that monitors open positions every 60 seconds.

    Args:
        trader:    Broker instance (PaperTrader, AlpacaTrader, or IBKRTrader).
        notifier:  Optional TelegramNotifier for alerts.
        db:        Optional Database for signal logging.
        interval:  Seconds between checks (default 60).
    """

    def __init__(
        self,
        trader: Any,
        notifier: Any | None = None,
        db: Any | None = None,
        interval: int = 60,
    ) -> None:
        self._trader = trader
        self._notifier = notifier
        self._interval = interval
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

        # Trailing stops: ticker → current trailing stop price.
        # Primary working state is this in-memory dict; the DB
        # (portfolio_positions.trailing_stop column) is the persistence
        # layer used to survive daemon restarts. Rehydrated below.
        self._trailing_stops: dict[str, float] = {}

        # Stuck-order cooldown: ticker → monotonic timestamp of the most
        # recent ORDER STUCK alert. While a ticker is in cooldown, we
        # skip evaluation so we don't submit a duplicate SELL on top of
        # the in-flight one (which IBKR did not cancel for us).
        self._stuck_orders: dict[str, float] = {}

        # Signal logger for audit trail
        from analytics.signal_logger import SignalLogger
        self._signal_logger = SignalLogger(db=db)

        # Direct DB handle so we can mark-to-market portfolio_positions every
        # cycle. Without this the EOD P&L summary reads stale current_value
        # (IBKRTrader.get_portfolio() writes 0 to current_value because IB
        # paper feeds don't include market prices).
        self._db = db

        # Rehydrate trailing stops from DB. Before 2026-05-12 the dict
        # was wiped on every restart, so any locked-in gain disappeared
        # until the next 60s cycle re-established the trail at the new
        # (lower) price. Best-effort — a DB error here is non-fatal;
        # the monitor simply re-trails on the next cycle.
        try:
            db_for_read = self._db
            if db_for_read is None:
                from storage.database import Database
                db_for_read = Database()
                self._db = db_for_read
            persisted = db_for_read.get_trailing_stops()
            if persisted:
                self._trailing_stops.update(persisted)
                log.info(
                    "PositionManager rehydrated %d trailing stop(s): %s",
                    len(persisted),
                    {t: f"${p:.2f}" for t, p in persisted.items()},
                )
            else:
                log.info("PositionManager rehydrate: no persisted trailing stops")
        except Exception as exc:
            log.warning(
                "PositionManager trailing-stop rehydrate failed (non-fatal): %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the monitor in a daemon thread."""
        if self._thread and self._thread.is_alive():
            log.warning("PositionManager already running")
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="PositionManager",
            daemon=True,
        )
        self._thread.start()
        log.info("PositionManager started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._shutdown.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            log.info("PositionManager stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Sleep-check loop that runs during US market hours.

        Every exception that could possibly escape from a cycle is
        caught here and logged as a warning — the position manager is
        a best-effort monitor and MUST NEVER crash the daemon, even
        when IB Gateway times out or the broker connection flaps.
        """
        log.info("PositionManager thread started")
        while not self._shutdown.is_set():
            # Belt-and-suspenders: wrap the whole cycle (market hours
            # check + position check + sleep) so a TimeoutError or any
            # other stray exception can never kill this thread.
            try:
                if not self._is_market_hours():
                    self._shutdown.wait(timeout=self._interval)
                    continue

                try:
                    self._check_all_positions()
                except TimeoutError as exc:
                    log.warning(
                        "PositionManager: request timed out (non-fatal): %s",
                        exc,
                    )
                except Exception as exc:
                    log.warning(
                        "PositionManager cycle error (non-fatal): %s", exc,
                    )

                self._shutdown.wait(timeout=self._interval)
            except Exception as exc:
                # Absolute safety net — nothing should escape _run_loop.
                log.warning(
                    "PositionManager thread-level error (swallowed): %s", exc,
                )
                self._shutdown.wait(timeout=self._interval)

        log.info("PositionManager thread exiting")

    # ------------------------------------------------------------------
    # Market hours check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_market_hours() -> bool:
        """Return True if it's a US-Eastern weekday between 9:30 and 16:00."""
        now = datetime.now(_NY_TZ)
        if now.weekday() >= 5:  # weekend
            return False

        now_minutes = now.hour * 60 + now.minute
        open_minutes = _MARKET_OPEN_LOCAL[0] * 60 + _MARKET_OPEN_LOCAL[1]
        close_minutes = _MARKET_CLOSE_LOCAL[0] * 60 + _MARKET_CLOSE_LOCAL[1]
        return open_minutes <= now_minutes < close_minutes

    # ------------------------------------------------------------------
    # Position checking
    # ------------------------------------------------------------------

    def _check_all_positions(self) -> list[dict]:
        """Iterate open positions and apply stop/profit/trailing logic."""
        log.info("PM heartbeat: _check_all_positions ENTRY")
        positions = self._get_open_positions()
        tickers = [p["ticker"] for p in positions]
        log.info("PM heartbeat: cycle running, %d positions to evaluate: %s",
                 len(positions), tickers)
        if not positions:
            log.info("PM heartbeat: _check_all_positions EXIT (no positions)")
            return []

        results: list[dict] = []
        for pos in positions:
            ticker = pos["ticker"]
            log.info("PM heartbeat: [%s] fetching price (sl=%s tp=%s)",
                     ticker, pos.get("stop_loss"), pos.get("take_profit"))
            current_price = self._fetch_current_price(ticker)
            if current_price is None:
                log.warning("No price for %s — skipping", ticker)
                continue
            log.info("PM heartbeat: [%s] price fetched = $%.2f", ticker, current_price)

            # Mark-to-market: persist live current_value so the EOD P&L
            # summary and dashboard see real numbers. Best-effort — never
            # block stop-loss evaluation if the DB write fails.
            self._mark_to_market(ticker, pos.get("shares", 0),
                                 pos.get("avg_price", 0), current_price)

            log.info("PM heartbeat: [%s] evaluating px=$%.2f sl=%s tp=%s",
                     ticker, current_price, pos.get("stop_loss"),
                     pos.get("take_profit"))
            result = self._evaluate_position(pos, current_price)
            log.info("PM heartbeat: [%s] eval result=%s", ticker, result)
            if result:
                results.append(result)

        log.info("PM heartbeat: _check_all_positions EXIT (results=%d)", len(results))
        return results

    def _mark_to_market(
        self,
        ticker: str,
        shares: int,
        avg_price: float,
        current_price: float,
    ) -> None:
        """Update portfolio_positions.current_value with the live mark.

        Runs once per ticker per 60s cycle during market hours. Without it,
        the column is whatever was written at the last broker sync — which
        for IBKR paper is 0.0 (no market data subscription).
        """
        if shares <= 0 or current_price <= 0:
            return
        try:
            db = self._db
            if db is None:
                from storage.database import Database
                db = Database()
                self._db = db
            db.set_portfolio_position(
                ticker=ticker,
                shares=shares,
                avg_price=avg_price,
                current_value=round(shares * current_price, 2),
            )
        except Exception as exc:
            log.warning(
                "Mark-to-market write failed for %s (non-fatal): %s",
                ticker, exc,
            )

    def _persist_trailing_stop(self, ticker: str, stop_price: float) -> None:
        """Best-effort DB write for the trailing-stop value.

        Errors are logged as warnings — the in-memory dict is authoritative
        for the current process; the DB is only consulted at startup
        rehydration. A failed write means a restart in the next few
        seconds would lose this update; we accept that risk rather than
        block the monitor on a transient SQLite error.
        """
        if self._db is None:
            return
        try:
            self._db.set_trailing_stop(ticker, stop_price)
        except Exception as exc:
            log.warning(
                "Trailing-stop persist failed for %s (non-fatal): %s",
                ticker, exc,
            )

    def _clear_persisted_trailing_stop(self, ticker: str) -> None:
        """Null out the trailing_stop column for *ticker*. Never raises."""
        if self._db is None:
            return
        try:
            self._db.set_trailing_stop(ticker, None)
        except Exception as exc:
            log.warning(
                "Trailing-stop clear failed for %s (non-fatal): %s",
                ticker, exc,
            )

    def _ensure_trader_connected(self) -> bool:
        """Re-establish broker connection if it was torn down between sessions.

        Returns True when the broker is connected (or reconnected
        successfully), False when reconnect failed or raised. Callers must
        honour the return value — proceeding to query the broker after a
        False return hits ``'NoneType' object has no attribute 'positions'``
        because ``IBKRTrader.disconnect()`` sets ``self._ib = None``.

        Non-IBKR traders (PaperTrader, AlpacaTrader) lack ``ensure_connected``
        and are assumed always-on.
        """
        ensure = getattr(self._trader, "ensure_connected", None)
        if ensure is None:
            return True
        try:
            ok = ensure()
        except Exception as exc:
            log.warning(
                "PositionManager reconnect raised (non-fatal): %s", exc,
            )
            return False
        if not ok:
            log.warning(
                "PositionManager reconnect returned False — "
                "skipping portfolio query this cycle",
            )
            return False
        return True

    def _get_open_positions(self) -> list[dict]:
        """
        Get open positions with stop_loss/take_profit from trade history.

        Returns list of dicts with: ticker, shares, avg_price, stop_loss,
        take_profit.
        """
        connected = self._ensure_trader_connected()
        log.info("PM heartbeat: ensure_connected=%s, is_connected=%s",
                 connected, getattr(self._trader, "is_connected", lambda: "n/a")())
        if not connected:
            return []
        try:
            portfolio = self._trader.get_portfolio()
            log.info("PM heartbeat: trader.get_portfolio() returned %d items",
                     len(portfolio) if portfolio else 0)
        except TimeoutError as exc:
            log.warning(
                "Portfolio fetch timed out (non-fatal): %s", exc,
            )
            return []
        except AttributeError as exc:
            # Race: scheduler thread set self._ib=None between our
            # ensure_connected() call and this get_portfolio() call.
            # Next cycle will reconnect; for now, skip quietly.
            log.warning(
                "Portfolio query raced with disconnect (non-fatal): %s", exc,
            )
            return []
        except Exception as exc:
            log.warning("Failed to get portfolio (non-fatal): %s", exc)
            return []

        if not portfolio:
            return []

        results: list[dict] = []
        for pos in portfolio:
            ticker = pos["ticker"]
            # Look up the original stop_loss / take_profit from trade history
            stop_loss = None
            take_profit = None
            try:
                trades = self._trader.get_trade_history(ticker=ticker, limit=1)
                if trades and trades[0].get("action") == "BUY":
                    stop_loss = trades[0].get("stop_loss")
                    take_profit = trades[0].get("take_profit")
            except Exception:
                pass

            results.append({
                "ticker": ticker,
                "shares": pos["shares"],
                "avg_price": pos["avg_price"],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })
        return results

    def _evaluate_position(self, pos: dict, current_price: float) -> dict | None:
        """
        Check a single position against stop-loss, take-profit, and
        trailing-stop rules.

        Returns an action dict if something was triggered, else None.
        """
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_price = pos["avg_price"]
        stop_loss = pos.get("stop_loss")
        take_profit = pos.get("take_profit")

        # Stuck-order cooldown: if a recent SELL hit IBKR's STOP_MAX_WAIT
        # ceiling without confirming, sit out this cycle so we don't pile
        # a duplicate on top of the in-flight order. Cooldown elapses
        # after _STUCK_ORDER_COOLDOWN_S, by which point either the
        # original filled or the user intervened manually.
        last_stuck = self._stuck_orders.get(ticker)
        if last_stuck is not None and (time.monotonic() - last_stuck) < _STUCK_ORDER_COOLDOWN_S:
            remaining = _STUCK_ORDER_COOLDOWN_S - (time.monotonic() - last_stuck)
            log.info(
                "[%s] Skipping eval - stuck-order cooldown (%.0fs remaining)",
                ticker, remaining,
            )
            return None

        # Use trailing stop if it has been set, otherwise fall back to original
        effective_stop = self._trailing_stops.get(ticker, stop_loss)

        # 1. Stop-loss check
        if effective_stop and current_price <= effective_stop:
            pnl_pct = (current_price - avg_price) / avg_price * 100
            result = self._close_position(ticker, shares, current_price)
            return self._handle_close_result(
                result, ticker, current_price, pnl_pct,
                trigger="stop_loss", level=effective_stop,
                alert_emoji="\U0001f534", alert_label="Stop-loss",
            )

        # 2. Take-profit check
        if take_profit and current_price >= take_profit:
            pnl_pct = (current_price - avg_price) / avg_price * 100
            result = self._close_position(ticker, shares, current_price)
            return self._handle_close_result(
                result, ticker, current_price, pnl_pct,
                trigger="take_profit", level=take_profit,
                alert_emoji="\u2705", alert_label="Take-profit",
            )

        # 3. Trailing stop: activate when position is up >2%
        if avg_price > 0:
            gain_pct = (current_price - avg_price) / avg_price * 100
            if gain_pct > _TRAILING_ACTIVATION_PCT:
                # Trail stop to lock in at least 1% gain
                lock_price = avg_price * (1 + _TRAILING_LOCK_PCT / 100)
                # Trail at (current_price - activation_buffer)
                trail_price = current_price * (1 - _TRAILING_ACTIVATION_PCT / 100)
                new_stop = max(lock_price, trail_price)

                current_trailing = self._trailing_stops.get(ticker)
                if current_trailing is None or new_stop > current_trailing:
                    self._trailing_stops[ticker] = new_stop
                    self._persist_trailing_stop(ticker, new_stop)
                    log.info("Trailing stop updated: %s → $%.2f", ticker, new_stop)
                    self._log_event(ticker, "HOLD", current_price,
                                    f"trailing_stop_updated to ${new_stop:.2f}")
                    return {"ticker": ticker, "action": "trailing_update",
                            "price": current_price, "new_stop": new_stop}

        return None

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _resolve_entry_strategy(self, ticker: str) -> str | None:
        """Look up the strategy that opened this position.

        Reads position_metadata.strategy.  Returns None if the row is
        missing, the column is unset, or the lookup raises — callers
        treat None as "attribution unknown" and the SELL still records.
        """
        if self._db is None:
            return None
        try:
            import sqlite3
            with sqlite3.connect(self._db.db_path, timeout=5.0) as conn:
                row = conn.execute(
                    "SELECT strategy FROM position_metadata WHERE ticker = ?",
                    (ticker.upper(),),
                ).fetchone()
            if row and row[0]:
                return row[0]
        except Exception as exc:
            log.warning(
                "[%s] entry-strategy lookup failed (non-fatal): %s", ticker, exc,
            )
        return None

    def _close_position(
        self, ticker: str, shares: int, price: float,
    ) -> dict | None:
        """Sell all shares via the configured broker.

        Returns the broker result dict (or ``None`` on exception). The
        caller inspects ``outcome``/``skipped`` to decide alerting and
        bookkeeping. Never raises — broker errors are logged as warnings
        so the monitor loop keeps running.
        """
        try:
            # Carry the entry strategy onto the exit row so attribution
            # surfaces in the per-strategy PnL views.  Failure-soft.
            entry_strategy = self._resolve_entry_strategy(ticker)
            result = self._trader.track_trade(
                ticker=ticker, action="SELL", shares=shares, price=price,
                strategy=entry_strategy, intended_price=price,
            )
            log.info("Close requested: %s %d shares @ $%.2f", ticker, shares, price)
            return result
        except TimeoutError as exc:
            log.warning(
                "Close position timed out for %s (non-fatal): %s", ticker, exc,
            )
            return None
        except Exception as exc:
            log.warning("Failed to close %s (non-fatal): %s", ticker, exc)
            return None

    def _handle_close_result(
        self,
        result: dict | None,
        ticker: str,
        current_price: float,
        pnl_pct: float,
        *,
        trigger: str,
        level: float,
        alert_emoji: str,
        alert_label: str,
    ) -> dict | None:
        """Dispatch alerts and bookkeeping based on the SELL outcome.

        - filled  → fire the normal "<label> hit" Telegram alert, clear
          trailing-stop state, return the trigger dict.
        - stuck   → fire an "ORDER STUCK" alert, register cooldown so the
          next cycles don't pile duplicate orders on the in-flight SELL.
          DO NOT clear trailing-stop state — the original is still live
          and may yet fill.
        - cancelled / timeout / None → log only. The next 60s cycle will
          re-evaluate and retry naturally if conditions still hold.
        """
        filled = bool(
            result and result.get("trade_id") and not result.get("skipped")
        )
        outcome = (result or {}).get("outcome")

        if filled:
            msg = f"{alert_emoji} {alert_label} hit: {ticker} {pnl_pct:+.1f}%"
            self._send_alert(msg)
            self._log_event(
                ticker, "SELL", current_price,
                f"{trigger}_triggered at ${level:.2f}",
            )
            # Belt-and-braces: a full close also wipes the
            # portfolio_positions row via _sync_position, but clear the
            # trailing stop explicitly so a partial-fill or stale row
            # never leaves a dangling trail.
            self._trailing_stops.pop(ticker, None)
            self._clear_persisted_trailing_stop(ticker)
            return {
                "ticker": ticker, "action": trigger,
                "price": current_price, "pnl_pct": pnl_pct,
            }

        if outcome == "stuck":
            self._stuck_orders[ticker] = time.monotonic()
            msg = (
                f"⚠️ ORDER STUCK: {ticker} {alert_label} SELL still "
                f"PreSubmitted after 5min. Order NOT cancelled — check IBKR "
                f"Gateway manually."
            )
            self._send_alert(msg)
            self._log_event(
                ticker, "SELL", current_price,
                f"{trigger}_order_stuck at ${level:.2f}",
            )
            return {
                "ticker": ticker, "action": f"{trigger}_stuck",
                "price": current_price, "pnl_pct": pnl_pct,
            }

        reason = (result or {}).get("skip_reason") or "no broker result"
        log.info(
            "[%s] %s SELL did not fill (%s); next cycle will retry",
            ticker, alert_label, reason,
        )
        return None

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_current_price(ticker: str) -> float | None:
        """Fetch latest 1-minute price via yfinance."""
        try:
            import yfinance as yf
            data = yf.Ticker(ticker).history(period="1d", interval="1m")
            if data.empty:
                return None
            return float(data["Close"].iloc[-1])
        except Exception as exc:
            log.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Logging & notifications
    # ------------------------------------------------------------------

    def _log_event(
        self,
        ticker: str,
        signal: str,
        price: float,
        detail: str,
    ) -> None:
        """Log decision to signal_events table (never raises)."""
        try:
            self._signal_logger.log({
                "ticker": ticker,
                "strategy": "PositionManager",
                "signal": signal,
                "price_at_signal": price,
                "bull_case": detail,
            })
        except Exception as exc:
            log.warning(
                "Signal log failed for %s (non-fatal): %s", ticker, exc,
            )

    def _send_alert(self, message: str) -> None:
        """Send Telegram alert (best-effort, never raises)."""
        log.info("ALERT: %s", message)
        if self._notifier is not None:
            try:
                self._notifier.send_price_alert(message)
            except Exception as exc:
                log.warning("Telegram alert failed: %s", exc)
