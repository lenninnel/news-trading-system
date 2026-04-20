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

log = logging.getLogger(__name__)

# US market window (UTC)
_MARKET_OPEN_HOUR = 14
_MARKET_OPEN_MIN = 30
_MARKET_CLOSE_HOUR = 21
_MARKET_CLOSE_MIN = 0

# Trailing-stop parameters
_TRAILING_ACTIVATION_PCT = 2.0   # position must be up >2% to activate
_TRAILING_LOCK_PCT = 1.0         # trail locks in at least 1% gain


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

        # Trailing stops: ticker → current trailing stop price
        self._trailing_stops: dict[str, float] = {}

        # Signal logger for audit trail
        from analytics.signal_logger import SignalLogger
        self._signal_logger = SignalLogger(db=db)

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
        """Return True if current UTC time is within 14:30–21:00 on a weekday."""
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:  # weekend
            return False

        now_minutes = now.hour * 60 + now.minute
        open_minutes = _MARKET_OPEN_HOUR * 60 + _MARKET_OPEN_MIN
        close_minutes = _MARKET_CLOSE_HOUR * 60 + _MARKET_CLOSE_MIN
        return open_minutes <= now_minutes < close_minutes

    # ------------------------------------------------------------------
    # Position checking
    # ------------------------------------------------------------------

    def _check_all_positions(self) -> list[dict]:
        """Iterate open positions and apply stop/profit/trailing logic."""
        positions = self._get_open_positions()
        if not positions:
            return []

        results: list[dict] = []
        for pos in positions:
            ticker = pos["ticker"]
            current_price = self._fetch_current_price(ticker)
            if current_price is None:
                log.warning("No price for %s — skipping", ticker)
                continue

            result = self._evaluate_position(pos, current_price)
            if result:
                results.append(result)

        return results

    def _ensure_trader_connected(self) -> None:
        """Re-establish broker connection if it was torn down between sessions.

        IBKRTrader's ``disconnect()`` sets ``self._ib = None`` at session
        end. Without this call, every subsequent portfolio query raises
        ``'NoneType' object has no attribute 'positions'`` until the
        next session reconnects. Non-IBKR traders simply lack
        ``ensure_connected`` and are a no-op here.
        """
        ensure = getattr(self._trader, "ensure_connected", None)
        if ensure is None:
            return
        try:
            ensure()
        except Exception as exc:
            log.debug(
                "PositionManager silent reconnect failed (non-fatal): %s", exc,
            )

    def _get_open_positions(self) -> list[dict]:
        """
        Get open positions with stop_loss/take_profit from trade history.

        Returns list of dicts with: ticker, shares, avg_price, stop_loss,
        take_profit.
        """
        self._ensure_trader_connected()
        try:
            portfolio = self._trader.get_portfolio()
        except TimeoutError as exc:
            log.warning(
                "Portfolio fetch timed out (non-fatal): %s", exc,
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

        # Use trailing stop if it has been set, otherwise fall back to original
        effective_stop = self._trailing_stops.get(ticker, stop_loss)

        # 1. Stop-loss check
        if effective_stop and current_price <= effective_stop:
            pnl_pct = (current_price - avg_price) / avg_price * 100
            self._close_position(ticker, shares, current_price)
            msg = f"\U0001f534 Stop-loss hit: {ticker} {pnl_pct:+.1f}%"
            self._send_alert(msg)
            self._log_event(ticker, "SELL", current_price,
                            f"stop_loss_triggered at ${effective_stop:.2f}")
            # Clean up trailing stop
            self._trailing_stops.pop(ticker, None)
            return {"ticker": ticker, "action": "stop_loss", "price": current_price,
                    "pnl_pct": pnl_pct}

        # 2. Take-profit check
        if take_profit and current_price >= take_profit:
            pnl_pct = (current_price - avg_price) / avg_price * 100
            self._close_position(ticker, shares, current_price)
            msg = f"\u2705 Take-profit hit: {ticker} {pnl_pct:+.1f}%"
            self._send_alert(msg)
            self._log_event(ticker, "SELL", current_price,
                            f"take_profit_triggered at ${take_profit:.2f}")
            self._trailing_stops.pop(ticker, None)
            return {"ticker": ticker, "action": "take_profit", "price": current_price,
                    "pnl_pct": pnl_pct}

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
                    log.info("Trailing stop updated: %s → $%.2f", ticker, new_stop)
                    self._log_event(ticker, "HOLD", current_price,
                                    f"trailing_stop_updated to ${new_stop:.2f}")
                    return {"ticker": ticker, "action": "trailing_update",
                            "price": current_price, "new_stop": new_stop}

        return None

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _close_position(self, ticker: str, shares: int, price: float) -> None:
        """Sell all shares via the configured broker.

        Never raises. Timeouts and other broker errors are logged as
        warnings so the monitor loop keeps running — we'd rather
        retry on the next cycle than tear down the daemon thread.
        """
        try:
            self._trader.track_trade(
                ticker=ticker, action="SELL", shares=shares, price=price,
            )
            log.info("Closed position: %s %d shares @ $%.2f", ticker, shares, price)
        except TimeoutError as exc:
            log.warning(
                "Close position timed out for %s (non-fatal): %s", ticker, exc,
            )
        except Exception as exc:
            log.warning("Failed to close %s (non-fatal): %s", ticker, exc)

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
