"""
PriceMonitor — real-time price alert and stop-loss tracker.

Monitors all open paper-portfolio positions and fires alerts when:
  • Stop-loss is hit (current price ≤ stop_loss)
  • Take-profit is hit (current price ≥ take_profit)
  • Intraday price moves > threshold % (default 5 %)
  • Volume spikes > N × 20-day average (default 3×)

Modes
-----
  Daemon      python3 monitoring/price_monitor.py --daemon
              Loops forever; checks every 60 s during market hours.
              Sleeps gracefully outside hours.

  Once        python3 monitoring/price_monitor.py --check-now
              Single pass over all open positions, then exit.

  Backfill    python3 monitoring/price_monitor.py --backfill
              Single pass; logs alerts to DB but suppresses Telegram.

Market hours (active if ANY market is open)
-------------------------------------------
  US (NYSE)   Mon–Fri  09:30–16:00  America/New_York
  EU (XETRA)  Mon–Fri  09:00–17:30  Europe/Berlin

Alert spam protection
---------------------
  max_alerts_per_day : 5 per (ticker, alert_type) pair
  alert_cooldown     : 300 s between same (ticker, alert_type) pair

Config (watchlist.yaml → monitoring section)
--------------------------------------------
  monitoring:
    enabled: true
    check_interval: 60
    market_hours_only: true
    auto_close_on_stop: false
    volume_spike_threshold: 3.0
    price_move_threshold: 5.0
    max_alerts_per_day: 5
    alert_cooldown: 300
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

# ── Path setup (allow running as __main__ from project root) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH                           # noqa: E402
from execution.paper_trader import PaperTrader                # noqa: E402
from storage.database import Database                         # noqa: E402

log = logging.getLogger(__name__)

# ── Timezone helpers ──────────────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    _TZ_ET  = ZoneInfo("America/New_York")
    _TZ_CET = ZoneInfo("Europe/Berlin")
except Exception:
    import pytz
    _TZ_ET  = pytz.timezone("America/New_York")
    _TZ_CET = pytz.timezone("Europe/Berlin")

# ── Price quote cache ─────────────────────────────────────────────────────────
# {ticker: (fetched_at_epoch, quote_dict)}
_QUOTE_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL   = 300  # 5 minutes

# ── Alert dataclass ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    ticker:       str
    alert_type:   str          # stop_loss | take_profit | price_move | volume_spike
    price:        float
    message:      str
    stop_loss:    "float | None"  = None
    take_profit:  "float | None"  = None
    change_pct:   "float | None"  = None
    volume_ratio: "float | None"  = None


# ══════════════════════════════════════════════════════════════════════════════
# PriceMonitor
# ══════════════════════════════════════════════════════════════════════════════

class PriceMonitor:
    """
    Real-time price alert and stop-loss monitor for open paper positions.

    Args:
        cfg:            Full config dict (from watchlist.yaml).
        notifier:       Optional TelegramNotifier instance.
        auto_close:     Close positions on stop-loss hit (overrides config).
        db_path:        SQLite file path.
    """

    # ── Hard defaults (overridden by config) ──────────────────────────────────
    DEFAULT_INTERVAL          = 60     # seconds
    DEFAULT_VOL_THRESHOLD     = 3.0   # × average volume
    DEFAULT_PRICE_THRESHOLD   = 5.0   # % intraday move
    DEFAULT_MAX_ALERTS_DAY    = 5
    DEFAULT_COOLDOWN          = 300   # seconds

    def __init__(
        self,
        cfg: dict | None           = None,
        notifier: Any              = None,
        auto_close: bool           = False,
        db_path: str               = DB_PATH,
    ) -> None:
        cfg = cfg or {}
        m   = cfg.get("monitoring", {})

        self._notifier          = notifier
        self._db                = Database(db_path)
        self._db_path           = db_path
        self._paper_trader      = PaperTrader(db_path)

        self._interval          = int(m.get("check_interval",        self.DEFAULT_INTERVAL))
        self._market_hours_only = bool(m.get("market_hours_only",    True))
        self._auto_close        = auto_close or bool(m.get("auto_close_on_stop", False))
        self._vol_threshold     = float(m.get("volume_spike_threshold", self.DEFAULT_VOL_THRESHOLD))
        self._price_threshold   = float(m.get("price_move_threshold",   self.DEFAULT_PRICE_THRESHOLD))
        self._max_per_day       = int(m.get("max_alerts_per_day",   self.DEFAULT_MAX_ALERTS_DAY))
        self._cooldown          = int(m.get("alert_cooldown",        self.DEFAULT_COOLDOWN))

        # Runtime state
        self._running        = False
        self._shutdown       = threading.Event()

        # Cooldown tracking: (ticker, alert_type) → last fire epoch
        self._last_alert: dict[tuple[str, str], float] = {}
        # Daily counter:   (ticker, alert_type, date_str) → count
        self._daily_count: dict[tuple[str, str, str], int] = {}

    # ------------------------------------------------------------------
    # Public modes
    # ------------------------------------------------------------------

    def run_daemon(self) -> None:
        """Block indefinitely; check every check_interval during market hours."""
        log.info("PriceMonitor daemon started (interval=%ds, market_hours_only=%s)",
                 self._interval, self._market_hours_only)

        # Graceful shutdown on SIGTERM / SIGINT
        def _handle_signal(sig, frame):  # type: ignore[return]
            log.info("Signal %s received — shutting down.", sig)
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT,  _handle_signal)

        self._running = True
        while not self._shutdown.is_set():
            if self._market_hours_only and not self._is_market_hours():
                wait = self._seconds_until_open()
                log.info("Market closed — sleeping %.0f min.", wait / 60)
                self._shutdown.wait(timeout=min(wait, self._interval * 10))
                continue

            try:
                alerts = self._run_once(notify=True)
                log.info("Check complete: %d position(s) checked, %d alert(s) fired.",
                         len(self._last_positions), len(alerts))
            except Exception as exc:
                log.error("Monitor cycle error: %s", exc, exc_info=True)

            self._shutdown.wait(timeout=self._interval)

        self._running = False
        log.info("PriceMonitor daemon stopped.")

    def check_now(self) -> list[Alert]:
        """Run one check pass with Telegram notifications. Returns fired alerts."""
        return self._run_once(notify=True)

    def backfill(self) -> list[Alert]:
        """Run one check pass; log to DB but suppress Telegram. Returns alerts."""
        return self._run_once(notify=False)

    # ------------------------------------------------------------------
    # Core check loop
    # ------------------------------------------------------------------

    def _run_once(self, notify: bool = True) -> list[Alert]:
        """Fetch quotes and check all open positions. Returns list of fired alerts."""
        positions = self._get_open_positions()
        self._last_positions = positions

        if not positions:
            log.info("No open positions to monitor.")
            return []

        all_alerts: list[Alert] = []
        for pos in positions:
            ticker = pos["ticker"]
            quote  = self._fetch_quote(ticker)
            if not quote or not quote.get("price"):
                log.warning("No quote data for %s — skipping.", ticker)
                continue

            alerts = self._check_position(pos, quote)
            for alert in alerts:
                if self._can_alert(alert.ticker, alert.alert_type):
                    self._fire_alert(alert, notify=notify)
                    all_alerts.append(alert)

                    if self._auto_close and alert.alert_type == "stop_loss":
                        self._auto_close_position(pos, quote["price"])

        return all_alerts

    # ------------------------------------------------------------------
    # Alert condition checks
    # ------------------------------------------------------------------

    def _check_position(self, pos: dict, quote: dict) -> list[Alert]:
        """Evaluate all alert conditions for one position. Returns triggered alerts."""
        alerts: list[Alert] = []
        ticker      = pos["ticker"]
        price       = quote["price"]
        prev_close  = quote.get("prev_close") or price
        stop_loss   = pos.get("stop_loss")
        take_profit = pos.get("take_profit")
        today_vol   = quote.get("today_volume", 0.0) or 0.0
        avg_vol     = quote.get("avg_volume", 0.0) or 0.0

        # ── 1. Stop-loss ──────────────────────────────────────────────────────
        if stop_loss and price <= stop_loss:
            pct = (price - stop_loss) / stop_loss * 100
            msg = (
                f"🔴 STOP LOSS HIT: {ticker} at ${price:.2f} "
                f"(stop: ${stop_loss:.2f}, {pct:+.1f}%)"
            )
            alerts.append(Alert(
                ticker=ticker, alert_type="stop_loss", price=price,
                message=msg, stop_loss=stop_loss,
                change_pct=pct,
            ))

        # ── 2. Take-profit ────────────────────────────────────────────────────
        elif take_profit and price >= take_profit:
            pct = (price - take_profit) / take_profit * 100
            msg = (
                f"🟢 TAKE PROFIT HIT: {ticker} at ${price:.2f} "
                f"(target: ${take_profit:.2f}, {pct:+.1f}%)"
            )
            alerts.append(Alert(
                ticker=ticker, alert_type="take_profit", price=price,
                message=msg, take_profit=take_profit,
                change_pct=pct,
            ))

        # ── 3. Intraday price move ────────────────────────────────────────────
        if prev_close > 0:
            intraday_pct = (price - prev_close) / prev_close * 100
            if abs(intraday_pct) >= self._price_threshold:
                direction = "down" if intraday_pct < 0 else "up"
                msg = (
                    f"⚠️ PRICE ALERT: {ticker} {direction} "
                    f"{intraday_pct:.1f}% to ${price:.2f}"
                )
                alerts.append(Alert(
                    ticker=ticker, alert_type="price_move", price=price,
                    message=msg, change_pct=intraday_pct,
                ))

        # ── 4. Volume spike ───────────────────────────────────────────────────
        if avg_vol > 0 and today_vol > 0:
            vol_ratio = today_vol / avg_vol
            if vol_ratio >= self._vol_threshold:
                msg = (
                    f"📊 VOLUME SPIKE: {ticker} volume "
                    f"{vol_ratio:.1f}x average"
                )
                alerts.append(Alert(
                    ticker=ticker, alert_type="volume_spike", price=price,
                    message=msg, volume_ratio=vol_ratio,
                ))

        return alerts

    # ------------------------------------------------------------------
    # Alert dispatch
    # ------------------------------------------------------------------

    def _fire_alert(self, alert: Alert, notify: bool = True) -> None:
        """Persist alert to DB, optionally send Telegram, update cooldown."""
        # Log to DB
        try:
            self._db.log_price_alert(
                ticker       = alert.ticker,
                alert_type   = alert.alert_type,
                price        = alert.price,
                message      = alert.message,
                stop_loss    = alert.stop_loss,
                take_profit  = alert.take_profit,
                change_pct   = alert.change_pct,
                volume_ratio = alert.volume_ratio,
            )
        except Exception as exc:
            log.warning("Could not persist price alert: %s", exc)

        log.info("ALERT: %s", alert.message)

        # Telegram
        if notify and self._notifier is not None:
            try:
                self._notifier.send_price_alert(alert.message)
            except Exception as exc:
                log.warning("Telegram send failed: %s", exc)

        # Update cooldown + daily counter
        key = (alert.ticker, alert.alert_type)
        self._last_alert[key] = time.time()
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_key  = (alert.ticker, alert.alert_type, date_str)
        self._daily_count[day_key] = self._daily_count.get(day_key, 0) + 1

    def _can_alert(self, ticker: str, alert_type: str) -> bool:
        """Return True if this (ticker, alert_type) passes cooldown + daily-limit checks."""
        key     = (ticker, alert_type)
        now     = time.time()
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_key = (ticker, alert_type, date_str)

        # Daily limit
        if self._daily_count.get(day_key, 0) >= self._max_per_day:
            log.debug("Daily limit reached for %s/%s — suppressing.", ticker, alert_type)
            return False

        # Cooldown
        last = self._last_alert.get(key, 0.0)
        if now - last < self._cooldown:
            remaining = int(self._cooldown - (now - last))
            log.debug("Cooldown active for %s/%s — %ds remaining.", ticker, alert_type, remaining)
            return False

        return True

    # ------------------------------------------------------------------
    # Auto-close
    # ------------------------------------------------------------------

    def _auto_close_position(self, pos: dict, price: float) -> None:
        """Sell all shares of *pos* at *price* via PaperTrader."""
        ticker = pos["ticker"]
        shares = pos["shares"]
        try:
            trade_id = self._paper_trader.track_trade(
                ticker=ticker, action="SELL", shares=shares, price=price,
            )
            log.info(
                "Auto-closed %s × %d sh @ $%.2f  (trade #%s)",
                ticker, shares, price, trade_id,
            )
            if self._notifier:
                self._notifier.send_price_alert(
                    f"🔒 AUTO-CLOSE: {ticker} — sold {shares} sh @ ${price:.2f} "
                    f"(stop-loss triggered)"
                )
        except Exception as exc:
            log.error("Auto-close failed for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Position retrieval
    # ------------------------------------------------------------------

    def _get_open_positions(self) -> list[dict]:
        """Return open positions merged with their stop_loss/take_profit levels."""
        portfolio = self._paper_trader.get_portfolio()
        if not portfolio:
            return []

        results: list[dict] = []
        with self._connect() as conn:
            for pos in portfolio:
                ticker = pos["ticker"]
                row    = conn.execute(
                    """
                    SELECT stop_loss, take_profit, price AS entry_price
                    FROM   trade_history
                    WHERE  ticker = ? AND action = 'BUY'
                    ORDER  BY id DESC LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()
                results.append({
                    "ticker":      ticker,
                    "shares":      pos["shares"],
                    "avg_price":   pos["avg_price"],
                    "stop_loss":   row["stop_loss"]   if row else None,
                    "take_profit": row["take_profit"] if row else None,
                })
        return results

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Quote fetching
    # ------------------------------------------------------------------

    def _fetch_quote(self, ticker: str) -> dict:
        """
        Fetch current price and volume for *ticker* via yfinance.

        Returns a dict with keys:
            price        (float | None)
            prev_close   (float | None)
            today_volume (float)   current day's volume
            avg_volume   (float)   20-day average daily volume

        Cached for CACHE_TTL seconds.  Falls back to last close on failure.
        """
        now    = time.time()
        cached = _QUOTE_CACHE.get(ticker)
        if cached and (now - cached[0]) < _CACHE_TTL:
            return cached[1]

        quote: dict = {}
        try:
            obj  = yf.Ticker(ticker)
            fast = obj.fast_info

            price      = _safe_float(getattr(fast, "last_price",      None))
            prev_close = _safe_float(getattr(fast, "previous_close",  None))

            # Intraday fallback: use previous_close if last_price unavailable
            if price is None:
                price = prev_close

            # Volume: download last 25 trading days
            hist = obj.history(period="25d", interval="1d", auto_adjust=True)
            if not hist.empty:
                today_vol = float(hist["Volume"].iloc[-1])
                avg_vol   = float(hist["Volume"].iloc[:-1].mean()) if len(hist) > 1 else today_vol
            else:
                today_vol = 0.0
                avg_vol   = 0.0

            quote = {
                "price":        price,
                "prev_close":   prev_close,
                "today_volume": today_vol,
                "avg_volume":   avg_vol,
            }
        except Exception as exc:
            log.warning("Quote fetch failed for %s: %s", ticker, exc)

        _QUOTE_CACHE[ticker] = (now, quote)
        return quote

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def _is_market_hours(self) -> bool:
        """Return True if US NYSE or EU XETRA is currently open."""
        now_et  = datetime.now(_TZ_ET)
        now_cet = datetime.now(_TZ_CET)
        weekday = now_et.weekday()   # 0=Mon … 4=Fri; 5=Sat, 6=Sun

        if weekday >= 5:             # Weekend — both markets closed
            return False

        # US NYSE: 09:30–16:00 ET
        us_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        us_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
        if us_open <= now_et <= us_close:
            return True

        # EU XETRA: 09:00–17:30 CET
        eu_open  = now_cet.replace(hour=9,  minute=0,  second=0, microsecond=0)
        eu_close = now_cet.replace(hour=17, minute=30, second=0, microsecond=0)
        if eu_open <= now_cet <= eu_close:
            return True

        return False

    def _market_status(self) -> str:
        """Return human-readable market status string."""
        now_et  = datetime.now(_TZ_ET)
        now_cet = datetime.now(_TZ_CET)
        weekday = now_et.weekday()

        if weekday >= 5:
            return "weekend"

        us_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        us_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
        eu_open  = now_cet.replace(hour=9,  minute=0,  second=0, microsecond=0)
        eu_close = now_cet.replace(hour=17, minute=30, second=0, microsecond=0)

        us = "US open" if us_open <= now_et <= us_close else "US closed"
        eu = "EU open" if eu_open <= now_cet <= eu_close else "EU closed"
        return f"{us}, {eu}"

    def _seconds_until_open(self) -> float:
        """Return seconds until the next market session opens."""
        now_et  = datetime.now(_TZ_ET)
        weekday = now_et.weekday()

        # Days until next weekday (0=Mon)
        if weekday >= 5:
            days_ahead = (7 - weekday)   # 5→2, 6→1
        else:
            days_ahead = 0

        import datetime as _dt
        next_open_local = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        if days_ahead == 0 and now_et >= next_open_local:
            # Past open today → tomorrow
            next_open_local = next_open_local + _dt.timedelta(days=1)
            if next_open_local.weekday() >= 5:
                next_open_local = next_open_local + _dt.timedelta(days=2)
        elif days_ahead > 0:
            next_open_local = next_open_local + _dt.timedelta(days=days_ahead)

        diff = (next_open_local - now_et).total_seconds()
        return max(diff, 60.0)


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> "float | None":
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    import yaml
    watchlist_path = PROJECT_ROOT / "config" / "watchlist.yaml"
    with open(watchlist_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("monitoring", {})
    cfg["monitoring"].setdefault("enabled",                True)
    cfg["monitoring"].setdefault("check_interval",         60)
    cfg["monitoring"].setdefault("market_hours_only",      True)
    cfg["monitoring"].setdefault("auto_close_on_stop",     False)
    cfg["monitoring"].setdefault("volume_spike_threshold", 3.0)
    cfg["monitoring"].setdefault("price_move_threshold",   5.0)
    cfg["monitoring"].setdefault("max_alerts_per_day",     5)
    cfg["monitoring"].setdefault("alert_cooldown",         300)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _print_check_header(monitor: PriceMonitor, mode: str) -> None:
    now_str = datetime.now(_TZ_ET).strftime("%Y-%m-%d %H:%M:%S ET")
    W = 66
    print(f"\n{'═' * W}")
    print(f"  Price Monitor — {mode}  ({now_str})")
    print(f"  Market status: {monitor._market_status()}")
    print(f"{'═' * W}")


def _print_alert(alert: Alert) -> None:
    print(f"  🚨 {alert.message}")


def _print_position_result(ticker: str, quote: dict, alerts: list[Alert]) -> None:
    price   = quote.get("price")
    prev    = quote.get("prev_close")
    avg_vol = quote.get("avg_volume", 0.0) or 0.0
    day_vol = quote.get("today_volume", 0.0) or 0.0

    if price and prev:
        chg = (price - prev) / prev * 100
        chg_str = f"{chg:+.1f}%"
    else:
        chg_str = "n/a"

    vol_ratio = day_vol / avg_vol if avg_vol > 0 else 0.0

    alert_count = len(alerts)
    status = f"{alert_count} alert(s)" if alert_count > 0 else "No alerts"
    print(
        f"  {ticker:<6}  ${price:.2f}" if price else f"  {ticker:<6}  n/a   ",
        end="",
    )
    print(
        f"  {chg_str:>7}  vol {vol_ratio:.1f}x avg"
        f"  → {status}"
    )


def main() -> None:
    import argparse
    from notifications.telegram_bot import TelegramNotifier

    parser = argparse.ArgumentParser(
        description="Price Monitor — stop-loss and price alert tracker",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, check every check_interval seconds during market hours.",
    )
    mode.add_argument(
        "--check-now",
        action="store_true",
        help="Single check pass, send Telegram alerts, then exit.",
    )
    mode.add_argument(
        "--backfill",
        action="store_true",
        help="Single check pass, log to DB, suppress Telegram, then exit.",
    )
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Automatically close positions when stop-loss is triggered.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Enable Telegram notifications (requires config/env vars).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    _build_logger(args.verbose)
    cfg      = _load_config()
    notifier = TelegramNotifier.from_config(cfg) if args.notify else None

    monitor = PriceMonitor(
        cfg        = cfg,
        notifier   = notifier,
        auto_close = args.auto_close,
    )

    # ── Daemon ────────────────────────────────────────────────────────────────
    if args.daemon:
        print(f"Starting PriceMonitor daemon (interval={monitor._interval}s)")
        monitor.run_daemon()
        return

    # ── Check-now / backfill ──────────────────────────────────────────────────
    mode_label = "Check Now" if args.check_now else "Backfill"
    _print_check_header(monitor, mode_label)

    positions = monitor._get_open_positions()
    if not positions:
        print("  No open positions to monitor.\n")
        return

    print(f"  Checking {len(positions)} open position(s)...\n")

    all_alerts: list[Alert] = []
    for pos in positions:
        ticker = pos["ticker"]
        quote  = monitor._fetch_quote(ticker)
        if not quote or not quote.get("price"):
            print(f"  {ticker:<6}  — quote unavailable")
            continue

        alerts = monitor._check_position(pos, quote)
        fired: list[Alert] = []
        for alert in alerts:
            if monitor._can_alert(alert.ticker, alert.alert_type):
                notify = args.check_now   # suppress during backfill
                monitor._fire_alert(alert, notify=notify)
                fired.append(alert)
                all_alerts.append(alert)

        _print_position_result(ticker, quote, fired)
        for alert in fired:
            _print_alert(alert)

    W = 66
    print(f"\n{'═' * W}")
    print(
        f"  {mode_label} complete: {len(positions)} position(s) checked, "
        f"{len(all_alerts)} alert(s) fired."
    )
    if args.auto_close:
        sl_alerts = [a for a in all_alerts if a.alert_type == "stop_loss"]
        if sl_alerts:
            print(f"  Auto-close: {len(sl_alerts)} position(s) closed.")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
