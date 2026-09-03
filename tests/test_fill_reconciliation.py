"""Fill reconciliation — no execution at the broker without a DB row.

Background (2026-08-25 diagnosis, fixed 2026-09-03): BUY market orders
that part-filled inside ORDER_FILL_TIMEOUT were cancelled for the
remainder and the filled shares were never written to trade_history /
portfolio_positions (UFPI/TXRH/CACI — 8 orphan SELLs, shares running
without stops).  The real IBKR sequence for TXRH 2026-08-10:

    13:45:32  placeOrder  BUY 132            status=PendingSubmit
    13:45:34  orderStatus Submitted filled=100 remaining=32 avg=207.35
    13:46:03  cancelOrder                    status=PendingCancel filled=100
    13:46:03  orderStatus Cancelled filled=100 remaining=32

These tests replay that shape against the trader with a scripted
Trade object and a real sqlite Database.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database  # noqa: E402
from execution import ibkr_trader as mod  # noqa: E402
from execution.ibkr_trader import IBKRTrader  # noqa: E402


# ── Scripted broker objects ──────────────────────────────────────────────

class _Status:
    def __init__(self, status="Submitted", filled=0.0, avg=0.0):
        self.status = status
        self.filled = filled
        self.avgFillPrice = avg
        self.lastFillPrice = avg
        self.remaining = 0.0


class _Event:
    """Minimal eventkit.Event stand-in (connect / disconnect / emit)."""

    def __init__(self):
        self._listeners = []

    def connect(self, fn):
        self._listeners.append(fn)

    def disconnect(self, fn):
        self._listeners = [f for f in self._listeners if f is not fn]

    def emit(self, *args):
        for fn in list(self._listeners):
            fn(*args)

    def __len__(self):
        return len(self._listeners)


class _Trade:
    """A Trade whose orderStatus follows a script, one step per poll.

    ``script`` is a list of (status, filled, avg).  The last entry is
    repeated once the script is exhausted.  ``on_cancel`` is applied
    when the trader calls cancelOrder (replaces the script).
    """

    def __init__(self, script, order_id=766, on_cancel=None):
        self._script = list(script)
        self._i = 0
        self.order = MagicMock()
        self.order.orderId = order_id
        self.fills = []
        self.log = []
        self.statusEvent = _Event()
        self._on_cancel = on_cancel
        self._cancel_script = None
        self._advance()

    def _advance(self):
        src = self._cancel_script if self._cancel_script is not None else self._script
        idx = min(self._i, len(src) - 1)
        st, filled, avg = src[idx]
        self.orderStatus = _Status(st, filled, avg)
        self._i += 1

    # the trader reads .orderStatus on every poll — make each read step
    @property
    def orderStatus(self):
        return self._status

    @orderStatus.setter
    def orderStatus(self, v):
        self._status = v

    def poll(self):
        self._advance()

    def cancelled(self):
        if self._on_cancel:
            self._cancel_script = list(self._on_cancel)
            self._i = 0
            self._advance()


class _IB(MagicMock):
    """MagicMock IB whose sleep() advances the scripted trade."""


def _make_trader(tmp_path, trade):
    db = Database(str(tmp_path / "t.db"))
    ib = _IB()
    ib.connect.return_value = None
    ib.isConnected.return_value = True
    ib.placeOrder.return_value = trade
    ib.sleep.side_effect = lambda *_a, **_k: trade.poll()
    ib.cancelOrder.side_effect = lambda *_a, **_k: trade.cancelled()
    trader = IBKRTrader(db=db, ib=ib)
    trader._new_ib_client = MagicMock(return_value=ib)
    trader._Stock = MagicMock()
    trader._MarketOrder = MagicMock()
    return trader, ib, db


def _rows(db):
    with sqlite3.connect(db.db_path) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM trade_history ORDER BY id"
        )]


_FAST = dict(ORDER_FILL_TIMEOUT=0.03, ORDER_POLL_INTERVAL=0.001,
             CANCEL_SETTLE_WAIT=0.03, STOP_EXTENDED_TIMEOUT=0.01,
             STOP_MAX_WAIT=0.05, STOP_SLOW_POLL_INTERVAL=0.001)


def _fast():
    return patch.multiple(mod, **_FAST)


# ── Schema ───────────────────────────────────────────────────────────────

class TestSchema:

    def test_reconciliation_columns_present_and_idempotent(self, tmp_path):
        p = str(tmp_path / "s.db")
        Database(p)
        Database(p)  # second init must not raise on duplicate columns
        with sqlite3.connect(p) as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(trade_history)")}
        assert {"fill_status", "requested_shares", "broker_order_id"} <= cols

    def test_log_trade_history_writes_new_columns(self, tmp_path):
        db = Database(str(tmp_path / "s.db"))
        tid = db.log_trade_history(
            "TXRH", "BUY", 100, 207.35, fill_status="partial",
            requested_shares=132, broker_order_id=766,
        )
        row = _rows(db)[0]
        assert row["id"] == tid
        assert (row["fill_status"], row["requested_shares"], row["broker_order_id"]) \
            == ("partial", 132, 766)

    def test_legacy_call_leaves_new_columns_null(self, tmp_path):
        db = Database(str(tmp_path / "s.db"))
        db.log_trade_history("AAPL", "BUY", 5, 100.0)
        row = _rows(db)[0]
        assert row["fill_status"] is None
        assert row["requested_shares"] is None


# ── BUY timeout with partial fill (the TXRH case) ────────────────────────

class TestBuyTimeoutPartialFill:

    def test_partial_recorded_on_timeout_cancel(self, tmp_path, caplog):
        trade = _Trade(
            script=[("Submitted", 0, 0), ("Submitted", 100, 207.35)],
            on_cancel=[("PendingCancel", 100, 207.35), ("Cancelled", 100, 207.35)],
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        caplog.set_level(logging.WARNING, logger="execution.ibkr_trader")
        with _fast():
            res = trader.track_trade(
                "TXRH", "BUY", 132, 207.10, stop_loss=200.0, take_profit=220.0,
                strategy="Combined", intended_price=207.10,
            )

        ib.cancelOrder.assert_called_once()
        # Result contract: a partial is an execution, not a skip
        assert res["skipped"] is not True
        assert res["partial"] is True
        assert res["outcome"] == "timeout"
        assert res["shares"] == 100 and res["filled_shares"] == 100
        assert res["unfilled_shares"] == 32
        assert res["price"] == 207.35
        assert res["trade_id"] is not None
        assert "partial fill 100/132" in res["skip_reason"]

        # DB: exactly the filled portion, flagged, with the broker id
        rows = _rows(db)
        assert len(rows) == 1
        r = rows[0]
        assert (r["action"], r["shares"], r["price"]) == ("BUY", 100, 207.35)
        assert r["fill_status"] == "partial"
        assert r["requested_shares"] == 132
        assert r["broker_order_id"] == 766
        assert (r["stop_loss"], r["take_profit"], r["strategy"]) == (200.0, 220.0, "Combined")
        assert (r["intended_price"], r["executed_price"]) == (207.10, 207.35)

        pos = db.get_portfolio_position("TXRH")
        assert pos["shares"] == 100 and pos["avg_price"] == 207.35

        # The order is terminal (Cancelled) → no late-fill watcher needed
        assert len(trade.statusEvent) == 0
        assert "PARTIAL FILL" in caplog.text

    def test_fill_that_beats_the_cancel_is_a_full_fill(self, tmp_path):
        trade = _Trade(
            script=[("Submitted", 0, 0), ("Submitted", 100, 207.35)],
            on_cancel=[("PendingCancel", 100, 207.35), ("Filled", 132, 207.40)],
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        with _fast():
            res = trader.track_trade("TXRH", "BUY", 132, 207.10)
        assert res.get("partial") is not True
        assert res["shares"] == 132 and res["price"] == 207.40
        r = _rows(db)[0]
        assert (r["shares"], r["fill_status"], r["requested_shares"]) == (132, "filled", 132)

    def test_timeout_with_nothing_filled_stays_skipped(self, tmp_path):
        trade = _Trade(
            script=[("Submitted", 0, 0)],
            on_cancel=[("Cancelled", 0, 0)],
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        with _fast():
            res = trader.track_trade("TXRH", "BUY", 132, 207.10)
        assert res["skipped"] is True and res["outcome"] == "timeout"
        assert res["filled_shares"] == 0
        assert _rows(db) == []
        assert db.get_portfolio_position("TXRH") is None

    def test_broker_cancel_after_partial_records_partial(self, tmp_path):
        """IBKR-side cancel (not ours) after a part-fill — same exposure."""
        trade = _Trade(
            script=[("Submitted", 0, 0), ("Submitted", 40, 50.0), ("Cancelled", 40, 50.0)],
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        with _fast():
            res = trader.track_trade("CACI", "BUY", 82, 49.9)
        ib.cancelOrder.assert_not_called()
        assert res["partial"] is True and res["outcome"] == "cancelled"
        r = _rows(db)[0]
        assert (r["shares"], r["fill_status"], r["requested_shares"]) == (40, "partial", 82)

    def test_cancel_still_pending_arms_late_fill_watch(self, tmp_path):
        """Cancel does not settle within CANCEL_SETTLE_WAIT → the filled
        portion is recorded now and a watcher records whatever lands
        later (here: 32 more shares, then Cancelled)."""
        trade = _Trade(
            script=[("Submitted", 0, 0), ("Submitted", 100, 207.35)],
            on_cancel=[("PendingCancel", 100, 207.35)],
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        with _fast():
            res = trader.track_trade("TXRH", "BUY", 132, 207.10, stop_loss=200.0)
        assert res["partial"] is True
        assert len(trade.statusEvent) == 1

        # 32 more shares fill at 207.60 before the cancel takes effect:
        # cumulative avg = (100*207.35 + 32*207.60)/132
        cum_avg = (100 * 207.35 + 32 * 207.60) / 132
        trade.orderStatus = _Status("Cancelled", 132, cum_avg)
        trade.statusEvent.emit(trade)

        rows = _rows(db)
        assert [(r["shares"], r["fill_status"]) for r in rows] == [
            (100, "partial"), (32, "late"),
        ]
        assert rows[1]["price"] == pytest.approx(207.60, abs=1e-3)
        assert rows[1]["stop_loss"] == 200.0  # SL/TP carried onto the late row
        assert db.get_portfolio_position("TXRH")["shares"] == 132
        assert len(trade.statusEvent) == 0  # detached on terminal status


# ── SELL stuck (extended wait) ───────────────────────────────────────────

class TestSellStuckLateFill:

    def _position(self, db, ticker, shares, avg):
        db.set_portfolio_position(ticker=ticker, shares=shares, avg_price=avg,
                                  current_value=shares * avg)

    def test_stuck_sell_late_full_fill_is_recorded(self, tmp_path, caplog):
        trade = _Trade(script=[("PreSubmitted", 0, 0)], order_id=461)
        trader, ib, db = _make_trader(tmp_path, trade)
        self._position(db, "VRT", 10, 300.0)
        caplog.set_level(logging.WARNING, logger="execution.ibkr_trader")
        with _fast():
            res = trader.track_trade("VRT", "SELL", 10, 360.0, strategy="Combined")
        assert res["skipped"] is True and res["outcome"] == "stuck"
        ib.cancelOrder.assert_not_called()
        assert _rows(db) == []
        assert "LATE-FILL WATCH armed" in caplog.text

        # ...the SELL fills two minutes later
        trade.orderStatus = _Status("Filled", 10, 358.30)
        trade.statusEvent.emit(trade)

        rows = _rows(db)
        assert len(rows) == 1
        r = rows[0]
        assert (r["action"], r["shares"], r["price"], r["fill_status"]) == ("SELL", 10, 358.30, "late")
        assert r["pnl"] == pytest.approx((358.30 - 300.0) * 10, abs=1e-6)
        assert r["strategy"] == "Combined"
        assert r["broker_order_id"] == 461
        assert db.get_portfolio_position("VRT") is None  # fully closed
        assert "LATE-FILL WATCH closed" in caplog.text

    def test_stuck_sell_partial_then_late_remainder(self, tmp_path):
        trade = _Trade(script=[("PreSubmitted", 0, 0), ("Submitted", 6, 359.0)])
        trader, ib, db = _make_trader(tmp_path, trade)
        self._position(db, "VRT", 10, 300.0)
        with _fast():
            res = trader.track_trade("VRT", "SELL", 10, 360.0)
        assert res["partial"] is True and res["outcome"] == "stuck"
        assert res["shares"] == 6 and res["unfilled_shares"] == 4
        assert db.get_portfolio_position("VRT")["shares"] == 4

        cum_avg = (6 * 359.0 + 4 * 357.0) / 10
        trade.orderStatus = _Status("Filled", 10, cum_avg)
        trade.statusEvent.emit(trade)

        rows = _rows(db)
        assert [(r["shares"], r["fill_status"]) for r in rows] == [(6, "partial"), (4, "late")]
        assert rows[1]["price"] == pytest.approx(357.0, abs=1e-3)
        assert db.get_portfolio_position("VRT") is None

    def test_late_watch_ignores_duplicate_events(self, tmp_path):
        trade = _Trade(script=[("PreSubmitted", 0, 0)])
        trader, ib, db = _make_trader(tmp_path, trade)
        self._position(db, "VRT", 10, 300.0)
        with _fast():
            trader.track_trade("VRT", "SELL", 10, 360.0)
        trade.orderStatus = _Status("Submitted", 10, 358.0)  # filled qty, not yet terminal
        trade.statusEvent.emit(trade)
        trade.statusEvent.emit(trade)  # duplicate status event
        trade.orderStatus = _Status("Filled", 10, 358.0)
        trade.statusEvent.emit(trade)
        assert len(_rows(db)) == 1

    def test_no_status_event_logs_loudly(self, tmp_path, caplog):
        trade = _Trade(script=[("PreSubmitted", 0, 0)])
        trade.statusEvent = None
        trader, ib, db = _make_trader(tmp_path, trade)
        caplog.set_level(logging.ERROR, logger="execution.ibkr_trader")
        with _fast():
            res = trader.track_trade("VRT", "SELL", 10, 360.0)
        assert res["outcome"] == "stuck"
        assert "LATE-FILL WATCH unavailable" in caplog.text


# ── Other order entry points share the path ──────────────────────────────

class TestOtherEntryPoints:

    def test_close_position_records_fill(self, tmp_path):
        trade = _Trade(script=[("Filled", 10, 155.0)])
        trader, ib, db = _make_trader(tmp_path, trade)
        db.set_portfolio_position(ticker="AAPL", shares=10, avg_price=150.0,
                                  current_value=1500.0)
        pos = MagicMock()
        pos.contract.symbol = "AAPL"
        pos.position = 10
        ib.positions.return_value = [pos]
        with _fast():
            assert trader.close_position("AAPL") is True
        r = _rows(db)[0]
        assert (r["action"], r["shares"], r["price"], r["fill_status"]) == ("SELL", 10, 155.0, "filled")
        assert db.get_portfolio_position("AAPL") is None

    def test_close_position_partial_returns_false_but_records(self, tmp_path):
        trade = _Trade(script=[("PreSubmitted", 0, 0), ("Submitted", 4, 155.0)])
        trader, ib, db = _make_trader(tmp_path, trade)
        pos = MagicMock()
        pos.contract.symbol = "AAPL"
        pos.position = 10
        ib.positions.return_value = [pos]
        with _fast():
            assert trader.close_position("AAPL") is False
        r = _rows(db)[0]
        assert (r["shares"], r["fill_status"], r["requested_shares"]) == (4, "partial", 10)

    def test_place_order_partial_status(self, tmp_path):
        trade = _Trade(
            script=[("Submitted", 0, 0), ("Submitted", 3, 20.0)],
            on_cancel=[("Cancelled", 3, 20.0)],
            order_id=99,
        )
        trader, ib, db = _make_trader(tmp_path, trade)
        with _fast():
            res = trader.place_order("PFE", 10, "BUY")
        assert res["status"] == "PartiallyFilled"
        assert res["order_id"] == 99 and res["filled_shares"] == 3
        assert _rows(db)[0]["shares"] == 3


# ── PositionManager consumes the partial contract ────────────────────────

class TestPositionManagerPartialExit:

    def _pm(self):
        from monitoring.position_manager import PositionManager
        trader = MagicMock()
        trader.get_portfolio.return_value = []
        notifier = MagicMock()
        pm = PositionManager(trader=trader, notifier=notifier)
        pm._log_event = MagicMock()
        return pm, notifier

    def _partial(self, outcome):
        return {
            "trade_id": 7, "ticker": "VRT", "action": "SELL", "shares": 6,
            "price": 359.0, "pnl": 354.0, "total_value": 2154.0,
            "partial": True, "outcome": outcome, "filled_shares": 6,
            "requested_shares": 10, "unfilled_shares": 4,
            "skip_reason": "IBKR order stuck: PreSubmitted — partial fill 6/10 recorded",
        }

    def test_partial_stuck_sets_cooldown_and_keeps_trail(self):
        pm, notifier = self._pm()
        pm._trailing_stops["VRT"] = 355.0
        out = pm._handle_close_result(
            self._partial("stuck"), "VRT", 358.0, 19.3,
            trigger="stop_loss", level=355.0, alert_emoji="x", alert_label="Stop-loss",
        )
        assert out["action"] == "stop_loss_partial"
        assert out["filled_shares"] == 6 and out["requested_shares"] == 10
        assert "VRT" in pm._stuck_orders           # remainder in flight → no retry
        assert pm._trailing_stops["VRT"] == 355.0   # position still open
        msg = notifier.send_price_alert.call_args[0][0]
        assert "PARTIAL EXIT" in msg and "6/10" in msg and "in flight" in msg
        assert "Stop-loss hit" not in msg

    def test_partial_cancelled_allows_retry(self):
        pm, notifier = self._pm()
        out = pm._handle_close_result(
            self._partial("cancelled"), "VRT", 358.0, 19.3,
            trigger="take_profit", level=360.0, alert_emoji="x", alert_label="Take-profit",
        )
        assert out["action"] == "take_profit_partial"
        assert "VRT" not in pm._stuck_orders         # remainder cancelled → retry next cycle
        msg = notifier.send_price_alert.call_args[0][0]
        assert "remainder cancelled" in msg

    def test_full_fill_unchanged(self):
        pm, notifier = self._pm()
        full = {"trade_id": 7, "ticker": "VRT", "action": "SELL", "shares": 10,
                "price": 358.0, "pnl": 0.0, "total_value": 3580.0}
        out = pm._handle_close_result(
            full, "VRT", 358.0, 19.3,
            trigger="stop_loss", level=355.0, alert_emoji="x", alert_label="Stop-loss",
        )
        assert out["action"] == "stop_loss"
        assert "hit" in notifier.send_price_alert.call_args[0][0]
