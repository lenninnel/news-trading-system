"""analytics/portfolio_view.py + the Database mark/snapshot methods
(2026-09-23): NAV = cash + positions at live marks, daily P&L vs the
previous close, broker sync keeps the mark.

Run: python3 -m pytest tests/test_portfolio_view.py -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analytics.portfolio_view import build_portfolio_view  # noqa: E402
from storage.database import Database  # noqa: E402

_NOW = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "v.db"))


def _view(db: Database, now=_NOW) -> dict:
    conn = sqlite3.connect(db.db_path)
    try:
        return build_portfolio_view(conn, now=now)
    finally:
        conn.close()


# ── Database: marks survive the broker sync ─────────────────────────────

def test_sync_keeps_existing_mark_and_recomputes_value(db):
    db.set_portfolio_position(ticker="AAPL", shares=81, avg_price=333.45,
                              current_value=27009.45, mark_price=333.44, mark_source="fill")
    db.mark_portfolio_position("AAPL", 81, 333.45, 342.98, "yfinance_1m")
    row = db.sync_portfolio_position("AAPL", 81, 333.45)      # the IBKR position list
    assert row["current_value"] == round(81 * 342.98, 2)
    stored = db.get_portfolio_position("AAPL")
    assert stored["mark_price"] == 342.98
    assert stored["mark_source"] == "yfinance_1m"
    assert stored["current_value"] == round(81 * 342.98, 2)


def test_sync_scales_mark_when_shares_change(db):
    db.mark_portfolio_position("AAPL", 81, 333.45, 340.0, "yfinance_1m")
    row = db.sync_portfolio_position("AAPL", 40, 333.45)       # partial exit at the broker
    assert row["current_value"] == 40 * 340.0
    assert db.get_portfolio_position("AAPL")["mark_price"] == 340.0


def test_sync_of_never_marked_row_uses_entry_value(db):
    row = db.sync_portfolio_position("MSFT", 10, 400.0)
    assert row["current_value"] == 4000.0
    assert db.get_portfolio_position("MSFT")["mark_price"] is None


def test_set_without_mark_leaves_mark_columns_alone(db):
    db.mark_portfolio_position("AAPL", 81, 333.45, 340.0, "yfinance_1m")
    db.set_portfolio_position(ticker="AAPL", shares=81, avg_price=333.45, current_value=1.0)
    stored = db.get_portfolio_position("AAPL")
    assert stored["mark_price"] == 340.0 and stored["current_value"] == 1.0


def test_mark_ignores_bad_price(db):
    db.mark_portfolio_position("AAPL", 81, 333.45, 0.0, "yfinance_1m")
    assert db.get_portfolio_position("AAPL") is None


def test_snapshot_rejects_zero_netliq(db):
    assert db.record_account_snapshot(net_liquidation=0.0, kind="pm") is None
    assert db.get_latest_account_snapshot() is None
    assert db.record_account_snapshot(net_liquidation=100.0, total_cash=90.0, kind="pm") == 1
    assert db.get_latest_account_snapshot()["total_cash"] == 90.0


# ── View ────────────────────────────────────────────────────────────────

def test_empty_db_is_honest(db):
    v = _view(db)
    assert v["positions"] == []
    assert v["value"] == 0.0 and v["cash"] == 0.0
    assert v["daily_pnl"] is None
    assert v["cash_source"] == "unavailable"
    assert any("daily P&L unavailable" in n for n in v["notes"])


def test_nav_is_cash_plus_marked_positions_and_reconciles_with_broker(db):
    db.set_portfolio_position(ticker="AAPL", shares=81, avg_price=333.45,
                              current_value=27009.45, mark_price=333.44, mark_source="fill")
    db.mark_portfolio_position("AAPL", 81, 333.45, 342.98, "yfinance_1m")
    db.sync_portfolio_position("AAPL", 81, 333.45)   # a session-start sync must not regress it
    db.record_account_snapshot(net_liquidation=271095.71, total_cash=243314.33,
                               prev_day_equity=270000.0, kind="session",
                               ts="2026-09-23T13:15:00+00:00")
    v = _view(db)
    assert v["positions_value"] == 27781.38
    assert v["cash"] == 243314.33
    assert v["value"] == v["nav"] == 271095.71
    assert v["cash_source"] == "ibkr:TotalCashValue"
    assert v["cash_as_of"].startswith("2026-09-23T13:15")
    assert v["daily_pnl"] == 1095.71 and v["daily_pnl_pct"] == 0.41
    assert "PreviousDayEquityWithLoanValue" in v["daily_pnl_basis"]
    assert v["broker_nav"] == 271095.71 and v["nav_minus_broker"] == 0.0
    p = v["positions"][0]
    assert (p["entry"], p["current"], p["pnl_pct"], p["mark_source"]) == (333.45, 342.98, 2.9, "yfinance_1m")
    assert p["unrealized_pnl"] == 771.93
    assert v["notes"] == []


def test_previous_close_falls_back_to_last_eod_snapshot(db):
    db.record_account_snapshot(net_liquidation=270000.0, total_cash=270000.0, kind="eod",
                               ts="2026-09-22T22:45:00+00:00")
    db.record_account_snapshot(net_liquidation=271000.0, total_cash=271000.0, kind="pm",
                               ts="2026-09-23T14:00:00+00:00")   # no prev_day_equity tag
    v = _view(db)
    assert v["prev_close_nav"] == 270000.0
    assert v["daily_pnl"] == 1000.0
    assert v["daily_pnl_basis"].startswith("NAV − previous close (eod snapshot 2026-09-22T22:45")


def test_today_eod_snapshot_is_not_used_as_previous_close(db):
    db.record_account_snapshot(net_liquidation=270000.0, total_cash=270000.0, kind="eod",
                               ts="2026-09-23T22:45:00+00:00")
    v = _view(db, now=datetime(2026, 9, 23, 23, 0, tzinfo=timezone.utc))
    assert v["daily_pnl"] is None


def test_realized_today_is_reported_separately(db):
    db.log_trade_history(ticker="AAPL", action="SELL", shares=81, price=337.37,
                         stop_loss=None, take_profit=None, pnl=317.33)
    db.record_account_snapshot(net_liquidation=271413.04, total_cash=271413.04,
                               prev_day_equity=271095.71, kind="pm")
    v = _view(db, now=datetime.now(timezone.utc))
    assert v["realized_today"] == 317.33
    assert v["daily_pnl"] == 317.33          # NAV moved by exactly the realised amount
    assert v["positions_value"] == 0.0


def test_unmarked_position_falls_back_and_says_so(db):
    db.sync_portfolio_position("MSFT", 10, 400.0)   # never marked
    v = _view(db)
    p = v["positions"][0]
    assert p["current"] == 400.0 and p["mark_source"].startswith("unmarked:")
    assert any("no live mark yet for MSFT" in n for n in v["notes"])


def test_pre_snapshot_cash_is_the_old_estimate_labelled(db):
    db.sync_portfolio_position("MSFT", 10, 400.0)
    db.log_risk_calculation(ticker="MSFT", signal="BUY", confidence=0.5, current_price=400.0,
                            account_balance=10_000.0, position_size_usd=4000.0, shares=10,
                            stop_loss=390.0, take_profit=420.0, risk_amount=100.0,
                            kelly_fraction=0.1, stop_pct=2.5)
    v = _view(db)
    assert v["cash"] == 6000.0
    assert v["cash_source"].startswith("estimate:")
    assert v["value"] == 10_000.0
    assert v["daily_pnl"] is None


def test_legacy_db_without_mark_columns_or_snapshots(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE portfolio_positions(ticker TEXT, shares INTEGER, avg_price REAL,
            current_value REAL, updated_at TEXT);
        CREATE TABLE trade_history(ticker TEXT, action TEXT, shares INTEGER, price REAL,
            pnl REAL, created_at TEXT);
        CREATE TABLE risk_calculations(id INTEGER PRIMARY KEY, account_balance REAL,
            created_at TEXT);
        INSERT INTO portfolio_positions VALUES ('AAPL', 10, 180.0, 1850.0, '2026-09-22T19:59:00');
        INSERT INTO risk_calculations(account_balance, created_at) VALUES (10000.0, '2026-09-23T13:15:00');
        """
    )
    conn.commit()
    v = build_portfolio_view(conn, now=_NOW)
    conn.close()
    p = v["positions"][0]
    assert p["current"] == 185.0 and p["mark_source"] == "unmarked:current_value"
    assert v["positions_value"] == 1850.0
    assert v["cash"] == 10000.0 - 1800.0
    assert v["value"] == 1850.0 + 8200.0
    assert v["daily_pnl"] is None
