"""
Tests for trade_history attribution + execution-quality split.

Covers the additive migration that adds four columns to trade_history:
    strategy        — per-trade attribution captured at execution time
    commission      — broker commission for the fill (None on paper)
    intended_price  — price the strategy saw at signal-trigger time
    executed_price  — actual fill price returned by the broker

Behavioural contracts under test:
  * The migration is idempotent — re-running Database() on a fresh DB
    is a no-op the second time (try/except sqlite3.OperationalError).
  * Existing rows keep NULL for the four new columns; legacy queries
    against `price` continue to return their pre-migration values.
  * PaperTrader threads ``strategy`` and ``intended_price`` from the
    caller through to trade_history; commission stays None; executed
    equals the caller's price.
  * IBKRTrader sums ``trade.fills[*].commissionReport.commission``
    across fills, stores ``avgFillPrice`` as executed_price, and uses
    the kwarg ``intended_price`` (falling back to ``price``) as the
    signal-trigger price.
  * Every new-column population is failure-soft: a malformed input
    that raises during capture must NOT prevent the trade row from
    being written.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database
from execution.paper_trader import PaperTrader


# ── helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch) -> Database:
    """Fresh DB at a temp path with the migration applied."""
    db_path = tmp_path / "nts_attr.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    return Database(str(db_path))


def _columns(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as c:
        return [r[1] for r in c.execute("PRAGMA table_info(trade_history)").fetchall()]


# ── 1. Migration idempotency ─────────────────────────────────────────


class TestMigrationIdempotency:

    def test_columns_present_after_first_init(self, tmp_db):
        cols = _columns(tmp_db.db_path)
        for new_col in ("strategy", "commission", "intended_price", "executed_price"):
            assert new_col in cols, f"missing {new_col}"

    def test_reinit_does_not_duplicate_columns(self, tmp_path):
        """Apply migration twice — second pass is a no-op."""
        db_path = str(tmp_path / "twice.db")
        Database(db_path)
        cols_a = _columns(db_path)
        # Second instantiation must not raise and must not add columns
        Database(db_path)
        cols_b = _columns(db_path)
        assert cols_a == cols_b
        # Each column appears exactly once
        for new_col in ("strategy", "commission", "intended_price", "executed_price"):
            assert cols_b.count(new_col) == 1

    def test_existing_rows_unchanged(self, tmp_path):
        """Pre-migration rows survive intact with NULL new-column values."""
        db_path = str(tmp_path / "preexisting.db")
        # Set up a DB with the legacy schema and a row
        with sqlite3.connect(db_path) as c:
            c.execute(
                """
                CREATE TABLE trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL, action TEXT NOT NULL,
                    shares INTEGER NOT NULL, price REAL NOT NULL,
                    stop_loss REAL, take_profit REAL,
                    pnl REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            c.execute(
                "INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) "
                "VALUES ('LEGACY', 'BUY', 7, 123.45, 0.0, '2026-04-01T00:00:00+00:00')"
            )
            c.commit()
        # Trigger migration
        Database(db_path)
        with sqlite3.connect(db_path) as c:
            row = c.execute(
                "SELECT ticker, price, strategy, commission, intended_price, executed_price "
                "FROM trade_history WHERE ticker='LEGACY'"
            ).fetchone()
        assert row == ("LEGACY", 123.45, None, None, None, None)


# ── 2. Direct Database.log_trade_history ────────────────────────────


class TestLogTradeHistoryNewFields:

    def test_all_four_fields_persisted(self, tmp_db):
        trade_id = tmp_db.log_trade_history(
            ticker="META", action="BUY", shares=3, price=502.10,
            stop_loss=485.0, take_profit=525.0, pnl=0.0,
            strategy="Combined", commission=1.42,
            intended_price=501.85, executed_price=502.10,
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT strategy, commission, intended_price, executed_price "
                "FROM trade_history WHERE id=?", (trade_id,),
            ).fetchone()
        assert row == ("Combined", 1.42, 501.85, 502.10)

    def test_defaults_are_null(self, tmp_db):
        """Calling without the new kwargs leaves them NULL — no silent defaults."""
        trade_id = tmp_db.log_trade_history(
            ticker="AAPL", action="BUY", shares=2, price=263.88,
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT strategy, commission, intended_price, executed_price "
                "FROM trade_history WHERE id=?", (trade_id,),
            ).fetchone()
        assert row == (None, None, None, None)


# ── 3. PaperTrader populates attribution on BUY/SELL ────────────────


class TestPaperTraderAttribution:

    def _trader(self, tmp_db) -> PaperTrader:
        return PaperTrader(db=tmp_db)

    @pytest.mark.parametrize("strat", ["Momentum", "Pullback", "NewsCatalyst", "Combined", "PEAD"])
    def test_buy_carries_strategy_label(self, tmp_db, strat):
        trader = self._trader(tmp_db)
        res = trader.track_trade(
            "META", "BUY", 3, 502.0,
            stop_loss=485.0, take_profit=525.0,
            strategy=strat, intended_price=501.5,
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT strategy, intended_price, executed_price, commission "
                "FROM trade_history WHERE id=?", (res["trade_id"],),
            ).fetchone()
        assert row == (strat, 501.5, 502.0, None)

    def test_intended_price_defaults_to_price_when_omitted(self, tmp_db):
        trader = self._trader(tmp_db)
        res = trader.track_trade(
            "AAPL", "BUY", 2, 263.88,
            stop_loss=258.6, take_profit=274.4, strategy="Pullback",
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT intended_price, executed_price FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()
        assert row == (263.88, 263.88)

    def test_sell_records_strategy(self, tmp_db):
        trader = self._trader(tmp_db)
        trader.track_trade("META", "BUY", 3, 502.0,
                           stop_loss=485.0, take_profit=525.0,
                           strategy="Combined")
        sell = trader.track_trade(
            "META", "SELL", 3, 515.0,
            strategy="Combined", intended_price=515.0,
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT action, strategy, intended_price FROM trade_history WHERE id=?",
                (sell["trade_id"],),
            ).fetchone()
        assert row == ("SELL", "Combined", 515.0)

    def test_legacy_price_column_unchanged(self, tmp_db):
        """trade_history.price must continue to equal the executed fill price
        so historical queries against `price` keep returning the same values
        they did before the migration."""
        trader = self._trader(tmp_db)
        res = trader.track_trade(
            "AAPL", "BUY", 1, 263.88,
            stop_loss=258.0, take_profit=275.0, strategy="Pullback",
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            price, executed = c.execute(
                "SELECT price, executed_price FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()
        assert price == 263.88
        assert price == executed


# ── 4. IBKR fill commission + executed price ────────────────────────


def _build_ibkr_trader(monkeypatch, tmp_db):
    import os
    from execution.ibkr_trader import IBKRTrader
    mock_ib = MagicMock()
    mock_ib.connect.return_value = None
    mock_ib.isConnected.return_value = True
    monkeypatch.setenv("IBKR_PAPER", "true")
    trader = IBKRTrader(db=tmp_db, ib=mock_ib)
    trader._new_ib_client = MagicMock(return_value=mock_ib)
    trader._Stock = MagicMock()
    trader._MarketOrder = MagicMock()
    return trader, mock_ib


def _make_filled_trade(avg_fill: float, commissions: list[float | None]):
    """Build a mock ib_insync Trade with a Filled status and N fills."""
    trade = MagicMock()
    # Force terminal status on the first poll
    trade.orderStatus.status = "Filled"
    trade.orderStatus.avgFillPrice = avg_fill
    trade.log = []
    fills = []
    for c in commissions:
        f = MagicMock()
        if c is None:
            f.commissionReport = None
        else:
            f.commissionReport = MagicMock()
            f.commissionReport.commission = c
        fills.append(f)
    trade.fills = fills
    return trade


class TestIBKRExecutionCapture:

    def test_executed_price_is_avg_fill_price(self, tmp_db, monkeypatch):
        trader, mock_ib = _build_ibkr_trader(monkeypatch, tmp_db)
        mock_ib.placeOrder.return_value = _make_filled_trade(
            avg_fill=502.10, commissions=[0.85],
        )
        res = trader.track_trade(
            "META", "BUY", 3, 501.50,
            stop_loss=485.0, take_profit=525.0,
            strategy="Combined", intended_price=501.50,
        )
        assert res.get("skipped") is not True
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT intended_price, executed_price, commission, strategy "
                "FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()
        assert row == (501.50, 502.10, 0.85, "Combined")

    def test_commission_sums_across_fills(self, tmp_db, monkeypatch):
        trader, mock_ib = _build_ibkr_trader(monkeypatch, tmp_db)
        mock_ib.placeOrder.return_value = _make_filled_trade(
            avg_fill=358.30, commissions=[0.50, 0.35, 0.40],
        )
        res = trader.track_trade(
            "AAPL", "BUY", 30, 357.0,
            stop_loss=345.0, take_profit=375.0, strategy="Combined",
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            commission = c.execute(
                "SELECT commission FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()[0]
        assert commission == pytest.approx(1.25)

    def test_commission_none_when_no_reports(self, tmp_db, monkeypatch):
        """If no Fill carries a CommissionReport, commission stays NULL."""
        trader, mock_ib = _build_ibkr_trader(monkeypatch, tmp_db)
        mock_ib.placeOrder.return_value = _make_filled_trade(
            avg_fill=263.88, commissions=[None, None],
        )
        res = trader.track_trade(
            "AAPL", "BUY", 2, 263.50,
            stop_loss=258.0, take_profit=275.0, strategy="Combined",
        )
        with sqlite3.connect(tmp_db.db_path) as c:
            commission = c.execute(
                "SELECT commission FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()[0]
        assert commission is None


# ── 5. Failure-soft guarantees ──────────────────────────────────────


class TestFailureSoft:

    def test_commission_parse_error_does_not_block_trade(self, tmp_db, monkeypatch):
        """An exception while summing commissions must not prevent the
        trade row from being written."""
        trader, mock_ib = _build_ibkr_trader(monkeypatch, tmp_db)
        # fills is a non-iterable poison value — iterating raises TypeError.
        # ibkr_trader wraps that in try/except so the trade still records.
        trade = MagicMock()
        trade.orderStatus.status = "Filled"
        trade.orderStatus.avgFillPrice = 502.10
        trade.log = []
        # __iter__ raises — `for f in fills` blows up
        bad_fills = MagicMock()
        bad_fills.__iter__ = MagicMock(side_effect=RuntimeError("poisoned"))
        trade.fills = bad_fills
        mock_ib.placeOrder.return_value = trade

        res = trader.track_trade(
            "META", "BUY", 3, 501.5,
            stop_loss=485.0, take_profit=525.0,
            strategy="Combined", intended_price=501.5,
        )
        # The trade still recorded
        assert res.get("skipped") is not True
        assert res.get("trade_id") is not None
        with sqlite3.connect(tmp_db.db_path) as c:
            row = c.execute(
                "SELECT strategy, commission, executed_price FROM trade_history WHERE id=?",
                (res["trade_id"],),
            ).fetchone()
        # Strategy and executed_price still populated; commission is NULL
        assert row[0] == "Combined"
        assert row[1] is None
        assert row[2] == 502.10

    def test_position_manager_strategy_lookup_failure_does_not_block_sell(
        self, tmp_db,
    ):
        """A poisoned position_metadata read returns None from the lookup
        helper and the SELL still records (with strategy=NULL)."""
        from monitoring.position_manager import PositionManager
        # Build a PositionManager with the bare minimum to call the helper
        pm = PositionManager.__new__(PositionManager)
        pm._db = tmp_db
        # Drop the position_metadata table so the SELECT raises
        with sqlite3.connect(tmp_db.db_path) as c:
            c.execute("DROP TABLE IF EXISTS position_metadata")
            c.commit()
        result = pm._resolve_entry_strategy("META")
        assert result is None  # warning logged, no exception escaped
