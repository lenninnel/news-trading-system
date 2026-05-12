"""
Tests for the drawdown halt feature.

Covers:
  • Database peak ratchet (only goes up)
  • Halt trigger at > threshold
  • Halt is one-shot (does not re-fire on subsequent dips)
  • Manual unlock clears the halt
  • reset_drawdown_peak overrides peak and clears halt
  • PortfolioManager.can_add_position refuses BUYs while halted
  • Database.log_portfolio_violation persists rows (critical 2)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ════════════════════════════════════════════════════════════════════════════
# Database-layer tests
# ════════════════════════════════════════════════════════════════════════════

class TestPortfolioPeakRatchet:
    def test_first_update_seeds_peak(self, tmp_db):
        state = tmp_db.update_portfolio_peak(100_000.0, 0.10)
        assert state["peak_value"] == 100_000.0
        assert state["halted"] is False
        assert state["drawdown_pct"] == 0.0
        assert state["newly_halted"] is False

    def test_peak_ratchets_up(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        state = tmp_db.update_portfolio_peak(105_000.0, 0.10)
        assert state["peak_value"] == 105_000.0
        assert state["halted"] is False

    def test_peak_does_not_ratchet_down(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        state = tmp_db.update_portfolio_peak(95_000.0, 0.10)
        # 5% drawdown, below the 10% threshold
        assert state["peak_value"] == 100_000.0
        assert state["halted"] is False
        assert state["drawdown_pct"] == pytest.approx(0.05)

    def test_zero_or_negative_current_is_noop(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        state = tmp_db.update_portfolio_peak(0.0, 0.10)
        assert state["peak_value"] == 100_000.0
        assert state["halted"] is False


class TestDrawdownHaltTrigger:
    def test_halt_triggers_above_threshold(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        # 11% drop > 10% threshold
        state = tmp_db.update_portfolio_peak(89_000.0, 0.10)
        assert state["halted"] is True
        assert state["newly_halted"] is True
        assert state["halted_value"] == 89_000.0
        assert state["halted_drawdown_pct"] == pytest.approx(0.11)
        assert "below peak" in state["halt_reason"]

    def test_halt_does_not_trigger_at_exactly_threshold(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        # exactly 10% — using `>` not `>=`
        state = tmp_db.update_portfolio_peak(90_000.0, 0.10)
        assert state["halted"] is False

    def test_halt_is_one_shot(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        state1 = tmp_db.update_portfolio_peak(85_000.0, 0.10)
        assert state1["newly_halted"] is True
        # Another dip — already halted, must NOT re-fire newly_halted
        state2 = tmp_db.update_portfolio_peak(80_000.0, 0.10)
        assert state2["halted"] is True
        assert state2["newly_halted"] is False
        # Halted snapshot frozen at the first trigger
        assert state2["halted_value"] == 85_000.0


class TestUnlock:
    def test_unlock_clears_halt(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        tmp_db.update_portfolio_peak(85_000.0, 0.10)
        assert tmp_db.get_drawdown_state()["halted"] is True

        cleared = tmp_db.unlock_drawdown_halt("test-user", "smoke check")
        assert cleared is True
        state = tmp_db.get_drawdown_state()
        assert state["halted"] is False
        assert state["unlocked_by"] == "test-user"
        assert state["unlock_reason"] == "smoke check"

    def test_unlock_when_not_halted_is_noop(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        assert tmp_db.unlock_drawdown_halt("test", "x") is False

    def test_peak_preserved_after_unlock(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        tmp_db.update_portfolio_peak(85_000.0, 0.10)
        tmp_db.unlock_drawdown_halt("test", "ok")
        state = tmp_db.get_drawdown_state()
        # Peak does NOT reset on unlock — operator must use --reset-peak
        assert state["peak_value"] == 100_000.0


class TestResetPeak:
    def test_reset_peak_overrides(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        state = tmp_db.reset_drawdown_peak(150_000.0, "ops", "deposit")
        assert state["peak_value"] == 150_000.0
        assert state["halted"] is False
        assert state["unlocked_by"] == "ops"

    def test_reset_peak_clears_halt(self, tmp_db):
        tmp_db.update_portfolio_peak(100_000.0, 0.10)
        tmp_db.update_portfolio_peak(85_000.0, 0.10)
        assert tmp_db.get_drawdown_state()["halted"] is True
        tmp_db.reset_drawdown_peak(110_000.0, "ops", "reset")
        assert tmp_db.get_drawdown_state()["halted"] is False

    def test_reset_peak_rejects_non_positive(self, tmp_db):
        with pytest.raises(ValueError):
            tmp_db.reset_drawdown_peak(0.0, "ops", "x")


# ════════════════════════════════════════════════════════════════════════════
# log_portfolio_violation — critical 2
# ════════════════════════════════════════════════════════════════════════════

class TestLogPortfolioViolation:
    def test_insert_returns_rowid(self, tmp_db):
        rowid = tmp_db.log_portfolio_violation(
            ticker="AAPL",
            violation_type="max_positions",
            reason="cap reached",
            strategy="momentum",
            amount_usd=1234.56,
        )
        assert isinstance(rowid, int) and rowid > 0

    def test_row_is_queryable(self, tmp_db):
        tmp_db.log_portfolio_violation(
            ticker="MSFT",
            violation_type="drawdown_halt",
            reason="dd 11%",
            strategy="pead",
            amount_usd=500.0,
        )
        import sqlite3
        conn = sqlite3.connect(tmp_db.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM portfolio_violations WHERE ticker='MSFT'"
        ).fetchone()
        conn.close()
        assert row["violation_type"] == "drawdown_halt"
        assert row["strategy"] == "pead"
        assert row["amount_usd"] == 500.0
        assert row["reason"] == "dd 11%"


# ════════════════════════════════════════════════════════════════════════════
# PortfolioManager gate
# ════════════════════════════════════════════════════════════════════════════

class TestPortfolioManagerDrawdownGate:
    def _make_pm(self, db_path):
        from execution.portfolio_manager import PortfolioManager
        # _fetch_sector hits yfinance — patch it for this test class
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            return PortfolioManager(account_balance=100_000.0, db_path=db_path)

    def test_can_add_position_blocked_when_halted(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        from storage.database import Database
        db = Database(db_path)
        db.update_portfolio_peak(100_000.0, 0.10)
        db.update_portfolio_peak(85_000.0, 0.10)  # trigger halt

        pm = self._make_pm(db_path)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("AAPL", "momentum", 5_000.0)
        assert allowed is False
        assert "Drawdown halt" in reason

        # And it must be persisted to portfolio_violations
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT violation_type, ticker FROM portfolio_violations "
            "WHERE violation_type='drawdown_halt' AND ticker='AAPL'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_can_add_position_passes_drawdown_gate_when_no_halt(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        from storage.database import Database
        Database(db_path)  # init schema only — no peak row yet

        pm = self._make_pm(db_path)
        # No peak ⇒ no halt ⇒ gate passes to subsequent checks. With an empty
        # portfolio and a small amount, can_add_position should allow.
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("AAPL", "momentum", 5_000.0)
        assert allowed is True, reason
