"""
Kill switch is BUY-only (R amendment 2026-08-26).

--stop-trading blocks new ENTRIES only; exits must always pass —
blocking them converts bounded risk into unbounded risk exactly when
the switch is pulled. --stop-all + SIGTERM remains the halt-everything
path.

Covers (IBKRTrader variants live in tests/test_ibkr_trader.py::TestKillSwitch):
  t1  BUY blocked with flag active (PaperTrader)
  t2  SELL passes with flag active (PaperTrader)
  t3  live-path regression: the actual production call chains —
      PositionManager exit chain and the coordinator PEAD entry chain —
      driven against a REAL PaperTrader with the REAL flag file active
  t4  lowercase "buy" is still blocked (guard sits after .upper())
"""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from emergency_stop import KillSwitch, TradingBlocked
from execution.paper_trader import PaperTrader
from storage.database import Database


# ── helpers ──────────────────────────────────────────────────────────

def _temp_db() -> Database:
    return Database(db_path=tempfile.mktemp(suffix=".db"))


def _kill_switch(active: bool):
    """Patch the flag check to a fixed state (unit tests)."""
    return patch(
        "execution.paper_trader.KillSwitch.is_trading_blocked",
        return_value=active,
    )


# ── t1 / t2 / t4: PaperTrader unit level ─────────────────────────────

class TestPaperTraderKillSwitch:

    def test_buy_blocked(self):
        trader = PaperTrader(db=_temp_db())
        with _kill_switch(True):
            with pytest.raises(TradingBlocked):
                trader.track_trade("AAPL", "BUY", 5, 200.0,
                                   stop_loss=190.0, take_profit=220.0)
        # Fill logic never reached: nothing recorded, no position
        assert trader.get_trade_history() == []
        assert trader.get_portfolio() == []

    def test_lowercase_buy_blocked(self):
        """Ordering trap: guard sits after .upper(), so 'buy' is caught."""
        trader = PaperTrader(db=_temp_db())
        with _kill_switch(True):
            with pytest.raises(TradingBlocked):
                trader.track_trade("AAPL", "buy", 5, 200.0,
                                   stop_loss=190.0, take_profit=220.0)
        assert trader.get_trade_history() == []

    def test_sell_passes_when_blocked(self):
        db = _temp_db()
        trader = PaperTrader(db=db)
        with _kill_switch(False):
            trader.track_trade("AAPL", "BUY", 5, 200.0,
                               stop_loss=190.0, take_profit=220.0)
        with _kill_switch(True):
            result = trader.track_trade("AAPL", "SELL", 5, 210.0)
        assert result["trade_id"] is not None
        assert result["pnl"] == 50.0
        assert trader.get_portfolio() == []  # position closed

    def test_invalid_action_raises_value_error_not_blocked(self):
        """Validation runs before the guard — garbage input stays ValueError."""
        trader = PaperTrader(db=_temp_db())
        with _kill_switch(True):
            with pytest.raises(ValueError, match="must be BUY or SELL"):
                trader.track_trade("AAPL", "SHORT", 5, 200.0)


# ── t3: live-path regression, exit chain ─────────────────────────────

class TestLivePathExitChain:
    """PositionManager._close_position → PaperTrader.track_trade(SELL)
    with the REAL emergency_stop.flag active — the SELL must fill."""

    def test_pm_stop_loss_sell_passes_with_flag_active(self):
        from monitoring.position_manager import PositionManager

        db = _temp_db()
        trader = PaperTrader(db=db)
        # Open the position while the switch is OFF (entries allowed)
        trader.track_trade("AAPL", "BUY", 5, 200.0,
                           stop_loss=190.0, take_profit=220.0)

        pm = PositionManager(trader=trader, notifier=MagicMock(), db=db)

        assert not KillSwitch.is_stopped(), \
            "stray emergency_stop.flag before test"
        KillSwitch.activate("stop_trading", "pytest t3 exit chain")
        try:
            with patch.object(pm, "_fetch_current_price", return_value=185.0):
                results = pm._check_all_positions()
        finally:
            KillSwitch.deactivate()

        # The stop-loss SELL went through the real guard and filled
        assert len(results) == 1
        assert results[0]["action"] == "stop_loss"
        sells = [t for t in trader.get_trade_history(ticker="AAPL")
                 if t["action"] == "SELL"]
        assert len(sells) == 1
        assert db.get_portfolio_position("AAPL") is None  # closed


# ── t3: live-path regression, entry chain ────────────────────────────

class TestLivePathEntryChain:
    """Coordinator PEAD fast path (coordinator.py ~:628) →
    PaperTrader.track_trade(BUY) with the REAL flag active —
    TradingBlocked must propagate and no order may be submitted."""

    @staticmethod
    def _make_coordinator(db, trader):
        from orchestrator.coordinator import Coordinator

        market_data = MagicMock()
        market_data.fetch = MagicMock(return_value={
            "price": 200.0, "name": "AAPL", "currency": "USD",
        })
        risk_agent = MagicMock()
        risk_agent.run = MagicMock(return_value={
            "ticker": "AAPL", "signal": "BUY", "direction": "BUY",
            "position_size_usd": 600.0, "shares": 3,
            "stop_loss": 190.0, "take_profit": 220.0,
            "risk_amount": 30.0, "kelly_fraction": 0.05, "stop_pct": 0.05,
            "skipped": False, "skip_reason": None,
            "event_risk_flag": "none", "days_to_earnings": None,
            "regime": "TRENDING_BULL", "calc_id": 1,
        })
        coord = Coordinator(
            news_feed=MagicMock(),
            market_data=market_data,
            sentiment_agent=MagicMock(),
            technical_agent=MagicMock(),
            risk_agent=risk_agent,
            regime_agent=MagicMock(),
            db=db,
            paper_trader=trader,
            reddit_feed=MagicMock(),
            stocktwits_feed=MagicMock(),
        )
        # Keep the test focused on the kill-switch chain: portfolio
        # gate open so execution is reached.
        coord._portfolio_manager = MagicMock()
        coord._portfolio_manager.can_add_position.return_value = (True, "")
        return coord

    @staticmethod
    def _run(coord):
        return asyncio.run(coord._run_pead_only_async(
            "AAPL",
            account_balance=10_000.0,
            execute=True,
            data_semaphore=asyncio.Semaphore(4),
            db_lock=asyncio.Lock(),
        ))

    def test_pead_buy_blocked_propagates(self):
        db = _temp_db()
        trader = PaperTrader(db=db)
        coord = self._make_coordinator(db, trader)

        with patch.object(coord, "_run_pead", return_value={
            "signal": "BUY", "confidence": 0.8, "indicators": {},
        }):
            assert not KillSwitch.is_stopped(), \
                "stray emergency_stop.flag before test"
            KillSwitch.activate("stop_trading", "pytest t3 entry chain")
            try:
                with pytest.raises(TradingBlocked):
                    self._run(coord)
            finally:
                KillSwitch.deactivate()

        # No order submitted, no position opened
        assert trader.get_trade_history() == []
        assert db.get_portfolio_position("AAPL") is None

    def test_pead_buy_executes_without_flag(self):
        """Control: the same harness reaches track_trade when the switch
        is off — proves the blocked test isn't vacuously passing."""
        db = _temp_db()
        trader = PaperTrader(db=db)
        coord = self._make_coordinator(db, trader)

        with patch.object(coord, "_run_pead", return_value={
            "signal": "BUY", "confidence": 0.8, "indicators": {},
        }):
            result = self._run(coord)

        assert result["execution"] is not None
        assert result["execution"]["trade_id"] is not None
        assert db.get_portfolio_position("AAPL") is not None
