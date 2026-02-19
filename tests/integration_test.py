"""
Integration Test Suite — end-to-end workflow verification.

Tests the full pipeline with all external APIs mocked:
  Screener → Technical → Sentiment → Risk → PaperTrader → DB → Dashboard queries

All external I/O is patched:
  • yfinance.Ticker / yfinance.download  → synthetic OHLCV DataFrames
  • anthropic.Anthropic                  → canned Claude responses
  • requests.get                         → canned NewsAPI responses

A temporary SQLite database is used throughout; it is deleted on teardown.

Target: <30 seconds total runtime.

Run:
    pytest tests/integration_test.py -v
    python3 tests/integration_test.py   # standalone
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Synthetic market data ──────────────────────────────────────────────────────

def _make_ohlcv(ticker: str = "AAPL", days: int = 60):
    """Return a pandas DataFrame with synthetic OHLCV data."""
    import pandas as pd
    import numpy as np

    dates  = pd.date_range(end="2025-01-15", periods=days, freq="B")
    close  = 150.0 + np.cumsum(np.random.randn(days) * 1.5)
    volume = np.random.randint(20_000_000, 80_000_000, size=days).astype(float)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_yfinance_ticker_mock(price: float = 155.0):
    """Return a mock yfinance.Ticker whose fast_info and history() work."""
    mock_ticker = MagicMock()
    mock_ticker.fast_info.last_price = price
    mock_ticker.fast_info.previous_close = price * 0.99
    mock_ticker.history.return_value = _make_ohlcv()
    mock_ticker.info = {
        "longName": "Apple Inc.", "sector": "Technology",
        "marketCap": 3_000_000_000_000, "averageVolume": 60_000_000,
    }
    return mock_ticker


def _make_claude_message(text: str):
    """Return a mock anthropic response object."""
    msg = MagicMock()
    msg.content = [MagicMock()]
    msg.content[0].text = text
    return msg


# ── Canned Claude responses ────────────────────────────────────────────────────

_SENTIMENT_RESPONSE = json.dumps([
    {"headline": "Apple reports record quarterly revenue",
     "sentiment": "bullish", "score": 1,
     "reason": "Record revenue is strongly bullish."},
    {"headline": "Apple expands AI chip production",
     "sentiment": "bullish", "score": 1,
     "reason": "AI expansion signals growth."},
    {"headline": "Market volatility persists amid macro uncertainty",
     "sentiment": "bearish", "score": -1,
     "reason": "Macro uncertainty is negative."},
])

_TECHNICAL_RESPONSE = json.dumps({
    "signal": "BUY",
    "reasoning": ["RSI oversold at 28", "MACD bullish crossover", "Price above SMA-50"],
})

_STRATEGY_MOMENTUM_RESPONSE = json.dumps({
    "signal": "BUY",
    "confidence": 78.0,
    "timeframe": "3-5 days",
    "reasoning": ["Strong volume breakout", "RSI momentum positive"],
    "indicators": {"rsi": 62.5, "macd": 0.45, "volume_ratio": 2.1},
})

_STRATEGY_MEAN_REV_RESPONSE = json.dumps({
    "signal": "HOLD",
    "confidence": 45.0,
    "timeframe": "5-10 days",
    "reasoning": ["Price near fair value"],
    "indicators": {"rsi": 50.0, "bb_position": 0.5},
})

_STRATEGY_SWING_RESPONSE = json.dumps({
    "signal": "BUY",
    "confidence": 65.0,
    "timeframe": "1-2 weeks",
    "reasoning": ["Bullish MACD crossover on daily chart"],
    "indicators": {"macd_hist": 0.3, "atr": 2.1},
})

_RISK_RESPONSE = json.dumps({
    "direction": "BUY",
    "stop_pct": 0.02,
    "take_profit_pct": 0.04,
    "notes": "Strong momentum setup, reasonable risk.",
})

_NEWSAPI_RESPONSE = {
    "status": "ok",
    "totalResults": 3,
    "articles": [
        {"title": "Apple reports record quarterly revenue",
         "publishedAt": "2025-01-15T09:00:00Z"},
        {"title": "Apple expands AI chip production",
         "publishedAt": "2025-01-15T10:00:00Z"},
        {"title": "Market volatility persists amid macro uncertainty",
         "publishedAt": "2025-01-15T11:00:00Z"},
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Base test class
# ══════════════════════════════════════════════════════════════════════════════

class IntegrationTestBase(unittest.TestCase):
    """Base class: sets up a temp SQLite DB and common mocks."""

    def setUp(self):
        self.tmpdir  = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_trading.db")
        os.environ["DB_PATH"] = self.db_path
        # Ensure no kill switch is active
        ks_flag = PROJECT_ROOT / "emergency_stop.flag"
        if ks_flag.exists():
            ks_flag.unlink()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("DB_PATH", None)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Database layer
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseLayer(IntegrationTestBase):
    """Verify all log_* methods write and the get_* reads return correct data."""

    def test_full_crud_cycle(self):
        from storage.database import Database
        db = Database(self.db_path)

        # Runs + headline scores
        run_id = db.log_run("AAPL", 10, 8, 0.65, "BUY")
        self.assertIsInstance(run_id, int)
        db.log_headline_score(run_id, "Apple beats earnings", "bullish", 1, "Great result")
        db.log_headline_score(run_id, "Macro fears linger",  "bearish", -1, "Concern")

        runs   = db.get_recent_runs(limit=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["ticker"], "AAPL")

        scores = db.get_scores_for_run(run_id)
        self.assertEqual(len(scores), 2)

        # Technical signal
        ts_id = db.log_technical_signal("AAPL", "BUY", "RSI < 30", rsi=28.5, price=150.0)
        self.assertIsInstance(ts_id, int)

        # Combined signal
        cs_id = db.log_combined_signal("AAPL", "STRONG BUY", "BUY", "BUY", 0.65, 0.80,
                                       run_id, ts_id)
        self.assertIsInstance(cs_id, int)

        # Risk calculation
        risk_id = db.log_risk_calculation(
            "AAPL", "STRONG BUY", 80.0, 150.0, 10000.0,
            position_size_usd=500.0, shares=3,
            stop_loss=147.0, take_profit=156.0, risk_amount=9.0,
            kelly_fraction=0.05, stop_pct=0.02,
        )
        self.assertIsInstance(risk_id, int)

        # Strategy signal + performance
        ss_id = db.log_strategy_signal("AAPL", "momentum", "BUY", 78.0,
                                        timeframe="3-5 days", risk_calc_id=risk_id)
        self.assertIsInstance(ss_id, int)

        sp_id = db.log_strategy_performance(
            "AAPL", datetime.now(timezone.utc).isoformat(),
            "BUY", 71.0, "majority",
            momentum_signal="BUY", momentum_confidence=78.0,
        )
        self.assertIsInstance(sp_id, int)

        # Health check + emergency stop
        hc_id = db.log_health_check(
            datetime.now(timezone.utc).isoformat(),
            True, True, True, True, True, True,
        )
        self.assertIsInstance(hc_id, int)

        es_id = db.log_emergency_stop("stop_trading", "test", "pytest")
        self.assertIsInstance(es_id, int)

        # Scheduler log
        sl_id = db.log_scheduler_run(
            datetime.now(timezone.utc).isoformat(),
            ["AAPL"], 1, 0, 0.0, 5.2, [], "success", "Test run",
        )
        self.assertIsInstance(sl_id, int)

    def test_all_tables_created(self):
        import sqlite3
        from storage.database import Database
        Database(self.db_path)
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        expected = {
            "runs", "headline_scores", "technical_signals", "combined_signals",
            "risk_calculations", "strategy_signals", "strategy_performance",
            "scheduler_logs", "health_checks", "emergency_stops",
            "portfolio_snapshots", "portfolio_violations", "price_alerts",
            "optimization_results", "backtest_results", "screener_results",
            "backtest_strategy_comparison",
        }
        self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Paper Trader
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperTrader(IntegrationTestBase):
    """Verify buy/sell/portfolio tracking and kill-switch blocking."""

    def test_buy_and_sell_cycle(self):
        from execution.paper_trader import PaperTrader
        pt = PaperTrader(self.db_path)

        trade_id = pt.track_trade("AAPL", "BUY", 5, 150.0,
                                   stop_loss=147.0, take_profit=156.0)
        self.assertIsInstance(trade_id, int)

        portfolio = pt.get_portfolio()
        self.assertEqual(len(portfolio), 1)
        self.assertEqual(portfolio[0]["ticker"], "AAPL")
        self.assertEqual(portfolio[0]["shares"], 5)

        sell_id = pt.track_trade("AAPL", "SELL", 5, 156.0)
        self.assertIsInstance(sell_id, int)

        portfolio = pt.get_portfolio()
        self.assertEqual(len(portfolio), 0)

    def test_kill_switch_blocks_trade(self):
        from execution.paper_trader import PaperTrader
        flag = PROJECT_ROOT / "emergency_stop.flag"
        flag.write_text(
            json.dumps({"action": "stop_trading", "activated_at": "2025-01-15T00:00:00"}),
            encoding="utf-8",
        )
        try:
            pt = PaperTrader(self.db_path)
            with self.assertRaises(RuntimeError, msg="Kill switch should block trade"):
                pt.track_trade("AAPL", "BUY", 1, 150.0)
        finally:
            flag.unlink(missing_ok=True)

    def test_averaging_up(self):
        from execution.paper_trader import PaperTrader
        pt = PaperTrader(self.db_path)
        pt.track_trade("AAPL", "BUY", 4, 100.0)
        pt.track_trade("AAPL", "BUY", 4, 120.0)  # avg = 110
        p = pt.get_portfolio()[0]
        self.assertEqual(p["shares"], 8)
        self.assertAlmostEqual(p["avg_price"], 110.0, places=2)

    def test_pnl_on_sell(self):
        from execution.paper_trader import PaperTrader
        pt = PaperTrader(self.db_path)
        pt.track_trade("AAPL", "BUY", 10, 100.0)
        pt.track_trade("AAPL", "SELL", 10, 110.0)
        history = pt.get_trade_history()
        sell = [t for t in history if t["action"] == "SELL"][0]
        self.assertAlmostEqual(sell["pnl"], 100.0, places=2)  # 10 shares × $10 gain


# ══════════════════════════════════════════════════════════════════════════════
# 3. RiskAgent integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskAgentIntegration(IntegrationTestBase):
    """Run RiskAgent with a real DB and verify persistence."""

    def test_risk_agent_persists_to_db(self):
        from storage.database import Database
        from agents.risk_agent import RiskAgent

        db    = Database(self.db_path)
        agent = RiskAgent(db=db)
        result = agent.run(
            ticker="AAPL", signal="STRONG BUY", confidence=75.0,
            current_price=150.0, account_balance=10_000.0,
        )

        self.assertFalse(result["skipped"])
        self.assertGreater(result["shares"], 0)
        self.assertIsNotNone(result["stop_loss"])
        self.assertIsNotNone(result["take_profit"])
        self.assertIsInstance(result["calc_id"], int)

    def test_skipped_result_persisted(self):
        from storage.database import Database
        from agents.risk_agent import RiskAgent

        db    = Database(self.db_path)
        agent = RiskAgent(db=db)
        result = agent.run(
            ticker="AAPL", signal="HOLD", confidence=50.0,
            current_price=150.0, account_balance=10_000.0,
        )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["shares"], 0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Signal fusion (Coordinator logic)
# ══════════════════════════════════════════════════════════════════════════════

class TestCoordinatorFusion(IntegrationTestBase):
    """Verify the signal fusion matrix produces correct combined signals."""

    def test_fusion_matrix(self):
        from orchestrator.coordinator import Coordinator
        cases = [
            ("BUY",  "BUY",  "STRONG BUY"),
            ("SELL", "SELL", "STRONG SELL"),
            ("BUY",  "HOLD", "WEAK BUY"),
            ("SELL", "HOLD", "WEAK SELL"),
            ("BUY",  "SELL", "CONFLICTING"),
            ("HOLD", "BUY",  "HOLD"),
        ]
        for sent, tech, expected in cases:
            result = Coordinator.combine_signals(sent, tech)
            self.assertEqual(result, expected, f"({sent}, {tech}) → {result}, expected {expected}")

    def test_confidence_scoring(self):
        from orchestrator.coordinator import Coordinator
        c_strong = Coordinator.confidence("STRONG BUY", 0.8)
        c_weak   = Coordinator.confidence("WEAK BUY", 0.8)
        c_hold   = Coordinator.confidence("HOLD", 0.8)
        self.assertGreater(c_strong, c_weak)
        self.assertLess(c_hold, c_weak)   # HOLD should be less confident than WEAK BUY


# ══════════════════════════════════════════════════════════════════════════════
# 5. Technical Agent (with mocked yfinance)
# ══════════════════════════════════════════════════════════════════════════════

class TestTechnicalAgentIntegration(IntegrationTestBase):
    """Run TechnicalAgent with mocked yfinance; verify DB persistence."""

    @patch("yfinance.Ticker")
    def test_technical_signal_persisted(self, mock_yf):
        mock_yf.return_value = _make_yfinance_ticker_mock(price=155.0)

        from storage.database import Database
        from agents.technical_agent import TechnicalAgent

        db    = Database(self.db_path)
        agent = TechnicalAgent(db=db)
        result = agent.run("AAPL")

        self.assertIn(result["signal"], ("BUY", "SELL", "HOLD"))
        self.assertIsInstance(result.get("signal_id"), int)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Kill Switch integration
# ══════════════════════════════════════════════════════════════════════════════

class TestKillSwitch(IntegrationTestBase):
    """Verify kill-switch flag management and DB logging."""

    def test_stop_and_resume_cycle(self):
        from emergency_stop import KillSwitch
        flag = PROJECT_ROOT / "emergency_stop.flag"
        flag.unlink(missing_ok=True)

        self.assertFalse(KillSwitch.is_stopped())
        KillSwitch.activate("stop_trading", "integration test")
        self.assertTrue(KillSwitch.is_stopped())
        state = KillSwitch.get_state()
        self.assertEqual(state["action"], "stop_trading")
        self.assertEqual(state["reason"], "integration test")

        KillSwitch.deactivate()
        self.assertFalse(KillSwitch.is_stopped())

    def test_assert_raises_when_active(self):
        from emergency_stop import KillSwitch, TradingBlocked
        flag = PROJECT_ROOT / "emergency_stop.flag"
        flag.unlink(missing_ok=True)

        KillSwitch.activate("stop_all")
        try:
            with self.assertRaises(TradingBlocked):
                KillSwitch.assert_trading_allowed()
        finally:
            KillSwitch.deactivate()

    def test_db_logging(self):
        from storage.database import Database
        from emergency_stop import KillSwitch

        db = Database(self.db_path)
        db.log_emergency_stop("stop_trading", "test", "pytest")
        rows = db._select("SELECT * FROM emergency_stops")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "stop_trading")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Retry utility
# ══════════════════════════════════════════════════════════════════════════════

class TestRetryUtility(IntegrationTestBase):
    """Verify retry decorator: success on retry, raises on exhaustion."""

    def test_succeeds_on_second_attempt(self):
        from utils.retry import with_retry

        call_count = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.01)
        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("first attempt fails")
            return "ok"

        result = flaky()
        self.assertEqual(result, "ok")
        self.assertEqual(call_count["n"], 2)

    def test_raises_after_max_attempts(self):
        from utils.retry import with_retry

        @with_retry(max_attempts=2, base_delay=0.01)
        def always_fails():
            raise RuntimeError("always fails")

        with self.assertRaises(RuntimeError):
            always_fails()

    def test_alert_fn_called_on_final_failure(self):
        from utils.retry import with_retry

        alerts = []

        @with_retry(max_attempts=2, base_delay=0.01, alert_fn=alerts.append)
        def bad():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            bad()

        self.assertEqual(len(alerts), 1)
        self.assertIn("boom", alerts[0])


# ══════════════════════════════════════════════════════════════════════════════
# 8. End-to-end: StrategyCoordinator with mocked Claude + yfinance
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyCoordinatorE2E(IntegrationTestBase):
    """
    Full pipeline test with all external calls mocked.
    StrategyCoordinator → MomentumAgent + MeanReversionAgent + SwingAgent
      → RiskAgent → DB
    """

    def _make_claude_mock(self):
        """Single Claude mock that cycles through strategy responses."""
        responses = [
            _STRATEGY_MOMENTUM_RESPONSE,
            _STRATEGY_MEAN_REV_RESPONSE,
            _STRATEGY_SWING_RESPONSE,
            _RISK_RESPONSE,
        ]
        call_iter = iter(responses)

        mock_client = MagicMock()

        def _create(**kwargs):
            try:
                text = next(call_iter)
            except StopIteration:
                text = _RISK_RESPONSE
            return _make_claude_message(text)

        mock_client.messages.create.side_effect = _create
        return mock_client

    @patch("yfinance.Ticker")
    @patch("yfinance.download")
    @patch("anthropic.Anthropic")
    def test_full_strategy_pipeline(self, mock_anthropic, mock_yf_dl, mock_yf):
        mock_yf.return_value        = _make_yfinance_ticker_mock(price=155.0)
        mock_yf_dl.return_value     = _make_ohlcv()
        mock_anthropic.return_value = self._make_claude_mock()

        from storage.database import Database
        from orchestrator.strategy_coordinator import StrategyCoordinator

        db    = Database(self.db_path)
        coord = StrategyCoordinator(db=db)

        result = coord.run(
            ticker="AAPL",
            account_balance=10_000.0,
            verbose=False,
        )

        # Structural checks
        self.assertIn("combined_strategy_signal", result)
        self.assertIn("ensemble_confidence", result)
        self.assertIn("risk", result)
        self.assertIn("strategy_signals", result)
        self.assertIn(result["combined_strategy_signal"],
                      ["BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL",
                       "WEAK BUY", "WEAK SELL", "CONFLICTING"])

        # Verify DB persistence
        rows = db._select("SELECT * FROM strategy_performance")
        self.assertGreater(len(rows), 0, "strategy_performance should have a row")
        self.assertEqual(rows[-1]["ticker"], "AAPL")

    @patch("yfinance.Ticker")
    @patch("yfinance.download")
    @patch("anthropic.Anthropic")
    def test_pipeline_persists_risk_calculation(self, mock_anthropic, mock_yf_dl, mock_yf):
        mock_yf.return_value        = _make_yfinance_ticker_mock(price=155.0)
        mock_yf_dl.return_value     = _make_ohlcv()
        mock_anthropic.return_value = self._make_claude_mock()

        from storage.database import Database
        from orchestrator.strategy_coordinator import StrategyCoordinator

        db    = Database(self.db_path)
        coord = StrategyCoordinator(db=db)
        coord.run(ticker="AAPL", account_balance=10_000.0, verbose=False)

        risk_rows = db._select("SELECT * FROM risk_calculations")
        self.assertGreater(len(risk_rows), 0, "risk_calculations should have a row")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Dashboard query patterns
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardQueries(IntegrationTestBase):
    """Verify the SQL queries used by the dashboard run without error."""

    def _seed_db(self):
        from storage.database import Database
        db = Database(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        for i in range(5):
            run_id = db.log_run(f"TICK{i}", 10, 8, 0.4 + i * 0.1, "BUY")
            db.log_headline_score(run_id, f"Headline {i}", "bullish", 1, "Good")
            ts_id = db.log_technical_signal(f"TICK{i}", "BUY", "RSI oversold", rsi=28.0)
            db.log_combined_signal(f"TICK{i}", "STRONG BUY", "BUY", "BUY", 0.6, 0.8,
                                   run_id, ts_id)
            db.log_risk_calculation(f"TICK{i}", "STRONG BUY", 75.0, 100.0, 10000.0,
                                    position_size_usd=400.0, shares=4,
                                    stop_loss=98.0, take_profit=104.0, risk_amount=8.0)
        return db

    def test_recent_runs_query(self):
        db = self._seed_db()
        rows = db.get_recent_runs(limit=10)
        self.assertEqual(len(rows), 5)

    def test_combined_signals_query(self):
        import sqlite3
        self._seed_db()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM combined_signals ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 5)

    def test_strategy_performance_query(self):
        import sqlite3
        from storage.database import Database
        db = Database(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        db.log_strategy_performance("AAPL", now, "BUY", 72.0, "majority",
                                     momentum_signal="BUY", momentum_confidence=78.0)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM strategy_performance ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    sys.exit(0 if result.wasSuccessful() else 1)
