"""
Tests for the strategy comparison framework.

Covers:
  - strategies.py: all 7 strategies are valid dicts
  - strategy_runner.py: parallel execution with mocked engine
  - strategy_report.py: report generation from sample results
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.strategies import (
    ALL_TICKERS,
    STRATEGIES,
    TICKER_SECTOR,
    TICKERS,
    _REQUIRED_KEYS,
)


# ══════════════════════════════════════════════════════════════════════
# 1. Strategy definitions
# ══════════════════════════════════════════════════════════════════════

class TestStrategyDefinitions:
    """All 7 strategies must have required keys and valid types."""

    def test_seven_strategies_defined(self):
        assert len(STRATEGIES) == 7

    def test_all_have_required_keys(self):
        for name, params in STRATEGIES.items():
            for key in _REQUIRED_KEYS:
                assert key in params, f"{name} missing '{key}'"

    def test_threshold_types_are_numeric(self):
        for name, params in STRATEGIES.items():
            assert isinstance(params["buy_threshold"], (int, float)), f"{name} buy_threshold"
            assert isinstance(params["sell_threshold"], (int, float)), f"{name} sell_threshold"
            assert isinstance(params["stop_loss_pct"], (int, float)), f"{name} stop_loss_pct"
            assert isinstance(params["take_profit_ratio"], (int, float)), f"{name} take_profit_ratio"

    def test_use_flags_are_bool(self):
        for name, params in STRATEGIES.items():
            assert isinstance(params["use_sentiment"], bool), f"{name} use_sentiment"
            assert isinstance(params["use_technical"], bool), f"{name} use_technical"

    def test_strategy_names_expected(self):
        expected = {
            "BASELINE", "TECHNICAL_ONLY", "SENTIMENT_ONLY", "MOMENTUM",
            "MEAN_REVERSION", "TREND_FOLLOWING", "NEWS_EVENT_DRIVEN",
        }
        assert set(STRATEGIES.keys()) == expected

    def test_mean_reversion_has_rsi_params(self):
        mr = STRATEGIES["MEAN_REVERSION"]
        assert "rsi_oversold" in mr
        assert "rsi_overbought" in mr
        assert mr["rsi_oversold"] < mr["rsi_overbought"]

    def test_trend_following_has_trend_alignment(self):
        tf = STRATEGIES["TREND_FOLLOWING"]
        assert tf.get("require_trend_alignment") is True


# ══════════════════════════════════════════════════════════════════════
# 2. Ticker universe
# ══════════════════════════════════════════════════════════════════════

class TestTickerUniverse:

    def test_all_tickers_count(self):
        assert len(ALL_TICKERS) == 16

    def test_sectors_covered(self):
        assert set(TICKERS.keys()) == {"AI_CHIPS", "DATACENTER", "GERMAN_TECH", "CRYPTO"}

    def test_ticker_sector_mapping(self):
        assert TICKER_SECTOR["NVDA"] == "AI_CHIPS"
        assert TICKER_SECTOR["BTC"] == "CRYPTO"
        assert TICKER_SECTOR["SAP.XETRA"] == "GERMAN_TECH"


# ══════════════════════════════════════════════════════════════════════
# 3. Strategy runner (mocked engine)
# ══════════════════════════════════════════════════════════════════════

def _fake_backtest_result():
    return {
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.05,
        "win_rate": 0.6,
        "total_return": 0.12,
        "trade_count": 10,
        "equity_curve": [10000, 10100, 10200],
        "trades": [{"pnl": 50, "exit": "take_profit"}],
    }


class TestStrategyRunner:

    @patch("backtest.strategy_runner.run_backtest", return_value=_fake_backtest_result())
    def test_run_comparison_returns_results(self, mock_bt):
        from backtest.strategy_runner import run_comparison
        data = run_comparison(
            tickers=["AAPL"],
            strategies=["BASELINE"],
            workers=1,
        )
        assert "results" in data
        assert "meta" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["ticker"] == "AAPL"
        assert data["results"][0]["strategy"] == "BASELINE"
        assert data["results"][0]["error"] is None

    @patch("backtest.strategy_runner.run_backtest", return_value=_fake_backtest_result())
    def test_multiple_tickers_and_strategies(self, mock_bt):
        from backtest.strategy_runner import run_comparison
        data = run_comparison(
            tickers=["AAPL", "NVDA"],
            strategies=["BASELINE", "MOMENTUM"],
            workers=1,
        )
        assert len(data["results"]) == 4  # 2 tickers × 2 strategies

    @patch("backtest.strategy_runner.run_backtest", side_effect=Exception("fail"))
    def test_failure_doesnt_stop_others(self, mock_bt):
        from backtest.strategy_runner import run_comparison
        data = run_comparison(
            tickers=["AAPL"],
            strategies=["BASELINE"],
            workers=1,
        )
        assert len(data["results"]) == 1
        assert data["results"][0]["error"] is not None

    def test_invalid_strategy_raises(self):
        from backtest.strategy_runner import run_comparison
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_comparison(strategies=["DOES_NOT_EXIST"], workers=1)

    @patch("backtest.strategy_runner.run_backtest", return_value=_fake_backtest_result())
    def test_save_results_creates_json(self, mock_bt):
        from backtest.strategy_runner import run_comparison, save_results
        data = run_comparison(
            tickers=["AAPL"], strategies=["BASELINE"], workers=1,
        )
        path = save_results(data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert "results" in loaded
        # Cleanup
        path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 4. Strategy report
# ══════════════════════════════════════════════════════════════════════

class TestStrategyReport:

    def _sample_data(self):
        return {
            "results": [
                {
                    "ticker": "AAPL", "strategy": "BASELINE", "sector": "DATACENTER",
                    "sharpe": 1.5, "total_return_pct": 12.0, "max_drawdown_pct": 5.0,
                    "win_rate": 60.0, "total_trades": 10, "elapsed_s": 1.0, "error": None,
                },
                {
                    "ticker": "AAPL", "strategy": "MOMENTUM", "sector": "DATACENTER",
                    "sharpe": 2.0, "total_return_pct": 18.0, "max_drawdown_pct": 8.0,
                    "win_rate": 55.0, "total_trades": 15, "elapsed_s": 1.0, "error": None,
                },
                {
                    "ticker": "NVDA", "strategy": "BASELINE", "sector": "AI_CHIPS",
                    "sharpe": 0.8, "total_return_pct": 5.0, "max_drawdown_pct": 3.0,
                    "win_rate": 50.0, "total_trades": 8, "elapsed_s": 1.0, "error": None,
                },
                {
                    "ticker": "BTC", "strategy": "BASELINE", "sector": "CRYPTO",
                    "sharpe": -0.5, "total_return_pct": -3.0, "max_drawdown_pct": 12.0,
                    "win_rate": 40.0, "total_trades": 5, "elapsed_s": 1.0, "error": None,
                },
            ],
            "meta": {
                "start_date": "2024-01-01", "end_date": "2025-01-01",
                "balance": 10000, "succeeded": 4, "failed": 0,
            },
        }

    def test_generate_report_creates_file(self):
        from backtest.strategy_report import generate_report
        path = generate_report(self._sample_data())
        assert path.exists()
        content = path.read_text()
        assert "Strategy Comparison Report" in content
        assert "Overall Winner" in content
        assert "MOMENTUM" in content  # should be winner (highest avg sharpe for AAPL)
        # Cleanup
        path.unlink(missing_ok=True)

    def test_report_has_all_sections(self):
        from backtest.strategy_report import generate_report
        path = generate_report(self._sample_data())
        content = path.read_text()
        for section in [
            "Overall Winner",
            "Strategy Rankings",
            "Ticker Rankings",
            "Sector Analysis",
            "Crypto vs Stocks",
            "Top 10",
            "Bottom 10",
            "Recommendation",
        ]:
            assert section in content, f"Missing section: {section}"
        path.unlink(missing_ok=True)

    def test_empty_results_handled(self):
        from backtest.strategy_report import generate_report
        data = {"results": [], "meta": {"start_date": "2024-01-01",
                "end_date": "2025-01-01", "balance": 10000}}
        path = generate_report(data)
        assert path.exists()
        path.unlink(missing_ok=True)
