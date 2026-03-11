"""
Tests for the backtest engine, walk-forward optimiser, and results persistence.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from backtest.engine import max_drawdown, run_backtest, sharpe_ratio
from optimization.optimizer import WalkForwardOptimizer, _add_months, _all_combos
from optimization.results import load_optimization_runs, save_optimization_run
from storage.database import Database


# ── Helpers ────────────────────────────────────────────────────────────

def _make_ohlcv(days: int = 200, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a mild uptrend."""
    rng = np.random.RandomState(123)
    dates = pd.bdate_range("2023-01-02", periods=days)
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0005, 0.015)))
    close = np.array(prices)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.randint(100_000, 1_000_000, size=days),
        },
        index=dates,
    )


def _temp_db() -> Database:
    return Database(db_path=tempfile.mktemp(suffix=".db"))


# ══════════════════════════════════════════════════════════════════════
# Sharpe ratio
# ══════════════════════════════════════════════════════════════════════

class TestSharpeRatio:
    def test_flat_curve_returns_zero(self):
        assert sharpe_ratio([100, 100, 100, 100]) == 0.0

    def test_single_point_returns_zero(self):
        assert sharpe_ratio([100]) == 0.0

    def test_empty_returns_zero(self):
        assert sharpe_ratio([]) == 0.0

    def test_positive_returns_positive(self):
        # Monotonically increasing curve → positive Sharpe
        curve = [100 + i for i in range(50)]
        assert sharpe_ratio(curve) > 0

    def test_negative_returns_negative(self):
        # Monotonically decreasing curve → negative Sharpe
        curve = [100 - i * 0.5 for i in range(50)]
        assert sharpe_ratio(curve) < 0


# ══════════════════════════════════════════════════════════════════════
# Max drawdown
# ══════════════════════════════════════════════════════════════════════

class TestMaxDrawdown:
    def test_no_drawdown(self):
        assert max_drawdown([100, 101, 102, 103]) == 0.0

    def test_known_drawdown(self):
        # Peak 200, trough 100 → 50% drawdown
        curve = [100, 200, 100, 150]
        dd = max_drawdown(curve)
        assert abs(dd - 0.5) < 1e-6

    def test_single_point(self):
        assert max_drawdown([100]) == 0.0

    def test_empty(self):
        assert max_drawdown([]) == 0.0


# ══════════════════════════════════════════════════════════════════════
# run_backtest
# ══════════════════════════════════════════════════════════════════════

class TestRunBacktest:
    def test_returns_expected_keys(self):
        df = _make_ohlcv(200)
        result = run_backtest(
            ticker="TEST",
            start_date="2023-04-01",
            end_date="2023-10-01",
            params={
                "buy_threshold": 0.20,
                "sell_threshold": -0.20,
                "stop_loss_pct": 0.02,
                "take_profit_ratio": 2.0,
            },
            ohlcv=df,
        )
        expected_keys = {
            "sharpe_ratio", "max_drawdown", "win_rate",
            "total_return", "trade_count", "equity_curve", "trades",
        }
        assert set(result.keys()) == expected_keys

    def test_equity_curve_not_empty(self):
        df = _make_ohlcv(200)
        result = run_backtest(
            ticker="TEST",
            start_date="2023-04-01",
            end_date="2023-10-01",
            params={"buy_threshold": 0.20, "sell_threshold": -0.20,
                    "stop_loss_pct": 0.02, "take_profit_ratio": 2.0},
            ohlcv=df,
        )
        assert len(result["equity_curve"]) > 0

    def test_empty_df_returns_empty_result(self):
        result = run_backtest(
            ticker="TEST",
            start_date="2023-04-01",
            end_date="2023-10-01",
            params={"buy_threshold": 0.20, "sell_threshold": -0.20,
                    "stop_loss_pct": 0.02, "take_profit_ratio": 2.0},
            ohlcv=pd.DataFrame(),
        )
        assert result["trade_count"] == 0
        assert result["sharpe_ratio"] == 0.0

    def test_reproducible_with_same_seed(self):
        df = _make_ohlcv(200)
        params = {"buy_threshold": 0.20, "sell_threshold": -0.20,
                  "stop_loss_pct": 0.02, "take_profit_ratio": 2.0}
        r1 = run_backtest("TEST", "2023-04-01", "2023-10-01", params, ohlcv=df, seed=99)
        r2 = run_backtest("TEST", "2023-04-01", "2023-10-01", params, ohlcv=df, seed=99)
        assert r1["equity_curve"] == r2["equity_curve"]
        assert r1["trade_count"] == r2["trade_count"]

    def test_different_seed_may_differ(self):
        df = _make_ohlcv(200)
        params = {"buy_threshold": 0.10, "sell_threshold": -0.10,
                  "stop_loss_pct": 0.02, "take_profit_ratio": 2.0}
        r1 = run_backtest("TEST", "2023-04-01", "2023-10-01", params, ohlcv=df, seed=1)
        r2 = run_backtest("TEST", "2023-04-01", "2023-10-01", params, ohlcv=df, seed=2)
        # Different seeds should produce different sentiment → different results
        # (not guaranteed but very likely with 200 days)
        assert r1["equity_curve"] != r2["equity_curve"] or r1["trades"] != r2["trades"]

    def test_start_before_data_returns_empty(self):
        df = _make_ohlcv(30)  # too few rows for warm-up
        result = run_backtest(
            ticker="TEST",
            start_date="2023-01-02",
            end_date="2023-03-01",
            params={"buy_threshold": 0.20, "sell_threshold": -0.20,
                    "stop_loss_pct": 0.02, "take_profit_ratio": 2.0},
            ohlcv=df,
        )
        assert result["trade_count"] == 0


# ══════════════════════════════════════════════════════════════════════
# Walk-forward optimizer helpers
# ══════════════════════════════════════════════════════════════════════

class TestOptimizerHelpers:
    def test_all_combos_count(self):
        combos = _all_combos()
        # 5 * 5 * 4 * 4 = 400
        assert len(combos) == 400

    def test_all_combos_keys(self):
        combos = _all_combos()
        expected_keys = {"buy_threshold", "sell_threshold", "stop_loss_pct", "take_profit_ratio"}
        assert set(combos[0].keys()) == expected_keys

    def test_add_months_simple(self):
        from datetime import date
        assert _add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)
        assert _add_months(date(2024, 1, 15), 6) == date(2024, 7, 15)

    def test_add_months_year_wrap(self):
        from datetime import date
        assert _add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)

    def test_add_months_day_clamp(self):
        from datetime import date
        # Jan 31 + 1 month = Feb 28 (or 29 in leap year)
        result = _add_months(date(2024, 1, 31), 1)
        assert result == date(2024, 2, 29)  # 2024 is a leap year
        result2 = _add_months(date(2023, 1, 31), 1)
        assert result2 == date(2023, 2, 28)


# ══════════════════════════════════════════════════════════════════════
# Walk-forward optimizer build_windows
# ══════════════════════════════════════════════════════════════════════

class TestBuildWindows:
    def test_windows_generated(self):
        opt = WalkForwardOptimizer(
            ticker="TEST", start="2024-01-01", end="2025-01-01",
        )
        windows = opt._build_windows()
        # With 6-month train + 1-month test, advancing 1 month at a time,
        # we need at least 7 months of data.  12 months → several windows.
        assert len(windows) > 0

    def test_windows_non_overlapping_test(self):
        opt = WalkForwardOptimizer(
            ticker="TEST", start="2024-01-01", end="2025-01-01",
        )
        windows = opt._build_windows()
        # Each window's test_end <= next window's test_start is NOT required
        # (walk-forward windows overlap on training), but test periods
        # advance by 1 month each step.
        for i in range(1, len(windows)):
            assert windows[i][2] > windows[i - 1][2]  # test_start advances

    def test_short_period_no_windows(self):
        opt = WalkForwardOptimizer(
            ticker="TEST", start="2024-01-01", end="2024-06-01",
        )
        windows = opt._build_windows()
        # 5 months < 6 train + 1 test → no valid windows
        assert len(windows) == 0


# ══════════════════════════════════════════════════════════════════════
# Results persistence
# ══════════════════════════════════════════════════════════════════════

class TestResultsPersistence:
    def test_save_and_load(self):
        db = _temp_db()
        run_id = save_optimization_run(
            db=db,
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2025-01-01",
            best_params={"buy_threshold": 0.20, "stop_loss_pct": 0.02},
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=0.8,
            total_windows=5,
            equity_curve=[10000, 10100, 10200],
        )
        assert isinstance(run_id, int)

        rows = load_optimization_runs(db, ticker="AAPL")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["best_params"]["buy_threshold"] == 0.20
        assert rows[0]["equity_curve"] == [10000, 10100, 10200]
        assert rows[0]["out_of_sample_sharpe"] == 0.8

    def test_load_all_tickers(self):
        db = _temp_db()
        save_optimization_run(
            db=db, ticker="AAPL", start_date="2024-01-01",
            end_date="2025-01-01", best_params={}, in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0, total_windows=0,
        )
        save_optimization_run(
            db=db, ticker="MSFT", start_date="2024-01-01",
            end_date="2025-01-01", best_params={}, in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0, total_windows=0,
        )
        rows = load_optimization_runs(db)
        assert len(rows) == 2

    def test_load_filter_by_ticker(self):
        db = _temp_db()
        save_optimization_run(
            db=db, ticker="AAPL", start_date="2024-01-01",
            end_date="2025-01-01", best_params={}, in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0, total_windows=0,
        )
        save_optimization_run(
            db=db, ticker="MSFT", start_date="2024-01-01",
            end_date="2025-01-01", best_params={}, in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0, total_windows=0,
        )
        rows = load_optimization_runs(db, ticker="MSFT")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "MSFT"

    def test_no_equity_curve(self):
        db = _temp_db()
        save_optimization_run(
            db=db, ticker="AAPL", start_date="2024-01-01",
            end_date="2025-01-01", best_params={"k": 1},
            in_sample_sharpe=0.0, out_of_sample_sharpe=0.0, total_windows=0,
        )
        rows = load_optimization_runs(db)
        assert rows[0]["equity_curve"] is None
