"""
Tests for the trend-following walk-forward optimizer.
"""

import os
import sys
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.trend_optimizer import (
    PARAM_GRID,
    _FIXED_PARAMS,
    _add_months,
    _build_combos,
    _build_windows,
    _extract_tunable,
    generate_report,
)


# ══════════════════════════════════════════════════════════════════════
# Parameter grid
# ══════════════════════════════════════════════════════════════════════

class TestParamGrid:

    def test_combos_exclude_invalid_sma(self):
        """sma_fast must be < sma_slow — combos with sma_fast >= sma_slow filtered."""
        combos = _build_combos()
        for c in combos:
            assert c["sma_fast"] < c["sma_slow"], f"Invalid: {c['sma_fast']} >= {c['sma_slow']}"

    def test_combos_count_default(self):
        """Default grid: 4 valid sma pairs × 2 × 2 × 2 × 3 × 3 × 2 = 576."""
        combos = _build_combos(full=False)
        # sma pairs from default: (20,100), (20,200), (50,100), (50,200) = 4 valid
        expected = 4 * 2 * 2 * 2 * 3 * 3 * 2
        assert len(combos) == expected

    def test_combos_count_full(self):
        """Full grid: 5 valid sma pairs × 3 × 3 × 3 × 4 × 4 × 2 = 4320."""
        combos = _build_combos(full=True)
        expected = 5 * 3 * 3 * 3 * 4 * 4 * 2
        assert len(combos) == expected

    def test_all_combos_have_fixed_params(self):
        combos = _build_combos()
        for c in combos:
            assert c["use_sentiment"] is False
            assert c["use_technical"] is True
            assert c["require_trend_alignment"] is True

    def test_extract_tunable_filters_fixed(self):
        combo = {**_FIXED_PARAMS, "sma_fast": 50, "sma_slow": 200, "rsi_period": 14,
                 "rsi_oversold": 30, "rsi_overbought": 70, "stop_loss_pct": 0.02,
                 "take_profit_ratio": 2.0, "require_volume_confirmation": False}
        tunable = _extract_tunable(combo)
        assert "use_sentiment" not in tunable
        assert "sma_fast" in tunable


# ══════════════════════════════════════════════════════════════════════
# Walk-forward windows
# ══════════════════════════════════════════════════════════════════════

class TestWalkForwardWindows:

    def test_windows_generated_for_2_years(self):
        windows = _build_windows(date(2023, 1, 1), date(2025, 1, 1))
        assert len(windows) >= 5  # 2 years with 6mo train + 2mo test, rolling 2mo

    def test_window_structure(self):
        windows = _build_windows(date(2023, 1, 1), date(2025, 1, 1))
        for tr_s, tr_e, te_s, te_e in windows:
            assert tr_s < tr_e
            assert tr_e == te_s  # test starts where train ends
            assert te_s < te_e

    def test_short_period_no_windows(self):
        windows = _build_windows(date(2024, 1, 1), date(2024, 6, 1))
        assert len(windows) == 0

    def test_add_months(self):
        assert _add_months(date(2024, 1, 1), 6) == date(2024, 7, 1)
        assert _add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)


# ══════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════

class TestReportGeneration:

    def _sample_data(self):
        return {
            "results": [
                {
                    "ticker": "AAPL", "sector": "DATACENTER",
                    "avg_oos_sharpe": 1.5, "avg_is_sharpe": 1.8,
                    "sharpe_gap": 0.3, "overfitted": False,
                    "total_oos_trades": 25, "low_trades": False,
                    "consensus_params": {"sma_fast": 50, "sma_slow": 200,
                                        "rsi_period": 14, "rsi_oversold": 30,
                                        "rsi_overbought": 70, "stop_loss_pct": 0.02,
                                        "take_profit_ratio": 2.0,
                                        "require_volume_confirmation": False},
                    "windows": [], "num_windows": 6, "num_combos": 2592,
                    "elapsed_s": 5.0, "error": None,
                },
                {
                    "ticker": "NVDA", "sector": "AI_CHIPS",
                    "avg_oos_sharpe": 0.8, "avg_is_sharpe": 2.0,
                    "sharpe_gap": 1.2, "overfitted": True,
                    "total_oos_trades": 15, "low_trades": False,
                    "consensus_params": {"sma_fast": 20, "sma_slow": 100,
                                        "rsi_period": 10, "rsi_oversold": 25,
                                        "rsi_overbought": 75, "stop_loss_pct": 0.015,
                                        "take_profit_ratio": 3.0,
                                        "require_volume_confirmation": True},
                    "windows": [], "num_windows": 6, "num_combos": 2592,
                    "elapsed_s": 5.0, "error": None,
                },
            ],
            "meta": {
                "opt_start": "2023-01-01", "opt_end": "2025-01-01",
                "num_combos": 2592, "num_windows": 6, "elapsed_s": 10.0,
            },
        }

    def test_report_creates_file(self):
        path = generate_report(self._sample_data())
        assert path.exists()
        content = path.read_text()
        assert "Trend-Following Deep Optimization" in content
        assert "AAPL" in content
        assert "OVERFIT" in content
        path.unlink(missing_ok=True)

    def test_report_has_sections(self):
        path = generate_report(self._sample_data())
        content = path.read_text()
        for section in ["Per-Ticker Results", "Sector Consensus", "Top 5",
                        "Production Readiness"]:
            assert section in content
        path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Engine extensions (volume confirmation, configurable SMA/RSI)
# ══════════════════════════════════════════════════════════════════════

class TestEngineExtensions:

    def _make_ohlcv(self, days=100):
        dates = pd.date_range(end="2024-06-01", periods=days, freq="B")
        close = 100.0 + np.cumsum(np.random.RandomState(42).randn(days) * 1.5)
        volume = np.random.RandomState(42).randint(1_000_000, 5_000_000, size=days).astype(float)
        return pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.01,
            "Low": close * 0.98, "Close": close, "Volume": volume,
        }, index=dates)

    def test_custom_rsi_period(self):
        from backtest.engine import _compute_indicators
        df = self._make_ohlcv(200)
        ind_14 = _compute_indicators(df, {"rsi_period": 14})
        ind_21 = _compute_indicators(df, {"rsi_period": 21})
        # Both should return RSI but values likely differ
        assert ind_14["rsi"] is not None
        assert ind_21["rsi"] is not None

    def test_custom_sma_windows(self):
        from backtest.engine import _compute_indicators
        df = self._make_ohlcv(250)
        ind = _compute_indicators(df, {"sma_fast": 20, "sma_slow": 100})
        assert ind["sma_50"] is not None  # named sma_50 but uses sma_fast
        assert ind["sma_200"] is not None  # named sma_200 but uses sma_slow

    def test_volume_confirmation_present(self):
        from backtest.engine import _compute_indicators
        df = self._make_ohlcv(100)
        ind = _compute_indicators(df)
        assert "volume_confirmed" in ind

    def test_backtest_with_trend_params(self):
        from backtest.engine import run_backtest
        df = self._make_ohlcv(300)
        result = run_backtest(
            ticker="TEST", start_date="2024-01-01", end_date="2024-06-01",
            params={
                "use_sentiment": False, "use_technical": True,
                "require_trend_alignment": True,
                "sma_fast": 50, "sma_slow": 200,
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
                "stop_loss_pct": 0.02, "take_profit_ratio": 2.0,
                "buy_threshold": 0.0, "sell_threshold": 0.0,
                "require_volume_confirmation": False,
            },
            ohlcv=df,
        )
        assert "sharpe_ratio" in result
        assert "trade_count" in result
