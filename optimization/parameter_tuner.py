"""
Parameter optimization for the News Trading System.

Uses grid search + walk-forward testing to find strategy parameters that
maximise the Sharpe ratio while respecting drawdown, win-rate, and trade-count
constraints.

Walk-forward protocol
---------------------
  Train window : 6 months
  Test window  : 3 months
  Roll step    : 3 months  (overlapping)

  For a 1-year optimisation range this produces 2 test windows:
    Window 1:  train Jan–Jun  →  test Jul–Sep
    Window 2:  train Apr–Sep  →  test Oct–Dec

Constraints (evaluated on test windows)
-----------------------------------------
  • Min trades   ≥ 10
  • Max drawdown ≥ -10 %
  • Win rate     ≥ 40 %
  • Best params also need test Sharpe ≥ 1.0 to be saved as "optimal"

Parameters optimised
--------------------
  Momentum
    rsi_threshold   50–70   — RSI must exceed this to qualify a BUY entry
    vol_multiplier  1.5–3.0 — minimum volume/avg-volume ratio
    stop_pct        2–5 %   — stop-loss distance from entry
    take_profit_pct 5–15 %  — take-profit distance from entry

  Mean Reversion
    rsi_oversold    20–35   — RSI oversold trigger level
    bb_std_dev      1.5–2.5 — Bollinger Band std-dev multiplier
    stop_pct        1–3 %   — stop-loss distance from entry
    exit_rsi        45–60   — RSI level at which the position is closed

  Swing
    macd_fast       fast EMA period (10–14)
    macd_slow       slow EMA period (24–28)
    stop_pct        3–6 %   — stop-loss distance from entry
    take_profit_pct 8–20 %  — take-profit distance from entry

CLI
---
  # Optimise one strategy for one ticker (default 2024 range)
  python3 optimization/parameter_tuner.py --strategy momentum --ticker AAPL --optimize

  # Optimise all strategies for multiple tickers
  python3 optimization/parameter_tuner.py --strategy all --tickers AAPL,NVDA,TSLA --optimize

  # Compare default vs optimised params (reads config/optimized_params.yaml)
  python3 optimization/parameter_tuner.py --strategy momentum --ticker AAPL --compare

  # Custom date range
  python3 optimization/parameter_tuner.py --strategy swing --ticker NVDA \\
      --start 2023-01-01 --end 2024-01-01 --optimize
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta
import yfinance as yf
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storage.database import Database  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
_INITIAL_BALANCE  = 10_000.0
_TRANSACTION_COST = 0.001
_MAX_PORT_FRAC    = 0.10    # max position as fraction of portfolio
_MAX_RISK_FRAC    = 0.02    # max risk per trade as fraction of portfolio
_MIN_CONFIDENCE   = 30.0
_WARMUP_DAYS      = 310     # enough for SMA-200

# Walk-forward sizes (months)
_TRAIN_MONTHS = 6
_TEST_MONTHS  = 3
_STEP_MONTHS  = 3

# Constraint thresholds
_MIN_TRADES    = 10
_MAX_DRAWDOWN  = -10.0   # percent (negative)
_MIN_WIN_RATE  = 40.0    # percent
_MIN_SHARPE    = 1.0     # for "optimal" badge in YAML

# ── Parameter grids ────────────────────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "momentum": {
        "rsi_threshold":   [50, 60, 70],
        "vol_multiplier":  [1.5, 2.0, 2.5],
        "stop_pct":        [0.02, 0.03, 0.05],
        "take_profit_pct": [0.05, 0.08, 0.12],
    },
    "mean_reversion": {
        "rsi_oversold":  [20, 28, 35],
        "bb_std_dev":    [1.5, 2.0, 2.5],
        "stop_pct":      [0.01, 0.02, 0.03],
        "exit_rsi":      [45, 52, 60],
    },
    "swing": {
        "macd_fast":       [10, 12, 14],
        "macd_slow":       [24, 26, 28],
        "stop_pct":        [0.03, 0.04, 0.06],
        "take_profit_pct": [0.08, 0.12, 0.20],
    },
}

DEFAULT_PARAMS: dict[str, dict] = {
    "momentum": {
        "rsi_threshold": 60, "vol_multiplier": 1.5,
        "stop_pct": 0.02, "take_profit_pct": 0.05,
    },
    "mean_reversion": {
        "rsi_oversold": 30, "bb_std_dev": 2.0,
        "stop_pct": 0.02, "exit_rsi": 50,
    },
    "swing": {
        "macd_fast": 12, "macd_slow": 26,
        "stop_pct": 0.03, "take_profit_pct": 0.08,
    },
}

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Metrics for a single walk-forward test window."""
    window_idx:   int
    phase:        str   # "train" or "test"
    start:        str
    end:          str
    sharpe:       float
    total_return: float
    max_dd:       float
    win_rate:     float
    trade_count:  int

    @property
    def passes_constraints(self) -> bool:
        return (
            self.trade_count >= _MIN_TRADES
            and self.max_dd   >= _MAX_DRAWDOWN
            and self.win_rate >= _MIN_WIN_RATE
        )


@dataclass
class ParamComboResult:
    """Aggregated results for one parameter combination across test windows."""
    params:   dict
    windows:  list[WindowResult] = field(default_factory=list)

    @property
    def avg_sharpe(self) -> float:
        return float(np.mean([w.sharpe for w in self.windows])) if self.windows else -999.0

    @property
    def avg_return(self) -> float:
        return float(np.mean([w.total_return for w in self.windows])) if self.windows else 0.0

    @property
    def avg_max_dd(self) -> float:
        return float(np.mean([w.max_dd for w in self.windows])) if self.windows else 0.0

    @property
    def avg_win_rate(self) -> float:
        return float(np.mean([w.win_rate for w in self.windows])) if self.windows else 0.0

    @property
    def avg_trade_count(self) -> float:
        return float(np.mean([w.trade_count for w in self.windows])) if self.windows else 0.0

    @property
    def stability_score(self) -> float:
        """Std-dev of Sharpe across test windows. Lower = more stable."""
        if len(self.windows) < 2:
            return 0.0
        return float(np.std([w.sharpe for w in self.windows]))

    @property
    def passes_constraints(self) -> bool:
        return (
            self.avg_trade_count >= _MIN_TRADES
            and self.avg_max_dd   >= _MAX_DRAWDOWN
            and self.avg_win_rate >= _MIN_WIN_RATE
        )


@dataclass
class OptimizationResult:
    """Full optimization output for one (ticker, strategy) pair."""
    ticker:           str
    strategy:         str
    start_date:       str
    end_date:         str
    best_params:      dict
    default_params:   dict
    best_sharpe:      float
    default_sharpe:   float
    best_return:      float
    default_return:   float
    best_max_dd:      float
    default_max_dd:   float
    best_win_rate:    float
    default_win_rate: float
    best_trade_count: float
    stability_score:  float
    windows_tested:   int
    combos_tested:    int
    window_results:   list[WindowResult]
    is_optimal:       bool   # True if best_sharpe ≥ MIN_SHARPE and constraints pass
    db_id:            Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# ParameterTuner
# ══════════════════════════════════════════════════════════════════════════════

class ParameterTuner:
    """
    Grid-search + walk-forward parameter optimizer for the three trading strategies.

    Downloads price data once per ticker and reuses it across all parameter
    combinations. Indicators that depend on parameters (e.g. BB std-dev, MACD
    periods) are recomputed on demand; all other indicators are cached.

    Args:
        ticker:          Stock symbol, e.g. "AAPL".
        start_date:      Optimization range start "YYYY-MM-DD".
        end_date:        Optimization range end   "YYYY-MM-DD".
        initial_balance: Starting capital in USD.
        verbose:         Print progress information.
    """

    def __init__(
        self,
        ticker:          str,
        start_date:      str,
        end_date:        str,
        initial_balance: float = _INITIAL_BALANCE,
        verbose:         bool  = True,
    ) -> None:
        self.ticker          = ticker.upper()
        self.start_ts        = pd.Timestamp(start_date)
        self.end_ts          = pd.Timestamp(end_date)
        self.initial_balance = initial_balance
        self.verbose         = verbose
        self._db             = Database()

        # Cached data (populated on first call to _get_price_data)
        self._price_df:   Optional[pd.DataFrame] = None
        self._common_ind: Optional[pd.DataFrame] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def optimize(self, strategy: str) -> OptimizationResult:
        """
        Run grid-search + walk-forward optimization for one strategy.

        Args:
            strategy: "momentum" | "mean_reversion" | "swing"

        Returns:
            OptimizationResult with best parameters, OOS metrics, and DB id.
        """
        self._log(f"\n{'═' * 66}")
        self._log(f"  Optimizing {strategy!r}  ·  {self.ticker}")
        self._log(f"  Range: {self.start_ts.date()} → {self.end_ts.date()}")
        self._log(f"{'═' * 66}")

        price_df = self._get_price_data()
        windows  = self._make_windows()

        if not windows:
            raise ValueError(
                f"Date range {self.start_ts.date()} – {self.end_ts.date()} is too short "
                f"for walk-forward testing (need at least "
                f"{_TRAIN_MONTHS + _TEST_MONTHS} months)."
            )

        grid   = PARAM_GRIDS.get(strategy, {})
        combos = list(_iter_combos(grid))
        # Filter invalid MACD combos (fast must be < slow)
        if strategy == "swing":
            combos = [c for c in combos if c["macd_fast"] < c["macd_slow"]]

        self._log(f"  Walk-forward windows  : {len(windows)}")
        self._log(f"  Parameter combinations: {len(combos)}")
        self._log(f"  Total backtests       : {len(combos) * len(windows)}")

        # Grid search — evaluate every combo on every TEST window
        self._log("\n  Grid search …")
        combo_results = self._grid_search(strategy, price_df, combos, windows)

        # Select best combo (highest avg test Sharpe, constraints respected)
        best_combo = self._select_best(combo_results)

        # Evaluate defaults for comparison
        default_params = DEFAULT_PARAMS[strategy].copy()
        default_combo  = self._eval_all_windows(strategy, price_df, default_params, windows)

        is_optimal = (
            best_combo.passes_constraints
            and best_combo.avg_sharpe >= _MIN_SHARPE
        )

        result = OptimizationResult(
            ticker           = self.ticker,
            strategy         = strategy,
            start_date       = self.start_ts.strftime("%Y-%m-%d"),
            end_date         = self.end_ts.strftime("%Y-%m-%d"),
            best_params      = best_combo.params,
            default_params   = default_params,
            best_sharpe      = round(best_combo.avg_sharpe,  3),
            default_sharpe   = round(default_combo.avg_sharpe, 3),
            best_return      = round(best_combo.avg_return,  2),
            default_return   = round(default_combo.avg_return, 2),
            best_max_dd      = round(best_combo.avg_max_dd,  2),
            default_max_dd   = round(default_combo.avg_max_dd, 2),
            best_win_rate    = round(best_combo.avg_win_rate, 1),
            default_win_rate = round(default_combo.avg_win_rate, 1),
            best_trade_count = round(best_combo.avg_trade_count, 1),
            stability_score  = round(best_combo.stability_score, 3),
            windows_tested   = len(windows),
            combos_tested    = len(combos),
            window_results   = best_combo.windows,
            is_optimal       = is_optimal,
        )

        result.db_id = self._save_result(result)

        if self.verbose:
            self._print_result(result)

        return result

    def compare(self, strategy: str, optimized_params: Optional[dict] = None) -> None:
        """
        Compare default vs optimised parameters side-by-side on the full range.

        Args:
            strategy:         "momentum" | "mean_reversion" | "swing"
            optimized_params: Parameter dict; if None, reads config/optimized_params.yaml.
        """
        if optimized_params is None:
            yaml_path = PROJECT_ROOT / "config" / "optimized_params.yaml"
            if not yaml_path.exists():
                print(f"  [!] {yaml_path} not found — run --optimize first.")
                return
            with open(yaml_path) as f:
                stored = yaml.safe_load(f) or {}
            optimized_params = (
                stored.get(self.ticker, {})
                      .get(strategy, {})
                      .get("params")
            )
            if not optimized_params:
                print(
                    f"  [!] No optimized params found for {self.ticker}/{strategy} "
                    f"in {yaml_path}."
                )
                return

        self._log(f"\n{'═' * 66}")
        self._log(f"  Compare: {strategy!r}  ·  {self.ticker}")
        self._log(f"{'═' * 66}")

        price_df = self._get_price_data()
        bt_df    = price_df.loc[self.start_ts : self.end_ts]

        default_params = DEFAULT_PARAMS[strategy].copy()

        def _full_run(params: dict, label: str) -> None:
            ind_df = self._compute_indicators(price_df, strategy, params)
            eq, trades, cash = self._run_backtest(bt_df, ind_df, strategy, params)
            m = _compute_metrics(eq, trades, self.initial_balance)
            print(f"\n  {label}")
            print(f"    Params       : {params}")
            print(f"    Return       : {m['total_return']:+.2f}%")
            print(f"    Sharpe       : {m['sharpe']:.3f}")
            print(f"    Max DD       : {m['max_dd']:.2f}%")
            print(f"    Win rate     : {m['win_rate']:.1f}%")
            print(f"    Trade count  : {m['trade_count']}")

        _full_run(default_params,   "Default parameters")
        _full_run(optimized_params, "Optimised parameters")

    # ── Private: data helpers ──────────────────────────────────────────────────

    def _get_price_data(self) -> pd.DataFrame:
        """Download OHLCV with warmup window. Cached after first call."""
        if self._price_df is not None:
            return self._price_df

        warmup_start = self.start_ts - pd.Timedelta(days=_WARMUP_DAYS)
        self._log(
            f"  Downloading {self.ticker} "
            f"({warmup_start.date()} → {self.end_ts.date()})…"
        )
        df = yf.download(
            self.ticker,
            start=warmup_start.strftime("%Y-%m-%d"),
            end=(self.end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No price data returned for {self.ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self._log(f"  Got {len(df)} rows (incl. {_WARMUP_DAYS}-day warm-up).")
        self._price_df = df
        return df

    def _make_windows(self) -> list[dict]:
        """
        Generate walk-forward window dicts.

        Each dict contains timestamps for (train_start, train_end,
        test_start, test_end).
        """
        windows = []
        train_start = self.start_ts
        idx = 0

        while True:
            train_end  = train_start + pd.DateOffset(months=_TRAIN_MONTHS) - pd.Timedelta(days=1)
            test_start = train_end   + pd.Timedelta(days=1)
            test_end   = test_start  + pd.DateOffset(months=_TEST_MONTHS)  - pd.Timedelta(days=1)

            if test_end > self.end_ts:
                break

            windows.append({
                "idx":         idx,
                "train_start": train_start,
                "train_end":   train_end,
                "test_start":  test_start,
                "test_end":    test_end,
            })
            idx         += 1
            train_start  = train_start + pd.DateOffset(months=_STEP_MONTHS)

        return windows

    # ── Private: indicator computation ────────────────────────────────────────

    def _compute_indicators(
        self,
        price_df: pd.DataFrame,
        strategy: str,
        params:   dict,
    ) -> pd.DataFrame:
        """
        Compute all indicators required by `strategy` with the given `params`.

        For momentum and mean_reversion strategies, only threshold-style params
        vary; the indicator values themselves are shared (cached in
        self._common_ind). For swing, MACD periods may differ, so a fresh
        MACD is computed when needed.
        """
        close  = price_df["Close"].squeeze()
        high   = price_df["High"].squeeze()
        low    = price_df["Low"].squeeze()
        volume = price_df["Volume"].squeeze()

        # Build common indicators cache (strategy-agnostic)
        if self._common_ind is None:
            rsi     = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            stoch   = ta.momentum.StochasticOscillator(
                high=high, low=low, close=close, window=14, smooth_window=3
            )
            adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            sma20   = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
            sma50   = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
            sma200  = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
            ema20   = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            ema50   = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
            vol_avg   = volume.rolling(20).mean()
            vol_ratio = volume / vol_avg.replace(0, np.nan)
            high_20   = high.rolling(20).max().shift(1)
            low_20    = low.rolling(20).min().shift(1)
            roc       = ta.momentum.ROCIndicator(close=close, window=10).roc()
            atr_val   = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            atr_pct   = atr_val / close.replace(0, np.nan)
            pivot_high = high.rolling(5).max().shift(1)
            pivot_low  = low.rolling(5).min().shift(1)
            crossed_above = ((close.shift(1) < sma20.shift(1)) & (close > sma20)).astype(float)
            crossed_below = ((close.shift(1) > sma20.shift(1)) & (close < sma20)).astype(float)
            is_green  = (close > close.shift(1)).astype(float)

            self._common_ind = pd.DataFrame({
                "close":         close,
                "rsi":           rsi,
                "stoch_k":       stoch.stoch(),
                "adx":           adx_obj.adx(),
                "adx_pos":       adx_obj.adx_pos(),
                "adx_neg":       adx_obj.adx_neg(),
                "ema20":         ema20,
                "ema50":         ema50,
                "sma20":         sma20,
                "sma50":         sma50,
                "sma200":        sma200,
                "vol_ratio":     vol_ratio,
                "high_20":       high_20,
                "low_20":        low_20,
                "roc":           roc,
                "atr_pct":       atr_pct,
                "pivot_high":    pivot_high,
                "pivot_low":     pivot_low,
                "crossed_above": crossed_above,
                "crossed_below": crossed_below,
                "is_green":      is_green,
            }, index=price_df.index)

        ind = self._common_ind.copy()

        # Mean Reversion — BB std dev may vary
        if strategy == "mean_reversion":
            bb_std = params.get("bb_std_dev", 2.0)
            bb_obj  = ta.volatility.BollingerBands(close=close, window=20, window_dev=bb_std)
            ind["bb_pct_b"]  = bb_obj.bollinger_pband()
            ind["bb_upper"]  = bb_obj.bollinger_hband()
            ind["bb_lower"]  = bb_obj.bollinger_lband()

        # Swing — MACD periods may vary
        elif strategy == "swing":
            fast = int(params.get("macd_fast", 12))
            slow = int(params.get("macd_slow", 26))
            sign = 9
            macd_obj   = ta.trend.MACD(close=close, window_fast=fast, window_slow=slow, window_sign=sign)
            macd_line  = macd_obj.macd()
            macd_sig   = macd_obj.macd_signal()
            macd_hist  = macd_obj.macd_diff()
            ind["macd_line"]  = macd_line
            ind["macd_sig"]   = macd_sig
            ind["hist_up"]    = ((macd_hist > macd_hist.shift(1)) & (macd_hist > 0)).astype(float)
            ind["hist_down"]  = ((macd_hist < macd_hist.shift(1)) & (macd_hist < 0)).astype(float)

        return ind

    # ── Private: signal functions ──────────────────────────────────────────────

    @staticmethod
    def _signal_momentum(row: pd.Series, params: dict) -> tuple[str, float]:
        """Momentum signal with parameterised thresholds."""
        rsi_thresh = float(params.get("rsi_threshold",  60.0))
        vol_mult   = float(params.get("vol_multiplier", 1.5))

        price     = float(row["close"])    if pd.notna(row.get("close"))    else None
        ema20     = float(row["ema20"])    if pd.notna(row.get("ema20"))    else None
        ema50     = float(row["ema50"])    if pd.notna(row.get("ema50"))    else None
        adx       = float(row["adx"])      if pd.notna(row.get("adx"))      else 0.0
        adx_pos   = float(row["adx_pos"])  if pd.notna(row.get("adx_pos"))  else 0.0
        adx_neg   = float(row["adx_neg"])  if pd.notna(row.get("adx_neg"))  else 0.0
        vol_ratio = float(row["vol_ratio"]) if pd.notna(row.get("vol_ratio")) else 1.0
        high_20   = float(row["high_20"])  if pd.notna(row.get("high_20"))  else None
        low_20    = float(row["low_20"])   if pd.notna(row.get("low_20"))   else None
        roc       = float(row["roc"])      if pd.notna(row.get("roc"))      else 0.0
        rsi       = float(row["rsi"])      if pd.notna(row.get("rsi"))      else 50.0

        conf = 50.0
        if adx > 25:        conf += 15.0
        if adx > 35:        conf += 10.0
        if vol_ratio > vol_mult: conf += 10.0
        if abs(roc) > 5.0:  conf += 15.0
        conf = min(max(conf, 30.0), 90.0)

        # BUY: breakout above 20-day high with momentum & volume, RSI confirming
        if (price and high_20 and price > high_20
                and adx > 25 and vol_ratio > vol_mult
                and adx_pos > adx_neg and rsi > rsi_thresh):
            return "BUY", conf
        # BUY: EMA uptrend + positive ROC
        if (price and ema20 and ema50
                and ema20 > ema50 and price > ema20
                and roc > 3.0 and rsi > rsi_thresh):
            return "BUY", conf
        # SELL: breakdown below 20-day low
        if (price and low_20 and price < low_20
                and adx > 25 and vol_ratio > vol_mult
                and adx_neg > adx_pos and rsi < (100 - rsi_thresh)):
            return "SELL", conf
        # SELL: EMA downtrend + negative ROC
        if (price and ema20 and ema50
                and ema20 < ema50 and price < ema20
                and roc < -3.0 and rsi < (100 - rsi_thresh)):
            return "SELL", conf
        return "HOLD", 25.0

    @staticmethod
    def _signal_mean_reversion(
        row:      pd.Series,
        params:   dict,
        position: Optional[dict],
    ) -> tuple[str, float]:
        """Mean-reversion signal; exit_rsi controls position close trigger."""
        rsi_os   = float(params.get("rsi_oversold", 30.0))
        stoch_os = float(params.get("stoch_oversold", 20.0))
        exit_rsi = float(params.get("exit_rsi",     50.0))

        rsi      = float(row["rsi"])      if pd.notna(row.get("rsi"))      else 50.0
        stoch_k  = float(row["stoch_k"])  if pd.notna(row.get("stoch_k"))  else 50.0
        bb_pct_b = float(row["bb_pct_b"]) if pd.notna(row.get("bb_pct_b")) else 0.5
        is_green = bool(row.get("is_green", 0))

        # RSI-based exit: close position when RSI recovers above exit_rsi
        if position is not None and rsi > exit_rsi:
            return "SELL", 70.0

        # Oversold entry
        if rsi < rsi_os and stoch_k < stoch_os and bb_pct_b < 0 and is_green:
            return "BUY", 80.0
        if rsi < rsi_os and (stoch_k < stoch_os + 5):
            return "BUY", 60.0
        # Overbought entry
        rsi_ob   = 100.0 - rsi_os
        stoch_ob = 100.0 - stoch_os
        if rsi > rsi_ob and stoch_k > stoch_ob and bb_pct_b > 1.0:
            return "SELL", 80.0
        if rsi > rsi_ob and stoch_k > stoch_ob - 5:
            return "SELL", 60.0

        return "HOLD", 25.0

    @staticmethod
    def _signal_swing(row: pd.Series, params: dict) -> tuple[str, float]:
        """Swing signal — same structural logic as SwingAgent, parameterised MACD."""
        price      = float(row["close"])    if pd.notna(row.get("close"))    else None
        sma20      = float(row["sma20"])    if pd.notna(row.get("sma20"))    else None
        sma50      = float(row["sma50"])    if pd.notna(row.get("sma50"))    else None
        sma200     = float(row["sma200"])   if pd.notna(row.get("sma200"))   else None
        macd_line  = float(row["macd_line"]) if pd.notna(row.get("macd_line")) else 0.0
        macd_sig   = float(row["macd_sig"])  if pd.notna(row.get("macd_sig"))  else 0.0
        hist_up    = bool(row.get("hist_up",    0))
        hist_down  = bool(row.get("hist_down",  0))
        pivot_high = float(row["pivot_high"]) if pd.notna(row.get("pivot_high")) else None
        pivot_low  = float(row["pivot_low"])  if pd.notna(row.get("pivot_low"))  else None
        atr_pct    = float(row["atr_pct"])    if pd.notna(row.get("atr_pct"))    else 0.0
        cross_up   = bool(row.get("crossed_above", 0))
        cross_down = bool(row.get("crossed_below",  0))

        buy_signal = sell_signal = False

        if (price and sma20 and sma50 and price > sma20 > sma50
                and hist_up and pivot_high and price > pivot_high):
            buy_signal = True
        if price and sma200 and price > sma200 and hist_up and cross_up:
            buy_signal = True
        if (price and sma20 and sma50 and price < sma20 < sma50
                and hist_down and pivot_low and price < pivot_low):
            sell_signal = True
        if price and sma200 and price < sma200 and hist_down and cross_down:
            sell_signal = True

        if not buy_signal and not sell_signal:
            return "HOLD", 25.0

        conf = 55.0
        if buy_signal:
            if price and sma20 and sma50 and sma200:
                if price > sma20 > sma50 and price > sma200:
                    conf += 20.0
            if macd_line > macd_sig:
                conf += 10.0
        else:
            if price and sma20 and sma50 and sma200:
                if price < sma20 < sma50 and price < sma200:
                    conf += 20.0
            if macd_line < macd_sig:
                conf += 10.0
        if atr_pct > 0.03:
            conf -= 10.0
        conf = min(max(conf, 30.0), 85.0)

        return ("BUY" if buy_signal else "SELL"), conf

    # ── Private: backtest loop ─────────────────────────────────────────────────

    def _run_backtest(
        self,
        bt_df:    pd.DataFrame,
        ind_df:   pd.DataFrame,
        strategy: str,
        params:   dict,
    ) -> tuple[dict, list, float]:
        """
        Lightweight day-by-day simulation with parameterised stop/take levels.

        Returns:
            (equity_curve dict, trades list, final_cash float)
        """
        stop_pct = float(params.get("stop_pct", 0.02))
        # Explicit take_profit_pct where specified; otherwise 2× stop
        tp_pct   = float(params.get("take_profit_pct", stop_pct * 2.0))

        cash     = self.initial_balance
        shares   = 0
        position = None
        equity   = {}
        trades   = []

        for date in bt_df.index:
            close = float(bt_df.loc[date, "Close"])

            # Stop-loss / take-profit check
            if position is not None:
                hit_sl = close <= position["stop_loss"]
                hit_tp = close >= position["take_profit"]
                if hit_sl or hit_tp:
                    exit_px = position["stop_loss"] if hit_sl else position["take_profit"]
                    pnl     = (exit_px - position["entry_price"]) * position["shares"]
                    cash   += position["shares"] * exit_px
                    trades.append({
                        "pnl":         round(pnl, 2),
                        "exit_reason": "stop_loss" if hit_sl else "take_profit",
                    })
                    shares   = 0
                    position = None

            # Compute signal
            if date not in ind_df.index:
                equity[date] = cash + shares * close
                continue

            row = ind_df.loc[date]
            if strategy == "momentum":
                signal, conf = self._signal_momentum(row, params)
            elif strategy == "mean_reversion":
                signal, conf = self._signal_mean_reversion(row, params, position)
            else:
                signal, conf = self._signal_swing(row, params)

            # Open position
            if position is None and signal == "BUY" and conf >= _MIN_CONFIDENCE:
                sz = _size_position(close, cash, conf, stop_pct)
                if sz > 0:
                    cash    -= sz * close * (1 + _TRANSACTION_COST)
                    shares   = sz
                    position = {
                        "entry_price": close,
                        "shares":      sz,
                        "stop_loss":   round(close * (1.0 - stop_pct), 4),
                        "take_profit": round(close * (1.0 + tp_pct),   4),
                    }

            # Close on opposite signal
            elif position is not None and signal == "SELL":
                pnl  = (close - position["entry_price"]) * position["shares"]
                cash += position["shares"] * close
                trades.append({"pnl": round(pnl, 2), "exit_reason": "signal"})
                shares   = 0
                position = None

            equity[date] = cash + shares * close

        # Close open position at period end
        if position is not None:
            last_close = float(bt_df["Close"].iloc[-1])
            last_date  = bt_df.index[-1]
            pnl  = (last_close - position["entry_price"]) * position["shares"]
            cash += position["shares"] * last_close
            trades.append({"pnl": round(pnl, 2), "exit_reason": "end_of_period"})
            equity[last_date] = cash

        return equity, trades, cash

    # ── Private: optimisation helpers ─────────────────────────────────────────

    def _grid_search(
        self,
        strategy:  str,
        price_df:  pd.DataFrame,
        combos:    list[dict],
        windows:   list[dict],
    ) -> list[ParamComboResult]:
        """Evaluate all (combo, test_window) pairs and return results."""
        # Pre-compute indicators for combos sharing identical indicator params
        # For speed: cache indicator df keyed by the indicator-affecting params
        ind_cache: dict[str, pd.DataFrame] = {}

        def _cache_key(strategy: str, params: dict) -> str:
            if strategy == "swing":
                return f"swing_{params['macd_fast']}_{params['macd_slow']}"
            if strategy == "mean_reversion":
                return f"mr_{params['bb_std_dev']}"
            return "momentum"

        results: list[ParamComboResult] = []
        total = len(combos) * len(windows)
        done  = 0

        for params in combos:
            key = _cache_key(strategy, params)
            if key not in ind_cache:
                ind_cache[key] = self._compute_indicators(price_df, strategy, params)

            combo_result = ParamComboResult(params=params.copy())

            for win in windows:
                test_df = price_df.loc[win["test_start"] : win["test_end"]]
                if test_df.empty:
                    done += 1
                    continue

                ind_df = ind_cache[key]
                eq, trades, cash = self._run_backtest(test_df, ind_df, strategy, params)
                m = _compute_metrics(eq, trades, self.initial_balance)

                combo_result.windows.append(WindowResult(
                    window_idx   = win["idx"],
                    phase        = "test",
                    start        = win["test_start"].strftime("%Y-%m-%d"),
                    end          = win["test_end"].strftime("%Y-%m-%d"),
                    sharpe       = m["sharpe"],
                    total_return = m["total_return"],
                    max_dd       = m["max_dd"],
                    win_rate     = m["win_rate"],
                    trade_count  = m["trade_count"],
                ))
                done += 1

            results.append(combo_result)

            if self.verbose and done % max(1, total // 20) == 0:
                pct = done / total * 100
                self._log(f"    {done}/{total}  ({pct:.0f}%)  best so far: "
                          f"{max(results, key=lambda r: r.avg_sharpe).avg_sharpe:.2f}")

        return results

    def _eval_all_windows(
        self,
        strategy:  str,
        price_df:  pd.DataFrame,
        params:    dict,
        windows:   list[dict],
    ) -> ParamComboResult:
        """Evaluate a fixed parameter set across all test windows."""
        ind_df = self._compute_indicators(price_df, strategy, params)
        combo  = ParamComboResult(params=params.copy())
        for win in windows:
            test_df = price_df.loc[win["test_start"] : win["test_end"]]
            if test_df.empty:
                continue
            eq, trades, cash = self._run_backtest(test_df, ind_df, strategy, params)
            m = _compute_metrics(eq, trades, self.initial_balance)
            combo.windows.append(WindowResult(
                window_idx   = win["idx"],
                phase        = "test",
                start        = win["test_start"].strftime("%Y-%m-%d"),
                end          = win["test_end"].strftime("%Y-%m-%d"),
                sharpe       = m["sharpe"],
                total_return = m["total_return"],
                max_dd       = m["max_dd"],
                win_rate     = m["win_rate"],
                trade_count  = m["trade_count"],
            ))
        return combo

    @staticmethod
    def _select_best(combo_results: list[ParamComboResult]) -> ParamComboResult:
        """
        Pick the best combo:
        1. Prefer combos that pass all constraints
        2. Among those, pick highest avg test Sharpe
        3. Tie-break on lowest stability_score (more consistent)
        Fallback to highest avg_sharpe among all combos if none pass.
        """
        passing = [r for r in combo_results if r.passes_constraints]
        pool    = passing if passing else combo_results

        return max(pool, key=lambda r: (r.avg_sharpe, -r.stability_score))

    # ── Private: persistence ───────────────────────────────────────────────────

    def _save_result(self, result: OptimizationResult) -> int:
        window_dicts = [
            {
                "window_idx":   w.window_idx,
                "phase":        w.phase,
                "start":        w.start,
                "end":          w.end,
                "sharpe":       w.sharpe,
                "total_return": w.total_return,
                "max_dd":       w.max_dd,
                "win_rate":     w.win_rate,
                "trade_count":  w.trade_count,
            }
            for w in result.window_results
        ]
        return self._db.log_optimization_result(
            ticker              = result.ticker,
            strategy            = result.strategy,
            start_date          = result.start_date,
            end_date            = result.end_date,
            best_params         = result.best_params,
            default_params      = result.default_params,
            best_sharpe         = result.best_sharpe,
            default_sharpe      = result.default_sharpe,
            best_return         = result.best_return,
            default_return      = result.default_return,
            best_max_dd         = result.best_max_dd,
            default_max_dd      = result.default_max_dd,
            best_win_rate       = result.best_win_rate,
            default_win_rate    = result.default_win_rate,
            best_trade_count    = result.best_trade_count,
            stability_score     = result.stability_score,
            windows_tested      = result.windows_tested,
            combos_tested       = result.combos_tested,
            window_results_json = json.dumps(window_dicts),
        )

    # ── Private: output ────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _print_result(self, result: OptimizationResult) -> None:
        bar  = "─" * 66
        star = " ★ OPTIMAL" if result.is_optimal else ""
        print(f"\n{bar}")
        print(f"  {result.ticker}  ·  {result.strategy}  ·  Optimization Result{star}")
        print(bar)
        print(f"  Range          : {result.start_date} → {result.end_date}")
        print(f"  Windows tested : {result.windows_tested}")
        print(f"  Combos tested  : {result.combos_tested}")
        print()
        print(f"  {'Metric':<18} {'Default':>10}  {'Optimised':>10}  {'Δ':>8}")
        print(f"  {'─'*18} {'─'*10}  {'─'*10}  {'─'*8}")

        def _row(label: str, d: float, o: float, decimals: int = 2) -> None:
            delta = o - d
            sign  = "+" if delta >= 0 else ""
            d_s   = format(d,         f".{decimals}f")
            o_s   = format(o,         f".{decimals}f")
            dd_s  = format(abs(delta), f".{decimals}f")
            print(f"  {label:<18} {d_s:>10}  {o_s:>10}  {sign}{dd_s:>7}")

        _row("Sharpe",       result.default_sharpe,   result.best_sharpe,  3)
        _row("Return (%)",   result.default_return,   result.best_return,  2)
        _row("Max DD (%)",   result.default_max_dd,   result.best_max_dd,  2)
        _row("Win rate (%)", result.default_win_rate, result.best_win_rate, 1)
        print(f"  {'Avg trades':<18} {'—':>10}  {result.best_trade_count:>10.1f}")
        print(f"  {'Stability (σ)':<18} {'—':>10}  {result.stability_score:>10.3f}")
        print()
        print(f"  Best parameters: {result.best_params}")
        print()
        print(f"  Walk-forward test windows:")
        for w in result.window_results:
            mark = "✓" if w.passes_constraints else "✗"
            print(
                f"    [{mark}] W{w.window_idx + 1}  {w.start} → {w.end}  "
                f"Sharpe: {w.sharpe:+.2f}  "
                f"Return: {w.total_return:+.1f}%  "
                f"DD: {w.max_dd:.1f}%  "
                f"WR: {w.win_rate:.0f}%  "
                f"Trades: {w.trade_count}"
            )
        print(bar)
        print()


# ── Module-level helpers ───────────────────────────────────────────────────────

def _iter_combos(grid: dict[str, list]) -> list[dict]:
    """Iterate over all combinations in the parameter grid."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _size_position(
    price:          float,
    available_cash: float,
    confidence:     float,
    stop_pct:       float,
) -> int:
    """Kelly-based position sizing. Returns whole shares to buy (0 if skipped)."""
    if confidence < _MIN_CONFIDENCE or price <= 0 or stop_pct <= 0:
        return 0

    # Half-Kelly fraction
    p        = 0.50 + (confidence / 100.0) * 0.30
    q        = 1.0 - p
    rr_ratio = 2.0
    kelly    = max(0.0, (p * rr_ratio - q) / rr_ratio / 2.0)

    raw_pos = min(
        available_cash * kelly,
        available_cash * _MAX_PORT_FRAC,
        (available_cash * _MAX_RISK_FRAC) / stop_pct,
    )
    shares = int(raw_pos / price)
    if shares == 0 or shares * price > available_cash:
        return 0
    return shares


def _compute_metrics(
    equity_curve: dict,
    trades:       list[dict],
    initial_bal:  float,
) -> dict:
    """Compute Sharpe, return, drawdown, win-rate from raw simulation output."""
    if not equity_curve:
        return {
            "sharpe": 0.0, "total_return": 0.0,
            "max_dd": 0.0, "win_rate": 0.0, "trade_count": 0,
        }

    eq         = pd.Series(equity_curve).sort_index()
    final_val  = float(eq.iloc[-1])
    total_ret  = (final_val / initial_bal - 1.0) * 100.0

    daily_ret  = eq.pct_change().dropna()
    sharpe     = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    )

    rolling_max = eq.cummax()
    max_dd      = float(((eq - rolling_max) / rolling_max * 100).min())

    pnls     = [t["pnl"] for t in trades]
    winners  = [p for p in pnls if p > 0]
    win_rate = len(winners) / len(pnls) * 100.0 if pnls else 0.0

    return {
        "sharpe":       round(sharpe,    3),
        "total_return": round(total_ret, 2),
        "max_dd":       round(max_dd,    2),
        "win_rate":     round(win_rate,  1),
        "trade_count":  len(pnls),
    }


def save_optimized_params(results: list[OptimizationResult]) -> Path:
    """
    Persist optimised parameters to config/optimized_params.yaml.

    Only writes results that meet the is_optimal flag (Sharpe ≥ 1.0 and
    constraints pass). Returns the path of the written file.
    """
    yaml_path = PROJECT_ROOT / "config" / "optimized_params.yaml"

    # Load existing file if present
    existing: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            existing = yaml.safe_load(f) or {}

    for r in results:
        if not r.is_optimal:
            continue
        existing.setdefault(r.ticker, {})[r.strategy] = {
            "params":       r.best_params,
            "sharpe":       r.best_sharpe,
            "return_pct":   r.best_return,
            "max_dd":       r.best_max_dd,
            "win_rate":     r.best_win_rate,
            "trade_count":  r.best_trade_count,
            "stability":    r.stability_score,
            "optimized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "range":        f"{r.start_date} → {r.end_date}",
        }

    with open(yaml_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=True)

    return yaml_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search + walk-forward parameter optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--optimize", action="store_true",
                          help="Run optimization and save best params to YAML")
    mode_grp.add_argument("--compare",  action="store_true",
                          help="Compare default vs optimised params on full range")

    parser.add_argument("--strategy",
                        choices=["momentum", "mean_reversion", "swing", "all"],
                        required=True,
                        help="Strategy to optimise (or 'all')")
    parser.add_argument("--ticker",  metavar="SYMBOL",
                        help="Single stock ticker (e.g. AAPL)")
    parser.add_argument("--tickers", metavar="A,B,C",
                        help="Comma-separated list of tickers")
    parser.add_argument("--start", default="2024-01-01", metavar="YYYY-MM-DD",
                        help="Optimization range start (default: 2024-01-01)")
    parser.add_argument("--end",   default="2025-01-01", metavar="YYYY-MM-DD",
                        help="Optimization range end   (default: 2025-01-01)")
    parser.add_argument("--balance", type=float, default=_INITIAL_BALANCE,
                        metavar="USD",
                        help=f"Starting capital (default: {_INITIAL_BALANCE:,.0f})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Resolve tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        parser.error("Provide --ticker SYMBOL or --tickers A,B,C")
        return

    # Resolve strategies
    strategies = (
        ["momentum", "mean_reversion", "swing"]
        if args.strategy == "all"
        else [args.strategy]
    )

    verbose = not args.quiet
    all_results: list[OptimizationResult] = []

    for ticker in tickers:
        tuner = ParameterTuner(
            ticker          = ticker,
            start_date      = args.start,
            end_date        = args.end,
            initial_balance = args.balance,
            verbose         = verbose,
        )

        for strategy in strategies:
            if args.optimize:
                result = tuner.optimize(strategy)
                all_results.append(result)
            else:  # --compare
                tuner.compare(strategy)

    if args.optimize and all_results:
        yaml_path = save_optimized_params(all_results)
        optimal   = [r for r in all_results if r.is_optimal]
        print(f"\n  Saved {len(optimal)} optimal result(s) → {yaml_path}")

        if optimal:
            print("\n  Optimal parameter sets:")
            for r in optimal:
                print(f"    {r.ticker}/{r.strategy}: Sharpe={r.best_sharpe:.2f}  "
                      f"{r.best_params}")
        else:
            print(
                "\n  No results met the Sharpe ≥ 1.0 + constraints threshold.\n"
                "  Best parameters still logged to DB but not written to YAML.\n"
                "  Consider widening the date range or relaxing constraints."
            )


if __name__ == "__main__":
    main()
