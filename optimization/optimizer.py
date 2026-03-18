"""
Walk-forward parameter optimisation.

Downloads OHLCV once, then rolls a 6-month train / 1-month test window
forward month by month.  For each window a grid search over param combos
selects the best Sharpe on the training set, then evaluates that combo on
the held-out test set.

Usage::

    from optimization.optimizer import WalkForwardOptimizer

    opt = WalkForwardOptimizer(ticker="AAPL", start="2024-01-01", end="2025-01-01")
    result = opt.run()
    print(result["best_params"], result["oos_sharpe"])
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import run_backtest

# ── Parameter grid ───────────────────────────────────────────────────

_GRID: dict[str, list[float]] = {
    "buy_threshold":    [0.10, 0.20, 0.30, 0.40, 0.50],
    "sell_threshold":   [-0.50, -0.40, -0.30, -0.20, -0.10],
    "stop_loss_pct":    [0.01, 0.02, 0.03, 0.04],
    "take_profit_ratio": [1.5, 2.0, 2.5, 3.0],
}

_TRAIN_MONTHS = 6
_TEST_MONTHS = 1


def _all_combos() -> list[dict[str, float]]:
    keys = list(_GRID.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*_GRID.values())]


# ── Worker (top-level for pickling) ──────────────────────────────────

def _eval_params(args: tuple) -> tuple[dict, float]:
    """Evaluate one parameter combo.  Returns (params, sharpe)."""
    params, ticker, start, end, ohlcv_dict = args
    # Reconstruct DataFrame from serialised dict
    ohlcv = pd.DataFrame(ohlcv_dict)
    ohlcv.index = pd.to_datetime(ohlcv.index)
    result = run_backtest(
        ticker=ticker,
        start_date=start,
        end_date=end,
        params=params,
        ohlcv=ohlcv,
    )
    return params, result["sharpe_ratio"]


# ── Walk-forward optimizer ───────────────────────────────────────────

class WalkForwardOptimizer:
    """
    Walk-forward optimiser with grid search and multiprocessing.

    Args:
        ticker:   Stock ticker symbol.
        start:    ISO date string for the overall start.
        end:      ISO date string for the overall end.
        workers:  Number of parallel workers (default: CPU count).
    """

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        workers: int | None = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.start = date.fromisoformat(start)
        self.end = date.fromisoformat(end)
        self.workers = workers or min(mp.cpu_count(), 8)

    def run(self) -> dict[str, Any]:
        """
        Execute walk-forward optimisation.

        Returns:
            dict with keys:
                best_params      (dict):  Parameter combo with best OOS Sharpe.
                oos_sharpe       (float): Average out-of-sample Sharpe.
                oos_results      (list):  Per-window results.
                equity_curve     (list):  Concatenated OOS equity curves.
                all_windows      (int):   Number of train/test windows.
        """
        # Download OHLCV once — include 90-day warm-up before start
        warmup = self.start - timedelta(days=120)
        start_str = warmup.strftime("%Y-%m-%d")
        end_str = self.end.strftime("%Y-%m-%d")
        try:
            from data.alpaca_data import AlpacaDataClient
            alpaca = AlpacaDataClient()
            df = alpaca.get_bars(
                self.ticker, "1Day", limit=500,
                start=start_str, end=end_str,
            )
        except Exception:
            import yfinance as yf
            df = yf.download(
                self.ticker, start=start_str, end=end_str, progress=False,
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            return {
                "best_params": {},
                "oos_sharpe": 0.0,
                "oos_results": [],
                "equity_curve": [],
                "all_windows": 0,
            }

        # Build rolling windows
        windows = self._build_windows()
        combos = _all_combos()

        oos_results: list[dict] = []
        oos_curves: list[list[float]] = []

        for train_start, train_end, test_start, test_end in windows:
            # Grid search on training set
            best_params = self._grid_search(
                combos, df,
                train_start.isoformat(), train_end.isoformat(),
            )

            # Evaluate best on test set
            test_result = run_backtest(
                ticker=self.ticker,
                start_date=test_start.isoformat(),
                end_date=test_end.isoformat(),
                params=best_params,
                ohlcv=df,
            )
            oos_results.append({
                "train": (train_start.isoformat(), train_end.isoformat()),
                "test": (test_start.isoformat(), test_end.isoformat()),
                "best_params": best_params,
                "is_sharpe": 0.0,  # filled below if needed
                "oos_sharpe": test_result["sharpe_ratio"],
                "oos_return": test_result["total_return"],
                "oos_trades": test_result["trade_count"],
            })
            if test_result["equity_curve"]:
                oos_curves.append(test_result["equity_curve"])

        # Aggregate
        avg_oos_sharpe = float(np.mean([r["oos_sharpe"] for r in oos_results])) if oos_results else 0.0

        # Best params = the one most frequently chosen across windows
        if oos_results:
            param_strs = [str(sorted(r["best_params"].items())) for r in oos_results]
            from collections import Counter
            most_common = Counter(param_strs).most_common(1)[0][0]
            # Find the actual dict
            for r in oos_results:
                if str(sorted(r["best_params"].items())) == most_common:
                    best_params = r["best_params"]
                    break
        else:
            best_params = {}

        # Concatenate OOS equity curves (normalised)
        flat_curve: list[float] = []
        for curve in oos_curves:
            if not curve:
                continue
            if not flat_curve:
                flat_curve.extend(curve)
            else:
                # Scale this window so it starts where the last one ended
                scale = flat_curve[-1] / curve[0] if curve[0] != 0 else 1.0
                flat_curve.extend(v * scale for v in curve[1:])

        return {
            "best_params": best_params,
            "oos_sharpe": round(avg_oos_sharpe, 4),
            "oos_results": oos_results,
            "equity_curve": flat_curve,
            "all_windows": len(windows),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_windows(self) -> list[tuple[date, date, date, date]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        windows = []
        cursor = self.start
        while True:
            train_start = cursor
            train_end = _add_months(train_start, _TRAIN_MONTHS)
            test_start = train_end
            test_end = _add_months(test_start, _TEST_MONTHS)
            if test_end > self.end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            cursor = _add_months(cursor, _TEST_MONTHS)
        return windows

    def _grid_search(
        self,
        combos: list[dict],
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
    ) -> dict[str, float]:
        """Find the param combo with the highest Sharpe on the training window."""
        # Serialise DataFrame for multiprocessing
        ohlcv_dict = df.to_dict()

        tasks = [
            (combo, self.ticker, train_start, train_end, ohlcv_dict)
            for combo in combos
        ]

        results: list[tuple[dict, float]]
        if self.workers > 1 and len(tasks) > 10:
            with mp.Pool(processes=self.workers) as pool:
                results = pool.map(_eval_params, tasks)
        else:
            results = [_eval_params(t) for t in tasks]

        # Pick highest Sharpe
        best = max(results, key=lambda x: x[1])
        return best[0]


def _add_months(d: date, months: int) -> date:
    """Add *months* calendar months to a date."""
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    # Clamp day to valid range
    import calendar
    max_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, max_day))
