#!/usr/bin/env python3
"""
Standalone pullback strategy backtest with walk-forward validation.

Strategy rules (from strategies/pullback.py):
  Entry (>= 3 of 4 conditions):
    1. Price > SMA50 (uptrend confirmed)
    2. RSI(14) prev < 45 AND current RSI > prev RSI (dipped then rising)
    3. Price within 5% of SMA50: 0 <= (price - SMA50) / SMA50 <= 0.05
    4. Stochastic(14,3) %K prev < 30 AND current %K > prev %K (rising from oversold)
  Exit:
    - RSI > 65 (profit target via RSI)
    - Price > 20-day high (breakout exit)
  Risk:
    - Stop-loss: 2% below entry
    - Take-profit: 2x risk = 4% above entry

Walk-forward windows (OOS only):
  Window 1: Train 2024-03-20..2025-03-20, Test 2025-03-20..2025-09-20
  Window 2: Train 2024-09-20..2025-09-20, Test 2025-09-20..2026-03-20

Research only -- does NOT modify any existing project files.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ta

# Add project root to path so we can import from backtest.engine
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import _download_ohlcv, sharpe_ratio, max_drawdown


# ── Configuration ────────────────────────────────────────────────────

TICKERS = ["AAPL", "CEG", "VST", "MSFT"]
FULL_START = "2024-03-20"
FULL_END = "2026-03-20"
INITIAL_CAPITAL = 100_000.0
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04  # 2x risk
RISK_PER_TRADE = 0.02
MAX_POSITION_PCT = 0.10

WALK_FORWARD_WINDOWS = [
    {
        "train_start": "2024-03-20",
        "train_end": "2025-03-20",
        "test_start": "2025-03-20",
        "test_end": "2025-09-20",
    },
    {
        "train_start": "2024-09-20",
        "train_end": "2025-09-20",
        "test_start": "2025-09-20",
        "test_end": "2026-03-20",
    },
]


# ── Indicator computation ────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all pullback indicators vectorized. Returns a copy with new columns."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    # Flatten MultiIndex columns if needed
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]

    # RSI(14)
    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    out["rsi"] = rsi_indicator.rsi()
    out["rsi_prev"] = out["rsi"].shift(1)

    # SMA(50)
    sma50_indicator = ta.trend.SMAIndicator(close=close, window=50)
    out["sma50"] = sma50_indicator.sma_indicator()

    # Stochastic(14, 3)
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_k_prev"] = out["stoch_k"].shift(1)

    # 20-day rolling high (for exit)
    out["high_20"] = high.rolling(20).max()

    # Distance from SMA50 as ratio
    out["sma50_dist"] = (close - out["sma50"]) / out["sma50"]

    return out


# ── Signal generation ────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True where >= 3 of 4 entry conditions met."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Condition 1: Price > SMA50
    c1 = close > df["sma50"]

    # Condition 2: RSI prev < 45 AND current RSI > prev RSI
    c2 = (df["rsi_prev"] < 45) & (df["rsi"] > df["rsi_prev"])

    # Condition 3: Price within 5% of SMA50 (0 <= dist <= 0.05)
    c3 = (df["sma50_dist"] >= 0) & (df["sma50_dist"] <= 0.05)

    # Condition 4: Stochastic %K prev < 30 AND current %K > prev %K
    c4 = (df["stoch_k_prev"] < 30) & (df["stoch_k"] > df["stoch_k_prev"])

    conditions_met = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
    return conditions_met >= 3


# ── Trade simulation ─────────────────────────────────────────────────

def simulate_trades(
    df: pd.DataFrame,
    entry_signals: pd.Series,
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Simulate trades on the OOS window.

    Uses training data rows (before start_date) for indicator warm-up.
    Only trades within [start_date, end_date).
    """
    sim_start = pd.Timestamp(start_date)
    sim_end = pd.Timestamp(end_date)

    # Get the close column as a 1D series
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    high_col = df["High"]
    if isinstance(high_col, pd.DataFrame):
        high_col = high_col.iloc[:, 0]

    low_col = df["Low"]
    if isinstance(low_col, pd.DataFrame):
        low_col = low_col.iloc[:, 0]

    # Filter to simulation range
    sim_mask = (df.index >= sim_start) & (df.index <= sim_end)
    sim_indices = df.index[sim_mask]

    if len(sim_indices) == 0:
        return _empty_sim_result(initial_capital)

    cash = initial_capital
    position = None  # dict with entry, shares, stop, tp
    equity_curve = []
    trade_pnls = []

    for dt in sim_indices:
        price = float(close.loc[dt])
        hi = float(high_col.loc[dt])
        lo = float(low_col.loc[dt])
        rsi_val = float(df.loc[dt, "rsi"]) if not pd.isna(df.loc[dt, "rsi"]) else None
        high_20_val = float(df.loc[dt, "high_20"]) if not pd.isna(df.loc[dt, "high_20"]) else None

        # ── Check exits if in position ──
        if position is not None:
            exit_reason = None

            # Stop-loss: hit if low goes below stop
            if lo <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "stop_loss"

            # Take-profit: hit if high goes above TP
            elif hi >= position["tp"]:
                exit_price = position["tp"]
                exit_reason = "take_profit"

            # RSI exit: RSI > 65
            elif rsi_val is not None and rsi_val > 65:
                exit_price = price
                exit_reason = "rsi_exit"

            # Breakout exit: price > 20-day high (use previous bar's 20-day high to avoid look-ahead)
            elif high_20_val is not None and price > high_20_val:
                # high_20 includes current bar, so compare with price
                # Actually, the 20-day high rolling includes current bar.
                # To avoid look-ahead, we should use the high_20 from the prior bar.
                # But since we're checking if the current close exceeds the prior high_20,
                # let's use the shifted version.
                pass  # handled below

            if exit_reason is None:
                # Check breakout exit using previous bar's 20-day high
                idx_pos = df.index.get_loc(dt)
                if idx_pos > 0:
                    prev_dt = df.index[idx_pos - 1]
                    prev_high_20 = df.loc[prev_dt, "high_20"]
                    if not pd.isna(prev_high_20) and price > float(prev_high_20):
                        exit_price = price
                        exit_reason = "breakout_exit"

            if exit_reason is not None:
                pnl = (exit_price - position["entry"]) * position["shares"]
                cash += position["shares"] * exit_price
                trade_pnls.append(pnl)
                position = None

        # ── Check entry if flat ──
        if position is None and entry_signals.get(dt, False):
            if cash > 0 and not pd.isna(df.loc[dt, "sma50"]):
                entry_price = price
                stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
                tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

                # Position sizing: risk 2% of account, cap at 10%
                risk_per_share = entry_price * STOP_LOSS_PCT
                if risk_per_share > 0:
                    risk_budget = cash * RISK_PER_TRADE / risk_per_share
                    max_budget = cash * MAX_POSITION_PCT / entry_price
                    shares = int(min(risk_budget, max_budget))

                    if shares > 0 and shares * entry_price <= cash:
                        cash -= shares * entry_price
                        position = {
                            "entry": entry_price,
                            "shares": shares,
                            "stop": stop_price,
                            "tp": tp_price,
                        }

        # Mark to market
        if position is not None:
            equity_curve.append(cash + position["shares"] * price)
        else:
            equity_curve.append(cash)

    # Close remaining position at end
    if position is not None:
        last_price = float(close.loc[sim_indices[-1]])
        pnl = (last_price - position["entry"]) * position["shares"]
        trade_pnls.append(pnl)
        cash += position["shares"] * last_price
        equity_curve[-1] = cash

    # Metrics
    wins = sum(1 for p in trade_pnls if p > 0)
    total_trades = len(trade_pnls)
    wr = wins / total_trades if total_trades > 0 else 0.0
    total_ret = (equity_curve[-1] / initial_capital - 1) if equity_curve else 0.0

    return {
        "sharpe": round(sharpe_ratio(equity_curve), 4) if len(equity_curve) > 1 else 0.0,
        "win_rate": round(wr, 4),
        "max_drawdown": round(max_drawdown(equity_curve), 4) if len(equity_curve) > 1 else 0.0,
        "total_return": round(total_ret, 4),
        "trade_count": total_trades,
        "final_equity": round(equity_curve[-1], 2) if equity_curve else initial_capital,
    }


def _empty_sim_result(capital: float) -> dict:
    return {
        "sharpe": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "trade_count": 0,
        "final_equity": capital,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    results = {
        "strategy": "pullback",
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "tickers": {},
        "summary": {"passed_tickers": [], "failed_tickers": []},
    }

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  Processing {ticker}")
        print(f"{'='*60}")

        # Download data once for the full period (engine adds 300-day warmup)
        print(f"  Downloading {ticker} data ({FULL_START} to {FULL_END})...")
        df_raw = _download_ohlcv(ticker, FULL_START, FULL_END)

        if df_raw.empty or len(df_raw) < 100:
            print(f"  WARNING: Insufficient data for {ticker} ({len(df_raw)} rows). Skipping.")
            results["tickers"][ticker] = {
                "windows": [],
                "avg_sharpe": 0.0,
                "avg_win_rate": 0.0,
                "avg_max_drawdown": 0.0,
                "total_trades": 0,
                "passed": False,
            }
            results["summary"]["failed_tickers"].append(ticker)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        print(f"  Downloaded {len(df_raw)} bars ({df_raw.index[0].date()} to {df_raw.index[-1].date()})")

        # Compute indicators on full dataset
        print("  Computing indicators...")
        df_ind = compute_indicators(df_raw)

        # Generate entry signals on full dataset
        entry_signals = generate_signals(df_ind)

        # Walk-forward windows
        ticker_windows = []
        for w_idx, window in enumerate(WALK_FORWARD_WINDOWS):
            print(f"\n  Window {w_idx + 1}: Test {window['test_start']} to {window['test_end']}")

            sim_result = simulate_trades(
                df_ind,
                entry_signals,
                start_date=window["test_start"],
                end_date=window["test_end"],
                initial_capital=INITIAL_CAPITAL,
            )

            window_result = {
                "train": f"{window['train_start']} to {window['train_end']}",
                "test": f"{window['test_start']} to {window['test_end']}",
                "sharpe": sim_result["sharpe"],
                "win_rate": sim_result["win_rate"],
                "max_drawdown": sim_result["max_drawdown"],
                "total_return": sim_result["total_return"],
                "trade_count": sim_result["trade_count"],
            }
            ticker_windows.append(window_result)

            print(f"    Sharpe:      {sim_result['sharpe']:.4f}")
            print(f"    Win Rate:    {sim_result['win_rate']:.4f}")
            print(f"    Max DD:      {sim_result['max_drawdown']:.4f}")
            print(f"    Total Ret:   {sim_result['total_return']:.4f}")
            print(f"    Trades:      {sim_result['trade_count']}")

        # Aggregate metrics
        sharpes = [w["sharpe"] for w in ticker_windows]
        win_rates = [w["win_rate"] for w in ticker_windows]
        max_dds = [w["max_drawdown"] for w in ticker_windows]
        total_trades = sum(w["trade_count"] for w in ticker_windows)

        avg_sharpe = round(np.mean(sharpes), 4) if sharpes else 0.0
        avg_wr = round(np.mean(win_rates), 4) if win_rates else 0.0
        avg_dd = round(np.mean(max_dds), 4) if max_dds else 0.0

        # Pass criteria: Sharpe > 0.5 AND win_rate > 0.45
        passed = bool(avg_sharpe > 0.5 and avg_wr > 0.45)

        results["tickers"][ticker] = {
            "windows": ticker_windows,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_wr,
            "avg_max_drawdown": avg_dd,
            "total_trades": total_trades,
            "passed": passed,
        }

        if passed:
            results["summary"]["passed_tickers"].append(ticker)
        else:
            results["summary"]["failed_tickers"].append(ticker)

        print(f"\n  {ticker} SUMMARY: avg_sharpe={avg_sharpe}, avg_wr={avg_wr}, "
              f"avg_dd={avg_dd}, trades={total_trades}, passed={passed}")

    # Save results
    output_path = Path(__file__).resolve().parent / "pullback_backtest.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  PULLBACK STRATEGY BACKTEST SUMMARY")
    print("=" * 60)
    for ticker, data in results["tickers"].items():
        status = "PASSED" if data["passed"] else "FAILED"
        print(f"  {ticker:6s}  Sharpe={data['avg_sharpe']:+.4f}  "
              f"WR={data['avg_win_rate']:.4f}  "
              f"DD={data['avg_max_drawdown']:.4f}  "
              f"Trades={data['total_trades']:3d}  [{status}]")

    print(f"\n  Passed: {results['summary']['passed_tickers']}")
    print(f"  Failed: {results['summary']['failed_tickers']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
