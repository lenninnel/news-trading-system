#!/usr/bin/env python3
"""
Standalone momentum strategy backtest with walk-forward validation.

Strategy rules (from strategies/momentum.py):
  Entry (3 technical conditions — sentiment skipped):
    1. Price > SMA20 > SMA50  (uptrend)
    2. RSI(14) between 50-65  (momentum zone)
    3. Volume > 1.3x 20-day average (confirmation)
  Exit:
    - RSI > 75 (overbought)
    - Price < SMA20 (trend break)
  Risk:
    - Stop-loss: 1.5% below entry
    - Take-profit: 3x ATR(14) above entry
  Position sizing:
    - Risk 2% of account per trade
    - Cap at 10% of account per position
    - Starting capital: $100,000
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import ta

# Add project root to path so we can import from backtest.engine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtest.engine import _download_ohlcv, sharpe_ratio, max_drawdown

# ── Configuration ─────────────────────────────────────────────────────

TICKERS = ["MSFT", "NVDA", "DELL", "GOOGL", "META"]
FULL_START = "2024-03-20"
FULL_END = "2026-03-20"
INITIAL_CAPITAL = 100_000.0
STOP_LOSS_PCT = 0.015       # 1.5%
RISK_PER_TRADE = 0.02       # 2% of account
MAX_POSITION_PCT = 0.10     # 10% of account

# Walk-forward windows (out-of-sample only reported)
WINDOWS = [
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

# Pass threshold
SHARPE_THRESHOLD = 0.5
WIN_RATE_THRESHOLD = 0.45


# ── Indicator computation (vectorised) ────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to the DataFrame in place."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["SMA20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    df["ATR14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    vol_avg = volume.rolling(20).mean()
    df["VolRatio"] = volume / vol_avg.replace(0, np.nan)

    # 20-day high (for potential breakout reference)
    df["High20"] = high.rolling(20).max()

    return df


# ── Signal generation ─────────────────────────────────────────────────

def generate_entry_signals(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True where all 3 technical entry conditions met."""
    price = df["Close"]
    sma20 = df["SMA20"]
    sma50 = df["SMA50"]
    rsi = df["RSI14"]
    vol_ratio = df["VolRatio"]

    # Condition 1: Uptrend — price > SMA20 > SMA50
    cond_uptrend = (price > sma20) & (sma20 > sma50)

    # Condition 2: RSI in momentum zone (50-65)
    cond_rsi = (rsi >= 50) & (rsi <= 65)

    # Condition 3: Volume > 1.3x average
    cond_volume = vol_ratio > 1.3

    # All 3 conditions met (sentiment skipped for pure technical test)
    return cond_uptrend & cond_rsi & cond_volume


# ── Trade simulation ──────────────────────────────────────────────────

def simulate_trades(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Simulate momentum trades on the test window.

    Uses rows before test_start for indicator warm-up (already computed).
    Only takes trades within [test_start, test_end].
    """
    # Slice to test period (indicators already computed on full data)
    test_mask = (df.index >= pd.Timestamp(test_start)) & (
        df.index <= pd.Timestamp(test_end)
    )
    if not test_mask.any():
        return _empty_result(initial_capital)

    test_idx = df.index[test_mask]
    entry_signals = generate_entry_signals(df)

    # Simulation state
    cash = initial_capital
    position = None  # dict: {shares, entry_price, stop_loss, take_profit}
    equity_curve = []
    trade_results = []  # list of dicts with pnl, exit_reason, etc.

    for dt in test_idx:
        row = df.loc[dt]
        price = float(row["Close"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        rsi = float(row["RSI14"]) if not pd.isna(row["RSI14"]) else None
        sma20 = float(row["SMA20"]) if not pd.isna(row["SMA20"]) else None

        # ── Check exits first (if in position) ───────────────────────
        if position is not None:
            exit_reason = None
            exit_price = None

            # Stop-loss check (intraday low)
            if low_price <= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_reason = "stop_loss"

            # Take-profit check (intraday high)
            elif high_price >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_reason = "take_profit"

            # RSI overbought exit
            elif rsi is not None and rsi > 75:
                exit_price = price  # exit at close
                exit_reason = "rsi_overbought"

            # Trend break: price < SMA20
            elif sma20 is not None and price < sma20:
                exit_price = price  # exit at close
                exit_reason = "trend_break"

            if exit_reason is not None:
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                cash += position["shares"] * exit_price
                trade_results.append({
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "shares": position["shares"],
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "return_pct": round(
                        (exit_price / position["entry_price"] - 1) * 100, 2
                    ),
                })
                position = None

        # ── Check entry (if flat) ────────────────────────────────────
        if position is None and bool(entry_signals.loc[dt]):
            atr = float(row["ATR14"]) if not pd.isna(row["ATR14"]) else None
            if atr is not None and atr > 0 and price > 0:
                # Position sizing: risk 2% of account
                risk_amount = cash * RISK_PER_TRADE
                shares_from_risk = int(risk_amount / (price * STOP_LOSS_PCT))
                # Cap at 10% of account
                max_shares = int((cash * MAX_POSITION_PCT) / price)
                shares = min(shares_from_risk, max_shares)

                if shares > 0 and shares * price <= cash:
                    stop_loss = round(price * (1.0 - STOP_LOSS_PCT), 2)
                    take_profit = round(price + 3.0 * atr, 2)

                    cash -= shares * price
                    position = {
                        "shares": shares,
                        "entry_price": price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }

        # ── Mark to market ────────────────────────────────────────────
        if position is not None:
            equity_curve.append(cash + position["shares"] * price)
        else:
            equity_curve.append(cash)

    # Close remaining position at last close
    if position is not None:
        last_price = float(df.loc[test_idx[-1], "Close"])
        pnl = (last_price - position["entry_price"]) * position["shares"]
        cash += position["shares"] * last_price
        trade_results.append({
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "shares": position["shares"],
            "pnl": round(pnl, 2),
            "exit_reason": "end_of_period",
            "return_pct": round(
                (last_price / position["entry_price"] - 1) * 100, 2
            ),
        })
        equity_curve[-1] = cash
        position = None

    # ── Compute metrics ───────────────────────────────────────────────
    if not equity_curve:
        return _empty_result(initial_capital)

    wins = sum(1 for t in trade_results if t["pnl"] > 0)
    total_trades = len(trade_results)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_return = (equity_curve[-1] / initial_capital) - 1.0

    return {
        "sharpe": round(sharpe_ratio(equity_curve), 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "total_return": round(total_return, 4),
        "trade_count": total_trades,
        "final_equity": round(equity_curve[-1], 2),
        "trades": trade_results,
    }


def _empty_result(capital: float) -> dict:
    return {
        "sharpe": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "trade_count": 0,
        "final_equity": capital,
        "trades": [],
    }


# ── Main: walk-forward backtest ───────────────────────────────────────

def run_backtest() -> dict:
    """Run the full walk-forward momentum backtest across all tickers."""
    results = {
        "strategy": "momentum",
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "parameters": {
            "stop_loss_pct": STOP_LOSS_PCT,
            "risk_per_trade": RISK_PER_TRADE,
            "max_position_pct": MAX_POSITION_PCT,
            "initial_capital": INITIAL_CAPITAL,
            "entry_conditions": [
                "price > SMA20 > SMA50 (uptrend)",
                "RSI(14) between 50-65 (momentum zone)",
                "volume > 1.3x 20-day average",
                "sentiment skipped (pure technical)",
            ],
            "exit_conditions": [
                "RSI > 75 (overbought)",
                "price < SMA20 (trend break)",
                "stop-loss: 1.5% below entry",
                "take-profit: 3x ATR(14) above entry",
            ],
        },
        "tickers": {},
        "summary": {
            "passed_tickers": [],
            "failed_tickers": [],
        },
    }

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  Processing {ticker}")
        print(f"{'='*60}")

        # Download data once for the full period
        print(f"  Downloading data for {ticker} ({FULL_START} to {FULL_END})...")
        try:
            df = _download_ohlcv(ticker, FULL_START, FULL_END)
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            results["tickers"][ticker] = {
                "windows": [],
                "avg_sharpe": 0.0,
                "avg_win_rate": 0.0,
                "avg_max_drawdown": 0.0,
                "total_trades": 0,
                "passed": False,
                "error": str(e),
            }
            results["summary"]["failed_tickers"].append(ticker)
            continue

        if df.empty or len(df) < 100:
            print(f"  WARNING: Insufficient data for {ticker} ({len(df)} rows)")
            results["tickers"][ticker] = {
                "windows": [],
                "avg_sharpe": 0.0,
                "avg_win_rate": 0.0,
                "avg_max_drawdown": 0.0,
                "total_trades": 0,
                "passed": False,
                "error": "insufficient_data",
            }
            results["summary"]["failed_tickers"].append(ticker)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print(f"  Data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

        # Compute indicators on full dataset
        df = compute_indicators(df)

        # Walk-forward windows
        ticker_windows = []
        sharpes = []
        win_rates = []
        max_dds = []
        total_trade_count = 0

        for w_idx, window in enumerate(WINDOWS):
            print(f"\n  Window {w_idx + 1}: "
                  f"Train {window['train_start']} to {window['train_end']}, "
                  f"Test {window['test_start']} to {window['test_end']}")

            # Run simulation on OOS (test) window
            # Indicators are already computed on full data — no look-ahead
            result = simulate_trades(
                df,
                test_start=window["test_start"],
                test_end=window["test_end"],
                initial_capital=INITIAL_CAPITAL,
            )

            window_result = {
                "train": f"{window['train_start']} to {window['train_end']}",
                "test": f"{window['test_start']} to {window['test_end']}",
                "sharpe": result["sharpe"],
                "win_rate": result["win_rate"],
                "max_drawdown": result["max_drawdown"],
                "total_return": result["total_return"],
                "trade_count": result["trade_count"],
            }
            ticker_windows.append(window_result)

            sharpes.append(result["sharpe"])
            win_rates.append(result["win_rate"])
            max_dds.append(result["max_drawdown"])
            total_trade_count += result["trade_count"]

            # Print exit reason breakdown
            exit_reasons = {}
            for t in result["trades"]:
                reason = t["exit_reason"]
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            print(f"    Trades: {result['trade_count']}, "
                  f"Win rate: {result['win_rate']:.1%}, "
                  f"Sharpe: {result['sharpe']:.3f}, "
                  f"Return: {result['total_return']:.2%}, "
                  f"Max DD: {result['max_drawdown']:.2%}")
            if exit_reasons:
                print(f"    Exit breakdown: {exit_reasons}")

        # Aggregate metrics
        avg_sharpe = round(np.mean(sharpes), 4) if sharpes else 0.0
        avg_win_rate = round(np.mean(win_rates), 4) if win_rates else 0.0
        avg_max_dd = round(np.mean(max_dds), 4) if max_dds else 0.0
        passed = bool(avg_sharpe > SHARPE_THRESHOLD and avg_win_rate > WIN_RATE_THRESHOLD)

        results["tickers"][ticker] = {
            "windows": ticker_windows,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
            "avg_max_drawdown": avg_max_dd,
            "total_trades": total_trade_count,
            "passed": passed,
        }

        if passed:
            results["summary"]["passed_tickers"].append(ticker)
        else:
            results["summary"]["failed_tickers"].append(ticker)

        print(f"\n  {ticker} SUMMARY: Sharpe={avg_sharpe:.3f}, "
              f"WinRate={avg_win_rate:.1%}, MaxDD={avg_max_dd:.2%}, "
              f"Trades={total_trade_count}, "
              f"{'PASSED' if passed else 'FAILED'}")

    return results


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MOMENTUM STRATEGY BACKTEST — Walk-Forward Validation")
    print("=" * 60)
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Period: {FULL_START} to {FULL_END}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Pass criteria: Sharpe > {SHARPE_THRESHOLD}, "
          f"Win rate > {WIN_RATE_THRESHOLD:.0%}")

    results = run_backtest()

    # Save JSON (custom encoder for numpy types)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "momentum_backtest.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Print final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    for ticker, data in results["tickers"].items():
        status = "PASS" if data["passed"] else "FAIL"
        print(f"  {ticker:6s} | Sharpe={data['avg_sharpe']:+.3f} | "
              f"WinRate={data['avg_win_rate']:.1%} | "
              f"MaxDD={data['avg_max_drawdown']:.2%} | "
              f"Trades={data['total_trades']:3d} | {status}")

    print(f"\n  Passed: {results['summary']['passed_tickers'] or 'None'}")
    print(f"  Failed: {results['summary']['failed_tickers'] or 'None'}")
    print(f"\n  Results saved to: {output_path}")
