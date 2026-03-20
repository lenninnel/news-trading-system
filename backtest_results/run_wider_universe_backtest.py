#!/usr/bin/env python3
"""
Wider Universe Backtest — Momentum & Pullback strategies on 10 tickers.

Downloads 2 years of daily OHLCV via EODHD (with Alpaca/yfinance fallback),
runs walk-forward validation on both strategies, picks the best fit per ticker.

Output: backtest_results/wider_universe_backtest.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import ta

# Ensure project root is on sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import sharpe_ratio, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

TICKERS = ["AMZN", "AMD", "TSLA", "JPM", "BAC", "GS", "XOM", "CVX", "UNH", "PFE"]

START_DATE = "2024-03-20"
END_DATE = "2026-03-20"

ACCOUNT_BALANCE = 100_000.0
RISK_PER_TRADE = 0.02     # 2% of account
MAX_POSITION_PCT = 0.10   # 10% of account

# Walk-forward windows
WINDOWS = [
    {
        "name": "W1",
        "train_start": "2024-03-20",
        "train_end": "2025-03-20",
        "test_start": "2025-03-20",
        "test_end": "2025-09-20",
    },
    {
        "name": "W2",
        "train_start": "2024-09-20",
        "train_end": "2025-09-20",
        "test_start": "2025-09-20",
        "test_end": "2026-03-20",
    },
]

# Passing thresholds
MIN_SHARPE = 0.3
MIN_WIN_RATE = 0.40
MAX_DRAWDOWN_LIMIT = 0.20

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "wider_universe_backtest.json"


# ── Data Download ─────────────────────────────────────────────────────

def _download_eodhd(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try downloading daily bars from EODHD. Returns None on failure."""
    try:
        from config.settings import EODHD_API_TOKEN
        token = EODHD_API_TOKEN
    except Exception:
        token = os.environ.get("EODHD_API_TOKEN", "")

    if not token:
        log.warning("EODHD token not available, will use fallback")
        return None

    url = f"https://eodhd.com/api/eod/{ticker}.US"
    params = {
        "from": start,
        "to": end,
        "period": "d",
        "api_token": token,
        "fmt": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list):
            log.warning("EODHD returned empty/invalid data for %s", ticker)
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Normalise column names to match engine expectations
        rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
        df.rename(columns=rename, inplace=True)

        # Use adjusted close as Close if available
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        log.info("EODHD: downloaded %d bars for %s", len(df), ticker)
        return df

    except Exception as exc:
        log.warning("EODHD download failed for %s: %s", ticker, exc)
        return None


def _download_fallback(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fall back to the existing infrastructure (Alpaca -> yfinance)."""
    from backtest.engine import _download_ohlcv
    log.info("Falling back to _download_ohlcv for %s", ticker)
    return _download_ohlcv(ticker, start, end)


def download_data(ticker: str) -> pd.DataFrame:
    """Download OHLCV data for ticker, trying EODHD first."""
    # Add warmup period (300 days before start) for indicators
    warmup_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=300)).strftime("%Y-%m-%d")

    df = _download_eodhd(ticker, warmup_start, END_DATE)
    if df is None or df.empty or len(df) < 50:
        df = _download_fallback(ticker, START_DATE, END_DATE)

    if df is None or df.empty:
        log.error("No data available for %s", ticker)
        return pd.DataFrame()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ── Indicator Computation ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators needed by both strategies."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    # SMAs
    out["SMA20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    out["SMA50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    # RSI
    out["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    out["RSI_prev"] = out["RSI"].shift(1)

    # ATR
    out["ATR"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # Volume metrics
    out["Vol_SMA20"] = volume.rolling(20).mean()
    out["Vol_Ratio"] = volume / out["Vol_SMA20"]

    # Stochastic oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    out["Stoch_K"] = stoch.stoch()
    out["Stoch_K_prev"] = out["Stoch_K"].shift(1)

    # 20-day high (for pullback exit)
    out["High_20"] = high.rolling(20).max()

    return out


# ── Strategy Signals ──────────────────────────────────────────────────

def momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate momentum strategy entry/exit signals.

    Entry: 3 of 3 conditions:
      1. Price > SMA20 > SMA50
      2. RSI(14) between 50-65
      3. Volume > 1.3x 20-day avg

    Exit: RSI > 75 OR price < SMA20
    Stop-loss: 1.5% below entry
    Take-profit: 3x ATR(14) above entry
    """
    out = df.copy()

    cond1 = (out["Close"] > out["SMA20"]) & (out["SMA20"] > out["SMA50"])
    cond2 = (out["RSI"] >= 50) & (out["RSI"] <= 65)
    cond3 = out["Vol_Ratio"] > 1.3

    out["entry"] = cond1 & cond2 & cond3

    out["exit_rsi"] = out["RSI"] > 75
    out["exit_sma"] = out["Close"] < out["SMA20"]
    out["exit_signal"] = out["exit_rsi"] | out["exit_sma"]

    return out


def pullback_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pullback strategy entry/exit signals.

    Entry: 3 of 4 conditions:
      1. Price > SMA50
      2. RSI prev < 45 AND RSI rising
      3. Price within 5% of SMA50
      4. Stochastic K prev < 30 AND K rising

    Exit: RSI > 65 OR price > 20-day high
    Stop-loss: 2% below entry
    Take-profit: 4% above entry
    """
    out = df.copy()

    cond1 = out["Close"] > out["SMA50"]
    cond2 = (out["RSI_prev"] < 45) & (out["RSI"] > out["RSI_prev"])
    cond3 = (out["Close"] / out["SMA50"] - 1).abs() <= 0.05
    cond4 = (out["Stoch_K_prev"] < 30) & (out["Stoch_K"] > out["Stoch_K_prev"])

    # Require 3 of 4
    score = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + cond4.astype(int)
    out["entry"] = score >= 3

    out["exit_rsi"] = out["RSI"] > 65
    out["exit_high"] = out["Close"] > out["High_20"].shift(1)  # shifted to avoid look-ahead
    out["exit_signal"] = out["exit_rsi"] | out["exit_high"]

    return out


# ── Backtest Engine ───────────────────────────────────────────────────

def run_strategy_backtest(
    df: pd.DataFrame,
    strategy: str,
    test_start: str,
    test_end: str,
) -> dict:
    """Run a single strategy backtest on a date range.

    Returns dict with sharpe_ratio, max_drawdown, win_rate, total_trades,
    total_return, equity_curve.
    """
    # Compute indicators on the full data (no look-ahead — we only use data up to each bar)
    df_ind = compute_indicators(df)

    # Apply strategy signals
    if strategy == "momentum":
        df_sig = momentum_signals(df_ind)
        stop_loss_pct = 0.015   # 1.5%
        use_atr_tp = True       # 3x ATR
    elif strategy == "pullback":
        df_sig = pullback_signals(df_ind)
        stop_loss_pct = 0.02    # 2%
        use_atr_tp = False      # 4% fixed
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Filter to test period
    test_mask = (df_sig.index >= pd.Timestamp(test_start)) & (df_sig.index <= pd.Timestamp(test_end))
    if not test_mask.any():
        return _empty_metrics()

    test_start_idx = int(np.argmax(test_mask))
    test_end_idx = len(df_sig) - 1 - int(np.argmax(test_mask[::-1]))

    # Simulation
    cash = ACCOUNT_BALANCE
    pos_shares = 0
    pos_entry = 0.0
    pos_stop = 0.0
    pos_tp = 0.0
    in_position = False

    equity_curve = []
    trade_pnls = []

    close_arr = df_sig["Close"].values
    high_arr = df_sig["High"].values
    low_arr = df_sig["Low"].values
    entry_arr = df_sig["entry"].values
    exit_arr = df_sig["exit_signal"].values
    atr_arr = df_sig["ATR"].values if "ATR" in df_sig.columns else np.zeros(len(df_sig))

    for i in range(test_start_idx, test_end_idx + 1):
        price = float(close_arr[i])
        hi = float(high_arr[i])
        lo = float(low_arr[i])

        # Check stop-loss / take-profit
        if in_position:
            # Stop-loss hit
            if lo <= pos_stop:
                pnl = (pos_stop - pos_entry) * pos_shares
                cash += pos_shares * pos_stop
                trade_pnls.append(pnl)
                in_position = False
                pos_shares = 0
            # Take-profit hit
            elif hi >= pos_tp:
                pnl = (pos_tp - pos_entry) * pos_shares
                cash += pos_shares * pos_tp
                trade_pnls.append(pnl)
                in_position = False
                pos_shares = 0
            # Strategy exit signal
            elif bool(exit_arr[i]):
                pnl = (price - pos_entry) * pos_shares
                cash += pos_shares * price
                trade_pnls.append(pnl)
                in_position = False
                pos_shares = 0

        # Check entry
        if not in_position and bool(entry_arr[i]) and cash > 0:
            # Position sizing: risk 2% of account, cap at 10%
            risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE  # $2,000
            shares_by_risk = int(risk_amount / (price * stop_loss_pct))
            max_shares = int(ACCOUNT_BALANCE * MAX_POSITION_PCT / price)
            shares = min(shares_by_risk, max_shares, int(cash / price))

            if shares > 0:
                cash -= shares * price
                pos_shares = shares
                pos_entry = price
                pos_stop = round(price * (1.0 - stop_loss_pct), 2)

                if use_atr_tp:
                    atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else price * 0.02
                    pos_tp = round(price + 3.0 * atr_val, 2)
                else:
                    pos_tp = round(price * 1.04, 2)  # 4% take-profit

                in_position = True

        # Mark-to-market
        if in_position:
            equity_curve.append(cash + pos_shares * price)
        else:
            equity_curve.append(cash)

    # Close any remaining position
    if in_position:
        last_price = float(close_arr[test_end_idx])
        pnl = (last_price - pos_entry) * pos_shares
        trade_pnls.append(pnl)
        equity_curve[-1] = cash + pos_shares * last_price

    # Metrics
    if not equity_curve:
        return _empty_metrics()

    sr = sharpe_ratio(equity_curve)
    mdd = max_drawdown(equity_curve)
    wins = sum(1 for p in trade_pnls if p > 0)
    total_trades = len(trade_pnls)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_ret = (equity_curve[-1] / ACCOUNT_BALANCE - 1) if equity_curve else 0.0

    return {
        "sharpe_ratio": round(sr, 4),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "total_return": round(total_ret, 4),
    }


def _empty_metrics() -> dict:
    return {
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "total_return": 0.0,
    }


# ── Walk-Forward Validation ───────────────────────────────────────────

def walk_forward(df: pd.DataFrame, strategy: str) -> dict:
    """Run walk-forward validation across both windows.

    Returns averaged metrics across test windows.
    """
    window_results = []

    for w in WINDOWS:
        log.info("  %s %s: test %s -> %s", strategy, w["name"], w["test_start"], w["test_end"])
        result = run_strategy_backtest(df, strategy, w["test_start"], w["test_end"])
        window_results.append(result)
        log.info("    Sharpe=%.4f  WR=%.2f%%  DD=%.2f%%  Trades=%d",
                 result["sharpe_ratio"], result["win_rate"] * 100,
                 result["max_drawdown"] * 100, result["total_trades"])

    # Average across windows
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in window_results])
    avg_win_rate = np.mean([r["win_rate"] for r in window_results])
    avg_max_dd = np.mean([r["max_drawdown"] for r in window_results])
    total_trades = sum(r["total_trades"] for r in window_results)
    avg_return = np.mean([r["total_return"] for r in window_results])

    passed = (
        avg_sharpe >= MIN_SHARPE
        and avg_win_rate >= MIN_WIN_RATE
        and avg_max_dd <= MAX_DRAWDOWN_LIMIT
        and total_trades >= 3  # need at least a few trades
    )

    return {
        "avg_sharpe": round(float(avg_sharpe), 4),
        "avg_win_rate": round(float(avg_win_rate), 4),
        "avg_max_drawdown": round(float(avg_max_dd), 4),
        "total_trades": int(total_trades),
        "avg_return": round(float(avg_return), 4),
        "passed": bool(passed),
        "windows": window_results,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("WIDER UNIVERSE BACKTEST — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("Tickers: %s", ", ".join(TICKERS))
    log.info("Period: %s to %s", START_DATE, END_DATE)
    log.info("Account: $%,.0f | Risk/trade: %.0f%% | Max position: %.0f%%",
             ACCOUNT_BALANCE, RISK_PER_TRADE * 100, MAX_POSITION_PCT * 100)
    log.info("=" * 70)

    results = {
        "strategy": "wider_universe",
        "run_date": "2026-03-20",
        "tickers": {},
        "summary": {
            "momentum_fits": [],
            "pullback_fits": [],
            "skip": [],
        },
    }

    for ticker in TICKERS:
        log.info("")
        log.info("━" * 50)
        log.info("Processing %s", ticker)
        log.info("━" * 50)

        df = download_data(ticker)
        if df.empty or len(df) < 100:
            log.warning("Insufficient data for %s (%d bars), skipping", ticker, len(df))
            results["tickers"][ticker] = {
                "momentum": _empty_metrics() | {"passed": False},
                "pullback": _empty_metrics() | {"passed": False},
                "best_strategy": "neither",
                "recommendation": "skip",
            }
            results["summary"]["skip"].append(ticker)
            continue

        log.info("Data: %d bars from %s to %s",
                 len(df), df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))

        # Run both strategies
        log.info("--- Momentum Strategy ---")
        mom_result = walk_forward(df, "momentum")

        log.info("--- Pullback Strategy ---")
        pb_result = walk_forward(df, "pullback")

        # Determine best strategy
        if mom_result["passed"] and pb_result["passed"]:
            if mom_result["avg_sharpe"] >= pb_result["avg_sharpe"]:
                best = "momentum"
            else:
                best = "pullback"
        elif mom_result["passed"]:
            best = "momentum"
        elif pb_result["passed"]:
            best = "pullback"
        else:
            best = "neither"

        recommendation = "add_to_watchlist" if best != "neither" else "skip"

        ticker_result = {
            "momentum": {
                "avg_sharpe": mom_result["avg_sharpe"],
                "avg_win_rate": mom_result["avg_win_rate"],
                "avg_max_drawdown": mom_result["avg_max_drawdown"],
                "total_trades": mom_result["total_trades"],
                "avg_return": mom_result["avg_return"],
                "passed": mom_result["passed"],
            },
            "pullback": {
                "avg_sharpe": pb_result["avg_sharpe"],
                "avg_win_rate": pb_result["avg_win_rate"],
                "avg_max_drawdown": pb_result["avg_max_drawdown"],
                "total_trades": pb_result["total_trades"],
                "avg_return": pb_result["avg_return"],
                "passed": pb_result["passed"],
            },
            "best_strategy": best,
            "recommendation": recommendation,
        }
        results["tickers"][ticker] = ticker_result

        # Populate summary lists
        if best == "momentum":
            results["summary"]["momentum_fits"].append(ticker)
        elif best == "pullback":
            results["summary"]["pullback_fits"].append(ticker)
        else:
            results["summary"]["skip"].append(ticker)

        log.info("  >> Best: %s | Recommendation: %s", best.upper(), recommendation)

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log.info("")
    log.info("=" * 70)
    log.info("Results saved to %s", OUTPUT_FILE)
    log.info("=" * 70)

    # Print summary
    print("\n" + "=" * 70)
    print("WIDER UNIVERSE BACKTEST — SUMMARY")
    print("=" * 70)
    print(f"\nMomentum fits:  {results['summary']['momentum_fits'] or '(none)'}")
    print(f"Pullback fits:  {results['summary']['pullback_fits'] or '(none)'}")
    print(f"Skip:           {results['summary']['skip'] or '(none)'}")

    print("\n{:<6} {:<12} {:<8} {:<8} {:<8} {:<6} {:<8} {:<8} {:<8} {:<6} {:<12} {:<12}".format(
        "Tick", "Mom Sharpe", "WR%", "DD%", "Ret%", "#Tr",
        "PB Sharpe", "WR%", "DD%", "#Tr", "Best", "Rec"))
    print("-" * 120)

    for tick, data in results["tickers"].items():
        m = data["momentum"]
        p = data["pullback"]
        print("{:<6} {:<12.4f} {:<8.1f} {:<8.1f} {:<8.2f} {:<6} {:<8.4f} {:<8.1f} {:<8.1f} {:<6} {:<12} {:<12}".format(
            tick,
            m["avg_sharpe"], m["avg_win_rate"] * 100, m["avg_max_drawdown"] * 100,
            m.get("avg_return", 0) * 100, m["total_trades"],
            p["avg_sharpe"], p["avg_win_rate"] * 100, p["avg_max_drawdown"] * 100,
            p["total_trades"],
            data["best_strategy"], data["recommendation"]))

    print("\n" + "=" * 70)

    # Also print the JSON
    print("\nFull JSON output:")
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
