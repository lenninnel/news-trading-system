"""
Day-by-day backtesting engine.

Replays historical OHLCV data bar-by-bar, applies TechnicalAgent signal
rules with configurable thresholds, simulates a sentiment score, and
tracks an equity curve with realistic stop-loss / take-profit exits.

No live API calls — uses pre-downloaded price data and deterministic
random sentiment so results are reproducible.

Usage::

    from backtest.engine import run_backtest

    result = run_backtest(
        ticker="AAPL",
        start_date="2024-06-01",
        end_date="2025-06-01",
        params={"buy_threshold": 0.30, "sell_threshold": -0.30,
                "stop_loss_pct": 0.02, "take_profit_ratio": 2.0},
    )
    print(result["sharpe_ratio"], result["max_drawdown"])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ta


# ── Indicator helpers (lifted from TechnicalAgent, no DB/yf deps) ────

def _compute_indicators(df_slice: pd.DataFrame) -> dict:
    """Compute technical indicators on a DataFrame slice (look-ahead safe)."""
    close: pd.Series = df_slice["Close"].squeeze()
    if len(close) < 26:
        return {}

    def latest(s: pd.Series) -> float | None:
        c = s.dropna()
        return float(c.iloc[-1]) if not c.empty else None

    def prev(s: pd.Series) -> float | None:
        c = s.dropna()
        return float(c.iloc[-2]) if len(c) >= 2 else None

    rsi_s = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    macd_s = macd_obj.macd()
    sig_s = macd_obj.macd_signal()

    cur_macd, cur_sig = latest(macd_s), latest(sig_s)
    prv_macd, prv_sig = prev(macd_s), prev(sig_s)
    macd_bull = macd_bear = False
    if all(v is not None for v in (cur_macd, cur_sig, prv_macd, prv_sig)):
        macd_bull = (prv_macd <= prv_sig) and (cur_macd > cur_sig)
        macd_bear = (prv_macd >= prv_sig) and (cur_macd < cur_sig)

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

    sma50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator() if len(close) >= 50 else pd.Series(dtype=float)
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator() if len(close) >= 200 else pd.Series(dtype=float)

    return {
        "rsi": latest(rsi_s),
        "macd_bull_cross": macd_bull,
        "macd_bear_cross": macd_bear,
        "bb_upper": latest(bb.bollinger_hband()),
        "bb_lower": latest(bb.bollinger_lband()),
        "price": float(close.iloc[-1]),
        "sma_50": latest(sma50),
        "sma_200": latest(sma200),
    }


def _signal_from_indicators(ind: dict, params: dict | None = None) -> str:
    """BUY / SELL / HOLD from indicator dict.

    Supports optional params:
        rsi_oversold   (float): RSI buy threshold (default 30).
        rsi_overbought (float): RSI sell threshold (default 70).
        require_trend_alignment (bool): Only signal when SMA50 > SMA200 (BUY)
            or SMA50 < SMA200 (SELL).  Default False.
    """
    params = params or {}
    rsi_os = params.get("rsi_oversold", 30)
    rsi_ob = params.get("rsi_overbought", 70)
    trend_req = params.get("require_trend_alignment", False)

    rsi = ind.get("rsi")
    price = ind.get("price")
    bb_low = ind.get("bb_lower")
    bb_up = ind.get("bb_upper")
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")

    buy = (rsi is not None and rsi < rsi_os) or \
          ind.get("macd_bull_cross", False) or \
          (price is not None and bb_low is not None and price < bb_low)
    sell = (rsi is not None and rsi > rsi_ob) or \
           ind.get("macd_bear_cross", False) or \
           (price is not None and bb_up is not None and price > bb_up)

    # Trend alignment filter: require SMA50 vs SMA200 confirmation
    if trend_req and sma50 is not None and sma200 is not None:
        if buy and sma50 <= sma200:
            buy = False   # no buying in a downtrend
        if sell and sma50 >= sma200:
            sell = False  # no shorting in an uptrend

    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return "HOLD"


# ── Data download with fallbacks ─────────────────────────────────────

_CRYPTO_SYMBOLS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX"}


def _download_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV with yfinance; fall back to Binance for crypto."""
    warmup_start = pd.Timestamp(start_date) - pd.Timedelta(days=300)

    # Crypto tickers → Binance klines (yfinance crypto is unreliable)
    if ticker.upper() in _CRYPTO_SYMBOLS:
        return _download_binance(ticker.upper(), warmup_start, pd.Timestamp(end_date))

    # German tickers: convert .XETRA → .DE for yfinance
    yf_ticker = ticker
    if ticker.upper().endswith(".XETRA"):
        yf_ticker = ticker.rsplit(".", 1)[0] + ".DE"

    import yfinance as yf
    df = yf.download(
        yf_ticker,
        start=warmup_start.strftime("%Y-%m-%d"),
        end=end_date,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_binance(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download daily OHLCV from Binance public klines API."""
    import requests

    pair = f"{symbol}USDT"
    all_rows = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while cursor_ms < end_ms:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": pair,
                "interval": "1d",
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        for k in klines:
            all_rows.append({
                "Date": pd.to_datetime(k[0], unit="ms"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Volume": float(k[5]),
            })
        cursor_ms = int(klines[-1][6]) + 1  # close_time + 1ms

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index("Date")
    df.sort_index(inplace=True)
    return df


# ── Metrics ──────────────────────────────────────────────────────────

def sharpe_ratio(equity_curve: list[float], risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio from a daily equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if np.std(returns) == 0:
        return 0.0
    daily_rf = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = returns - daily_rf
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252))


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.15 = 15%)."""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / peak
    return float(np.max(dd))


# ── Engine ───────────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    params: dict[str, float],
    account_balance: float = 10_000.0,
    seed: int = 42,
    ohlcv: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run a day-by-day backtest for *ticker*.

    Args:
        ticker:          Stock ticker symbol.
        start_date:      ISO date string (YYYY-MM-DD).
        end_date:        ISO date string (YYYY-MM-DD).
        params:          Dict with keys: buy_threshold, sell_threshold,
                         stop_loss_pct, take_profit_ratio.
        account_balance: Starting cash.
        seed:            Random seed for sentiment simulation.
        ohlcv:           Pre-downloaded DataFrame (avoids re-downloading
                         in walk-forward loops).  Must cover at least
                         50 trading days before start_date for warm-up.

    Returns:
        dict with keys: sharpe_ratio, max_drawdown, win_rate,
        total_return, trade_count, equity_curve, trades.
    """
    buy_thresh = params.get("buy_threshold", 0.30)
    sell_thresh = params.get("sell_threshold", -0.30)
    stop_pct = params.get("stop_loss_pct", 0.02)
    tp_ratio = params.get("take_profit_ratio", 2.0)
    use_sentiment = params.get("use_sentiment", True)
    use_technical = params.get("use_technical", True)

    # -- Download data (or reuse) --
    if ohlcv is not None:
        df = ohlcv.copy()
    else:
        df = _download_ohlcv(ticker, start_date, end_date)

    if df.empty or len(df) < 50:
        return _empty_result(account_balance)

    # Locate the simulation start index (first row >= start_date)
    sim_mask = df.index >= pd.Timestamp(start_date)
    if not sim_mask.any():
        return _empty_result(account_balance)
    sim_start_idx = int(np.argmax(sim_mask))

    # Need at least 26 rows of warm-up before simulation start
    if sim_start_idx < 26:
        return _empty_result(account_balance)

    # -- Simulate --
    rng = np.random.RandomState(seed)
    cash = account_balance
    position: dict | None = None  # {shares, entry, stop, tp, direction}
    equity_curve: list[float] = []
    trades: list[dict] = []

    for day_idx in range(sim_start_idx, len(df)):
        row = df.iloc[day_idx]
        price = float(row["Close"])

        # Check stop-loss / take-profit on open positions
        if position is not None:
            high = float(row["High"])
            low = float(row["Low"])
            closed = False

            if position["direction"] == "BUY":
                if low <= position["stop"]:
                    pnl = (position["stop"] - position["entry"]) * position["shares"]
                    cash += position["shares"] * position["stop"]
                    trades.append({"pnl": pnl, "exit": "stop_loss"})
                    position = None
                    closed = True
                elif high >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["shares"]
                    cash += position["shares"] * position["tp"]
                    trades.append({"pnl": pnl, "exit": "take_profit"})
                    position = None
                    closed = True
            else:  # SELL (short)
                if high >= position["stop"]:
                    pnl = (position["entry"] - position["stop"]) * position["shares"]
                    cash += position["shares"] * (2 * position["entry"] - position["stop"])
                    trades.append({"pnl": pnl, "exit": "stop_loss"})
                    position = None
                    closed = True
                elif low <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["shares"]
                    cash += position["shares"] * (2 * position["entry"] - position["tp"])
                    trades.append({"pnl": pnl, "exit": "take_profit"})
                    position = None
                    closed = True

        # Compute indicators on data visible up to today (no look-ahead)
        df_visible = df.iloc[:day_idx + 1]
        ind = _compute_indicators(df_visible)
        if not ind:
            equity = cash + (position["shares"] * price if position else 0)
            equity_curve.append(equity)
            continue

        tech_signal = _signal_from_indicators(ind, params) if use_technical else "HOLD"

        # Simulated sentiment score
        sentiment_score = float(rng.normal(0, 0.3))

        # Combine: sentiment threshold determines sentiment signal
        if use_sentiment:
            if sentiment_score >= buy_thresh:
                sent_signal = "BUY"
            elif sentiment_score <= sell_thresh:
                sent_signal = "SELL"
            else:
                sent_signal = "HOLD"
        else:
            sent_signal = "HOLD"

        # Fusion logic depends on which sources are active
        combined = None
        if use_sentiment and use_technical:
            # Both must agree
            if sent_signal == "BUY" and tech_signal == "BUY":
                combined = "BUY"
            elif sent_signal == "SELL" and tech_signal == "SELL":
                combined = "SELL"
        elif use_technical and not use_sentiment:
            # Technical only — trade on technical signal alone
            if tech_signal in ("BUY", "SELL"):
                combined = tech_signal
        elif use_sentiment and not use_technical:
            # Sentiment only — trade on sentiment signal alone
            if sent_signal in ("BUY", "SELL"):
                combined = sent_signal

        # Enter new position if flat
        if combined is not None and position is None and cash > 0:
            risk_budget = cash * 0.02 / stop_pct
            position_budget = min(cash * 0.10, risk_budget)
            shares = int(position_budget / price)
            if shares > 0:
                cost = shares * price
                cash -= cost

                if combined == "BUY":
                    stop = round(price * (1 - stop_pct), 2)
                    tp_price = round(price * (1 + stop_pct * tp_ratio), 2)
                else:
                    stop = round(price * (1 + stop_pct), 2)
                    tp_price = round(price * (1 - stop_pct * tp_ratio), 2)

                position = {
                    "shares": shares,
                    "entry": price,
                    "stop": stop,
                    "tp": tp_price,
                    "direction": combined,
                }

        # Mark-to-market equity
        if position is not None:
            if position["direction"] == "BUY":
                mtm = position["shares"] * price
            else:
                mtm = position["shares"] * (2 * position["entry"] - price)
            equity = cash + mtm
        else:
            equity = cash
        equity_curve.append(equity)

    # Close any remaining position at last price
    if position is not None:
        last_price = float(df.iloc[-1]["Close"])
        if position["direction"] == "BUY":
            pnl = (last_price - position["entry"]) * position["shares"]
        else:
            pnl = (position["entry"] - last_price) * position["shares"]
        trades.append({"pnl": pnl, "exit": "end_of_period"})

    # -- Metrics --
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_return_pct = (equity_curve[-1] / account_balance - 1) if equity_curve else 0.0

    return {
        "sharpe_ratio": round(sharpe_ratio(equity_curve), 4) if equity_curve else 0.0,
        "max_drawdown": round(max_drawdown(equity_curve), 4) if equity_curve else 0.0,
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
        "total_return": round(total_return_pct, 4),
        "trade_count": len(trades),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _empty_result(balance: float) -> dict:
    return {
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "trade_count": 0,
        "equity_curve": [balance],
        "trades": [],
    }
