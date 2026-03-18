"""
Day-by-day backtesting engine.

Replays historical OHLCV data bar-by-bar, applies TechnicalAgent signal
rules with configurable thresholds, simulates a sentiment score, and
tracks an equity curve with realistic stop-loss / take-profit exits.

Indicators are pre-computed once for the full series using vectorised
numpy / pandas operations, then the simulation loop runs over numpy
arrays — no per-bar re-computation.

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

def _compute_indicators(df_slice: pd.DataFrame, params: dict | None = None) -> dict:
    """Compute technical indicators on a DataFrame slice (look-ahead safe).

    Configurable via *params*:
        rsi_period (int): RSI window (default 14).
        sma_fast   (int): Fast SMA period (default 50).
        sma_slow   (int): Slow SMA period (default 200).
    """
    params = params or {}
    rsi_period = int(params.get("rsi_period", 14))
    sma_fast = int(params.get("sma_fast", 50))
    sma_slow = int(params.get("sma_slow", 200))

    close: pd.Series = df_slice["Close"].squeeze()
    if len(close) < 26:
        return {}

    def latest(s: pd.Series) -> float | None:
        c = s.dropna()
        return float(c.iloc[-1]) if not c.empty else None

    def prev(s: pd.Series) -> float | None:
        c = s.dropna()
        return float(c.iloc[-2]) if len(c) >= 2 else None

    rsi_s = ta.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
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

    sma_fast_s = ta.trend.SMAIndicator(close=close, window=sma_fast).sma_indicator() if len(close) >= sma_fast else pd.Series(dtype=float)
    sma_slow_s = ta.trend.SMAIndicator(close=close, window=sma_slow).sma_indicator() if len(close) >= sma_slow else pd.Series(dtype=float)

    # Volume confirmation: RVOL > 1.5
    volume_confirmed = False
    if "Volume" in df_slice.columns:
        vol = df_slice["Volume"].squeeze()
        if len(vol) >= 20 and not vol.empty:
            avg_vol = float(vol.iloc[-20:].mean())
            if avg_vol > 0:
                rvol = float(vol.iloc[-1]) / avg_vol
                volume_confirmed = rvol > 1.5

    return {
        "rsi": latest(rsi_s),
        "macd_bull_cross": macd_bull,
        "macd_bear_cross": macd_bear,
        "bb_upper": latest(bb.bollinger_hband()),
        "bb_lower": latest(bb.bollinger_lband()),
        "price": float(close.iloc[-1]),
        "sma_50": latest(sma_fast_s),
        "sma_200": latest(sma_slow_s),
        "volume_confirmed": volume_confirmed,
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

    # Trend alignment filter: require SMA_fast vs SMA_slow confirmation
    if trend_req and sma50 is not None and sma200 is not None:
        if buy and sma50 <= sma200:
            buy = False   # no buying in a downtrend
        if sell and sma50 >= sma200:
            sell = False  # no shorting in an uptrend

    # Volume confirmation filter
    if params.get("require_volume_confirmation", False):
        if not ind.get("volume_confirmed", False):
            buy = False
            sell = False

    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return "HOLD"


# ── Vectorised indicator pre-computation ─────────────────────────────

def _ema_np(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, seeded with SMA of first *period* values."""
    n = len(data)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _ema_of_valid(data: np.ndarray, period: int) -> np.ndarray:
    """EMA of array that may have leading NaNs."""
    n = len(data)
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    valid = ~np.isnan(data)
    count = 0
    start = -1
    for i in range(n):
        if valid[i]:
            count += 1
            if count >= period:
                start = i - period + 1
                break
        else:
            count = 0
    if start < 0:
        return out
    end = start + period
    out[end - 1] = np.nanmean(data[start:end])
    for i in range(end, n):
        if np.isnan(data[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _precompute_indicators(df: pd.DataFrame, params: dict) -> dict[str, np.ndarray]:
    """Compute all indicator arrays once for the full DataFrame."""
    close = df["Close"].values.astype(np.float64)
    if close.ndim > 1:
        close = close[:, 0]
    n = len(close)

    rsi_period = int(params.get("rsi_period", 14))
    sma_fast_p = int(params.get("sma_fast", 50))
    sma_slow_p = int(params.get("sma_slow", 200))

    # ── RSI (Wilder's smoothing) ─────────────────────────────────────
    rsi = np.full(n, np.nan)
    if n > rsi_period:
        delta = np.diff(close)
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        ag = np.mean(gain[:rsi_period])
        al = np.mean(loss[:rsi_period])
        rsi[rsi_period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        for i in range(rsi_period, len(delta)):
            ag = (ag * (rsi_period - 1) + gain[i]) / rsi_period
            al = (al * (rsi_period - 1) + loss[i]) / rsi_period
            rsi[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    # ── MACD (12 / 26 / 9) ──────────────────────────────────────────
    ema12 = _ema_np(close, 12)
    ema26 = _ema_np(close, 26)
    macd_line = ema12 - ema26
    macd_sig = _ema_of_valid(macd_line, 9)

    # Crossover detection
    macd_bull = np.zeros(n, dtype=bool)
    macd_bear = np.zeros(n, dtype=bool)
    valid_macd = (~np.isnan(macd_line)) & (~np.isnan(macd_sig))
    for i in range(1, n):
        if valid_macd[i] and valid_macd[i - 1]:
            if macd_line[i - 1] <= macd_sig[i - 1] and macd_line[i] > macd_sig[i]:
                macd_bull[i] = True
            if macd_line[i - 1] >= macd_sig[i - 1] and macd_line[i] < macd_sig[i]:
                macd_bear[i] = True

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────
    s_close = pd.Series(close)
    bb_mean = s_close.rolling(20).mean().values
    bb_std = s_close.rolling(20).std(ddof=0).values
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std

    # ── SMA fast / slow ──────────────────────────────────────────────
    sma_fast = s_close.rolling(sma_fast_p).mean().values if n >= sma_fast_p else np.full(n, np.nan)
    sma_slow = s_close.rolling(sma_slow_p).mean().values if n >= sma_slow_p else np.full(n, np.nan)

    # ── Volume RVOL ──────────────────────────────────────────────────
    if "Volume" in df.columns:
        volume = df["Volume"].values.astype(np.float64)
        if volume.ndim > 1:
            volume = volume[:, 0]
        vol_avg = pd.Series(volume).rolling(20).mean().values
        rvol = np.where((vol_avg > 0) & (~np.isnan(vol_avg)), volume / vol_avg, 0.0)
    else:
        rvol = np.zeros(n)

    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    if high.ndim > 1:
        high = high[:, 0]
    if low.ndim > 1:
        low = low[:, 0]

    return {
        "close": close,
        "high": high,
        "low": low,
        "rsi": rsi,
        "macd_bull": macd_bull,
        "macd_bear": macd_bear,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rvol": rvol,
    }


def _precompute_signals(ind: dict[str, np.ndarray], params: dict) -> np.ndarray:
    """Convert pre-computed indicators to signal array (1=BUY, -1=SELL, 0=HOLD)."""
    n = len(ind["close"])
    rsi_os = params.get("rsi_oversold", 30)
    rsi_ob = params.get("rsi_overbought", 70)
    trend_req = params.get("require_trend_alignment", False)
    vol_req = params.get("require_volume_confirmation", False)

    rsi = ind["rsi"]
    close = ind["close"]
    bb_lower = ind["bb_lower"]
    bb_upper = ind["bb_upper"]
    sma_fast = ind["sma_fast"]
    sma_slow = ind["sma_slow"]
    rvol = ind["rvol"]

    valid_rsi = ~np.isnan(rsi)
    valid_bbl = ~np.isnan(bb_lower)
    valid_bbu = ~np.isnan(bb_upper)

    buy = ((valid_rsi & (rsi < rsi_os))
           | ind["macd_bull"]
           | (valid_bbl & (close < bb_lower)))
    sell = ((valid_rsi & (rsi > rsi_ob))
            | ind["macd_bear"]
            | (valid_bbu & (close > bb_upper)))

    if trend_req:
        both_valid = (~np.isnan(sma_fast)) & (~np.isnan(sma_slow))
        buy = buy & ~(both_valid & (sma_fast <= sma_slow))
        sell = sell & ~(both_valid & (sma_fast >= sma_slow))

    if vol_req:
        vol_ok = rvol > 1.5
        buy = buy & vol_ok
        sell = sell & vol_ok

    signals = np.zeros(n, dtype=np.int8)
    signals[sell] = -1
    signals[buy] = 1   # BUY takes priority when both are true
    return signals


# ── Data download with fallbacks ─────────────────────────────────────

_CRYPTO_SYMBOLS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX"}


def _download_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV via Alpaca (US stocks), Binance (crypto), or yfinance (XETRA)."""
    warmup_start = pd.Timestamp(start_date) - pd.Timedelta(days=300)

    # Crypto tickers -> Binance klines
    if ticker.upper() in _CRYPTO_SYMBOLS:
        return _download_binance(ticker.upper(), warmup_start, pd.Timestamp(end_date))

    # German tickers: use yfinance as Alpaca does not cover XETRA
    from config.settings import is_german_ticker
    if is_german_ticker(ticker):
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

    # US stocks: use Alpaca
    try:
        from data.alpaca_data import AlpacaDataClient
        alpaca = AlpacaDataClient()
        df = alpaca.get_bars(
            ticker,
            "1Day",
            limit=500,
            start=warmup_start.strftime("%Y-%m-%d"),
            end=end_date,
        )
        return df
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Alpaca backtest download failed for %s: %s — falling back to yfinance", ticker, exc
        )
        import yfinance as yf
        df = yf.download(
            ticker,
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

    # -- Data --
    if ohlcv is not None:
        df = ohlcv
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    else:
        df = _download_ohlcv(ticker, start_date, end_date)

    if df.empty or len(df) < 50:
        return _empty_result(account_balance)

    # Locate the simulation start index (first row >= start_date)
    sim_mask = df.index >= pd.Timestamp(start_date)
    if not sim_mask.any():
        return _empty_result(account_balance)
    sim_start = int(np.argmax(sim_mask))

    # Need at least 26 rows of warm-up before simulation start
    if sim_start < 26:
        return _empty_result(account_balance)

    n_total = len(df)
    n_sim = n_total - sim_start

    # -- Pre-compute indicators & signals once for the full series --
    indicators = _precompute_indicators(df, params)
    if use_technical:
        tech_signals = _precompute_signals(indicators, params)
    else:
        tech_signals = np.zeros(n_total, dtype=np.int8)

    # -- Pre-generate all sentiment scores --
    rng = np.random.RandomState(seed)
    sent_scores = rng.normal(0, 0.3, size=n_sim)

    # -- Extract arrays for the tight loop --
    close = indicators["close"]
    high = indicators["high"]
    low = indicators["low"]

    # -- Simulation (scalar state, numpy reads) --
    cash = account_balance
    pos_shares = 0
    pos_entry = 0.0
    pos_stop = 0.0
    pos_tp = 0.0
    pos_dir = 0        # 0=flat, 1=long, -1=short
    eq = np.empty(n_sim, dtype=np.float64)
    trade_pnls: list[float] = []
    trade_exits: list[str] = []

    for i in range(n_sim):
        idx = sim_start + i
        price = close[idx]

        # ── Stop-loss / take-profit ──────────────────────────────────
        if pos_dir != 0:
            hi = high[idx]
            lo = low[idx]
            if pos_dir == 1:
                if lo <= pos_stop:
                    pnl = (pos_stop - pos_entry) * pos_shares
                    cash += pos_shares * pos_stop
                    trade_pnls.append(pnl)
                    trade_exits.append("stop_loss")
                    pos_dir = 0
                elif hi >= pos_tp:
                    pnl = (pos_tp - pos_entry) * pos_shares
                    cash += pos_shares * pos_tp
                    trade_pnls.append(pnl)
                    trade_exits.append("take_profit")
                    pos_dir = 0
            else:  # short
                if hi >= pos_stop:
                    pnl = (pos_entry - pos_stop) * pos_shares
                    cash += pos_shares * (2.0 * pos_entry - pos_stop)
                    trade_pnls.append(pnl)
                    trade_exits.append("stop_loss")
                    pos_dir = 0
                elif lo <= pos_tp:
                    pnl = (pos_entry - pos_tp) * pos_shares
                    cash += pos_shares * (2.0 * pos_entry - pos_tp)
                    trade_pnls.append(pnl)
                    trade_exits.append("take_profit")
                    pos_dir = 0

        # ── Signal fusion ────────────────────────────────────────────
        t = int(tech_signals[idx])

        if use_sentiment:
            s = sent_scores[i]
            sent = 1 if s >= buy_thresh else (-1 if s <= sell_thresh else 0)
        else:
            sent = 0

        combined = 0
        if use_sentiment and use_technical:
            if sent == t and t != 0:
                combined = t
        elif use_technical:
            combined = t
        elif use_sentiment:
            combined = sent

        # ── Entry ────────────────────────────────────────────────────
        if combined != 0 and pos_dir == 0 and cash > 0:
            risk_budget = cash * 0.02 / stop_pct
            position_budget = min(cash * 0.10, risk_budget)
            shares = int(position_budget / price)
            if shares > 0:
                cash -= shares * price
                pos_shares = shares
                pos_entry = price
                if combined == 1:
                    pos_stop = round(price * (1.0 - stop_pct), 2)
                    pos_tp = round(price * (1.0 + stop_pct * tp_ratio), 2)
                    pos_dir = 1
                else:
                    pos_stop = round(price * (1.0 + stop_pct), 2)
                    pos_tp = round(price * (1.0 - stop_pct * tp_ratio), 2)
                    pos_dir = -1

        # ── Mark-to-market ───────────────────────────────────────────
        if pos_dir == 1:
            eq[i] = cash + pos_shares * price
        elif pos_dir == -1:
            eq[i] = cash + pos_shares * (2.0 * pos_entry - price)
        else:
            eq[i] = cash

    # Close remaining position at last price
    if pos_dir != 0:
        last_price = close[-1]
        if pos_dir == 1:
            pnl = (last_price - pos_entry) * pos_shares
        else:
            pnl = (pos_entry - last_price) * pos_shares
        trade_pnls.append(pnl)
        trade_exits.append("end_of_period")

    # -- Metrics --
    equity_curve = eq.tolist()
    trades = [{"pnl": p, "exit": e} for p, e in zip(trade_pnls, trade_exits)]
    wins = sum(1 for p in trade_pnls if p > 0)
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
