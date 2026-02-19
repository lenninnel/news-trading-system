"""
Backtesting engine for the News Trading System.

Simulates the full daily pipeline over historical price data:
  - Technical indicators computed from actual OHLCV history (no lookahead bias)
  - Sentiment mocked with reproducible random signals (or fixed mode)
  - Risk sizing via the same Kelly Criterion logic as the live system
  - Position management: one position at a time, stop-loss and take-profit
    checked at each day's close

Metrics
-------
  Total return (%)     vs buy-and-hold benchmark
  Sharpe ratio         annualised, zero risk-free rate
  Max drawdown (%)     peak-to-trough over the full equity curve
  Win rate (%)         winning closed trades / all closed trades
  Avg win / avg loss   mean P&L on winners and losers
  Trade count          number of completed round-trips

Visualisations (plotly, single HTML file)
-----------------------------------------
  Row 1 — Equity curve (strategy vs buy-and-hold) + trade entry markers
  Row 2 — Drawdown (%)
  Row 3 — Monthly returns heatmap

Usage
-----
  python3 backtest/engine.py --ticker AAPL --start 2024-01-01 --end 2025-01-01
  python3 backtest/engine.py --ticker NVDA --start 2024-01-01 --end 2025-01-01 \\
      --balance 25000 --sentiment bullish --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta
import yfinance as yf
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.coordinator import Coordinator  # noqa: E402
from storage.database import Database             # noqa: E402

# ── Risk-sizing constants (mirror agents/risk_agent.py) ───────────────────────
_MIN_CONFIDENCE         = 30.0
_MAX_PORTFOLIO_FRACTION = 0.10
_MAX_RISK_FRACTION      = 0.02
_TRANSACTION_COST       = 0.001
_REWARD_RISK_RATIO      = 2.0
_WIN_PROB_BASE          = 0.50
_WIN_PROB_RANGE         = 0.30
_STOP_PCT               = {"STRONG": 0.02, "WEAK": 0.01}


def _kelly(confidence: float) -> float:
    p = _WIN_PROB_BASE + (confidence / 100.0) * _WIN_PROB_RANGE
    q = 1.0 - p
    return max(0.0, (p * _REWARD_RISK_RATIO - q) / _REWARD_RISK_RATIO / 2.0)


def _parse_signal(signal: str) -> tuple[str, str]:
    s = signal.upper().strip()
    if s == "STRONG BUY":   return "BUY",  "STRONG"
    if s == "WEAK BUY":     return "BUY",  "WEAK"
    if s == "BUY":          return "BUY",  "STRONG"
    if s == "STRONG SELL":  return "SELL", "STRONG"
    if s == "WEAK SELL":    return "SELL", "WEAK"
    if s == "SELL":         return "SELL", "STRONG"
    return "HOLD", "NONE"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    entry_price: float
    shares:      int
    stop_loss:   float
    take_profit: float
    signal:      str
    confidence:  float
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: str                    = ""
    pnl:         float                  = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.shares
        return (self.pnl / cost * 100) if cost else 0.0


@dataclass
class BacktestResult:
    ticker:                  str
    start_date:              str
    end_date:                str
    initial_balance:         float
    final_balance:           float
    total_return_pct:        float
    buy_and_hold_return_pct: float
    sharpe_ratio:            float
    max_drawdown_pct:        float
    win_rate_pct:            float
    avg_win:                 float
    avg_loss:                float
    total_trades:            int
    winning_trades:          int
    losing_trades:           int
    equity_curve:            pd.Series = field(repr=False)
    buy_and_hold_curve:      pd.Series = field(repr=False)
    drawdown_series:         pd.Series = field(repr=False)
    trades:                  list[Trade] = field(repr=False, default_factory=list)
    avg_hold_days:           float         = 0.0
    db_id:                   Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Simulates the trading pipeline over historical price data day-by-day.

    Args:
        ticker:          Stock ticker symbol (e.g. "AAPL").
        start_date:      Inclusive start date  "YYYY-MM-DD".
        end_date:        Inclusive end date    "YYYY-MM-DD".
        initial_balance: Starting cash in USD.
        sentiment_mode:  "random"  — per-date seeded random signal (default)
                         "bullish" — every day is BUY sentiment
                         "bearish" — every day is SELL sentiment
                         "neutral" — every day is HOLD sentiment
        verbose:         Print progress to stdout.
    """

    _WARMUP_DAYS = 100   # extra history before start_date for indicator warm-up

    def __init__(
        self,
        ticker:          str,
        start_date:      str,
        end_date:        str,
        initial_balance: float = 10_000.0,
        sentiment_mode:  str   = "random",
        verbose:         bool  = True,
    ) -> None:
        self.ticker          = ticker.upper()
        self.start_ts        = pd.Timestamp(start_date)
        self.end_ts          = pd.Timestamp(end_date)
        self.initial_balance = initial_balance
        self.sentiment_mode  = sentiment_mode
        self.verbose         = verbose
        self._db             = Database()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the full backtest simulation and return a BacktestResult."""
        self._log(f"\n{'=' * 62}")
        self._log(f"  Backtest : {self.ticker}  "
                  f"{self.start_ts.date()} → {self.end_ts.date()}")
        self._log(f"  Balance  : ${self.initial_balance:,.2f}  |  "
                  f"Sentiment mode: {self.sentiment_mode}")
        self._log(f"{'=' * 62}")

        # 1. Download with warm-up window
        df = self._download_data()

        # 2. Compute all indicators on the full series (no lookahead)
        ind_df = self._compute_indicators(df)

        # 3. Slice to requested backtest range
        bt_df = df.loc[self.start_ts : self.end_ts]
        if bt_df.empty:
            raise ValueError(
                f"No trading data for {self.ticker} in range "
                f"{self.start_ts.date()} – {self.end_ts.date()}"
            )
        self._log(f"  Trading days : {len(bt_df)}")

        # 4. Simulate
        equity_curve, trades, final_cash = self._run_loop(bt_df, ind_df)

        # 5. Compute metrics
        result = self._compute_metrics(equity_curve, trades, final_cash, bt_df)

        # 6. Persist
        result.db_id = self._save_result(result)

        if self.verbose:
            self._print_summary(result)

        return result

    def plot(
        self,
        result:    BacktestResult,
        show:      bool             = True,
        save_path: Optional[str]    = None,
    ) -> go.Figure:
        """
        Render equity curve, drawdown, and monthly returns heatmap.

        Args:
            result:    BacktestResult from run().
            show:      Call fig.show() to open in browser.
            save_path: If set, write the figure to this HTML path.

        Returns:
            The combined plotly Figure.
        """
        eq = result.equity_curve
        bh = result.buy_and_hold_curve
        dd = result.drawdown_series

        monthly_z, month_labels, year_labels = self._monthly_matrix(eq)

        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.42, 0.25, 0.33],
            vertical_spacing=0.09,
            subplot_titles=[
                f"{self.ticker} — Equity Curve vs Buy & Hold",
                "Drawdown (%)",
                "Monthly Returns (%)",
            ],
        )

        # ── Row 1: equity curve ───────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name="Strategy",
            line=dict(color="#4fc3f7", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=bh.index, y=bh.values,
            name="Buy & Hold",
            line=dict(color="#90a4ae", width=1.5, dash="dash"),
        ), row=1, col=1)

        # Trade entry/exit markers
        for t in result.trades:
            if t.entry_date in eq.index:
                is_buy = "BUY" in t.signal
                fig.add_trace(go.Scatter(
                    x=[t.entry_date],
                    y=[float(eq.get(t.entry_date, np.nan))],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if is_buy else "triangle-down",
                        color="#00e676" if is_buy else "#ff1744",
                        size=11,
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>ENTRY {'BUY' if is_buy else 'SELL'}</b><br>"
                        f"Date: {t.entry_date.date()}<br>"
                        f"Price: ${t.entry_price:.2f}<br>"
                        f"Signal: {t.signal}<br>"
                        f"Conf: {t.confidence:.0f}%<br>"
                        f"Shares: {t.shares}<extra></extra>"
                    ),
                ), row=1, col=1)

        # ── Row 2: drawdown ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ef5350", width=1),
            fillcolor="rgba(239,83,80,0.20)",
            showlegend=False,
        ), row=2, col=1)

        # Max drawdown annotation
        if not dd.empty:
            min_idx = dd.idxmin()
            fig.add_annotation(
                x=min_idx, y=float(dd.min()),
                text=f"Max DD: {dd.min():.1f}%",
                showarrow=True, arrowhead=2,
                font=dict(color="#ef5350", size=11),
                row=2, col=1,
            )

        # ── Row 3: monthly heatmap ────────────────────────────────────────────
        if monthly_z:
            fig.add_trace(go.Heatmap(
                z=monthly_z,
                x=month_labels,
                y=year_labels,
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:+.1f}%" if v is not None else "—" for v in row]
                      for row in monthly_z],
                texttemplate="%{text}",
                textfont=dict(size=11),
                showscale=True,
                colorbar=dict(thickness=12, len=0.30, y=0.13, title="%"),
            ), row=3, col=1)

        alpha = result.total_return_pct - result.buy_and_hold_return_pct
        fig.update_layout(
            height=950,
            template="plotly_dark",
            margin=dict(l=65, r=40, t=80, b=40),
            legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
            title=dict(
                text=(
                    f"{self.ticker} Backtest  |  "
                    f"{result.start_date} → {result.end_date}  |  "
                    f"Return: <b>{result.total_return_pct:+.1f}%</b>  "
                    f"B&H: {result.buy_and_hold_return_pct:+.1f}%  "
                    f"Alpha: {alpha:+.1f}%  |  "
                    f"Sharpe: {result.sharpe_ratio:.2f}  "
                    f"MaxDD: {result.max_drawdown_pct:.1f}%  "
                    f"WR: {result.win_rate_pct:.0f}%"
                ),
                font=dict(size=13),
            ),
        )
        fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        if save_path:
            fig.write_html(save_path)
            self._log(f"  Chart saved → {save_path}")

        if show:
            fig.show()

        return fig

    # ── Private: data ──────────────────────────────────────────────────────────

    def _download_data(self, warmup_days: Optional[int] = None) -> pd.DataFrame:
        days         = warmup_days if warmup_days is not None else self._WARMUP_DAYS
        warmup_start = self.start_ts - pd.Timedelta(days=days)
        self._log(
            f"  Downloading {self.ticker} "
            f"({warmup_start.date()} → {self.end_ts.date()})..."
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
        self._log(f"  Got {len(df)} rows (incl. {self._WARMUP_DAYS}-day warm-up).")
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute all indicators for the full series. O(n) — done once."""
        close = df["Close"].squeeze()

        rsi        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd_obj   = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        macd_line  = macd_obj.macd()
        macd_sig   = macd_obj.macd_signal()
        macd_bull  = (macd_line.shift(1) <= macd_sig.shift(1)) & (macd_line > macd_sig)
        macd_bear  = (macd_line.shift(1) >= macd_sig.shift(1)) & (macd_line < macd_sig)
        sma20      = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50      = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        bb         = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

        return pd.DataFrame({
            "close":     close,
            "rsi":       rsi,
            "macd_bull": macd_bull.astype(bool),
            "macd_bear": macd_bear.astype(bool),
            "sma_20":    sma20,
            "sma_50":    sma50,
            "bb_upper":  bb.bollinger_hband(),
            "bb_lower":  bb.bollinger_lband(),
        }, index=df.index)

    # ── Private: signal helpers ────────────────────────────────────────────────

    def _tech_signal(self, row: pd.Series) -> str:
        rsi   = float(row["rsi"])   if pd.notna(row["rsi"])   else None
        price = float(row["close"]) if pd.notna(row["close"]) else None
        bb_lo = float(row["bb_lower"]) if pd.notna(row["bb_lower"]) else None
        bb_hi = float(row["bb_upper"]) if pd.notna(row["bb_upper"]) else None

        buy = bool(
            (rsi is not None and rsi < 30) or
            bool(row["macd_bull"]) or
            (price is not None and bb_lo is not None and price < bb_lo)
        )
        sell = bool(
            (rsi is not None and rsi > 70) or
            bool(row["macd_bear"]) or
            (price is not None and bb_hi is not None and price > bb_hi)
        )
        if buy:  return "BUY"
        if sell: return "SELL"
        return "HOLD"

    def _mock_sentiment(self, date: pd.Timestamp) -> tuple[str, float]:
        """Return (signal, avg_score) for a trading day — no API calls."""
        if self.sentiment_mode == "bullish": return "BUY",  0.70
        if self.sentiment_mode == "bearish": return "SELL", -0.70
        if self.sentiment_mode == "neutral": return "HOLD",  0.00

        # "random" — seeded by date + ticker so results are reproducible
        seed = int(date.strftime("%Y%m%d")) + abs(hash(self.ticker)) % 100_000
        r    = np.random.default_rng(seed).random()
        if r < 0.40: return "BUY",  0.65
        if r < 0.65: return "SELL", -0.65
        return "HOLD", 0.00

    def _size_position(
        self,
        signal:         str,
        confidence:     float,
        price:          float,
        available_cash: float,
    ) -> dict:
        """Kelly-based position sizing — no DB writes (mirrors RiskAgent)."""
        direction, strength = _parse_signal(signal)
        if confidence < _MIN_CONFIDENCE or direction == "HOLD" or price <= 0:
            return {"shares": 0, "stop_loss": None, "take_profit": None}

        stop_pct = _STOP_PCT[strength]
        kelly    = _kelly(confidence)

        raw_pos = min(
            available_cash * kelly,
            available_cash * _MAX_PORTFOLIO_FRACTION,
            (available_cash * _MAX_RISK_FRACTION) / stop_pct,
        )
        shares = int(raw_pos * (1.0 - _TRANSACTION_COST) / price)

        if shares == 0 or shares * price > available_cash:
            return {"shares": 0, "stop_loss": None, "take_profit": None}

        stop_loss   = round(price * (1.0 - stop_pct), 4)
        take_profit = round(price * (1.0 + stop_pct * _REWARD_RISK_RATIO), 4)
        return {"shares": shares, "stop_loss": stop_loss, "take_profit": take_profit}

    # ── Private: simulation loop ───────────────────────────────────────────────

    def _run_loop(
        self,
        bt_df:  pd.DataFrame,
        ind_df: pd.DataFrame,
    ) -> tuple[dict, list[Trade], float]:
        cash:         float            = self.initial_balance
        shares:       int              = 0
        position:     Optional[dict]   = None
        equity_curve: dict             = {}
        trades:       list[Trade]      = []

        for date in bt_df.index:
            close = float(bt_df.loc[date, "Close"])

            # ── 1. Stop-loss / take-profit check ──────────────────────────────
            if position is not None:
                hit_sl = close <= position["stop_loss"]
                hit_tp = close >= position["take_profit"]
                if hit_sl or hit_tp:
                    exit_px = position["stop_loss"] if hit_sl else position["take_profit"]
                    pnl     = (exit_px - position["entry_price"]) * position["shares"]
                    cash   += position["shares"] * exit_px
                    t       = position["trade"]
                    t.exit_date   = date
                    t.exit_price  = round(exit_px, 4)
                    t.exit_reason = "stop_loss" if hit_sl else "take_profit"
                    t.pnl         = round(pnl, 2)
                    trades.append(t)
                    shares   = 0
                    position = None

            # ── 2. Compute today's signal ──────────────────────────────────────
            if date in ind_df.index:
                row      = ind_df.loc[date]
                tech     = self._tech_signal(row)
                sent, sc = self._mock_sentiment(date)
                combined = Coordinator.combine_signals(sent, tech)
                conf     = Coordinator.confidence(combined, sc) * 100   # → 0–100

                # ── 3. Open position ──────────────────────────────────────────
                if position is None and combined in ("STRONG BUY", "WEAK BUY"):
                    sz = self._size_position(combined, conf, close, cash)
                    if sz["shares"] > 0:
                        cash   -= sz["shares"] * close
                        shares  = sz["shares"]
                        position = {
                            "entry_price": close,
                            "shares":      sz["shares"],
                            "stop_loss":   sz["stop_loss"],
                            "take_profit": sz["take_profit"],
                            "trade": Trade(
                                entry_date  = date,
                                entry_price = close,
                                shares      = sz["shares"],
                                stop_loss   = sz["stop_loss"],
                                take_profit = sz["take_profit"],
                                signal      = combined,
                                confidence  = round(conf, 1),
                            ),
                        }

                # ── 4. Close position on opposite signal ──────────────────────
                elif position is not None and combined in ("STRONG SELL", "WEAK SELL"):
                    pnl  = (close - position["entry_price"]) * position["shares"]
                    cash += position["shares"] * close
                    t    = position["trade"]
                    t.exit_date   = date
                    t.exit_price  = close
                    t.exit_reason = "signal"
                    t.pnl         = round(pnl, 2)
                    trades.append(t)
                    shares   = 0
                    position = None

            # ── 5. Record portfolio value at close ─────────────────────────────
            equity_curve[date] = cash + shares * close

        # ── Close any open position at end of period ──────────────────────────
        if position is not None:
            last_close = float(bt_df["Close"].iloc[-1])
            last_date  = bt_df.index[-1]
            pnl  = (last_close - position["entry_price"]) * position["shares"]
            cash += position["shares"] * last_close
            t    = position["trade"]
            t.exit_date   = last_date
            t.exit_price  = last_close
            t.exit_reason = "end_of_period"
            t.pnl         = round(pnl, 2)
            trades.append(t)

        return equity_curve, trades, cash

    # ── Private: metrics ───────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        equity_curve: dict,
        trades:       list[Trade],
        final_cash:   float,
        bt_df:        pd.DataFrame,
    ) -> BacktestResult:
        eq          = pd.Series(equity_curve).sort_index()
        final_val   = float(eq.iloc[-1]) if not eq.empty else final_cash
        total_ret   = (final_val / self.initial_balance - 1.0) * 100

        # Sharpe ratio (annualised, risk-free = 0)
        daily_ret = eq.pct_change().dropna()
        sharpe    = (
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
        )

        # Drawdown
        rolling_max = eq.cummax()
        drawdown    = (eq - rolling_max) / rolling_max * 100
        max_dd      = float(drawdown.min())

        # Trade stats
        closed   = [t for t in trades if t.exit_price is not None]
        winners  = [t for t in closed if t.pnl > 0]
        losers   = [t for t in closed if t.pnl < 0]
        win_rate = len(winners) / len(closed) * 100 if closed else 0.0
        avg_win  = float(np.mean([t.pnl for t in winners])) if winners else 0.0
        avg_loss = float(np.mean([t.pnl for t in losers]))  if losers  else 0.0

        # Average holding period (calendar days)
        hold_days = [
            (t.exit_date - t.entry_date).days
            for t in closed
            if t.exit_date is not None
        ]
        avg_hold = float(np.mean(hold_days)) if hold_days else 0.0

        # Buy-and-hold benchmark (normalised to initial_balance)
        bh_prices = bt_df["Close"].squeeze()
        bh_curve  = bh_prices / float(bh_prices.iloc[0]) * self.initial_balance
        bh_ret    = (float(bh_prices.iloc[-1]) / float(bh_prices.iloc[0]) - 1.0) * 100

        return BacktestResult(
            ticker                  = self.ticker,
            start_date              = self.start_ts.strftime("%Y-%m-%d"),
            end_date                = self.end_ts.strftime("%Y-%m-%d"),
            initial_balance         = self.initial_balance,
            final_balance           = round(final_val, 2),
            total_return_pct        = round(total_ret, 2),
            buy_and_hold_return_pct = round(bh_ret, 2),
            sharpe_ratio            = round(sharpe, 3),
            max_drawdown_pct        = round(max_dd, 2),
            win_rate_pct            = round(win_rate, 1),
            avg_win                 = round(avg_win, 2),
            avg_loss                = round(avg_loss, 2),
            total_trades            = len(closed),
            winning_trades          = len(winners),
            losing_trades           = len(losers),
            equity_curve            = eq,
            buy_and_hold_curve      = bh_curve,
            drawdown_series         = drawdown,
            trades                  = trades,
            avg_hold_days           = round(avg_hold, 1),
        )

    # ── Strategy comparison ────────────────────────────────────────────────────

    # Extra warmup days needed to calculate SMA-200 (≈ 200 trading days ≈ 280 cal days)
    _STRATEGY_WARMUP_DAYS = 310

    def compare_strategies(
        self,
        strategies: Optional[list] = None,
    ) -> dict:
        """
        Run all three strategy agents on the same ticker/date range and compare.

        Each strategy uses its own indicator-based signal logic (same logic as
        the live agents) applied day-by-day to historical OHLCV data.

        Args:
            strategies: Subset to compare. Defaults to all three:
                        ["momentum", "mean_reversion", "swing"].

        Returns:
            dict with keys:
                ticker          (str)
                start_date      (str)
                end_date        (str)
                strategies      (dict[str, BacktestResult])
                buy_and_hold    (dict with curve/return_pct/sharpe)
                comparison_df   (pd.DataFrame)  — metrics side-by-side
                winner          (str)            — strategy with best Sharpe
        """
        if strategies is None:
            strategies = ["momentum", "mean_reversion", "swing"]

        self._log(f"\n{'=' * 66}")
        self._log(f"  Strategy Comparison : {self.ticker}  "
                  f"{self.start_ts.date()} → {self.end_ts.date()}")
        self._log(f"  Strategies : {', '.join(strategies)}")
        self._log(f"{'=' * 66}")

        # Download with extra warmup for SMA-200
        df = self._download_data(warmup_days=self._STRATEGY_WARMUP_DAYS)

        # Pre-compute all indicators once (shared across strategies)
        ind_df = self._compute_all_indicators(df)

        # Slice to requested backtest range
        bt_df = df.loc[self.start_ts : self.end_ts]
        if bt_df.empty:
            raise ValueError(
                f"No trading data for {self.ticker} in range "
                f"{self.start_ts.date()} – {self.end_ts.date()}"
            )
        self._log(f"  Trading days : {len(bt_df)}")

        # Buy-and-hold benchmark
        bh_prices  = bt_df["Close"].squeeze()
        bh_curve   = bh_prices / float(bh_prices.iloc[0]) * self.initial_balance
        bh_ret     = (float(bh_prices.iloc[-1]) / float(bh_prices.iloc[0]) - 1.0) * 100
        bh_daily   = bh_curve.pct_change().dropna()
        bh_sharpe  = (
            float(bh_daily.mean() / bh_daily.std() * np.sqrt(252))
            if len(bh_daily) > 1 and bh_daily.std() > 0 else 0.0
        )
        bh_roll_max = bh_curve.cummax()
        bh_dd_min   = float(((bh_curve - bh_roll_max) / bh_roll_max * 100).min())

        signal_fns = {
            "momentum":       self._signal_momentum,
            "mean_reversion": self._signal_mean_reversion,
            "swing":          self._signal_swing,
        }

        results: dict = {}
        for strat in strategies:
            fn = signal_fns.get(strat)
            if fn is None:
                self._log(f"  [!] Unknown strategy {strat!r} — skipped.")
                continue
            self._log(f"\n  ── {strat.replace('_', ' ').title()} ──")
            eq, trades, final_cash = self._run_strategy_loop(bt_df, ind_df, fn)
            result = self._compute_metrics(eq, trades, final_cash, bt_df)
            # Persist to DB
            self._db.log_strategy_comparison(
                ticker        = self.ticker,
                start_date    = self.start_ts.strftime("%Y-%m-%d"),
                end_date      = self.end_ts.strftime("%Y-%m-%d"),
                strategy      = strat,
                total_return  = result.total_return_pct,
                sharpe        = result.sharpe_ratio,
                max_dd        = result.max_drawdown_pct,
                win_rate      = result.win_rate_pct,
                trade_count   = result.total_trades,
                avg_hold_days = result.avg_hold_days,
            )
            results[strat] = result
            if self.verbose:
                self._log(
                    f"    Return: {result.total_return_pct:+.2f}%  "
                    f"Sharpe: {result.sharpe_ratio:.2f}  "
                    f"MaxDD: {result.max_drawdown_pct:.1f}%  "
                    f"WR: {result.win_rate_pct:.0f}%  "
                    f"Trades: {result.total_trades}  "
                    f"Avg hold: {result.avg_hold_days:.1f}d"
                )

        # Persist buy-and-hold row
        self._db.log_strategy_comparison(
            ticker        = self.ticker,
            start_date    = self.start_ts.strftime("%Y-%m-%d"),
            end_date      = self.end_ts.strftime("%Y-%m-%d"),
            strategy      = "buy_and_hold",
            total_return  = round(bh_ret, 2),
            sharpe        = round(bh_sharpe, 3),
            max_dd        = round(bh_dd_min, 2),
            win_rate      = 100.0,
            trade_count   = 1,
            avg_hold_days = float(len(bt_df)),
        )

        # Determine winner by Sharpe ratio
        winner = max(results, key=lambda s: results[s].sharpe_ratio) if results else "—"

        # Build comparison DataFrame
        _LABELS = {
            "momentum":       "Momentum",
            "mean_reversion": "Mean Rev.",
            "swing":          "Swing",
        }
        rows = []
        for strat in strategies:
            if strat not in results:
                continue
            r = results[strat]
            rows.append({
                "Strategy":      _LABELS.get(strat, strat),
                "Return (%)":    r.total_return_pct,
                "Sharpe":        r.sharpe_ratio,
                "Max DD (%)":    r.max_drawdown_pct,
                "Win Rate (%)":  r.win_rate_pct,
                "Trades":        r.total_trades,
                "Avg Hold (d)":  r.avg_hold_days,
            })
        rows.append({
            "Strategy":      "Buy & Hold",
            "Return (%)":    round(bh_ret, 2),
            "Sharpe":        round(bh_sharpe, 3),
            "Max DD (%)":    round(bh_dd_min, 2),
            "Win Rate (%)":  100.0,
            "Trades":        1,
            "Avg Hold (d)":  float(len(bt_df)),
        })
        comparison_df = pd.DataFrame(rows)

        return {
            "ticker":       self.ticker,
            "start_date":   self.start_ts.strftime("%Y-%m-%d"),
            "end_date":     self.end_ts.strftime("%Y-%m-%d"),
            "strategies":   results,
            "buy_and_hold": {
                "curve":      bh_curve,
                "return_pct": round(bh_ret, 2),
                "sharpe":     round(bh_sharpe, 3),
                "max_dd":     round(bh_dd_min, 2),
            },
            "comparison_df": comparison_df,
            "winner":        winner,
        }

    def plot_comparison(
        self,
        comparison: dict,
        show:       bool           = True,
        save_path:  Optional[str]  = None,
    ) -> go.Figure:
        """
        Render combined equity curves (one line per strategy + buy-and-hold).

        Args:
            comparison: Return value of compare_strategies().
            show:       Call fig.show() to open in browser.
            save_path:  If set, save interactive HTML to this path.

        Returns:
            The plotly Figure.
        """
        _COLORS = {
            "momentum":       "#4fc3f7",
            "mean_reversion": "#a5d6a7",
            "swing":          "#ffcc80",
            "buy_and_hold":   "#90a4ae",
        }
        _LABELS = {
            "momentum":       "Momentum",
            "mean_reversion": "Mean Reversion",
            "swing":          "Swing",
            "buy_and_hold":   "Buy & Hold",
        }

        winner    = comparison["winner"]
        ticker    = comparison["ticker"]
        start     = comparison["start_date"]
        end       = comparison["end_date"]
        results   = comparison["strategies"]
        bh_curve  = comparison["buy_and_hold"]["curve"]
        bh_ret    = comparison["buy_and_hold"]["return_pct"]

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.10,
            subplot_titles=[
                f"{ticker} — Strategy Comparison: Equity Curves",
                "Drawdown (%)",
            ],
        )

        # Equity curves
        for strat, result in results.items():
            eq = result.equity_curve
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                name=_LABELS.get(strat, strat)
                     + (" ★" if strat == winner else ""),
                line=dict(color=_COLORS.get(strat, "#ffffff"), width=2),
                hovertemplate=(
                    f"<b>{_LABELS.get(strat, strat)}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}<extra></extra>"
                ),
            ), row=1, col=1)

        # Buy & hold
        fig.add_trace(go.Scatter(
            x=bh_curve.index, y=bh_curve.values,
            name=f"Buy & Hold ({bh_ret:+.1f}%)",
            line=dict(color=_COLORS["buy_and_hold"], width=1.5, dash="dash"),
        ), row=1, col=1)

        # Drawdown curves
        for strat, result in results.items():
            fig.add_trace(go.Scatter(
                x=result.drawdown_series.index,
                y=result.drawdown_series.values,
                name=_LABELS.get(strat, strat),
                line=dict(color=_COLORS.get(strat, "#ffffff"), width=1),
                showlegend=False,
            ), row=2, col=1)

        winner_result = results.get(winner)
        title_suffix  = (
            f"Winner: {_LABELS.get(winner, winner)} "
            f"(Sharpe {winner_result.sharpe_ratio:.2f})"
            if winner_result else ""
        )

        fig.update_layout(
            height=820,
            template="plotly_dark",
            margin=dict(l=65, r=40, t=80, b=40),
            legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
            title=dict(
                text=(
                    f"{ticker} Strategy Comparison  |  {start} → {end}  |  "
                    f"{title_suffix}"
                ),
                font=dict(size=13),
            ),
        )
        fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        if save_path:
            fig.write_html(save_path)
            self._log(f"  Chart saved → {save_path}")
        if show:
            fig.show()
        return fig

    # ── Private: all-strategy indicators ──────────────────────────────────────

    def _compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-compute every indicator needed by all three strategy signal functions.
        Done once and shared across strategies for efficiency.
        """
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # ── Momentum indicators ───────────────────────────────────────────────
        ema20   = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        ema50   = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        adx     = adx_obj.adx()
        adx_pos = adx_obj.adx_pos()
        adx_neg = adx_obj.adx_neg()
        vol_avg   = volume.rolling(20).mean()
        vol_ratio = volume / vol_avg.replace(0, np.nan)
        # shift(1) avoids same-bar look-ahead (mirrors momentum_agent iloc[-2])
        high_20 = high.rolling(20).max().shift(1)
        low_20  = low.rolling(20).min().shift(1)
        roc     = ta.momentum.ROCIndicator(close=close, window=10).roc()

        # ── Mean-reversion indicators ──────────────────────────────────────────
        rsi      = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        stoch    = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        stoch_k  = stoch.stoch()
        bb_obj   = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_pct_b = bb_obj.bollinger_pband()
        wr       = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14
        ).williams_r()
        is_green = (close > close.shift(1)).astype(float)

        # ── Swing indicators ──────────────────────────────────────────────────
        sma20  = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        macd_obj  = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        macd_line = macd_obj.macd()
        macd_sig  = macd_obj.macd_signal()
        macd_hist = macd_obj.macd_diff()
        hist_up   = ((macd_hist > macd_hist.shift(1)) & (macd_hist > 0)).astype(float)
        hist_down = ((macd_hist < macd_hist.shift(1)) & (macd_hist < 0)).astype(float)
        # shift(1) avoids look-ahead (mirrors swing_agent iloc[-2])
        pivot_high = high.rolling(5).max().shift(1)
        pivot_low  = low.rolling(5).min().shift(1)
        atr_val    = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        atr_pct = atr_val / close.replace(0, np.nan)
        crossed_above = ((close.shift(1) < sma20.shift(1)) & (close > sma20)).astype(float)
        crossed_below = ((close.shift(1) > sma20.shift(1)) & (close < sma20)).astype(float)

        return pd.DataFrame({
            "close":         close,
            # momentum
            "ema20":         ema20,
            "ema50":         ema50,
            "adx":           adx,
            "adx_pos":       adx_pos,
            "adx_neg":       adx_neg,
            "vol_ratio":     vol_ratio,
            "high_20":       high_20,
            "low_20":        low_20,
            "roc":           roc,
            # mean reversion
            "rsi":           rsi,
            "stoch_k":       stoch_k,
            "bb_pct_b":      bb_pct_b,
            "williams_r":    wr,
            "is_green":      is_green,
            # swing
            "sma20":         sma20,
            "sma50":         sma50,
            "sma200":        sma200,
            "macd_line":     macd_line,
            "macd_sig":      macd_sig,
            "hist_up":       hist_up,
            "hist_down":     hist_down,
            "pivot_high":    pivot_high,
            "pivot_low":     pivot_low,
            "atr_pct":       atr_pct,
            "crossed_above": crossed_above,
            "crossed_below": crossed_below,
        }, index=df.index)

    # ── Private: per-strategy signal functions ─────────────────────────────────

    @staticmethod
    def _signal_momentum(row: pd.Series) -> tuple:
        """Return (signal, confidence) using MomentumAgent logic."""
        price     = float(row["close"])  if pd.notna(row.get("close"))     else None
        ema20     = float(row["ema20"])  if pd.notna(row.get("ema20"))     else None
        ema50     = float(row["ema50"])  if pd.notna(row.get("ema50"))     else None
        adx       = float(row["adx"])    if pd.notna(row.get("adx"))       else 0.0
        adx_pos   = float(row["adx_pos"]) if pd.notna(row.get("adx_pos")) else 0.0
        adx_neg   = float(row["adx_neg"]) if pd.notna(row.get("adx_neg")) else 0.0
        vol_ratio = float(row["vol_ratio"]) if pd.notna(row.get("vol_ratio")) else 1.0
        high_20   = float(row["high_20"]) if pd.notna(row.get("high_20")) else None
        low_20    = float(row["low_20"])  if pd.notna(row.get("low_20"))   else None
        roc       = float(row["roc"])     if pd.notna(row.get("roc"))      else 0.0

        conf = 50.0
        if adx > 25:        conf += 15.0
        if adx > 35:        conf += 10.0
        if vol_ratio > 1.5: conf += 10.0
        if abs(roc) > 5.0:  conf += 15.0
        conf = min(max(conf, 30.0), 90.0)

        if (price and high_20 and price > high_20
                and adx > 25 and vol_ratio > 1.2 and adx_pos > adx_neg):
            return "BUY", conf
        if (price and ema20 and ema50
                and ema20 > ema50 and price > ema20 and roc > 3.0):
            return "BUY", conf
        if (price and low_20 and price < low_20
                and adx > 25 and vol_ratio > 1.2 and adx_neg > adx_pos):
            return "SELL", conf
        if (price and ema20 and ema50
                and ema20 < ema50 and price < ema20 and roc < -3.0):
            return "SELL", conf
        return "HOLD", 25.0

    @staticmethod
    def _signal_mean_reversion(row: pd.Series) -> tuple:
        """Return (signal, confidence) using MeanReversionAgent logic."""
        rsi      = float(row["rsi"])       if pd.notna(row.get("rsi"))       else 50.0
        stoch_k  = float(row["stoch_k"])   if pd.notna(row.get("stoch_k"))   else 50.0
        bb_pct_b = float(row["bb_pct_b"])  if pd.notna(row.get("bb_pct_b"))  else 0.5
        wr       = float(row["williams_r"]) if pd.notna(row.get("williams_r")) else -50.0
        is_green = bool(row.get("is_green", 0))

        if rsi < 30 and stoch_k < 20 and bb_pct_b < 0 and is_green:
            return "BUY", 80.0
        if rsi < 35 and (stoch_k < 25 or wr < -80):
            return "BUY", 60.0
        if rsi > 70 and stoch_k > 80 and bb_pct_b > 1:
            return "SELL", 80.0
        if rsi > 65 and (stoch_k > 75 or wr > -20):
            return "SELL", 60.0
        return "HOLD", 25.0

    @staticmethod
    def _signal_swing(row: pd.Series) -> tuple:
        """Return (signal, confidence) using SwingAgent logic."""
        price      = float(row["close"])   if pd.notna(row.get("close"))   else None
        sma20      = float(row["sma20"])   if pd.notna(row.get("sma20"))   else None
        sma50      = float(row["sma50"])   if pd.notna(row.get("sma50"))   else None
        sma200     = float(row["sma200"])  if pd.notna(row.get("sma200"))  else None
        macd_line  = float(row["macd_line"]) if pd.notna(row.get("macd_line")) else 0.0
        macd_sig   = float(row["macd_sig"])  if pd.notna(row.get("macd_sig"))  else 0.0
        hist_up    = bool(row.get("hist_up",    0))
        hist_down  = bool(row.get("hist_down",  0))
        pivot_high = float(row["pivot_high"]) if pd.notna(row.get("pivot_high")) else None
        pivot_low  = float(row["pivot_low"])  if pd.notna(row.get("pivot_low"))  else None
        atr_pct    = float(row["atr_pct"])    if pd.notna(row.get("atr_pct"))    else 0.0
        cross_up   = bool(row.get("crossed_above", 0))
        cross_down = bool(row.get("crossed_below", 0))

        buy_signal  = False
        sell_signal = False

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

        if buy_signal:
            return "BUY", conf
        return "SELL", conf

    def _run_strategy_loop(
        self,
        bt_df:     pd.DataFrame,
        ind_df:    pd.DataFrame,
        signal_fn,
    ) -> tuple:
        """
        Generic day-by-day simulation using a strategy-specific signal function.

        Args:
            bt_df:     Price DataFrame sliced to the backtest date range.
            ind_df:    Full indicator DataFrame (pre-computed from download window).
            signal_fn: Callable(row) → (signal: str, confidence: float).

        Returns:
            (equity_curve dict, trades list, final_cash float)
        """
        cash:         float          = self.initial_balance
        shares:       int            = 0
        position:     Optional[dict] = None
        equity_curve: dict           = {}
        trades:       list           = []

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
                    t       = position["trade"]
                    t.exit_date   = date
                    t.exit_price  = round(exit_px, 4)
                    t.exit_reason = "stop_loss" if hit_sl else "take_profit"
                    t.pnl         = round(pnl, 2)
                    trades.append(t)
                    shares   = 0
                    position = None

            # Compute today's signal
            if date in ind_df.index:
                row    = ind_df.loc[date]
                signal, conf = signal_fn(row)

                if position is None and signal == "BUY":
                    sz = self._size_position("BUY", conf, close, cash)
                    if sz["shares"] > 0:
                        cash    -= sz["shares"] * close
                        shares   = sz["shares"]
                        position = {
                            "entry_price": close,
                            "shares":      sz["shares"],
                            "stop_loss":   sz["stop_loss"],
                            "take_profit": sz["take_profit"],
                            "trade": Trade(
                                entry_date  = date,
                                entry_price = close,
                                shares      = sz["shares"],
                                stop_loss   = sz["stop_loss"],
                                take_profit = sz["take_profit"],
                                signal      = "BUY",
                                confidence  = round(conf, 1),
                            ),
                        }
                elif position is not None and signal == "SELL":
                    pnl  = (close - position["entry_price"]) * position["shares"]
                    cash += position["shares"] * close
                    t    = position["trade"]
                    t.exit_date   = date
                    t.exit_price  = close
                    t.exit_reason = "signal"
                    t.pnl         = round(pnl, 2)
                    trades.append(t)
                    shares   = 0
                    position = None

            equity_curve[date] = cash + shares * close

        # Close any open position at end of period
        if position is not None:
            last_close = float(bt_df["Close"].iloc[-1])
            last_date  = bt_df.index[-1]
            pnl  = (last_close - position["entry_price"]) * position["shares"]
            cash += position["shares"] * last_close
            t    = position["trade"]
            t.exit_date   = last_date
            t.exit_price  = last_close
            t.exit_reason = "end_of_period"
            t.pnl         = round(pnl, 2)
            trades.append(t)

        return equity_curve, trades, cash

    # ── Private: chart helpers ─────────────────────────────────────────────────

    def _monthly_matrix(
        self, eq: pd.Series
    ) -> tuple[list[list], list[str], list[str]]:
        """Build (z_matrix, month_labels, year_labels) for the heatmap."""
        if len(eq) < 2:
            return [], [], []

        monthly  = eq.resample("ME").last().pct_change().dropna() * 100
        monthly.index = monthly.index.to_period("M")

        years  = sorted(monthly.index.year.unique().tolist())
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        z: list[list] = []
        for yr in years:
            row = []
            for mo in range(1, 13):
                period = pd.Period(year=yr, month=mo, freq="M")
                val    = float(monthly.loc[period]) if period in monthly.index else None
                row.append(val)
            z.append(row)

        return z, months, [str(y) for y in years]

    # ── Private: persistence ───────────────────────────────────────────────────

    def _save_result(self, result: BacktestResult) -> int:
        trades_json = json.dumps([
            {
                "entry_date":  str(t.entry_date.date()),
                "exit_date":   str(t.exit_date.date()) if t.exit_date else None,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "shares":      t.shares,
                "signal":      t.signal,
                "confidence":  t.confidence,
                "pnl":         t.pnl,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ])
        return self._db.log_backtest_result(
            ticker                  = result.ticker,
            start_date              = result.start_date,
            end_date                = result.end_date,
            initial_balance         = result.initial_balance,
            final_balance           = result.final_balance,
            total_return_pct        = result.total_return_pct,
            buy_and_hold_return_pct = result.buy_and_hold_return_pct,
            sharpe_ratio            = result.sharpe_ratio,
            max_drawdown_pct        = result.max_drawdown_pct,
            win_rate_pct            = result.win_rate_pct,
            avg_win                 = result.avg_win,
            avg_loss                = result.avg_loss,
            total_trades            = result.total_trades,
            sentiment_mode          = self.sentiment_mode,
            trades_json             = trades_json,
        )

    # ── Private: output ────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _print_summary(self, result: BacktestResult) -> None:
        bar   = "=" * 62
        alpha = result.total_return_pct - result.buy_and_hold_return_pct
        print(f"\n{bar}")
        print(f"  {self.ticker}  Backtest Results  (DB id #{result.db_id})")
        print(bar)
        print(f"  Period          : {result.start_date} → {result.end_date}")
        print(f"  Starting capital: ${result.initial_balance:>12,.2f}")
        print(f"  Ending capital  : ${result.final_balance:>12,.2f}")
        print(f"  Total return    : {result.total_return_pct:>+.2f}%")
        print(f"  Buy & hold      : {result.buy_and_hold_return_pct:>+.2f}%")
        print(f"  Alpha           : {alpha:>+.2f}%")
        print(f"  Sharpe ratio    : {result.sharpe_ratio:.3f}")
        print(f"  Max drawdown    : {result.max_drawdown_pct:.2f}%")
        print(f"  Trades          : {result.total_trades}"
              f"  (W: {result.winning_trades}  L: {result.losing_trades})")
        print(f"  Win rate        : {result.win_rate_pct:.1f}%")
        print(f"  Avg win / loss  : "
              f"+${result.avg_win:.2f}  /  -${abs(result.avg_loss):.2f}")
        if result.trades:
            print(f"\n  Trade log:")
            for t in result.trades:
                sign = "+" if t.pnl >= 0 else "-"
                print(
                    f"    {str(t.entry_date.date())} → "
                    f"{str(t.exit_date.date()) if t.exit_date else 'open':>10}  "
                    f"{t.signal:<14}  "
                    f"{t.shares} sh @ ${t.entry_price:.2f}  "
                    f"P&L: {sign}${abs(t.pnl):.2f}  [{t.exit_reason}]"
                )
        print(f"{bar}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the News Trading System on historical price data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",    required=True,               help="Stock ticker (e.g. AAPL)")
    parser.add_argument("--start",     required=True,               help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       required=True,               help="End date YYYY-MM-DD")
    parser.add_argument("--balance",   type=float, default=10_000,  metavar="USD",
                        help="Initial balance (default: 10000)")
    parser.add_argument("--sentiment", default="random",
                        choices=["random", "bullish", "bearish", "neutral"],
                        help="Sentiment mode (default: random)")
    parser.add_argument("--no-plot",   action="store_true",         help="Skip chart display")
    parser.add_argument("--save-html", metavar="PATH",
                        help="Save interactive chart as HTML (auto-named if omitted)")
    args = parser.parse_args()

    engine = BacktestEngine(
        ticker          = args.ticker,
        start_date      = args.start,
        end_date        = args.end,
        initial_balance = args.balance,
        sentiment_mode  = args.sentiment,
    )
    result = engine.run()

    if not args.no_plot:
        save_path = args.save_html or str(
            PROJECT_ROOT / "backtest" / f"{args.ticker}_{args.start}_{args.end}.html"
        )
        engine.plot(result, show=not args.no_plot, save_path=save_path)


if __name__ == "__main__":
    main()
