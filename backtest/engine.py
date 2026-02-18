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

    def _download_data(self) -> pd.DataFrame:
        warmup_start = self.start_ts - pd.Timedelta(days=self._WARMUP_DAYS)
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
        )

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
