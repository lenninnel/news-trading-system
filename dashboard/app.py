"""
News Trading System — Streamlit Dashboard.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Ensure project root is on the path so storage.database can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import DB_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="News Trading System",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame."""
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = ["Overview", "Signals", "Portfolio", "History", "Backtesting"]
page = st.sidebar.radio("Navigation", PAGES)

# ---------------------------------------------------------------------------
# Auto-refresh countdown (bottom of sidebar)
# ---------------------------------------------------------------------------

REFRESH_INTERVAL = 60  # seconds

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, REFRESH_INTERVAL - int(elapsed))

st.sidebar.markdown("---")
st.sidebar.caption(f"Auto-refresh in {remaining}s")

if elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()


# ===================================================================
# PAGE: Overview
# ===================================================================

def page_overview() -> None:
    st.title("Overview")

    # Broker mode banner
    mode = os.environ.get("TRADING_MODE", "paper_local").lower()
    labels = {
        "paper_local": ("LOCAL PAPER (simulated)", "info"),
        "alpaca_paper": ("ALPACA PAPER (sandbox)", "warning"),
        "alpaca_live": ("ALPACA LIVE (real money)", "error"),
    }
    label, level = labels.get(mode, (mode, "info"))
    getattr(st, level)(f"Broker mode: **{label}**")

    # --- Market regime ---
    regime_row = _query(
        "SELECT regime FROM risk_calculations "
        "WHERE regime IS NOT NULL ORDER BY id DESC LIMIT 1"
    )
    if not regime_row.empty:
        regime = regime_row.iloc[0]["regime"]
        _REGIME_DISPLAY = {
            "TRENDING_BULL": "TRENDING BULL :green_circle:",
            "TRENDING_BEAR": "TRENDING BEAR :red_circle:",
            "RANGING":       "RANGING :yellow_circle:",
            "HIGH_VOL":      "HIGH VOL :zap:",
        }
        st.subheader(_REGIME_DISPLAY.get(regime, regime))

    # --- KPIs ---
    portfolio = _query("SELECT * FROM portfolio_positions ORDER BY ticker")
    trades = _query("SELECT * FROM trade_history ORDER BY id DESC")

    total_value = portfolio["current_value"].sum() if not portfolio.empty else 0.0
    open_count = len(portfolio)
    total_pnl = trades["pnl"].sum() if not trades.empty else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value", f"${total_value:,.2f}")
    c2.metric("Open Positions", open_count)
    c3.metric("Total Realised PnL", f"${total_pnl:,.2f}")

    st.markdown("---")

    # --- Last 5 combined signals ---
    st.subheader("Recent Signals")
    signals = _query(
        "SELECT ticker, combined_signal, confidence, sentiment_score, "
        "created_at FROM combined_signals ORDER BY id DESC LIMIT 5"
    )
    if signals.empty:
        st.info("No signals recorded yet. Run an analysis first.")
    else:
        st.dataframe(signals, width="stretch", hide_index=True)


# ===================================================================
# PAGE: Signals
# ===================================================================

def page_signals() -> None:
    st.title("Signals")

    # --- Filters ---
    col_a, col_b, col_c = st.columns(3)

    all_tickers = _query(
        "SELECT DISTINCT ticker FROM combined_signals ORDER BY ticker"
    )
    ticker_list = ["All"] + (all_tickers["ticker"].tolist() if not all_tickers.empty else [])
    ticker_filter = col_a.selectbox("Ticker", ticker_list)

    signal_types = [
        "All", "STRONG BUY", "STRONG SELL", "WEAK BUY",
        "WEAK SELL", "CONFLICTING", "HOLD",
    ]
    signal_filter = col_b.selectbox("Signal", signal_types)

    today = date.today()
    date_range = col_c.date_input(
        "Date range",
        value=(today - timedelta(days=30), today),
    )

    # Build query
    clauses: list[str] = []
    params: list = []

    if ticker_filter != "All":
        clauses.append("ticker = ?")
        params.append(ticker_filter)
    if signal_filter != "All":
        clauses.append("combined_signal = ?")
        params.append(signal_filter)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        clauses.append("date(created_at) >= date(?)")
        params.append(start.isoformat())
        clauses.append("date(created_at) <= date(?)")
        params.append(end.isoformat())

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    # Join with technical_signals to get volume_confirmed,
    # and with risk_calculations to get event_risk_flag.
    sql = f"""
        SELECT
            cs.ticker,
            cs.combined_signal,
            printf('%.0f%%', cs.confidence * 100) AS confidence,
            COALESCE(rc.event_risk_flag, 'none')  AS event_risk_flag,
            CASE WHEN ts.volume_confirmed = 1 THEN 'Yes' ELSE 'No' END
                AS volume_confirmed,
            cs.created_at
        FROM combined_signals cs
        LEFT JOIN technical_signals ts ON cs.technical_id = ts.id
        LEFT JOIN risk_calculations rc
            ON rc.ticker = cs.ticker
            AND rc.created_at = (
                SELECT MAX(r2.created_at) FROM risk_calculations r2
                WHERE r2.ticker = cs.ticker
                AND date(r2.created_at) = date(cs.created_at)
            )
        {where}
        ORDER BY cs.id DESC
        LIMIT 200
    """
    df = _query(sql, tuple(params))

    if df.empty:
        st.info("No signals match the current filters.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"{len(df)} signal(s) shown (max 200)")


# ===================================================================
# PAGE: Portfolio
# ===================================================================

def page_portfolio() -> None:
    st.title("Portfolio")

    positions = _query("SELECT * FROM portfolio_positions ORDER BY ticker")

    if positions.empty:
        st.info("No open positions.")
        return

    # Fetch live prices via yfinance
    import yfinance as yf

    tickers = positions["ticker"].tolist()
    live_prices: dict[str, float | None] = {}
    try:
        data = yf.download(tickers, period="1d", progress=False, threads=True)
        close = data.get("Close")
        if close is not None:
            if isinstance(close, pd.Series):
                # Single ticker returns a Series
                live_prices[tickers[0]] = (
                    float(close.iloc[-1]) if not close.empty else None
                )
            else:
                for t in tickers:
                    if t in close.columns:
                        val = close[t].dropna()
                        live_prices[t] = float(val.iloc[-1]) if not val.empty else None
    except Exception:
        pass  # fall back to stored current_value

    rows = []
    for _, pos in positions.iterrows():
        t = pos["ticker"]
        live = live_prices.get(t)
        mkt_value = live * pos["shares"] if live else pos["current_value"]
        cost_basis = pos["avg_price"] * pos["shares"]
        unrealised = mkt_value - cost_basis
        rows.append({
            "Ticker": t,
            "Shares": pos["shares"],
            "Avg Price": f"${pos['avg_price']:.2f}",
            "Live Price": f"${live:.2f}" if live else "N/A",
            "Market Value": f"${mkt_value:,.2f}",
            "Unrealised PnL": f"${unrealised:+,.2f}",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ===================================================================
# PAGE: History
# ===================================================================

def page_history() -> None:
    st.title("Trade History")

    col_a, col_b = st.columns(2)

    all_tickers = _query(
        "SELECT DISTINCT ticker FROM trade_history ORDER BY ticker"
    )
    ticker_list = ["All"] + (all_tickers["ticker"].tolist() if not all_tickers.empty else [])
    ticker_filter = col_a.selectbox("Ticker", ticker_list, key="hist_ticker")

    today = date.today()
    date_range = col_b.date_input(
        "Date range",
        value=(today - timedelta(days=90), today),
        key="hist_dates",
    )

    clauses: list[str] = []
    params: list = []

    if ticker_filter != "All":
        clauses.append("ticker = ?")
        params.append(ticker_filter)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        clauses.append("date(created_at) >= date(?)")
        params.append(start.isoformat())
        clauses.append("date(created_at) <= date(?)")
        params.append(end.isoformat())

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    df = _query(
        f"SELECT ticker, action, shares, price, pnl, created_at "
        f"FROM trade_history {where} ORDER BY id DESC LIMIT 500",
        tuple(params),
    )

    if df.empty:
        st.info("No trades recorded yet.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        total_pnl = df["pnl"].sum()
        st.caption(f"{len(df)} trade(s) shown  |  Net PnL: ${total_pnl:,.2f}")


# ===================================================================
# PAGE: Backtesting
# ===================================================================

def page_backtesting() -> None:
    st.title("Backtesting")

    # --- Run a new optimisation ---
    st.subheader("Run Walk-Forward Optimisation")

    col_a, col_b, col_c = st.columns(3)
    ticker = col_a.text_input("Ticker", value="AAPL").upper()
    today = date.today()
    start = col_b.date_input("Start date", value=today - timedelta(days=365))
    end = col_c.date_input("End date", value=today)

    if st.button("Run Optimisation"):
        if not ticker:
            st.warning("Enter a ticker symbol.")
            return
        if start >= end:
            st.warning("Start date must be before end date.")
            return

        with st.spinner("Running walk-forward optimisation... this may take a few minutes."):
            try:
                from optimization.optimizer import WalkForwardOptimizer
                from optimization.results import save_optimization_run
                from storage.database import Database

                opt = WalkForwardOptimizer(
                    ticker=ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
                result = opt.run()

                # Compute average IS Sharpe if available
                is_sharpes = [r.get("is_sharpe", 0.0) for r in result.get("oos_results", [])]
                avg_is = sum(is_sharpes) / len(is_sharpes) if is_sharpes else 0.0

                db = Database()
                save_optimization_run(
                    db=db,
                    ticker=ticker,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    best_params=result["best_params"],
                    in_sample_sharpe=avg_is,
                    out_of_sample_sharpe=result["oos_sharpe"],
                    total_windows=result["all_windows"],
                    equity_curve=result["equity_curve"],
                )

                st.success("Optimisation complete!")
                st.session_state["last_opt_result"] = result
            except Exception as e:
                st.error(f"Optimisation failed: {e}")

    # --- Display last result ---
    result = st.session_state.get("last_opt_result")
    if result:
        st.markdown("---")
        st.subheader("Latest Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("OOS Sharpe", f"{result['oos_sharpe']:.4f}")
        c2.metric("Windows", result["all_windows"])
        c3.metric("Parameters", len(result.get("best_params", {})))

        if result.get("best_params"):
            st.write("**Best Parameters:**")
            st.json(result["best_params"])

        if result.get("equity_curve") and len(result["equity_curve"]) > 1:
            st.write("**Out-of-Sample Equity Curve:**")
            st.line_chart(result["equity_curve"])

        if result.get("oos_results"):
            st.write("**Per-Window Results:**")
            window_rows = []
            for r in result["oos_results"]:
                window_rows.append({
                    "Train": f"{r['train'][0]} → {r['train'][1]}",
                    "Test": f"{r['test'][0]} → {r['test'][1]}",
                    "OOS Sharpe": f"{r['oos_sharpe']:.4f}",
                    "OOS Return": f"{r['oos_return']:.2%}",
                    "Trades": r["oos_trades"],
                })
            st.dataframe(pd.DataFrame(window_rows), width="stretch", hide_index=True)

    # --- Past runs ---
    st.markdown("---")
    st.subheader("Past Optimisation Runs")

    runs = _query(
        "SELECT id, ticker, start_date, end_date, best_params, "
        "out_of_sample_sharpe, total_windows, created_at "
        "FROM optimization_runs ORDER BY id DESC LIMIT 20"
    )

    if runs.empty:
        st.info("No optimisation runs recorded yet.")
    else:
        st.dataframe(runs, width="stretch", hide_index=True)


# ===================================================================
# Router
# ===================================================================

_ROUTER = {
    "Overview": page_overview,
    "Signals": page_signals,
    "Portfolio": page_portfolio,
    "History": page_history,
    "Backtesting": page_backtesting,
}

_ROUTER[page]()
