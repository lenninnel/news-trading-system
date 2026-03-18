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
# Password protection (opt-in via DASHBOARD_PASSWORD env var)
# ---------------------------------------------------------------------------

_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")

if _DASHBOARD_PASSWORD:
    if not st.session_state.get("authenticated"):
        st.title("News Trading System")
        password = st.text_input("Enter dashboard password", type="password")
        if st.button("Login"):
            if password == _DASHBOARD_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _resolve_db_path() -> str:
    """Use Railway persistent volume if available and writable."""
    railway_dir = "/data"
    if os.path.isdir(railway_dir) and os.access(railway_dir, os.W_OK):
        return os.path.join(railway_dir, "news_trading.db")
    return DB_PATH


_DB_PATH = _resolve_db_path()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
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
    try:
        regime_row = _query(
            "SELECT regime FROM risk_calculations "
            "WHERE regime IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return
    if not regime_row.empty:
        regime = regime_row.iloc[0]["regime"]
        _REGIME_DISPLAY = {
            "TRENDING_BULL": "TRENDING BULL 🟢",
            "TRENDING_BEAR": "TRENDING BEAR 🔴",
            "RANGING":       "RANGING 🟡",
            "HIGH_VOL":      "HIGH VOL ⚡",
        }
        st.subheader(_REGIME_DISPLAY.get(regime, regime))

    # --- KPIs ---
    # Use Alpaca portfolio if available (source of truth for live/paper)
    alpaca_val, alpaca_df = _try_alpaca_portfolio()

    if alpaca_val is not None:
        total_value = alpaca_val
        open_count = len(alpaca_df) if alpaca_df is not None and not alpaca_df.empty else 0
    else:
        try:
            portfolio = _query(
                "SELECT COALESCE(SUM(current_value), 0) AS total_value, "
                "COUNT(*) AS cnt FROM portfolio_positions"
            )
        except sqlite3.OperationalError:
            st.info("No data yet — run the pipeline first.")
            return
        if not portfolio.empty:
            total_value = portfolio.iloc[0]["total_value"]
            open_count = portfolio.iloc[0]["cnt"]
        else:
            total_value = 0.0
            open_count = 0

    try:
        pnl_row = _query(
            "SELECT COALESCE(SUM(pnl), 0) AS total_pnl FROM trade_history"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return
    total_pnl = pnl_row.iloc[0]["total_pnl"] if not pnl_row.empty else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value", f"${total_value:,.2f}")
    c2.metric("Open Positions", open_count)
    c3.metric("Total Realised PnL", f"${total_pnl:,.2f}")

    st.markdown("---")

    # --- Last 5 combined signals ---
    st.subheader("Recent Signals")
    try:
        signals = _query(
            "SELECT ticker, combined_signal, confidence, sentiment_score, "
            "created_at FROM combined_signals ORDER BY id DESC LIMIT 5"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return
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

    try:
        all_tickers = _query(
            "SELECT DISTINCT ticker FROM combined_signals ORDER BY ticker"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return
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
        clauses.append("cs.ticker = ?")
        params.append(ticker_filter)
    if signal_filter != "All":
        clauses.append("cs.combined_signal = ?")
        params.append(signal_filter)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        clauses.append("date(cs.created_at) >= date(?)")
        params.append(start.isoformat())
        clauses.append("date(cs.created_at) <= date(?)")
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
            CASE WHEN ts.golden_cross_recent = 1 THEN '⭐ Golden Cross'
                 WHEN ts.death_cross_recent = 1 THEN '💀 Death Cross'
                 ELSE '' END AS ma_cross,
            CASE WHEN ts.bull_flag_detected = 1 THEN '🚩 Bull Flag' ELSE '' END AS patterns,
            CASE WHEN ts.wedge_breakout = 1 THEN '📐 ' || COALESCE(ts.wedge_type, '') || ' Wedge Breakout'
                 ELSE '' END AS wedge,
            COALESCE(printf('%.1f', ts.adx), '') AS adx,
            COALESCE(printf('%.1f%%', ts.ma200_distance_pct), '') AS ma200_dist,
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
    try:
        df = _query(sql, tuple(params))
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return

    if df.empty:
        st.info("No signals match the current filters.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"{len(df)} signal(s) shown (max 200)")

    # Pattern alerts (recent golden crosses, bull flags, wedge breakouts)
    try:
        patterns_df = _query("""
            SELECT ts.ticker, ts.signal, ts.price,
                CASE WHEN ts.golden_cross_recent = 1 THEN '⭐ Golden Cross' ELSE '' END ||
                CASE WHEN ts.bull_flag_detected = 1 THEN ' 🚩 Bull Flag' ELSE '' END ||
                CASE WHEN ts.wedge_breakout = 1 THEN ' 📐 Wedge Breakout' ELSE '' END AS patterns,
                printf('%.1f', ts.adx) AS adx,
                printf('%.1f%%', ts.ma200_distance_pct) AS ma200_dist,
                ts.trend_strength,
                ts.created_at
            FROM technical_signals ts
            WHERE ts.golden_cross_recent = 1
               OR ts.bull_flag_detected = 1
               OR ts.wedge_breakout = 1
            ORDER BY ts.id DESC
            LIMIT 20
        """)
        if not patterns_df.empty:
            st.markdown("---")
            st.subheader("Pattern Alerts")
            st.dataframe(patterns_df, width="stretch", hide_index=True)
    except sqlite3.OperationalError:
        pass  # new columns may not exist yet on older DBs


# ===================================================================
# PAGE: Portfolio
# ===================================================================

def _try_alpaca_portfolio() -> tuple[float | None, pd.DataFrame | None]:
    """Try to read portfolio from Alpaca API. Returns (value, positions_df) or (None, None)."""
    try:
        import alpaca_trade_api as tradeapi
        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        mode = os.environ.get("ALPACA_MODE", "paper").lower()
        if not key or not secret:
            return None, None
        base_url = "https://api.alpaca.markets" if mode == "live" \
            else "https://paper-api.alpaca.markets"
        api = tradeapi.REST(key_id=key, secret_key=secret,
                            base_url=base_url, api_version="v2")
        account = api.get_account()
        positions = api.list_positions()
        portfolio_value = float(account.portfolio_value)
        rows = []
        for p in positions:
            rows.append({
                "Ticker": p.symbol,
                "Shares": int(p.qty),
                "Side": "SHORT" if float(p.qty) < 0 else "LONG",
                "Avg Price": f"${float(p.avg_entry_price):.2f}",
                "Live Price": f"${float(p.current_price):.2f}",
                "Market Value": f"${float(p.market_value):,.2f}",
                "Unrealised PnL": f"${float(p.unrealized_pl):+,.2f}",
                "PnL %": f"{float(p.unrealized_plpc) * 100:+.2f}%",
            })
        return portfolio_value, pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return None, None


def page_portfolio() -> None:
    st.title("Portfolio")

    # Try Alpaca API first (source of truth for live/paper trading)
    alpaca_value, alpaca_df = _try_alpaca_portfolio()

    if alpaca_value is not None:
        st.success(f"Live from Alpaca | Portfolio Value: **${alpaca_value:,.2f}**")
        if alpaca_df is not None and not alpaca_df.empty:
            st.dataframe(alpaca_df, width="stretch", hide_index=True)
        else:
            st.info("No open positions on Alpaca.")
        alpaca_mode = os.environ.get("ALPACA_MODE", "paper").lower()
        mode_label = "live" if alpaca_mode == "live" else "paper"
        st.caption(f"Data from Alpaca {mode_label} trading API")
        return

    # Fallback: local DB
    st.warning("Alpaca unavailable — showing local DB (may be stale)")

    try:
        positions = _query("SELECT * FROM portfolio_positions ORDER BY ticker")
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return

    if positions.empty:
        st.info("No open positions.")
        return

    # Fetch live prices via Alpaca
    from data.alpaca_data import AlpacaDataClient

    tickers = positions["ticker"].tolist()
    live_prices: dict[str, float | None] = {}
    try:
        alpaca = AlpacaDataClient()
        for t in tickers:
            try:
                live_prices[t] = alpaca.get_current_price(t)
            except Exception:
                live_prices[t] = None
    except Exception:
        pass

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

    try:
        all_tickers = _query(
            "SELECT DISTINCT ticker FROM trade_history ORDER BY ticker"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return
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

    try:
        df = _query(
            f"SELECT ticker, action, shares, price, pnl, created_at "
            f"FROM trade_history {where} ORDER BY id DESC LIMIT 500",
            tuple(params),
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return

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

    try:
        runs = _query(
            "SELECT id, ticker, start_date, end_date, best_params, "
            "out_of_sample_sharpe, total_windows, created_at "
            "FROM optimization_runs ORDER BY id DESC LIMIT 20"
        )
    except sqlite3.OperationalError:
        st.info("No data yet — run the pipeline first.")
        return

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
