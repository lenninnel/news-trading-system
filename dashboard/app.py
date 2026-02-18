"""
Streamlit dashboard for the News Trading System.

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the project root is importable (config/, storage/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DB_PATH  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh (60 s) ───────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="autorefresh")
except ImportError:
    pass  # degrades gracefully — use the manual refresh button

# ── DB helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame. Cached for 60 s."""
    import sqlite3
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params if params else None)


def fmt_usd(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"${float(v):,.2f}"


def signal_color(sig: str) -> str:
    return {
        "STRONG BUY": "🟢",
        "WEAK BUY":   "🟩",
        "BUY":        "🟢",
        "STRONG SELL":"🔴",
        "WEAK SELL":  "🟥",
        "SELL":       "🔴",
        "HOLD":       "🟡",
        "CONFLICTING":"🟠",
    }.get(sig, "⚪")


def badge(sig: str) -> str:
    return f"{signal_color(sig)} {sig}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Trading System")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Signals", "Portfolio", "History", "Agents"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"DB: `{DB_PATH}`")
st.sidebar.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Overview")

    portfolio   = query("SELECT * FROM portfolio WHERE shares > 0")
    trade_hist  = query("SELECT * FROM trade_history ORDER BY timestamp DESC")
    all_signals = query("SELECT * FROM combined_signals ORDER BY created_at DESC")
    today_sigs  = query(
        "SELECT * FROM combined_signals WHERE date(created_at) = date('now')"
    )

    total_value   = portfolio["current_value"].sum() if not portfolio.empty else 0.0
    open_pos      = len(portfolio) if not portfolio.empty else 0
    total_trades  = len(trade_hist) if not trade_hist.empty else 0
    signals_today = len(today_sigs) if not today_sigs.empty else 0

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Value",  fmt_usd(total_value),
              help="Entry-based (shares × avg price)")
    k2.metric("Open Positions",   open_pos)
    k3.metric("Total Trades",     total_trades)
    k4.metric("Signals Today",    signals_today)

    st.markdown("---")

    col_l, col_r = st.columns([1.6, 1])

    # ── Recent signals ────────────────────────────────────────────────────────
    with col_l:
        st.subheader("Recent Signals")
        if all_signals.empty:
            st.info("No signals yet. Run `python main.py <TICKER>`.")
        else:
            d = all_signals[
                ["created_at", "ticker", "combined_signal",
                 "confidence", "sentiment_signal", "technical_signal"]
            ].head(10).copy()
            d["combined_signal"] = d["combined_signal"].apply(badge)
            d["confidence"]      = (d["confidence"] * 100).round(1).astype(str) + "%"
            d["created_at"]      = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            d.columns = ["Time", "Ticker", "Signal", "Conf.", "Sentiment", "Technical"]
            st.dataframe(d, use_container_width=True, hide_index=True)

    # ── Recent trades ─────────────────────────────────────────────────────────
    with col_r:
        st.subheader("Recent Trades")
        if trade_hist.empty:
            st.info("No trades yet.\nRun with `--execute`.")
        else:
            d = trade_hist[
                ["timestamp", "ticker", "action", "shares", "price", "pnl"]
            ].head(8).copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%m-%d %H:%M")
            d["price"]     = d["price"].apply(fmt_usd)
            d["pnl"]       = d["pnl"].apply(
                lambda x: f"+${x:.2f}" if x > 0 else (f"-${abs(x):.2f}" if x < 0 else "—")
            )
            d.columns = ["Time", "Ticker", "Action", "Shares", "Price", "P&L"]
            st.dataframe(d, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Charts row ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    SIGNAL_COLORS = {
        "STRONG BUY":  "#00c853",
        "WEAK BUY":    "#69f0ae",
        "HOLD":        "#ffd740",
        "CONFLICTING": "#ff9100",
        "WEAK SELL":   "#ff6d6d",
        "STRONG SELL": "#b71c1c",
    }

    with c1:
        st.subheader("Signal Distribution")
        if not all_signals.empty:
            dist = all_signals["combined_signal"].value_counts().reset_index()
            dist.columns = ["Signal", "Count"]
            fig = px.bar(
                dist, x="Signal", y="Count", color="Signal",
                color_discrete_map=SIGNAL_COLORS,
            )
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signals yet.")

    with c2:
        st.subheader("Win / Loss Ratio")
        sells = query("SELECT pnl FROM trade_history WHERE action='SELL'")
        if not sells.empty:
            wins      = int((sells["pnl"] > 0).sum())
            losses    = int((sells["pnl"] < 0).sum())
            breakeven = int((sells["pnl"] == 0).sum())
            if wins + losses + breakeven > 0:
                fig = px.pie(
                    values=[wins, losses, breakeven],
                    names=["Win", "Loss", "Breakeven"],
                    color_discrete_sequence=["#00c853", "#b71c1c", "#ffd740"],
                    hole=0.4,
                )
                fig.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No closed trades yet.")
        else:
            st.info("No closed trades yet.")

    with c3:
        st.subheader("Portfolio Value Over Time")
        if not trade_hist.empty:
            th = trade_hist.copy()
            th["timestamp"] = pd.to_datetime(th["timestamp"])
            th = th.sort_values("timestamp")
            th["flow"] = th.apply(
                lambda r: r["shares"] * r["price"] if r["action"] == "BUY"
                          else -r["shares"] * r["price"],
                axis=1,
            )
            th["deployed"] = th["flow"].cumsum()
            fig = px.line(
                th, x="timestamp", y="deployed",
                labels={"timestamp": "Time", "deployed": "Deployed Capital ($)"},
            )
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade history yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Signals":
    st.title("Combined Signals")

    signals = query("SELECT * FROM combined_signals ORDER BY created_at DESC")
    if signals.empty:
        st.info("No signals recorded. Run `python main.py <TICKER>` to generate some.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    tickers   = ["All"] + sorted(signals["ticker"].unique().tolist())
    sig_types = ["All"] + sorted(signals["combined_signal"].unique().tolist())

    with f1:
        sel_ticker = st.selectbox("Ticker", tickers)
    with f2:
        sel_signal = st.selectbox("Signal type", sig_types)
    with f3:
        min_conf = st.slider("Min confidence (%)", 0, 100, 0)

    filtered = signals.copy()
    if sel_ticker != "All":
        filtered = filtered[filtered["ticker"] == sel_ticker]
    if sel_signal != "All":
        filtered = filtered[filtered["combined_signal"] == sel_signal]
    filtered = filtered[filtered["confidence"] * 100 >= min_conf]

    st.caption(f"{len(filtered)} signal(s) shown")

    d = filtered[[
        "created_at", "ticker", "combined_signal", "confidence",
        "sentiment_signal", "technical_signal", "sentiment_score",
    ]].copy()
    d["combined_signal"]  = d["combined_signal"].apply(badge)
    d["confidence"]       = (d["confidence"] * 100).round(1).astype(str) + "%"
    d["sentiment_score"]  = d["sentiment_score"].round(3)
    d["created_at"]       = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    d.columns = ["Time", "Ticker", "Signal", "Confidence", "Sentiment", "Technical", "Sent. Score"]
    st.dataframe(d, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio":
    st.title("Portfolio")

    portfolio = query("SELECT * FROM portfolio ORDER BY ticker")
    closed    = query(
        "SELECT SUM(pnl) AS total_pnl, COUNT(*) AS count "
        "FROM trade_history WHERE action='SELL' AND pnl != 0"
    )

    open_df      = portfolio[portfolio["shares"] > 0] if not portfolio.empty else pd.DataFrame()
    total_value  = open_df["current_value"].sum() if not open_df.empty else 0.0
    realized_pnl = (
        closed["total_pnl"].iloc[0]
        if not closed.empty and closed["total_pnl"].iloc[0] is not None
        else 0.0
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Open Position Value", fmt_usd(total_value),
              help="Entry-based: shares × avg price")
    k2.metric(
        "Realised P&L",
        fmt_usd(realized_pnl),
        delta=f"{realized_pnl:+.2f}" if realized_pnl else None,
        delta_color="normal",
    )
    k3.metric("Open Positions", len(open_df))

    st.markdown("---")

    # ── Open positions ────────────────────────────────────────────────────────
    st.subheader("Open Positions")
    if open_df.empty:
        st.info("No open positions. Run `python main.py <TICKER> --execute` to log trades.")
    else:
        d = open_df.copy()
        d["avg_price"]     = d["avg_price"].apply(fmt_usd)
        d["current_value"] = d["current_value"].apply(fmt_usd)
        d.columns = ["Ticker", "Shares", "Avg Price", "Entry Value"]
        st.dataframe(d, use_container_width=True, hide_index=True)

    # ── All tickers (incl. closed) ────────────────────────────────────────────
    if not portfolio.empty:
        st.markdown("---")
        st.subheader("All Tickers (incl. flat)")
        d = portfolio.copy()
        d["avg_price"]     = d["avg_price"].apply(fmt_usd)
        d["current_value"] = d["current_value"].apply(fmt_usd)
        d.columns = ["Ticker", "Shares", "Avg Price", "Entry Value"]
        st.dataframe(d, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    st.title("Trade History")

    trades = query("SELECT * FROM trade_history ORDER BY timestamp DESC")
    if trades.empty:
        st.info("No trades recorded yet. Run with `--execute`.")
        st.stop()

    trades["timestamp"] = pd.to_datetime(trades["timestamp"])

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    tickers = ["All"] + sorted(trades["ticker"].unique().tolist())
    actions = ["All", "BUY", "SELL"]

    with f1:
        sel_ticker = st.selectbox("Ticker", tickers, key="hist_ticker")
    with f2:
        sel_action = st.selectbox("Action", actions)
    with f3:
        date_range = st.date_input(
            "Date range",
            value=(trades["timestamp"].min().date(), trades["timestamp"].max().date()),
        )

    filtered = trades.copy()
    if sel_ticker != "All":
        filtered = filtered[filtered["ticker"] == sel_ticker]
    if sel_action != "All":
        filtered = filtered[filtered["action"] == sel_action]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start) &
            (filtered["timestamp"].dt.date <= end)
        ]

    st.caption(f"{len(filtered)} trade(s) shown")

    # ── Table ─────────────────────────────────────────────────────────────────
    d = filtered.copy()
    d["timestamp"]   = d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    d["price"]       = d["price"].apply(fmt_usd)
    d["stop_loss"]   = d["stop_loss"].apply(fmt_usd)
    d["take_profit"] = d["take_profit"].apply(fmt_usd)
    d["pnl"]         = d["pnl"].apply(
        lambda x: f"+${x:.2f}" if x > 0 else (f"-${abs(x):.2f}" if x < 0 else "—")
    )
    d = d.drop(columns=["id"])
    d.columns = ["Time", "Ticker", "Action", "Shares", "Price", "Stop Loss", "Take Profit", "P&L"]
    st.dataframe(d, use_container_width=True, hide_index=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    if not filtered.empty:
        st.markdown("---")
        st.subheader("Summary")
        sells = filtered[filtered["action"] == "SELL"]
        realized = sells["pnl"].sum() if not sells.empty else 0.0

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Trades", len(filtered))
        s2.metric("Buys",  len(filtered[filtered["action"] == "BUY"]))
        s3.metric("Sells", len(sells))
        s4.metric("Realised P&L", fmt_usd(realized),
                  delta=f"{realized:+.2f}" if realized else None,
                  delta_color="normal")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Agents":
    st.title("Agent Outputs")

    tab_sent, tab_tech, tab_risk = st.tabs(["Sentiment", "Technical", "Risk"])

    # ── Sentiment ─────────────────────────────────────────────────────────────
    with tab_sent:
        st.subheader("Sentiment Runs")
        runs = query("SELECT * FROM runs ORDER BY created_at DESC")
        if runs.empty:
            st.info("No sentiment runs recorded.")
        else:
            tickers   = ["All"] + sorted(runs["ticker"].unique().tolist())
            sel       = st.selectbox("Ticker", tickers, key="sent_ticker")
            filt_runs = runs if sel == "All" else runs[runs["ticker"] == sel]

            d = filt_runs[[
                "created_at", "ticker", "signal", "avg_score",
                "headlines_analysed", "headlines_fetched",
            ]].copy()
            d["signal"]     = d["signal"].apply(badge)
            d["avg_score"]  = d["avg_score"].round(3)
            d["created_at"] = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            d.columns = ["Time", "Ticker", "Signal", "Avg Score", "Analysed", "Fetched"]
            st.dataframe(d, use_container_width=True, hide_index=True)

            # Headline drill-down
            st.markdown("---")
            st.subheader("Headline Scores")
            run_ids = filt_runs["id"].tolist()
            if run_ids:
                ph   = ",".join("?" * len(run_ids))
                hdls = query(
                    f"SELECT hs.sentiment, hs.score, hs.headline, hs.reason, r.ticker "
                    f"FROM headline_scores hs JOIN runs r ON r.id = hs.run_id "
                    f"WHERE hs.run_id IN ({ph}) ORDER BY hs.id DESC",
                    tuple(run_ids),
                )
                if not hdls.empty:
                    hdls.columns = ["Sentiment", "Score", "Headline", "Reason", "Ticker"]
                    hdls = hdls[["Ticker", "Sentiment", "Score", "Headline", "Reason"]]
                    st.dataframe(hdls, use_container_width=True, hide_index=True)

    # ── Technical ─────────────────────────────────────────────────────────────
    with tab_tech:
        st.subheader("Technical Signals")
        tech = query("SELECT * FROM technical_signals ORDER BY created_at DESC")
        if tech.empty:
            st.info("No technical signals recorded.")
        else:
            tickers   = ["All"] + sorted(tech["ticker"].unique().tolist())
            sel       = st.selectbox("Ticker", tickers, key="tech_ticker")
            filt_tech = tech if sel == "All" else tech[tech["ticker"] == sel]

            d = filt_tech[[
                "created_at", "ticker", "signal", "price",
                "rsi", "macd", "sma_20", "sma_50", "bb_upper", "bb_lower",
            ]].copy()
            d["signal"]     = d["signal"].apply(badge)
            d["created_at"] = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            for col in ["price", "rsi", "macd", "sma_20", "sma_50", "bb_upper", "bb_lower"]:
                d[col] = d[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            d.columns = ["Time", "Ticker", "Signal", "Price", "RSI", "MACD", "SMA20", "SMA50", "BB↑", "BB↓"]
            st.dataframe(d, use_container_width=True, hide_index=True)

            # Reasoning expanders
            st.markdown("---")
            st.subheader("Reasoning (latest 5)")
            for _, row in filt_tech.head(5).iterrows():
                label = f"{row['ticker']} — {str(row['created_at'])[:16]}"
                with st.expander(label):
                    reasoning = str(row.get("reasoning", "") or "")
                    parts = [p.strip() for p in reasoning.split(";") if p.strip()]
                    if parts:
                        for p in parts:
                            st.write(f"• {p}")
                    else:
                        st.write("No reasoning recorded.")

    # ── Risk ──────────────────────────────────────────────────────────────────
    with tab_risk:
        st.subheader("Risk Calculations")
        risk = query("SELECT * FROM risk_calculations ORDER BY created_at DESC")
        if risk.empty:
            st.info("No risk calculations recorded.")
        else:
            tickers   = ["All"] + sorted(risk["ticker"].unique().tolist())
            sel       = st.selectbox("Ticker", tickers, key="risk_ticker")
            filt_risk = risk if sel == "All" else risk[risk["ticker"] == sel]

            d = filt_risk[[
                "created_at", "ticker", "signal", "confidence",
                "current_price", "position_size_usd", "shares",
                "stop_loss", "take_profit", "risk_amount",
                "skipped", "skip_reason",
            ]].copy()
            d["signal"]           = d["signal"].apply(badge)
            d["confidence"]       = d["confidence"].apply(lambda x: f"{x:.1f}%")
            d["skipped"]          = d["skipped"].apply(lambda x: "Yes" if x else "No")
            d["created_at"]       = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            for col in ["current_price", "position_size_usd", "stop_loss", "take_profit", "risk_amount"]:
                d[col] = d[col].apply(fmt_usd)
            d.columns = [
                "Time", "Ticker", "Signal", "Conf.",
                "Price", "Position $", "Shares",
                "SL", "TP", "Risk $",
                "Skipped", "Skip Reason",
            ]
            st.dataframe(d, use_container_width=True, hide_index=True)
