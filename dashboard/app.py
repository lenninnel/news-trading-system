"""
Streamlit dashboard for the News Trading System.

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys
from datetime import date, datetime

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
    ["Overview", "Signals", "Portfolio", "History", "Agents", "Backtesting", "Screener"],
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
    import plotly.graph_objects as go
    from execution.portfolio_manager import PortfolioManager

    st.title("Portfolio")

    # ── Sidebar balance input ──────────────────────────────────────────────────
    pm_balance = st.sidebar.number_input(
        "Account Balance ($)",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1_000.0,
        key="pm_balance",
        help="Used to compute deployment % and capacity limits.",
    )

    @st.cache_data(ttl=60)
    def _load_pm_data(balance: float):
        pm      = PortfolioManager(account_balance=balance)
        div     = pm.get_diversification_metrics()
        cap     = pm.capacity_summary()
        risk    = pm.check_risk_limits()
        rebal   = pm.rebalance_if_needed()
        corr_df = pm.get_correlation_matrix()
        return pm, div, cap, risk, rebal, corr_df

    pm, div, cap, risk, rebal, corr_df = _load_pm_data(pm_balance)

    portfolio = query("SELECT * FROM portfolio ORDER BY ticker")
    closed    = query(
        "SELECT SUM(pnl) AS total_pnl, COUNT(*) AS count "
        "FROM trade_history WHERE action='SELL' AND pnl != 0"
    )

    open_df      = portfolio[portfolio["shares"] > 0] if not portfolio.empty else pd.DataFrame()
    total_value  = div["total_value"]
    realized_pnl = (
        closed["total_pnl"].iloc[0]
        if not closed.empty and closed["total_pnl"].iloc[0] is not None
        else 0.0
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Position Value", fmt_usd(total_value),
              help="Entry-based: shares × avg price")
    k2.metric(
        "Realised P&L",
        fmt_usd(realized_pnl),
        delta=f"{realized_pnl:+.2f}" if realized_pnl else None,
        delta_color="normal",
    )
    k3.metric("Open Positions", f"{cap['positions_used']}/{cap['positions_max']}")
    k4.metric(
        "Capital Deployed",
        f"{cap['deployed_pct']:.1%}",
        help=f"Max: {PortfolioManager.MAX_DEPLOYED_PCT:.0%} of ${pm_balance:,.0f}",
    )
    k5.metric("Cash Reserve", fmt_usd(cap["cash_reserve"]))

    st.markdown("---")

    # ── Warnings banner ───────────────────────────────────────────────────────
    all_warnings = cap["warnings"] + risk["warnings"]
    if all_warnings:
        for w in all_warnings:
            st.warning(f"⚠️ {w}")
    if rebal:
        for act in rebal:
            colour = "error" if act["action"] in ("partial_close", "block_sector") else "warning"
            getattr(st, colour)(f"{'🔴' if colour == 'error' else '⚠️'} {act['reason']}")

    # ── Row 1: Position limits + sector pie ───────────────────────────────────
    col_lim, col_pie, col_strat = st.columns([1.2, 1.4, 1])

    with col_lim:
        st.subheader("Position Limits")

        def _bar(used, mx, label):
            pct   = used / mx if mx > 0 else 0.0
            color = "#e74c3c" if pct >= 1.0 else ("#f39c12" if pct >= 0.80 else "#2ecc71")
            st.markdown(f"**{label}** — {used}/{mx}")
            st.progress(min(pct, 1.0))

        _bar(cap["positions_used"], cap["positions_max"], "Total positions")

        dep_pct = cap["deployed_usd"] / cap["deploy_max_usd"] if cap["deploy_max_usd"] > 0 else 0.0
        st.markdown(
            f"**Capital deployed** — {dep_pct:.0%}  "
            f"(${cap['deployed_usd']:,.0f} / ${cap['deploy_max_usd']:,.0f})"
        )
        st.progress(min(dep_pct, 1.0))

        st.markdown("**By strategy**")
        for strat, info in cap["by_strategy"].items():
            pct = info["used"] / info["max"] if info["max"] > 0 else 0.0
            st.markdown(f"  {strat.replace('_', ' ').title()}: {info['used']}/{info['max']}")
            st.progress(min(pct, 1.0))

    with col_pie:
        st.subheader("Sector Diversification")
        if div["by_sector"]:
            sector_labels = list(div["by_sector"].keys())
            sector_values = list(div["by_sector"].values())
            fig_pie = px.pie(
                values=sector_values,
                names=sector_labels,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_traces(textinfo="label+percent", pull=[0.02] * len(sector_labels))
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                height=260,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Sector concentration bars
            for sector, pct in sorted(div["sector_pcts"].items(), key=lambda x: -x[1]):
                icon = "🔴" if pct > PortfolioManager.MAX_SECTOR_PCT else (
                    "⚠️" if pct > PortfolioManager.MAX_SECTOR_PCT * PortfolioManager.WARN_THRESHOLD
                    else "🟢"
                )
                st.caption(
                    f"{icon} {sector}: {div['by_sector'].get(sector, 0)} pos · {pct:.0%} of portfolio"
                )
        else:
            st.info("No open positions.")

    with col_strat:
        st.subheader("Risk Metrics")

        def _metric_row(label, value, warn_val=None, fmt="{:.2f}"):
            if value is None:
                st.metric(label, "—")
            else:
                icon = ""
                if warn_val is not None and abs(value) > warn_val:
                    icon = " ⚠️"
                st.metric(label, fmt.format(value) + icon)

        _metric_row("Portfolio Beta",       risk["beta"],
                    warn_val=1.5, fmt="{:.2f}")
        _metric_row("Volatility (ann.)",    risk["volatility"],
                    warn_val=0.30, fmt="{:.1%}")
        _metric_row("Max Concentration",    risk["max_concentration"],
                    warn_val=PortfolioManager.MAX_POSITION_PCT, fmt="{:.1%}")
        _metric_row("Avg Correlation",      risk["avg_correlation"],
                    warn_val=PortfolioManager.WARN_CORRELATION, fmt="{:.2f}")
        _metric_row("Cash Reserve",         cap["cash_reserve"] / pm_balance if pm_balance else None,
                    fmt="{:.1%}")

    st.markdown("---")

    # ── Row 2: Correlation heatmap + open positions table ─────────────────────
    col_hm, col_pos = st.columns([1, 1.4])

    with col_hm:
        st.subheader("Correlation Heatmap (30d)")
        if not corr_df.empty and corr_df.shape[0] >= 2:
            fig_hm = go.Figure(
                go.Heatmap(
                    z        = corr_df.values,
                    x        = corr_df.columns.tolist(),
                    y        = corr_df.index.tolist(),
                    colorscale = "RdYlGn_r",
                    zmin     = -1,
                    zmax     = 1,
                    text     = [[f"{v:.2f}" for v in row] for row in corr_df.values],
                    texttemplate = "%{text}",
                    showscale = True,
                )
            )
            fig_hm.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            if risk["avg_correlation"] is not None:
                ac = risk["avg_correlation"]
                color = "red" if ac > PortfolioManager.MAX_CORRELATION else (
                    "orange" if ac > PortfolioManager.WARN_CORRELATION else "green"
                )
                st.markdown(
                    f"Average pairwise: "
                    f"<span style='color:{color};font-weight:bold'>{ac:.3f}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Need ≥ 2 open positions to compute correlations.")

    with col_pos:
        st.subheader("Open Positions")
        positions = div.get("positions", [])
        if positions:
            total_val = div["total_value"]
            rows = []
            for p in sorted(positions, key=lambda x: -x["current_value"]):
                weight = p["current_value"] / total_val if total_val > 0 else 0.0
                rows.append({
                    "Ticker":    p["ticker"],
                    "Shares":    p["shares"],
                    "Avg Price": fmt_usd(p["avg_price"]),
                    "Value":     fmt_usd(p["current_value"]),
                    "Weight":    f"{weight:.1%}",
                    "Strategy":  p["strategy"].replace("_", " ").title(),
                    "Sector":    p["sector"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        elif not open_df.empty:
            # Fall back to raw portfolio table (no metadata registered yet)
            d = open_df.copy()
            d["avg_price"]     = d["avg_price"].apply(fmt_usd)
            d["current_value"] = d["current_value"].apply(fmt_usd)
            d.columns = ["Ticker", "Shares", "Avg Price", "Entry Value"]
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.caption(
                "Strategy/sector data not available. "
                "Use PortfolioManager.register_position() when opening trades."
            )
        else:
            st.info("No open positions. Run with `--execute` to log trades.")

    st.markdown("---")

    # ── Portfolio violations log ───────────────────────────────────────────────
    st.subheader("Recent Portfolio Violations")
    violations = query(
        "SELECT created_at, ticker, strategy, amount_usd, violation_type, reason "
        "FROM portfolio_violations ORDER BY id DESC LIMIT 25"
    )
    if violations.empty:
        st.info("No violations recorded — all trades passed portfolio checks.")
    else:
        violations["created_at"] = pd.to_datetime(violations["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        violations.columns = ["Time", "Ticker", "Strategy", "Amount ($)", "Type", "Reason"]
        st.dataframe(violations, use_container_width=True, hide_index=True)

    # ── Historical snapshot chart ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Snapshots Over Time")
    snaps = query(
        "SELECT snapshot_at, open_positions, total_value, deployed_pct, "
        "avg_correlation, portfolio_beta "
        "FROM portfolio_snapshots ORDER BY snapshot_at"
    )
    if snaps.empty:
        st.info(
            "No snapshots saved yet. Run `python3 -m execution.portfolio_manager "
            "--balance 10000 --save-snapshot` to capture one."
        )
    else:
        snaps["snapshot_at"] = pd.to_datetime(snaps["snapshot_at"])
        fig_snaps = px.line(
            snaps,
            x="snapshot_at",
            y="total_value",
            title="Portfolio Value Over Snapshots",
            labels={"snapshot_at": "Time", "total_value": "Entry Value ($)"},
        )
        fig_snaps.update_layout(margin=dict(t=30, b=10))
        st.plotly_chart(fig_snaps, use_container_width=True)

        cols_show = [c for c in ["snapshot_at", "open_positions", "total_value",
                                  "deployed_pct", "avg_correlation", "portfolio_beta"]
                     if c in snaps.columns]
        d = snaps[cols_show].copy()
        if "deployed_pct" in d.columns:
            d["deployed_pct"] = (d["deployed_pct"] * 100).round(1).astype(str) + "%"
        if "avg_correlation" in d.columns:
            d["avg_correlation"] = d["avg_correlation"].round(3)
        if "portfolio_beta" in d.columns:
            d["portfolio_beta"] = d["portfolio_beta"].round(2)
        d.columns = [c.replace("_", " ").title() for c in d.columns]
        st.dataframe(d.tail(20), use_container_width=True, hide_index=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtesting":
    st.title("Backtesting")

    tab_run, tab_saved, tab_compare = st.tabs(
        ["▶  Run Backtest", "📋  Saved Results", "🔀  Strategy Comparison"]
    )

    # ── Tab: Run ──────────────────────────────────────────────────────────────
    with tab_run:
        COMMON_TICKERS = [
            "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL",
            "AMZN", "META", "NFLX", "AMD", "INTC",
            "SPY", "QQQ", "JPM", "BAC", "COIN",
        ]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bt_ticker = st.selectbox("Ticker", COMMON_TICKERS, key="bt_ticker")
        with c2:
            bt_start = st.date_input("Start date", value=date(2024, 1, 1), key="bt_start")
        with c3:
            bt_end = st.date_input("End date", value=date(2025, 1, 1), key="bt_end")
        with c4:
            bt_sentiment = st.selectbox(
                "Sentiment mode",
                ["random", "bullish", "bearish", "neutral"],
                key="bt_sentiment",
                help=(
                    "random — per-date seeded random signal (reproducible)\n"
                    "bullish/bearish/neutral — fixed sentiment every day"
                ),
            )

        c5, _ = st.columns([1, 3])
        with c5:
            bt_balance = st.number_input(
                "Initial balance ($)", min_value=1_000, max_value=1_000_000,
                value=10_000, step=1_000, key="bt_balance",
            )

        run_clicked = st.button("▶  Run Backtest", type="primary")

        if run_clicked:
            if bt_start >= bt_end:
                st.error("Start date must be before end date.")
            else:
                with st.spinner(
                    f"Running {bt_ticker} backtest "
                    f"({bt_start} → {bt_end}, sentiment={bt_sentiment})…"
                ):
                    try:
                        from backtest.engine import BacktestEngine  # lazy import
                        engine = BacktestEngine(
                            ticker          = bt_ticker,
                            start_date      = str(bt_start),
                            end_date        = str(bt_end),
                            initial_balance = float(bt_balance),
                            sentiment_mode  = bt_sentiment,
                            verbose         = False,
                        )
                        result = engine.run()
                        fig    = engine.plot(result, show=False, save_path=None)
                        st.session_state["bt_result"] = result
                        st.session_state["bt_engine"] = engine
                        st.session_state["bt_fig"]    = fig
                        st.cache_data.clear()   # refresh saved-results tab
                        st.success(
                            f"Backtest complete — "
                            f"{result.total_return_pct:+.2f}% return  |  "
                            f"{result.total_trades} trades  |  "
                            f"saved to DB #{result.db_id}"
                        )
                    except Exception as exc:
                        st.error(f"Backtest failed: {exc}")
                        st.session_state.pop("bt_result", None)

        # ── Results ───────────────────────────────────────────────────────────
        if "bt_result" in st.session_state:
            result = st.session_state["bt_result"]
            fig    = st.session_state["bt_fig"]
            alpha  = result.total_return_pct - result.buy_and_hold_return_pct

            st.markdown("---")
            st.subheader(
                f"{result.ticker}  ·  {result.start_date} → {result.end_date}  "
                f"·  sentiment: {st.session_state['bt_sentiment']}"
            )

            # ── KPI strip ─────────────────────────────────────────────────────
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric(
                "Total Return", f"{result.total_return_pct:+.2f}%",
                delta=f"{alpha:+.1f}% vs B&H", delta_color="normal",
            )
            k2.metric("Buy & Hold",   f"{result.buy_and_hold_return_pct:+.2f}%")
            k3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
            k4.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
            k5.metric("Win Rate",     f"{result.win_rate_pct:.1f}%")
            k6.metric("Trades",       result.total_trades,
                      help=f"W: {result.winning_trades}  L: {result.losing_trades}")

            # ── Chart (equity + drawdown + heatmap) ───────────────────────────
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # ── Stats + trade log side by side ────────────────────────────────
            col_stats, col_trades = st.columns([1, 1.6])

            with col_stats:
                st.subheader("Performance Summary")
                rows = [
                    ("Ticker",            result.ticker),
                    ("Period",            f"{result.start_date} → {result.end_date}"),
                    ("Initial Capital",   fmt_usd(result.initial_balance)),
                    ("Final Capital",     fmt_usd(result.final_balance)),
                    ("Total Return",      f"{result.total_return_pct:+.2f}%"),
                    ("Buy & Hold Return", f"{result.buy_and_hold_return_pct:+.2f}%"),
                    ("Alpha",             f"{alpha:+.2f}%"),
                    ("Sharpe Ratio",      f"{result.sharpe_ratio:.3f}"),
                    ("Max Drawdown",      f"{result.max_drawdown_pct:.2f}%"),
                    ("Total Trades",      str(result.total_trades)),
                    ("Winning Trades",    str(result.winning_trades)),
                    ("Losing Trades",     str(result.losing_trades)),
                    ("Win Rate",          f"{result.win_rate_pct:.1f}%"),
                    ("Avg Win",           fmt_usd(result.avg_win)),
                    ("Avg Loss",          fmt_usd(result.avg_loss)),
                ]
                st.dataframe(
                    pd.DataFrame(rows, columns=["Metric", "Value"]),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_trades:
                st.subheader(f"Trade Log ({result.total_trades} trades)")
                if result.trades:
                    trade_rows = []
                    for t in result.trades:
                        trade_rows.append({
                            "Entry Date":  str(t.entry_date.date()),
                            "Exit Date":   str(t.exit_date.date()) if t.exit_date else "open",
                            "Signal":      t.signal,
                            "Conf.":       f"{t.confidence:.0f}%",
                            "Shares":      t.shares,
                            "Entry $":     fmt_usd(t.entry_price),
                            "Exit $":      fmt_usd(t.exit_price),
                            "P&L":         f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}",
                            "Exit Reason": t.exit_reason,
                        })
                    st.dataframe(
                        pd.DataFrame(trade_rows),
                        use_container_width=True,
                        hide_index=True,
                        height=440,
                    )
                else:
                    st.info("No trades were executed in this period.")

            # ── Download ──────────────────────────────────────────────────────
            st.markdown("---")
            dl1, dl2 = st.columns(2)

            if result.trades:
                csv_data = pd.DataFrame([
                    {
                        "ticker":       result.ticker,
                        "entry_date":   str(t.entry_date.date()),
                        "exit_date":    str(t.exit_date.date()) if t.exit_date else "",
                        "signal":       t.signal,
                        "confidence":   t.confidence,
                        "shares":       t.shares,
                        "entry_price":  t.entry_price,
                        "exit_price":   t.exit_price if t.exit_price else "",
                        "stop_loss":    t.stop_loss,
                        "take_profit":  t.take_profit,
                        "pnl":          t.pnl,
                        "exit_reason":  t.exit_reason,
                    }
                    for t in result.trades
                ]).to_csv(index=False)

                with dl1:
                    st.download_button(
                        label="⬇  Download Trade Log (CSV)",
                        data=csv_data,
                        file_name=(
                            f"backtest_{result.ticker}_"
                            f"{result.start_date}_{result.end_date}.csv"
                        ),
                        mime="text/csv",
                    )

            # Download equity curve
            eq_csv = result.equity_curve.rename("portfolio_value").reset_index()
            eq_csv.columns = ["date", "portfolio_value"]
            with dl2:
                st.download_button(
                    label="⬇  Download Equity Curve (CSV)",
                    data=eq_csv.to_csv(index=False),
                    file_name=(
                        f"equity_{result.ticker}_"
                        f"{result.start_date}_{result.end_date}.csv"
                    ),
                    mime="text/csv",
                )

    # ── Tab: Saved Results ────────────────────────────────────────────────────
    with tab_saved:
        st.subheader("Saved Backtest Results")
        saved = query(
            "SELECT * FROM backtest_results ORDER BY created_at DESC"
        )

        if saved.empty:
            st.info("No saved backtests yet. Run one using the ▶ Run Backtest tab.")
        else:
            # Filters
            sf1, sf2 = st.columns(2)
            saved_tickers = ["All"] + sorted(saved["ticker"].unique().tolist())
            saved_sents   = ["All", "random", "bullish", "bearish", "neutral"]
            with sf1:
                sel_st = st.selectbox("Ticker", saved_tickers, key="saved_bt_ticker")
            with sf2:
                sel_ss = st.selectbox("Sentiment mode", saved_sents, key="saved_bt_sent")

            filt = saved.copy()
            if sel_st != "All":
                filt = filt[filt["ticker"] == sel_st]
            if sel_ss != "All":
                filt = filt[filt["sentiment_mode"] == sel_ss]

            st.caption(f"{len(filt)} saved backtest(s)")

            d = filt[[
                "created_at", "ticker", "start_date", "end_date",
                "initial_balance", "final_balance",
                "total_return_pct", "buy_and_hold_return_pct",
                "sharpe_ratio", "max_drawdown_pct",
                "win_rate_pct", "total_trades", "sentiment_mode",
            ]].copy()

            d["initial_balance"]         = d["initial_balance"].apply(fmt_usd)
            d["final_balance"]           = d["final_balance"].apply(fmt_usd)
            d["total_return_pct"]        = d["total_return_pct"].apply(lambda x: f"{x:+.2f}%")
            d["buy_and_hold_return_pct"] = d["buy_and_hold_return_pct"].apply(lambda x: f"{x:+.2f}%")
            d["sharpe_ratio"]            = d["sharpe_ratio"].apply(lambda x: f"{x:.3f}")
            d["max_drawdown_pct"]        = d["max_drawdown_pct"].apply(lambda x: f"{x:.2f}%")
            d["win_rate_pct"]            = d["win_rate_pct"].apply(lambda x: f"{x:.1f}%")
            d["created_at"]              = pd.to_datetime(d["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

            d.columns = [
                "Run At", "Ticker", "Start", "End",
                "Initial $", "Final $",
                "Return", "B&H Return",
                "Sharpe", "Max DD",
                "Win Rate", "Trades", "Sentiment",
            ]
            st.dataframe(d, use_container_width=True, hide_index=True)

            # Download all saved results
            st.download_button(
                label="⬇  Download All Results (CSV)",
                data=filt.drop(columns=["trades_json"], errors="ignore").to_csv(index=False),
                file_name="backtest_results_all.csv",
                mime="text/csv",
            )


    # ── Tab: Strategy Comparison ──────────────────────────────────────────────
    with tab_compare:
        st.subheader("Strategy Comparison")
        st.caption(
            "Backtest Momentum, Mean Reversion, and Swing strategies on the "
            "same ticker / date range and compare performance side-by-side."
        )

        _STRATEGY_COLORS = {
            "Momentum":    "#4fc3f7",
            "Mean Rev.":   "#a5d6a7",
            "Swing":       "#ffcc80",
            "Buy & Hold":  "#90a4ae",
        }

        COMPARE_TICKERS = [
            "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL",
            "AMZN", "META", "NFLX", "AMD", "INTC",
            "SPY", "QQQ",
        ]

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            cmp_ticker = st.selectbox("Ticker", COMPARE_TICKERS, key="cmp_ticker")
        with cc2:
            cmp_start = st.date_input(
                "Start date", value=date(2024, 1, 1), key="cmp_start"
            )
        with cc3:
            cmp_end = st.date_input(
                "End date", value=date(2025, 1, 1), key="cmp_end"
            )
        with cc4:
            cmp_balance = st.number_input(
                "Initial balance ($)", min_value=1_000, max_value=1_000_000,
                value=10_000, step=1_000, key="cmp_balance",
            )

        cmp_clicked = st.button("▶  Run Comparison", type="primary", key="cmp_run")

        if cmp_clicked:
            if cmp_start >= cmp_end:
                st.error("Start date must be before end date.")
            else:
                with st.spinner(
                    f"Running all 3 strategies on {cmp_ticker} "
                    f"({cmp_start} → {cmp_end})…"
                ):
                    try:
                        from backtest.engine import BacktestEngine
                        cmp_engine = BacktestEngine(
                            ticker          = cmp_ticker,
                            start_date      = str(cmp_start),
                            end_date        = str(cmp_end),
                            initial_balance = float(cmp_balance),
                            verbose         = False,
                        )
                        cmp_result = cmp_engine.compare_strategies()
                        cmp_fig    = cmp_engine.plot_comparison(
                            cmp_result, show=False, save_path=None
                        )
                        st.session_state["cmp_result"] = cmp_result
                        st.session_state["cmp_fig"]    = cmp_fig
                        st.cache_data.clear()
                        winner_label = {
                            "momentum":       "Momentum",
                            "mean_reversion": "Mean Reversion",
                            "swing":          "Swing",
                        }.get(cmp_result["winner"], cmp_result["winner"])
                        st.success(
                            f"Done — winner: **{winner_label}** "
                            f"(best Sharpe ratio)"
                        )
                    except Exception as exc:
                        st.error(f"Comparison failed: {exc}")
                        st.session_state.pop("cmp_result", None)

        if "cmp_result" in st.session_state:
            cmp_r   = st.session_state["cmp_result"]
            cmp_f   = st.session_state["cmp_fig"]
            df_cmp  = cmp_r["comparison_df"]
            winner  = cmp_r["winner"]
            results = cmp_r["strategies"]

            _W_LABELS = {
                "momentum":       "Momentum",
                "mean_reversion": "Mean Reversion",
                "swing":          "Swing",
            }
            winner_label = _W_LABELS.get(winner, winner)

            st.markdown("---")
            st.subheader(
                f"{cmp_r['ticker']}  ·  "
                f"{cmp_r['start_date']} → {cmp_r['end_date']}"
            )

            # ── Winner highlight ──────────────────────────────────────────────
            if winner in results:
                wr = results[winner]
                st.success(
                    f"🏆 **Winner: {winner_label}**  —  "
                    f"Sharpe {wr.sharpe_ratio:.2f}  |  "
                    f"Return {wr.total_return_pct:+.2f}%  |  "
                    f"Win Rate {wr.win_rate_pct:.0f}%  |  "
                    f"{wr.total_trades} trades"
                )

            # ── KPI strip (one column per strategy + B&H) ─────────────────────
            strat_keys = [s for s in ["momentum", "mean_reversion", "swing"]
                          if s in results]
            cols = st.columns(len(strat_keys) + 1)
            for col, strat in zip(cols, strat_keys):
                r     = results[strat]
                label = _W_LABELS.get(strat, strat)
                delta = "★ Winner" if strat == winner else None
                col.metric(
                    label,
                    f"{r.total_return_pct:+.2f}%",
                    delta=delta,
                    delta_color="off" if delta else "normal",
                    help=(
                        f"Sharpe: {r.sharpe_ratio:.2f}  "
                        f"MaxDD: {r.max_drawdown_pct:.1f}%  "
                        f"WR: {r.win_rate_pct:.0f}%  "
                        f"Trades: {r.total_trades}"
                    ),
                )
            bh = cmp_r["buy_and_hold"]
            cols[-1].metric(
                "Buy & Hold",
                f"{bh['return_pct']:+.2f}%",
                help=f"Sharpe: {bh['sharpe']:.2f}  MaxDD: {bh['max_dd']:.1f}%",
            )

            # ── Equity curves chart ───────────────────────────────────────────
            st.plotly_chart(cmp_f, use_container_width=True)

            st.markdown("---")

            # ── Metrics table ─────────────────────────────────────────────────
            st.subheader("Performance Table")

            disp = df_cmp.copy()
            disp["Return (%)"]   = disp["Return (%)"].apply(lambda x: f"{x:+.2f}%")
            disp["Sharpe"]       = disp["Sharpe"].apply(lambda x: f"{x:.2f}")
            disp["Max DD (%)"]   = disp["Max DD (%)"].apply(lambda x: f"{x:.1f}%")
            disp["Win Rate (%)"] = disp["Win Rate (%)"].apply(lambda x: f"{x:.0f}%")
            disp["Avg Hold (d)"] = disp["Avg Hold (d)"].apply(lambda x: f"{x:.1f}d")
            disp.columns = [
                "Strategy", "Return", "Sharpe", "Max DD",
                "Win Rate", "Trades", "Avg Hold",
            ]
            st.dataframe(disp, use_container_width=True, hide_index=True)

            # ── Trade distribution ────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Trade Distribution")
            td_cols = st.columns(len(strat_keys))
            for col, strat in zip(td_cols, strat_keys):
                r     = results[strat]
                label = _W_LABELS.get(strat, strat)
                with col:
                    st.markdown(f"**{label}**")
                    if r.trades:
                        import plotly.express as px_local
                        pnl_vals = [t.pnl for t in r.trades if t.exit_price]
                        fig_hist = px_local.histogram(
                            x=pnl_vals,
                            nbins=20,
                            labels={"x": "P&L ($)"},
                            color_discrete_sequence=[
                                _STRATEGY_COLORS.get(
                                    {"momentum": "Momentum",
                                     "mean_reversion": "Mean Rev.",
                                     "swing": "Swing"}.get(strat, strat),
                                    "#4fc3f7"
                                )
                            ],
                        )
                        fig_hist.add_vline(x=0, line_dash="dash",
                                           line_color="gray", opacity=0.6)
                        fig_hist.update_layout(
                            height=200,
                            margin=dict(t=5, b=5, l=5, r=5),
                            showlegend=False,
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("No trades.")

            # ── Download ──────────────────────────────────────────────────────
            st.markdown("---")
            st.download_button(
                label="⬇  Download Comparison CSV",
                data=df_cmp.to_csv(index=False),
                file_name=(
                    f"strategy_comparison_{cmp_r['ticker']}_"
                    f"{cmp_r['start_date']}_{cmp_r['end_date']}.csv"
                ),
                mime="text/csv",
            )

        # ── Saved comparison results ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("Saved Comparison Results")
        saved_cmp = query(
            "SELECT * FROM backtest_strategy_comparison ORDER BY created_at DESC"
        )
        if saved_cmp.empty:
            st.info(
                "No saved comparisons yet. "
                "Run one using the ▶ Run Comparison button above."
            )
        else:
            sf1, sf2 = st.columns(2)
            sv_tickers = ["All"] + sorted(saved_cmp["ticker"].unique().tolist())
            sv_strats  = ["All"] + sorted(saved_cmp["strategy"].unique().tolist())
            with sf1:
                sel_sv_t = st.selectbox("Ticker", sv_tickers, key="sv_cmp_ticker")
            with sf2:
                sel_sv_s = st.selectbox("Strategy", sv_strats, key="sv_cmp_strat")

            filt_cmp = saved_cmp.copy()
            if sel_sv_t != "All":
                filt_cmp = filt_cmp[filt_cmp["ticker"] == sel_sv_t]
            if sel_sv_s != "All":
                filt_cmp = filt_cmp[filt_cmp["strategy"] == sel_sv_s]

            st.caption(f"{len(filt_cmp)} result(s)")

            d_sv = filt_cmp[[
                "created_at", "ticker", "start_date", "end_date", "strategy",
                "total_return", "sharpe", "max_dd", "win_rate",
                "trade_count", "avg_hold_days",
            ]].copy()
            d_sv["total_return"]  = d_sv["total_return"].apply(lambda x: f"{x:+.2f}%")
            d_sv["sharpe"]        = d_sv["sharpe"].apply(lambda x: f"{x:.2f}")
            d_sv["max_dd"]        = d_sv["max_dd"].apply(lambda x: f"{x:.1f}%")
            d_sv["win_rate"]      = d_sv["win_rate"].apply(lambda x: f"{x:.0f}%")
            d_sv["avg_hold_days"] = d_sv["avg_hold_days"].apply(lambda x: f"{x:.1f}d")
            d_sv["created_at"]    = pd.to_datetime(d_sv["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            d_sv.columns = [
                "Run At", "Ticker", "Start", "End", "Strategy",
                "Return", "Sharpe", "Max DD", "Win Rate",
                "Trades", "Avg Hold",
            ]
            st.dataframe(d_sv, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Screener":
    st.title("🔍 Stock Screener")
    st.caption(
        "Scans DAX 40 · MDAX · SDAX · TecDAX · S&P 500 · NASDAQ 100 · "
        "EURO STOXX 50 · FTSE 100 · CAC 40 for momentum candidates."
    )

    tab_run, tab_saved = st.tabs(["▶  Run Screener", "📋  Saved Runs"])

    # ── Shared colour maps ─────────────────────────────────────────────────────
    COUNTRY_COLORS = {
        "DE": "#1565c0", "US": "#b71c1c", "GB": "#880e4f",
        "FR": "#1b5e20", "EU": "#37474f",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Run Screener
    # ══════════════════════════════════════════════════════════════════════════
    with tab_run:
        # ── Controls ──────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            sc_markets = st.multiselect(
                "Markets to scan",
                options=["US", "DE", "EU"],
                default=["US", "DE", "EU"],
                key="sc_markets",
            )
        with c2:
            sc_focus = st.selectbox(
                "Focus market",
                options=["DE", "US", "EU", "None"],
                index=0,
                key="sc_focus",
                help="Gets +0.2 priority bonus and a minimum 10-stock quota in results.",
            )
        with c3:
            sc_top = st.slider(
                "Top candidates", min_value=10, max_value=80,
                value=40, step=5, key="sc_top",
            )

        st.caption(
            "**Tip:** Live fetch of 150–500 tickers from yfinance. "
            "Expect 30–90 s depending on network speed and selected markets."
        )

        if st.button("▶  Run Screener", type="primary", key="sc_run_btn"):
            if not sc_markets:
                st.error("Select at least one market to scan.")
            else:
                focus_val = None if sc_focus == "None" else sc_focus
                with st.spinner(
                    f"Scanning {', '.join(sc_markets)} "
                    f"(focus: {focus_val or '—'}, top {sc_top})…"
                ):
                    try:
                        from agents.screener_agent import ScreenerAgent  # lazy import
                        sc_result = ScreenerAgent().run(
                            markets=sc_markets,
                            focus_market=focus_val,
                            top=sc_top,
                        )
                        st.session_state["sc_result"] = sc_result
                        st.cache_data.clear()
                        st.success(
                            f"Done — {sc_result['screened']} candidates from "
                            f"{sc_result['universe_size']} tickers → "
                            f"top {len(sc_result['candidates'])} returned."
                        )
                    except Exception as exc:
                        st.error(f"Screener failed: {exc}")
                        st.session_state.pop("sc_result", None)

        # ── Results ───────────────────────────────────────────────────────────
        if "sc_result" in st.session_state:
            sc_result  = st.session_state["sc_result"]
            candidates = sc_result["candidates"]

            if not candidates:
                st.warning(
                    "No candidates passed the filters. "
                    "Markets may be closed or the move thresholds weren't met."
                )
            else:
                df = pd.DataFrame(candidates)

                st.markdown("---")

                # ── KPI strip ─────────────────────────────────────────────────
                de_count  = int((df["country"] == "DE").sum())
                us_count  = int((df["country"] == "US").sum())
                top_row   = df.iloc[0]

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Universe",      sc_result["universe_size"])
                k2.metric("Passed filters", sc_result["screened"])
                k3.metric("Returned",      len(candidates))
                k4.metric("🇩🇪 DE stocks",  de_count)
                k5.metric("🇺🇸 US stocks",  us_count)
                k6.metric(
                    "Top pick",
                    top_row["ticker"],
                    help=f"Hotness {top_row['hotness']:.2f}  |  "
                         f"{top_row['price_change']:+.2f}%  |  "
                         f"{top_row['volume_ratio']:.1f}× vol",
                )

                st.markdown("---")

                # ── Charts row ────────────────────────────────────────────────
                ch1, ch2, ch3 = st.columns([2, 1, 1])

                with ch1:
                    st.subheader("Hotness Ranking — Top 20")
                    top20 = df.head(20).copy()
                    fig_bar = px.bar(
                        top20.sort_values("hotness"),
                        x="hotness",
                        y="ticker",
                        color="market",
                        orientation="h",
                        labels={"hotness": "Hotness (0–10)", "ticker": ""},
                        hover_data=["price_change", "volume_ratio", "rsi",
                                    "country", "exchange"],
                        height=520,
                    )
                    fig_bar.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis={"categoryorder": "total ascending"},
                        legend=dict(
                            orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1,
                        ),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with ch2:
                    st.subheader("By Country")
                    ctry = df["country"].value_counts().reset_index()
                    ctry.columns = ["Country", "Count"]
                    fig_pie = px.pie(
                        ctry,
                        names="Country",
                        values="Count",
                        color="Country",
                        color_discrete_map=COUNTRY_COLORS,
                        hole=0.4,
                        height=260,
                    )
                    fig_pie.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.subheader("By Index")
                    mkts = df["market"].value_counts().reset_index()
                    mkts.columns = ["Index", "Count"]
                    fig_idx = px.bar(
                        mkts.sort_values("Count"),
                        x="Count",
                        y="Index",
                        orientation="h",
                        labels={"Count": "Candidates", "Index": ""},
                        height=240,
                    )
                    fig_idx.update_layout(
                        margin=dict(t=10, b=10),
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_idx, use_container_width=True)

                with ch3:
                    st.subheader("RSI Distribution")
                    rsi_df = df["rsi"].dropna()
                    if not rsi_df.empty:
                        fig_rsi = px.histogram(
                            df.dropna(subset=["rsi"]),
                            x="rsi",
                            nbins=15,
                            color="country",
                            color_discrete_map=COUNTRY_COLORS,
                            labels={"rsi": "RSI", "count": "Count"},
                            height=240,
                        )
                        fig_rsi.add_vline(x=30, line_dash="dash",
                                          line_color="green", opacity=0.7,
                                          annotation_text="Oversold")
                        fig_rsi.add_vline(x=70, line_dash="dash",
                                          line_color="red", opacity=0.7,
                                          annotation_text="Overbought")
                        fig_rsi.update_layout(
                            margin=dict(t=10, b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.info("No RSI data.")

                    st.subheader("Price Change Distribution")
                    fig_chg = px.histogram(
                        df,
                        x="price_change",
                        nbins=15,
                        color="country",
                        color_discrete_map=COUNTRY_COLORS,
                        labels={"price_change": "Price Change (%)", "count": "Count"},
                        height=240,
                    )
                    fig_chg.add_vline(x=0, line_dash="dash",
                                      line_color="gray", opacity=0.5)
                    fig_chg.update_layout(
                        margin=dict(t=10, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_chg, use_container_width=True)

                st.markdown("---")

                # ── Volume spike vs price move scatter ────────────────────────
                st.subheader("Volume Spike vs. Price Move")
                fig_sc = px.scatter(
                    df,
                    x="volume_ratio",
                    y="price_change",
                    size="hotness",
                    color="market",
                    text="ticker",
                    hover_data=["rsi", "hotness", "country", "exchange", "avg_volume"],
                    labels={
                        "volume_ratio": "Volume Ratio  (today ÷ 20-day avg)",
                        "price_change": "Price Change (%)",
                    },
                    height=440,
                    size_max=30,
                )
                fig_sc.update_traces(textposition="top center", textfont_size=9)
                fig_sc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                fig_sc.update_layout(margin=dict(t=20, b=20))
                st.plotly_chart(fig_sc, use_container_width=True)

                st.markdown("---")

                # ── Candidates table ──────────────────────────────────────────
                st.subheader(f"All Candidates ({len(df)})")

                tf1, tf2, tf3 = st.columns(3)
                with tf1:
                    tbl_mkts = st.multiselect(
                        "Filter by index",
                        options=sorted(df["market"].unique().tolist()),
                        default=[],
                        key="sc_tbl_mkt",
                        placeholder="All indices",
                    )
                with tf2:
                    tbl_ctry = st.multiselect(
                        "Filter by country",
                        options=sorted(df["country"].unique().tolist()),
                        default=[],
                        key="sc_tbl_ctry",
                        placeholder="All countries",
                    )
                with tf3:
                    min_hot = st.slider(
                        "Min hotness", 0.0, 10.0, 0.0, 0.5, key="sc_min_hot"
                    )

                tbl = df.copy()
                if tbl_mkts:
                    tbl = tbl[tbl["market"].isin(tbl_mkts)]
                if tbl_ctry:
                    tbl = tbl[tbl["country"].isin(tbl_ctry)]
                tbl = tbl[tbl["hotness"] >= min_hot]

                st.caption(f"{len(tbl)} candidate(s) shown")

                d = tbl[[
                    "ticker", "market", "exchange", "country",
                    "hotness", "price_change", "volume_ratio", "rsi", "avg_volume", "price",
                ]].copy()
                d["hotness"]      = d["hotness"].apply(lambda x: f"{x:.2f}")
                d["price_change"] = d["price_change"].apply(lambda x: f"{x:+.2f}%")
                d["volume_ratio"] = d["volume_ratio"].apply(lambda x: f"{x:.1f}×")
                d["rsi"]          = d["rsi"].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                )
                d["avg_volume"]   = d["avg_volume"].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "—"
                )
                d["price"]        = d["price"].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                )
                d.columns = [
                    "Ticker", "Index", "Exchange", "Country",
                    "Hotness", "Chg%", "Vol×", "RSI", "Avg Vol", "Price",
                ]
                st.dataframe(d, use_container_width=True, hide_index=True)

                st.download_button(
                    label="⬇  Download Candidates (CSV)",
                    data=tbl.to_csv(index=False),
                    file_name=f"screener_{sc_result['run_at'][:10]}.csv",
                    mime="text/csv",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Saved Runs
    # ══════════════════════════════════════════════════════════════════════════
    with tab_saved:
        st.subheader("Saved Screener Runs")
        saved = query(
            "SELECT * FROM screener_results ORDER BY run_at DESC, hotness DESC"
        )

        if saved.empty:
            st.info(
                "No saved runs yet. "
                "Use the ▶ Run Screener tab — results are persisted automatically."
            )
        else:
            # ── Run selector ──────────────────────────────────────────────────
            run_dates = sorted(saved["run_at"].unique().tolist(), reverse=True)
            run_labels = {
                r: (
                    f"{r[:19].replace('T', ' ')} UTC  "
                    f"({int((saved['run_at'] == r).sum())} stocks)"
                )
                for r in run_dates[:50]
            }

            sv1, sv2, sv3 = st.columns(3)
            with sv1:
                sel_run = st.selectbox(
                    "Select run",
                    options=list(run_labels.keys()),
                    format_func=lambda r: run_labels.get(r, r),
                    key="sc_saved_run",
                )
            with sv2:
                sv_mkts = st.multiselect(
                    "Filter by index",
                    options=sorted(saved["market"].unique().tolist()),
                    default=[],
                    key="sc_sv_mkt",
                    placeholder="All indices",
                )
            with sv3:
                sv_ctry = st.multiselect(
                    "Filter by country",
                    options=sorted(saved["country"].dropna().unique().tolist()),
                    default=[],
                    key="sc_sv_ctry",
                    placeholder="All countries",
                )

            run_df = saved[saved["run_at"] == sel_run].copy()
            if sv_mkts:
                run_df = run_df[run_df["market"].isin(sv_mkts)]
            if sv_ctry:
                run_df = run_df[run_df["country"].isin(sv_ctry)]
            run_df = run_df.sort_values("hotness", ascending=False)

            st.caption(f"{len(run_df)} candidate(s) in this run")

            # ── Table ─────────────────────────────────────────────────────────
            d = run_df[[
                "ticker", "name", "market", "exchange", "country",
                "hotness", "price_change", "volume_ratio", "rsi", "avg_volume",
            ]].copy()
            d["hotness"]      = d["hotness"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )
            d["price_change"] = d["price_change"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
            )
            d["volume_ratio"] = d["volume_ratio"].apply(
                lambda x: f"{x:.1f}×" if pd.notna(x) else "—"
            )
            d["rsi"]          = d["rsi"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )
            d["avg_volume"]   = d["avg_volume"].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "—"
            )
            d.columns = [
                "Ticker", "Name", "Index", "Exchange", "Country",
                "Hotness", "Chg%", "Vol×", "RSI", "Avg Vol",
            ]
            st.dataframe(d, use_container_width=True, hide_index=True)

            # ── Charts for saved run ───────────────────────────────────────────
            if len(run_df) >= 3:
                st.markdown("---")
                sv_ch1, sv_ch2 = st.columns([1.6, 1])

                with sv_ch1:
                    st.subheader("Hotness Ranking — Top 20")
                    top20_sv = run_df.head(20).copy()
                    fig_sv = px.bar(
                        top20_sv.sort_values("hotness"),
                        x="hotness",
                        y="ticker",
                        color="market",
                        orientation="h",
                        labels={"hotness": "Hotness Score", "ticker": ""},
                        height=460,
                    )
                    fig_sv.update_layout(
                        margin=dict(t=10, b=10),
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_sv, use_container_width=True)

                with sv_ch2:
                    st.subheader("By Country")
                    ctry_sv = run_df["country"].value_counts().reset_index()
                    ctry_sv.columns = ["Country", "Count"]
                    fig_sv_pie = px.pie(
                        ctry_sv,
                        names="Country",
                        values="Count",
                        color="Country",
                        color_discrete_map=COUNTRY_COLORS,
                        hole=0.4,
                        height=220,
                    )
                    fig_sv_pie.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_sv_pie, use_container_width=True)

                    st.subheader("By Index")
                    mkt_sv = run_df["market"].value_counts().reset_index()
                    mkt_sv.columns = ["Index", "Count"]
                    fig_sv_idx = px.bar(
                        mkt_sv.sort_values("Count"),
                        x="Count",
                        y="Index",
                        orientation="h",
                        labels={"Count": "Stocks", "Index": ""},
                        height=220,
                    )
                    fig_sv_idx.update_layout(
                        margin=dict(t=10, b=10),
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_sv_idx, use_container_width=True)

            # ── Download ──────────────────────────────────────────────────────
            st.download_button(
                label="⬇  Download This Run (CSV)",
                data=run_df.drop(
                    columns=["id", "metrics", "created_at"], errors="ignore"
                ).to_csv(index=False),
                file_name=f"screener_{sel_run[:10]}.csv",
                mime="text/csv",
            )
