# News Trading System — Project Roadmap
*Reviewed: March 2026 | Based on: github.com/lenninnel/news-trading-system*

---

## What's Already Built ✅

The system is production-ready at a solid v1 level. Here's a quick snapshot of what exists:

| Layer | What's Done |
|---|---|
| **Agents** | SentimentAgent (Claude AI), TechnicalAgent (RSI/MACD/BB/SMA-20/50), RiskAgent (Half-Kelly), ScreenerAgent (400+ tickers, 8 markets) |
| **Data** | NewsAPI headlines, Yahoo Finance OHLCV, 4-level fallback chain for both sources |
| **Signal fusion** | Rule-based: sentiment × technical → STRONG BUY / WEAK BUY / HOLD / WEAK SELL / STRONG SELL / CONFLICTING |
| **Risk sizing** | Half-Kelly, 2% max risk/trade, 10% max position, 2:1 take-profit |
| **Execution** | Paper trading with SQLite persistence (10 tables) |
| **Backtesting** | Day-by-day engine with Sharpe, max drawdown, win rate |
| **Dashboard** | Streamlit (7 pages): Overview, Signals, Portfolio, History, Agents, Backtesting, Screener |
| **Notifications** | Telegram bot (real-time signals + EOD summary) |
| **Scheduler** | Daily runner — daemon and cron modes |
| **Deployment** | Docker, Railway.app, Render support |
| **Tests** | 202 passing unit tests |
| **Docs** | Full documentation suite (user manual, architecture, runbook, etc.) |

---

## Roadmap

The roadmap is organised into four phases, prioritised by business value and build complexity.

---

### Phase 1 — Signal Quality Uplift
*Goal: Make existing signals smarter and more reliable*
*Estimated effort: 2–4 weeks*

#### 1.1 Multi-timeframe Technical Analysis
Currently the TechnicalAgent works on a single timeframe. Extending it to confirm signals across multiple timeframes (e.g. 1D + 4H + 1H) reduces false positives significantly.
- Add `timeframe` parameter to TechnicalAgent
- Require signal alignment across ≥2 timeframes before escalating to STRONG signal
- Store per-timeframe results in DB

#### 1.2 Volume & Momentum Confirmation
Price signals without volume confirmation are weaker. Add:
- Volume spike detection (volume > 1.5× 20-day average)
- On-Balance Volume (OBV) trend
- Relative Volume (RVOL) score factored into confidence

#### 1.3 Earnings & Event Calendar Awareness
Add a data source for upcoming earnings dates, FOMC meetings, and economic releases. Flag tickers within 3 days of a high-impact event and downgrade position size automatically (e.g. cap at 50% normal size).
- Integrate `yfinance` earnings calendar
- Add `EarningsAgent` or extend RiskAgent to handle event risk

#### 1.4 Sentiment Source Expansion
NewsAPI alone has limited coverage and a free-tier cap. Add:
- Reddit (r/wallstreetbets, r/investing) via PRAW — free
- StockTwits API — free tier available
- Aggregate into a weighted multi-source sentiment score

---

### Phase 2 — Live Trading Integration
*Goal: Graduate from paper trading to real execution*
*Estimated effort: 3–5 weeks*

#### 2.1 Broker Integration — Alpaca (Recommended First)
Alpaca has a free paper + live API, commission-free US stocks, and a clean Python SDK. It is the lowest-friction path to live execution.
- Create `execution/alpaca_trader.py` mirroring the `PaperTrader` interface
- Support market orders, limit orders, and bracket orders (entry + stop-loss + take-profit in one order)
- Sync open positions back from broker to dashboard

#### 2.2 Interactive Brokers (IBKR) — for EU/German Stocks
Since the screener already covers DAX / MDAX / Euro Stoxx 50, IBKR is the natural choice for European market access.
- Use `ib_insync` library
- Gate behind a `BROKER=ibkr` env var so users can switch

#### 2.3 Order Management Layer
Currently execution is fire-and-forget. Add proper order lifecycle management:
- Track order states: pending → filled → partially filled → cancelled
- Handle partial fills and re-route remainder
- Expose order status in the dashboard Portfolio page

#### 2.4 Live/Paper Mode Toggle
Add a `TRADING_MODE=paper|live` environment variable and ensure zero code changes are needed to switch. Include a confirmation prompt before placing any live order.

---

### Phase 3 — Intelligence Upgrades
*Goal: Replace rule-based signal fusion with learned models*
*Estimated effort: 4–6 weeks*

#### 3.1 ML Signal Fusion
Currently fusion is fully deterministic (`if sentiment==BUY and technical==BUY → STRONG BUY`). Replace or supplement this with a lightweight trained model:
- Feature vector: sentiment score, RSI, MACD histogram, BB position, volume ratio, market regime
- Target: forward 5-day return direction (from backtested data)
- Start with logistic regression or XGBoost (interpretable)
- Retrain weekly on rolling window of trade history

#### 3.2 Market Regime Detection
Signals that work in trending markets fail in ranging markets. Add a regime classifier:
- Use VIX level, SPY 50/200 SMA crossover, and 20-day realised volatility
- Tag each signal with regime: `TRENDING | RANGING | HIGH_VOL`
- Adjust signal thresholds and position sizing per regime

#### 3.3 Strategy Parameter Optimisation (Use existing `optimization/` folder)
The `optimization/` directory exists but its contents are unclear. Build out:
- Grid search / Bayesian optimisation over `BUY_THRESHOLD`, `SELL_THRESHOLD`, stop-loss %, take-profit ratio
- Walk-forward optimisation to prevent overfitting
- Store optimisation runs in DB, expose results in dashboard

#### 3.4 Alternative Data — SEC Filings & Insider Activity
- Parse 8-K and 10-Q filings from SEC EDGAR (free API)
- Score insider buying/selling activity
- Feed into sentiment score as a high-weight signal

---

### Phase 4 — Productionisation & Scale
*Goal: Make the system robust, observable, and scalable for more tickers and users*
*Estimated effort: 3–4 weeks*

#### 4.1 Observability Stack
The `monitoring/` directory exists. Build it out properly:
- Expose Prometheus metrics (signal count, API latency, error rate, trade count)
- Add a Grafana dashboard (can be provisioned via Docker Compose)
- Alerting on: API key failure, DB write errors, scheduler missed runs

#### 4.2 Replace SQLite with PostgreSQL
SQLite is fine for a single-user local setup but becomes a bottleneck under concurrent writes (dashboard + scheduler + screener running simultaneously). Switching to Postgres (one Railway service) unlocks:
- Concurrent reads/writes
- Full-text search on headline history
- Easier backup / point-in-time recovery

#### 4.3 Async Pipeline
The current pipeline is synchronous — analysing 40 screener candidates one-by-one is slow. Rewrite the orchestrator with `asyncio` or a thread pool:
- Parallel Claude API calls for sentiment (batch up to 10 tickers simultaneously)
- Parallel yfinance data fetches
- Target: reduce full screener run from ~8 min to ~1 min

#### 4.4 WebSocket Real-time Feeds
The daily scheduler fires once at 09:30. Add intraday streaming for high-priority watchlist tickers:
- Integrate Alpaca WebSocket data stream (free with account)
- Re-run technical analysis on each price update
- Push intraday signals to Telegram as they occur

#### 4.5 Email Notifications
The `notifications/` folder references SMTP but status is unclear. Fully implement:
- Daily digest email with HTML formatting (equity curve image, top signals table)
- Alert email on STRONG BUY/SELL signals

#### 4.6 Multi-user / SaaS Mode
If the project moves beyond personal use:
- Add simple JWT auth to the Streamlit dashboard
- Per-user watchlists and account balances stored in DB
- Usage metering for Claude API cost allocation

---

## Suggested Priority Order

```
Now (Phase 1):
  ✦ Multi-source sentiment (Reddit/StockTwits)      — quick win, high signal value
  ✦ Volume confirmation                              — low effort, reduces noise
  ✦ Earnings calendar awareness                      — reduces event blowup risk

Next (Phase 2):
  ✦ Alpaca live/paper broker integration             — biggest feature unlock
  ✦ Order lifecycle management                       — required for live trading safety

Then (Phase 3):
  ✦ Market regime detection                          — high ROI, relatively simple
  ✦ ML signal fusion                                 — more complex, needs trade history first

Later (Phase 4):
  ✦ Prometheus/Grafana observability                 — needed as scale grows
  ✦ Async pipeline                                   — needed when screener > 100 tickers
  ✦ PostgreSQL migration                             — needed for multi-user or heavy load
```

---

## Open Questions to Decide

1. **Real money or paper only?** — If moving to live trading, Phase 2 becomes the top priority.
2. **Which markets matter most?** — US-only (Alpaca) vs. EU-included (IBKR) changes broker choice.
3. **How many tickers to support?** — If >100, async pipeline (Phase 4.3) moves up.
4. **Solo project or multi-user?** — Multi-user mode (Phase 4.6) is significant scope.
5. **Optimisation folder** — What's already in `optimization/`? Worth reviewing before building Phase 3.3.

---

*Prepared for continuation session — March 2026*
