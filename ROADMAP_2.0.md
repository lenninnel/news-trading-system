# Roadmap 2.0 — News Trading System
*March 2026 — Next steps to scale the system*

---

## Data Sources
*Biggest signal quality impact*

- [ ] Replace NewsAPI → **Marketaux** (ticker-tagged sentiment, 80+ markets, free)
- [ ] Add **Finnhub** (60 req/min free, real-time US + WebSocket)
- [ ] Add **ApeWisdom** (free Reddit mention aggregator, no NLP needed)
- [ ] Add **Binance API** (free crypto OHLCV, replaces yfinance for crypto)
- [ ] Add **EODHD** (€30/month — only reliable fix for German XETRA data)
- [ ] Add **SEC EDGAR 8-K filings** (free, event-driven signals)
- [ ] Add **FRED API** (free macro overlay — VIX, yield curve, GDP)
- [ ] Add **Crypto Fear & Greed Index** (free, single endpoint)

---

## Execution

- [ ] **IBKR integration** for German/EU stocks (DAX, MDAX)
- [ ] **Live trading activation** (after 2 clean weeks on Alpaca paper)
- [ ] **Order lifecycle tracking** (filled / partial / cancelled states)

---

## Phase 5 — Intelligence Layer
*Replace rule-based fusion with learned models. Needs trade history from Phase 4.*

- [ ] **ML signal fusion** — XGBoost trained on accumulated trade history, retrained weekly
- [ ] **Walk-forward optimisation** dashboard page (visual parameter heatmaps)
- [ ] **Options flow** via Unusual Whales (~€45/month — informed positioning signals)
- [ ] **WebSocket real-time feeds** — intraday signals, not just 09:30 daily run
- [ ] **TradingView strategy import workflow**
  - Design strategies visually in TradingView using Pine Script
  - Export chart data (CSV) from TradingView
  - Claude Code converts Pine Script logic → Python
  - Run through existing backtest/engine.py + optimizer.py
  - Deploy winning strategies to live system
  - *Requires TradingView subscription (already have it)*

---

## Infrastructure

- [ ] **Prometheus + Grafana** observability (metrics dashboard)
- [ ] **PostgreSQL migration** (replace SQLite for concurrent writes)
- [ ] **WebSocket real-time feeds** (intraday signals, not just 09:30)
- [ ] **Email digest** (daily HTML summary with equity curve)

---

## Scale

- [ ] Expand screener to **crypto tickers**
- [ ] **Portfolio-level risk management** (not just per-trade Kelly)
- [ ] **Walk-forward optimisation** dashboard page
- [ ] **Backtesting** on crypto + German stocks
