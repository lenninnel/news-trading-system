# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Live broker integration (Alpaca / IBKR paper API)
- MacroAgent for economic indicator analysis (CPI, Fed rate decisions)
- Portfolio-level risk monitoring (total exposure, correlation limits)
- Automated MDAX / SDAX ticker list refresh from Deutsche Börse API

---

## [0.6.0] — 2026-02-18

### Added
- **ScreenerAgent** (`agents/screener_agent.py`): Multi-market momentum screener covering DAX 40, MDAX (~50), SDAX (~70), TecDAX (30), S&P 500, NASDAQ 100, EURO STOXX 50, FTSE 100, and CAC 40
  - Composite "hotness" score (0–10 scale): volume spike (30%), price change (30%), RSI extreme (20%), liquidity (10%), market priority (10%)
  - Market-type-specific filters: blue chips require 2× volume and 3% price move; mid/small-caps require 1.5× volume and 4% move
  - Focus-market quota: German stocks guaranteed ≥ 10 slots when `focus_market=DE`
  - In-process price cache (5-minute TTL) and constituent-list cache (24-hour TTL)
  - CLI: `python3 -m agents.screener_agent --markets US DE EU --focus DE --top 40`
- **Screener page** in dashboard (`dashboard/app.py`): Run Screener tab with live results, Saved Runs tab with historical data; includes hotness bar chart, country donut, RSI histogram, volume/price scatter, CSV export
- **`screener_results` table** in database: stores all screener runs with ticker, market, country, hotness, price, price_change, volume_ratio, volume, rsi, avg_volume
- **`UNIQUE INDEX`** on `screener_results (run_at, ticker)` with `INSERT OR IGNORE` upsert semantics
- **`_migrate_schema()`** method in `Database`: idempotently adds new columns to existing databases
- **`screener:` section** in `config/watchlist.yaml`: configures markets, focus market, top_candidates, german_indices, and optional threshold overrides
- Comprehensive documentation in `docs/` folder: USER_MANUAL.md, ARCHITECTURE.md, SETUP_GUIDE.md, OPERATIONS.md, DEVELOPMENT.md
- GitHub templates: `.github/ISSUE_TEMPLATE.md`, `.github/PULL_REQUEST_TEMPLATE.md`
- `CHANGELOG.md` (this file)

### Changed
- `DPW.DE` → `DHL.DE` in DAX 40 ticker list (Deutsche Post renamed to DHL Group in 2023)
- `MAN.DE` replaced with `GXI.DE` in MDAX list (MAN SE taken private in 2022)
- yfinance ERROR logs suppressed at `CRITICAL` level to silence delisted-ticker noise
- Batch downloads in ScreenerAgent now include per-chunk progress logging and 100ms rate-limit pause between chunks
- Added retry logic (`_download_with_retry`, max 2 retries with exponential backoff) to batch downloads
- SQLite connection timeout increased to 5 seconds; `check_same_thread=False` for concurrent Streamlit + CLI access
- Updated `README.md` with full feature list, architecture diagram, ASCII pipeline diagram, signal reference tables, and contributing guidelines

### Fixed
- `screener_results` table missing `price` and `volume` columns (required by candidate dicts)
- `log_screener_results()` did not persist `price` or `volume` fields

---

## [0.5.0] — 2026-02-17

### Added
- **Telegram notifications** (`notifications/telegram_bot.py`): Real-time alerts for trade signals, executed trades, and daily summaries
  - `TelegramNotifier.from_config()` factory method reads from `watchlist.yaml` + env vars
  - Optional inline dashboard URL button in messages
  - Non-blocking: all exceptions swallowed to never interrupt the trading pipeline
- `telegram:` section in `config/watchlist.yaml`
- `--notify` flag to `scheduler/daily_runner.py` and `main.py`
- `notifications/README.md` with full BotFather setup walkthrough

---

## [0.4.0] — 2026-02-16

### Added
- **RiskAgent** (`agents/risk_agent.py`): Half-Kelly Criterion position sizing
  - Position constraints: Kelly fraction, 10% portfolio cap, 2% max risk budget (tightest wins)
  - Stop-loss: 2% (STRONG signals), 1% (WEAK signals)
  - Take-profit: 2:1 reward-to-risk ratio
  - Safety gates: skip below 30% confidence, skip HOLD/CONFLICTING signals
  - Logs to `risk_calculations` table
- **`risk_calculations` table** in database
- Risk section in dashboard Agents page

---

## [0.3.0] — 2026-02-15

### Added
- **TechnicalAgent** (`agents/technical_agent.py`): Deterministic signal from RSI-14, MACD (12/26/9), SMA-20/50, Bollinger Bands (20, 2σ)
  - Signal rules: RSI < 30 / > 70, MACD crossovers, price vs. Bollinger Bands
  - Logs to `technical_signals` table
- **Signal fusion** in `Coordinator.combine_signals()`: merges sentiment + technical into STRONG BUY / STRONG SELL / WEAK BUY / WEAK SELL / CONFLICTING / HOLD
- **`combined_signals` table** in database
- **`Coordinator.run_combined()`**: full multi-agent pipeline wiring all four agents
- `--agent {sentiment,technical}` flag to `main.py` for single-agent mode
- `--execute` flag to `main.py` for paper-trade logging
- **Technical signals** and **Combined signals** pages in dashboard

### Changed
- Python 3.9 compatibility: removed walrus operator, replaced `str | None` with `Optional[str]`, added `from __future__ import annotations`
- Fixed Jinja2 prompt template escaping (braces in agent prompts)

---

## [0.2.0] — 2026-02-14

### Added
- **BacktestEngine** (`backtest/engine.py`): Day-by-day historical simulation
  - Metrics: total return, buy-and-hold return, Sharpe ratio, max drawdown, win rate, avg win/loss
  - Sentiment modes: `random`, `bullish`, `bearish`, `neutral`
  - Plotly visualisation: equity curve, drawdown chart, monthly returns heatmap
  - Logs to `backtest_results` table
- **PaperTrader** (`execution/paper_trader.py`): Simulated order execution
  - Tracks `portfolio` (open positions) and `trade_history` tables
  - Calculates realised P&L on SELL orders
- **Daily scheduler** (`scheduler/daily_runner.py`): Batch processor for multiple tickers
  - Daemon mode (fires at `schedule.time`) and immediate mode (`--now`)
  - Email failure alerts via SMTP
  - Cron installation helper (`scheduler/install_cron.sh`)
  - Logs to `scheduler_logs` table
- **Streamlit dashboard** (`dashboard/app.py`): Interactive 7-page UI
  - Overview, Signals, Portfolio, History, Agents, Backtesting, Screener pages
  - Plotly charts, filterable tables, CSV downloads
  - 60-second auto-refresh
- `config/watchlist.yaml`: Externalised scheduler and screener configuration
- `watchlist:`, `account:`, `execution:`, `schedule:`, `email:` config sections

---

## [0.1.0] — 2026-02-12 (Initial Release)

### Added
- **SentimentAgent** (`agents/sentiment_agent.py`): Claude-powered headline classifier
  - Classifies each headline as bullish (+1), neutral (0), or bearish (−1)
  - Returns structured JSON with sentiment, score, and one-sentence reason
  - Model: `claude-sonnet-4-6`
- **BaseAgent** (`agents/base_agent.py`): Abstract base class with `name` and `run()` interface
- **NewsFeed** (`data/news_feed.py`): NewsAPI adapter; fetches up to 10 English headlines per ticker
- **MarketData** (`data/market_data.py`): yfinance adapter for current price and fundamentals
- **Coordinator** (`orchestrator/coordinator.py`): Sentiment pipeline wiring
  - `run()`: fetch → score × N → aggregate → signal → log
  - Signal thresholds: BUY ≥ +0.30, SELL ≤ −0.30, HOLD otherwise
- **Database** (`storage/database.py`): SQLite persistence
  - Tables: `runs`, `headline_scores`
  - `CREATE TABLE IF NOT EXISTS` (idempotent schema init)
- **`config/settings.py`**: Centralised constants; loads from `.env` via `python-dotenv`
- **`main.py`**: CLI entry point: `python main.py TICKER`
- `requirements.txt` with initial dependency list
- Basic `README.md`

---

[Unreleased]: https://github.com/your-username/news-trading-system/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/your-username/news-trading-system/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/your-username/news-trading-system/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/your-username/news-trading-system/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/your-username/news-trading-system/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/your-username/news-trading-system/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-username/news-trading-system/releases/tag/v0.1.0
