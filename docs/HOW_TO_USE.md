# How to Use the News Trading System

## What this system does

The News Trading System is an AI-powered stock analysis tool that combines three sources of information — news headlines (via NewsAPI), social media sentiment (StockTwits and Reddit), and technical price indicators (RSI, MACD, Bollinger Bands) — to generate trading signals for individual stocks. It uses Claude to score each headline as bullish, neutral, or bearish, then a risk agent calculates position sizes with stop-loss and take-profit levels. The system can run as a one-off analysis, a daily batch screener, or a live trading pipeline connected to Alpaca.

---

## Trading Modes

The system supports three modes, controlled by the `TRADING_MODE` variable in `.env`:

| Mode | What it does | Risk level |
|---|---|---|
| `paper_local` | Logs trades to a local SQLite database. No real money involved. | None |
| `alpaca_paper` | Sends orders to Alpaca's paper-trading sandbox. Simulates real execution with fake money. | None |
| `alpaca_live` | Sends real orders to Alpaca with real money. Requires `LIVE_TRADING_CONFIRMED=true` as a safety gate. | Real money at risk |

Start with `paper_local` to understand the signals, then move to `alpaca_paper` to test execution, and only switch to `alpaca_live` when you're confident.

---

## Analyse a Single Stock

```bash
python3 main.py AAPL --execute
```

This runs the full pipeline for one ticker:
1. Fetches market data (price, name, currency)
2. Pulls 10 recent headlines from NewsAPI + StockTwits + Reddit
3. Claude scores each headline as bullish / neutral / bearish
4. Technical agent computes RSI, MACD, SMA-20, SMA-50, Bollinger Bands
5. Coordinator combines sentiment + technical into a final signal
6. Risk agent calculates position size, stop-loss, and take-profit
7. `--execute` logs the trade (or sends the order in Alpaca mode)

Without `--execute`, the system only analyses — no trade is recorded.

Other options:

```bash
python3 main.py NVDA --balance 25000 --execute    # custom account balance
python3 main.py TSLA --agent sentiment             # sentiment analysis only
python3 main.py AAPL --agent technical             # technical analysis only
```

---

## Run the Full Screener

```bash
python3 -m scheduler.daily_runner --now
```

This analyses 20 large-cap stocks in parallel using asyncio, with concurrency limits to avoid API rate limits. Results are logged to the database and printed to stdout.

Options:

```bash
python3 -m scheduler.daily_runner --now --watchlist AAPL,NVDA,TSLA
python3 -m scheduler.daily_runner --now --workers 8
python3 -m scheduler.daily_runner --now --balance 50000
python3 -m scheduler.daily_runner --now --no-execute    # analyse only, no trades
python3 -m scheduler.daily_runner --now --benchmark      # compare async vs sequential speed
```

---

## Reading the Signals

The system produces these combined signals:

| Signal | Meaning | What the system does |
|---|---|---|
| **STRONG BUY** | Sentiment and technicals both say buy | Full position (up to 10% of account) |
| **WEAK BUY** | One says buy, the other is neutral | Reduced position |
| **HOLD** | Both neutral, or no clear direction | No trade |
| **WEAK SELL** | One says sell, the other is neutral | Reduced short position |
| **STRONG SELL** | Sentiment and technicals both say sell | Full short position |
| **CONFLICTING** | One says buy, the other says sell | No trade (signals disagree) |

Each signal comes with a **confidence score** (0.0 to 1.0). The risk agent skips trades when confidence is below 0.30.

Additional modifiers:
- **Earnings guard**: Positions are capped at 50% during earnings week and 25% when earnings are imminent (within 2 trading days). Strong signals are downgraded to weak.
- **Market regime**: In a bear trend, positions are reduced 30%. In high volatility, positions are reduced 50% and weak/low-confidence signals are skipped entirely.
- **Volume confirmation**: When relative volume > 1.5x and On-Balance Volume supports the signal direction, confidence gets a boost.

---

## The Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501). Four pages:

- **Overview** — Latest signals for all analysed tickers, sentiment breakdown by source (NewsAPI / StockTwits / Reddit), recent trade history
- **Technical** — Charts for RSI, MACD, Bollinger Bands, and price action for each ticker
- **Risk** — Position sizing details: Kelly fraction, shares, stop-loss, take-profit, risk amount, earnings proximity
- **Backtesting** — Run walk-forward parameter optimisation: pick a ticker and date range, view in-sample vs out-of-sample Sharpe ratios, equity curves, and per-window results. Past optimisation runs are saved and listed.

---

## Running a Backtest

1. Open the dashboard: `streamlit run dashboard/app.py`
2. Navigate to the **Backtesting** page
3. Enter a ticker (e.g. `AAPL`), start date, and end date
4. Click **Run Optimisation**
5. The system runs a grid search over parameter combinations using walk-forward windows
6. Results show:
   - Best parameter combination
   - In-sample and out-of-sample Sharpe ratios
   - Equity curve chart
   - Per-window breakdown table
7. Results are saved to the database and appear in the "Past Runs" section

---

## Checking the Cron Job

```bash
# Verify the cron entry is installed
crontab -l

# Check the latest log output
cat logs/daily_runner.log

# Follow the log in real time
tail -f logs/daily_runner.log
```

If `crontab -l` shows nothing, install it:

```bash
bash scheduler/install_cron.sh
```

---

## Going Live

When you're ready to trade with real money:

1. **Test thoroughly** in `alpaca_paper` mode first
2. Review your signals and trade history in the dashboard
3. Update your `.env`:

```env
TRADING_MODE=alpaca_live
LIVE_TRADING_CONFIRMED=true
```

Both values must be set — `LIVE_TRADING_CONFIRMED` is a safety gate that prevents accidental live trading. If it's missing or set to anything other than `true`, the system will refuse to start in live mode.

4. Restart the system or scheduler

Live mode sends real market orders through Alpaca's API with bracket orders (stop-loss + take-profit attached automatically).

---

## Market Regime

The system automatically detects the current market regime before each analysis:

| Regime | How it's detected | Effect on trading |
|---|---|---|
| **Trending Bull** | SMA-50 of SPY is above SMA-200 and VIX < 25 | Normal trading, no adjustments |
| **Trending Bear** | SMA-50 of SPY is below SMA-200 and VIX < 25 | Position sizes reduced by 30% |
| **Ranging** | SMA-50 roughly equal to SMA-200 | Position sizes reduced by 20%, weak signals skipped |
| **High Volatility** | VIX > 25 (or realised volatility > 25% annualised if VIX unavailable) | Position sizes reduced by 50%, weak and low-confidence signals skipped entirely |

The regime is cached for 4 hours to avoid redundant calculations. You can see the current regime in the dashboard Overview page and in the daily runner output.
