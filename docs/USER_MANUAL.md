# User Manual — News Trading System

> **Audience**: Non-technical users who want to run the system and interpret its output.
> For installation help see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## Table of Contents

1. [What is the News Trading System?](#1-what-is-the-news-trading-system)
2. [Starting the System](#2-starting-the-system)
3. [Reading Trading Signals](#3-reading-trading-signals)
4. [Using the Dashboard](#4-using-the-dashboard)
5. [Using the Stock Screener](#5-using-the-stock-screener)
6. [Managing Your Watchlist](#6-managing-your-watchlist)
7. [Understanding Risk Calculations](#7-understanding-risk-calculations)
8. [FAQ](#8-faq)

---

## 1. What is the News Trading System?

The News Trading System is a tool that helps you decide whether to buy, sell, or hold a stock. It does this by:

1. **Reading the news** — Fetches recent headlines about the stock from the internet
2. **Understanding the news** — Uses AI (Claude by Anthropic) to assess whether each headline is positive or negative
3. **Checking the charts** — Looks at price momentum, volume, and technical indicators
4. **Combining both views** — Merges the news and chart signals into one actionable recommendation
5. **Sizing the trade** — Calculates how much money to risk based on your account balance

All analysis results are stored locally on your computer. Nothing is sent to external servers except the API calls to fetch news and run the AI.

> **Important**: This system is a research tool, not financial advice. Always apply your own judgement before making any investment decision.

---

## 2. Starting the System

### Option A — Analyse a single stock (recommended for beginners)

Open a terminal and run:

```bash
python main.py AAPL
```

Replace `AAPL` with any stock ticker (e.g. `NVDA`, `MSFT`, `BAYN.DE`).

The system will print a report like this:

```
============================================================
  AAPL  |  Apple Inc.  |  $189.30
============================================================
  Sentiment signal:   BUY   (avg score: +0.45)
  Technical signal:   BUY   (RSI: 28.4)
  Combined signal:    STRONG BUY
  Confidence:         78%
------------------------------------------------------------
  Position size:      $820.00  (4 shares)
  Stop-loss:          $185.52  (−2.0%)
  Take-profit:        $196.36  (+3.7%)
============================================================
```

### Option B — Open the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Option C — Run the daily scheduler once

```bash
python3 scheduler/daily_runner.py --now
```

This analyses every ticker in your watchlist and stores the results in the database.

### Option D — Run the scheduler automatically every weekday

```bash
python3 scheduler/daily_runner.py
```

Leave this running in the background. It will automatically fire at the time configured in `config/watchlist.yaml` (default: 09:30 local time, weekdays only).

---

## 3. Reading Trading Signals

### 3.1 The Sentiment Signal

The sentiment signal summarises what the news says about a stock.

| Signal | Meaning | When it appears |
|---|---|---|
| **BUY** | News is broadly positive | Average score ≥ +0.30 |
| **HOLD** | News is mixed or neutral | Score between −0.30 and +0.30 |
| **SELL** | News is broadly negative | Average score ≤ −0.30 |

Each headline is scored: +1 (bullish), 0 (neutral), or −1 (bearish). The average across all headlines gives the final score.

### 3.2 The Technical Signal

The technical signal looks at the stock's recent price behaviour.

| Signal | Typical trigger |
|---|---|
| **BUY** | RSI below 30 (oversold), MACD bullish crossover, or price below lower Bollinger Band |
| **HOLD** | No strong trend in either direction |
| **SELL** | RSI above 70 (overbought), MACD bearish crossover, or price above upper Bollinger Band |

### 3.3 The Combined Signal

The combined signal merges both views:

| Sentiment | Technical | Combined Signal | Strength |
|---|---|---|---|
| BUY | BUY | **STRONG BUY** | High conviction |
| SELL | SELL | **STRONG SELL** | High conviction |
| BUY | HOLD | **WEAK BUY** | Moderate |
| SELL | HOLD | **WEAK SELL** | Moderate |
| BUY | SELL | **CONFLICTING** | No position taken |
| SELL | BUY | **CONFLICTING** | No position taken |
| HOLD | any | **HOLD** | No position taken |

### 3.4 Confidence Score

The confidence score (0–100%) indicates how reliable the signal is:

- **≥ 70%** — High confidence; position sizing is close to maximum
- **50–69%** — Moderate confidence
- **30–49%** — Low confidence; smaller position size
- **< 30%** — Too uncertain; no position is taken

---

## 4. Using the Dashboard

Open [http://localhost:8501](http://localhost:8501) after running `streamlit run dashboard/app.py`.

### Overview page

The first page you see. It shows:

- **Portfolio KPIs** at the top: total value, number of open positions, trades today, signals today
- **Recent signals table**: the last combined signals with their direction and confidence
- **Open positions**: any currently active paper trades
- **Signal breakdown chart**: visual summary of today's signals

### Signals page

A filterable table of every combined signal ever generated. Use the drop-downs at the top to filter by:
- Ticker symbol
- Signal type (e.g. only STRONG BUY)
- Confidence level

### Portfolio page

Shows your current paper-trading positions:
- How many shares you hold for each ticker
- The average buy price
- Current value (updated when you re-run the analysis)
- Unrealised P&L

### History page

A complete log of every paper trade, with date-range and ticker filters. Shows both open and closed trades, along with realised profit/loss for closed positions.

### Agents page

A deep-dive into the individual agent outputs:
- **Sentiment tab**: Every analysis run with the full headline breakdown
- **Technical tab**: RSI, MACD, SMA, and Bollinger Band values for each analysis
- **Risk tab**: Position sizing details (Kelly fraction, stop-loss, take-profit)

### Backtesting page

Run a historical simulation to see how the strategy would have performed:

1. Enter a ticker symbol
2. Choose start and end dates
3. Set your account balance
4. Choose a sentiment mode (`random` is the most realistic)
5. Click **Run Backtest**

The page shows an equity curve, drawdown chart, monthly returns heatmap, and key metrics (Sharpe ratio, max drawdown, win rate).

### Screener page

Find the hottest stocks across multiple markets without analysing each one individually:

1. Choose which markets to scan (US, German/DE, European/EU)
2. Pick a focus market (German stocks get priority weighting)
3. Set how many candidates you want
4. Click **Run Screener**

Stocks are ranked by a "hotness" score (0–10) that rewards unusual volume, significant price moves, and extreme RSI readings. After reviewing the results, you can run the full analysis on any interesting ticker.

---

## 5. Using the Stock Screener

The screener monitors 400+ stocks across nine indices and surfaces the ones with the most momentum right now.

### How to read the screener results

| Column | Meaning |
|---|---|
| **Hotness** | Composite score 0–10. Higher = more activity |
| **Chg%** | Today's price change vs. yesterday |
| **VolRatio** | Today's volume vs. 20-day average (e.g. 3.0× = 3× normal volume) |
| **RSI** | Relative Strength Index (< 30 = oversold, > 70 = overbought) |
| **Market** | Index the stock belongs to (DAX, MDAX, SP500, etc.) |
| **Country** | Country of listing |

### What makes a stock "hot"?

The hotness formula weighs four factors:

| Factor | Weight | What it means |
|---|---|---|
| Volume spike | 30% | Unusual trading activity |
| Price move | 30% | Big price change (either direction) |
| RSI extreme | 20% | Very overbought or oversold |
| Liquidity | 10% | Larger, more liquid stocks get a small bonus |
| Market priority | 10% | German stocks get +0.2 priority when DE is the focus |

A stock with a hotness score of 7+ is significantly active and worth further analysis.

### Minimum filters

Stocks must pass these filters to appear in the results:

| Market type | Min volume ratio | Min price change |
|---|---|---|
| Blue chip (DAX, S&P 500, etc.) | 2.0× | 3% |
| Mid-cap (MDAX, TecDAX) | 1.5× | 4% |
| Small-cap (SDAX) | 1.5× | 4% |

---

## 6. Managing Your Watchlist

The daily scheduler uses the watchlist from `config/watchlist.yaml`. Open that file in a text editor to change it.

### Adding or removing tickers

```yaml
watchlist:
  - AAPL      # Apple
  - NVDA      # NVIDIA
  - TSLA      # Tesla
  - BAYN.DE   # Bayer (German market, note the .DE suffix)
```

For German stocks, append `.DE` to the ticker. For other European markets:
- **UK**: `.L` (e.g. `SHEL.L`)
- **France**: `.PA` (e.g. `MC.PA`)
- **Netherlands**: `.AS` (e.g. `ASML.AS`)

### Changing the schedule

```yaml
schedule:
  time: "09:30"         # Change to any HH:MM in your local time zone
  weekdays_only: true   # Set to false to run on weekends too
```

### Adjusting your paper-trading balance

```yaml
account:
  balance: 10000.0   # Change to your paper-trading amount in USD
```

---

## 7. Understanding Risk Calculations

When the system generates a BUY or SELL signal with sufficient confidence, it calculates a suggested position size.

### Position size example

Suppose you have a $10,000 account and the system generates a STRONG BUY for AAPL at $189.30 with 78% confidence:

| Parameter | Value | Explanation |
|---|---|---|
| Kelly fraction | 8.2% | Half-Kelly optimal allocation |
| Position cap | 10% | Maximum per single trade |
| Risk cap | 2% | Max loss allowed = $200 |
| **Position size** | **$820** | Determined by risk cap (tightest constraint) |
| **Shares** | **4** | $820 ÷ $189.30 = 4.33 → rounded down |
| Stop-loss | $185.52 | −2% from entry (STRONG signal) |
| Take-profit | $196.36 | +2× the risk distance above entry |
| Max risk | $20.00 | 4 shares × (189.30 − 185.52) |

### When no position is taken

The system will skip the trade (show "no position") when:

- Combined signal is HOLD or CONFLICTING
- Confidence is below 30%
- The calculated position rounds down to 0 shares

---

## 8. FAQ

**Q: The system shows "no candidates passed filters" for the screener. Is something broken?**

A: This is normal outside of trading hours. The screener looks at today's price change and volume vs. the 20-day average. When markets are closed, there is no new price movement, so few stocks meet the minimum thresholds. Run the screener during or shortly after market hours for best results.

---

**Q: How often should I run the analysis?**

A: Once per day, ideally shortly after market open (9:30–10:00 AM local time for US markets, 9:00–9:30 for European). The daily scheduler can handle this automatically. Avoid running multiple times per day as the news data does not change that frequently.

---

**Q: The sentiment signal is BUY but the technical signal is SELL. What should I do?**

A: A CONFLICTING signal means the two data sources disagree. The system will not suggest a position in this case. You can dig into the Agents page on the dashboard to see exactly which indicators triggered each signal, and form your own view.

---

**Q: Can I use this with real money?**

A: The system only paper-trades (simulated). It does not connect to any brokerage. If you want to use the signals with real money, you must execute the trades yourself. Always apply your own risk management.

---

**Q: The system says "NewsAPI limit reached". What does that mean?**

A: Free NewsAPI accounts allow 100 requests per day. Each ticker analysis uses one request. If you have more than 100 tickers in your watchlist, you will hit the limit. Upgrade to a paid NewsAPI plan, or reduce your watchlist.

---

**Q: Can I add stocks from markets other than the US and Germany?**

A: Yes. Any ticker that yfinance supports will work. For analysis via `main.py`, just use the ticker as listed on Yahoo Finance (e.g. `ASML.AS` for ASML on Euronext Amsterdam). For the screener, the EU market group includes EURO STOXX 50, FTSE 100, and CAC 40.

---

**Q: Where is my data stored?**

A: All data is stored locally in `news_trading.db` (a SQLite file in the project folder). No trading data is ever sent to external servers. Only the API calls to NewsAPI and Anthropic leave your machine.

---

**Q: How do I reset everything and start fresh?**

A: Delete `news_trading.db`. The system will recreate it on next run with an empty database. Your configuration (`config/watchlist.yaml` and `.env`) is not affected.

---

**Q: The dashboard shows an error about missing columns in the database.**

A: This usually means the database was created by an older version of the system. The simplest fix is to delete `news_trading.db` and let it be recreated. Alternatively, run `python -c "from storage.database import Database; Database()"` to trigger the migration.
