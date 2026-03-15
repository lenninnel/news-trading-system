# Trend-Following 2-Stage Optimization Report
**Period:** 2023-01-01 to 2025-01-01  
**Stage 1 combos:** 24 (coarse) + ~16 (fine) per window  
**Walk-forward windows:** 4 per ticker  
**Total evaluations:** 6,656  
**Runtime:** 1230.4s  

## Per-Ticker Results

| Ticker | Sector | OOS Sharpe | IS Sharpe | Gap | Trades | Status |
|--------|--------|-----------|----------|-----|--------|--------|
| MSFT         | AI_CHIPS   |     +3.07 |    +1.93 | -1.13 |    71 | OK         |
| META         | AI_CHIPS   |     +2.99 |    +2.17 | -0.82 |    51 | OK         |
| VST          | DATACENTER |     +2.69 |    +2.31 | -0.38 |    62 | OK         |
| AAPL         | DATACENTER |     +2.32 |    +1.73 | -0.59 |    57 | OK         |
| CEG          | DATACENTER |     +2.29 |    +1.87 | -0.42 |    59 | OK         |
| GOOGL        | AI_CHIPS   |     +2.10 |    +1.92 | -0.17 |    40 | OK         |
| DELL         | DATACENTER |     +1.97 |    +1.79 | -0.18 |    81 | OK         |
| AMD          | AI_CHIPS   |     +1.74 |    +2.57 | +0.83 |    49 | OVERFIT    |
| SAP.XETRA    | GERMAN_TECH |     +1.64 |    +1.25 | -0.39 |    31 | OK         |
| NVDA         | AI_CHIPS   |     +0.68 |    +2.52 | +1.83 |    37 | OVERFIT    |
| SOL          | CRYPTO     |     +0.53 |    +0.27 | -0.26 |   124 | OK         |
| TSLA         | AI_CHIPS   |     +0.29 |    +0.94 | +0.65 |    49 | OVERFIT    |
| SMCI         | DATACENTER |     +0.25 |    +1.69 | +1.45 |    40 | OVERFIT    |
| BTC          | CRYPTO     |     +0.00 |    +0.00 | +0.00 |     0 | LOW TRADES |
| ETH          | CRYPTO     |     +0.00 |    +0.00 | +0.00 |     0 | LOW TRADES |
| SIE.XETRA    | GERMAN_TECH |     -0.11 |    +1.04 | +1.16 |    38 | OVERFIT    |

## Sector Consensus Parameters

**AI_CHIPS** (from MSFT, OOS Sharpe +3.07):
  - require_volume_confirmation: False
  - rsi_overbought: 70
  - rsi_oversold: 30
  - rsi_period: 14
  - sma_fast: 10
  - sma_slow: 200
  - stop_loss_pct: 0.01
  - take_profit_ratio: 2.25

**CRYPTO** (from SOL, OOS Sharpe +0.53):
  - require_volume_confirmation: False
  - rsi_overbought: 70
  - rsi_oversold: 30
  - rsi_period: 14
  - sma_fast: 20
  - sma_slow: 125
  - stop_loss_pct: 0.035
  - take_profit_ratio: 2.5

**DATACENTER** (from VST, OOS Sharpe +2.69):
  - require_volume_confirmation: False
  - rsi_overbought: 70
  - rsi_oversold: 30
  - rsi_period: 14
  - sma_fast: 20
  - sma_slow: 200
  - stop_loss_pct: 0.015
  - take_profit_ratio: 2.5

**GERMAN_TECH** (from SAP.XETRA, OOS Sharpe +1.64):
  - require_volume_confirmation: False
  - rsi_overbought: 70
  - rsi_oversold: 30
  - rsi_period: 14
  - sma_fast: 10
  - sma_slow: 175
  - stop_loss_pct: 0.035
  - take_profit_ratio: 2.5

## Top 5 Ticker+Param Combinations (by OOS Sharpe)

1. **MSFT** — OOS Sharpe: +3.07 (IS: +1.93, trades: 71, VALID)
   SMA 10/200, RSI-14 (30/70), SL 1.0%, TP 2.25x, vol_confirm=False
2. **META** — OOS Sharpe: +2.99 (IS: +2.17, trades: 51, VALID)
   SMA 50/100, RSI-14 (30/70), SL 2.0%, TP 2.75x, vol_confirm=False
3. **VST** — OOS Sharpe: +2.69 (IS: +2.31, trades: 62, VALID)
   SMA 20/200, RSI-14 (30/70), SL 1.5%, TP 2.5x, vol_confirm=False
4. **AAPL** — OOS Sharpe: +2.32 (IS: +1.73, trades: 57, VALID)
   SMA 30/75, RSI-14 (30/70), SL 1.5%, TP 2.25x, vol_confirm=False
5. **CEG** — OOS Sharpe: +2.29 (IS: +1.87, trades: 59, VALID)
   SMA 50/200, RSI-14 (30/70), SL 3.0%, TP 2.0x, vol_confirm=False

## Production Readiness

**Production-ready:** MSFT, META, GOOGL, VST, CEG, DELL, AAPL, SAP.XETRA, SOL
**Skip (overfit/low trades/negative):** NVDA, AMD, TSLA, SMCI, SIE.XETRA, BTC, ETH

