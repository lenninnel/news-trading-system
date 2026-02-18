"""
ScreenerAgent — multi-market momentum screener for US, German, and EU stocks.

Scans a configurable universe that spans the DAX 40, MDAX, SDAX, TecDAX,
S&P 500, NASDAQ 100, EURO STOXX 50, FTSE 100, and CAC 40, then ranks every
ticker by a composite "hotness" score that rewards:
  • Unusual volume spikes
  • Significant intraday price moves
  • Extreme RSI readings (oversold/overbought)
  • Liquidity (market-cap proxy)
  • Market priority (German stocks get a local-advantage bonus)

German small and mid-caps (MDAX, SDAX, TecDAX) use relaxed filter thresholds
compared to blue-chip indices.

CLI::

    python3 -m agents.screener_agent                          # DE focus, top 40
    python3 -m agents.screener_agent --markets US DE EU --focus DE --top 40
    python3 -m agents.screener_agent --markets US --top 20    # US only

Hotness formula (all components normalised to [0, 1])::

    hotness = (vol_ratio_norm × 0.30)
            + (price_chg_norm × 0.30)
            + (rsi_extreme    × 0.20)
            + (liquidity_norm × 0.10)
            + (market_priority × 0.10)

    × 10   →  final score on a 0–10 scale

Note
----
Index constituent lists are curated approximations as of early 2026.
Constituents change quarterly; invalid or delisted tickers are silently
skipped during data fetch.  Update the _DAX_40 / _MDAX / _SDAX / _TECDAX
constants when doing a quarterly refresh.

Requires::

    pip install yfinance ta pandas numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import ta  # type: ignore
import yfinance as yf

# ── Path bootstrap (needed when run as __main__) ───────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.base_agent import BaseAgent
from storage.database import Database

log = logging.getLogger(__name__)

# Suppress yfinance's ERROR-level noise for delisted / invalid tickers.
# Those failures are handled gracefully — we don't need the stack traces.
for _yf_logger_name in ("yfinance", "yfinance.base", "yfinance.utils",
                         "yfinance.scrapers.history", "peewee"):
    logging.getLogger(_yf_logger_name).setLevel(logging.CRITICAL)

# ──────────────────────────────────────────────────────────────────────────────
# Universe definitions
# Note: Lists are approximate and require periodic updates.
# Delisted / invalid tickers are automatically skipped on data fetch.
# ──────────────────────────────────────────────────────────────────────────────

# DAX 40 — German blue-chip index (XETRA)
_DAX_40: list[str] = [
    "SAP.DE",  "SIE.DE",  "ALV.DE",  "MUV2.DE", "BMW.DE",  "VOW3.DE",
    "MBG.DE",  "DTE.DE",  "BAYN.DE", "BAS.DE",  "ADS.DE",  "RWE.DE",
    "EOAN.DE", "DBK.DE",  "IFX.DE",  "DHL.DE",  "DB1.DE",  "LIN.DE",
    "MRK.DE",  "HEI.DE",  "HEN3.DE", "FRE.DE",  "ZAL.DE",  "CON.DE",
    "VNA.DE",  "RHM.DE",  "AIR.DE",  "PAH3.DE", "P911.DE", "BNR.DE",
    "MTX.DE",  "SRT3.DE", "DHER.DE", "FME.DE",  "CBK.DE",  "HNR1.DE",
    "ENR.DE",  "SHL.DE",  "EVK.DE",  "SY1.DE",
]

# MDAX — German mid-cap (~50 stocks, XETRA)
_MDAX: list[str] = [
    "AFX.DE",  "AIXA.DE", "BC8.DE",  "BOSS.DE", "DWS.DE",  "EVD.DE",
    "FNTN.DE", "GBF.DE",  "HOT.DE",  "LEG.DE",  "MLP.DE",  "NDX1.DE",
    "PSM.DE",  "PUM.DE",  "QIA.DE",  "TAG.DE",  "TUI1.DE", "WCHA.DE",
    "O2D.DE",  "SDF.DE",  "RDC.DE",  "VBK.DE",  "MVV1.DE", "DIC.DE",
    "S92.DE",  "HAB.DE",  "LHA.DE",  "NDA.DE",  "SIX2.DE", "TKA.DE",
    "UTDI.DE", "1U1.DE",  "ECV.DE",  "LXS.DE",  "MBB.DE",  "GFT.DE",
    "HLAG.DE", "WAF.DE",  "SFQ.DE",  "KBX.DE",  "SGL.DE",  "OHB.DE",
    "FRA.DE",  "SZU.DE",  "GXI.DE",
]

# SDAX — German small-cap (~70 stocks, XETRA)
_SDAX: list[str] = [
    "PNE.DE",  "SMHN.DE", "NFON.DE", "HAWE.DE", "MBB.DE",  "JUN3.DE",
    "KSB.DE",  "ARL.DE",  "PRG.DE",  "DEQ.DE",  "HBM.DE",  "VH2.DE",
    "RWA.DE",  "AAD.DE",  "JEN.DE",  "MCH.DE",  "DKGR.DE", "TTK.DE",
    "WUW.DE",  "NB2.DE",  "AOX.DE",  "FPE3.DE", "SLT.DE",  "PDX.DE",
    "PSAN.DE", "DRW3.DE", "CWC.DE",  "WEPA.DE", "HABN.DE", "GHH.DE",
    "SBS.DE",  "M1MA.DE", "LNSX.DE", "SGMO.DE", "ABR.DE",  "TEG.DE",
    "IOS.DE",  "FNTN.DE", "MUM.DE",  "3V64.DE", "PFV.DE",  "FRST.DE",
    "EVT.DE",  "SLB.DE",  "GBF.DE",  "WL6.DE",  "HAOG.DE",
]

# TecDAX — German technology index (30 stocks, XETRA)
_TECDAX: list[str] = [
    "SAP.DE",  "IFX.DE",  "AIXA.DE", "AFX.DE",  "QIA.DE",  "NDX1.DE",
    "S92.DE",  "SRT3.DE", "FNTN.DE", "1U1.DE",  "UTDI.DE", "SHL.DE",
    "ZAL.DE",  "WAF.DE",  "SMHN.DE", "NFON.DE", "PNE.DE",  "ECV.DE",
    "VBK.DE",  "GFT.DE",  "SFQ.DE",  "BC8.DE",  "DWS.DE",  "O2D.DE",
    "RDC.DE",  "MBB.DE",  "OHB.DE",  "AIXA.DE", "HAWE.DE", "GBF.DE",
]

# EURO STOXX 50 — pan-European blue chips
_EURO_STOXX_50: list[str] = [
    # Germany (overlap with DAX intentional — deduplication handled in code)
    "SAP.DE",  "SIE.DE",  "ALV.DE",  "MUV2.DE", "BAS.DE",  "BAYN.DE",
    "ADS.DE",  "IFX.DE",  "MBG.DE",  "DTE.DE",  "ENR.DE",
    # France
    "MC.PA",   "OR.PA",   "SAN.PA",  "AIR.PA",  "BNP.PA",  "FP.PA",
    "ACA.PA",  "GLE.PA",  "SU.PA",   "KER.PA",  "CAP.PA",  "DSY.PA",
    "SAF.PA",  "CS.PA",   "AI.PA",   "DG.PA",   "RMS.PA",  "LR.PA",
    # Netherlands
    "ASML.AS", "PHIA.AS", "INGA.AS", "AD.AS",   "WKL.AS",  "RAND.AS",
    # Spain
    "BBVA.MC", "SAN.MC",  "IBE.MC",  "ITX.MC",  "AMS.MC",
    # Italy
    "ENI.MI",  "ISP.MI",  "UCG.MI",  "G.MI",    "RACE.MI",
    # Belgium/Finland
    "ABI.BR",  "NOKIA.HE",
]

# FTSE 100 — UK blue chips (London Stock Exchange)
_FTSE_100: list[str] = [
    "SHEL.L",  "AZN.L",   "ULVR.L",  "BATS.L",  "BP.L",    "HSBA.L",
    "GSK.L",   "RIO.L",   "DGE.L",   "BARC.L",  "LLOY.L",  "VOD.L",
    "AAL.L",   "GLEN.L",  "BHP.L",   "PRU.L",   "REL.L",   "CPG.L",
    "WPP.L",   "BA.L",    "RR.L",    "IMB.L",   "NWG.L",   "STAN.L",
    "NG.L",    "SSE.L",   "CRH.L",   "EXPN.L",  "LSEG.L",  "FLTR.L",
]

# CAC 40 — French blue chips (Euronext Paris)
_CAC_40: list[str] = [
    "MC.PA",   "OR.PA",   "SAN.PA",  "AI.PA",   "AIR.PA",  "BNP.PA",
    "FP.PA",   "SU.PA",   "DG.PA",   "RMS.PA",  "KER.PA",  "ACA.PA",
    "GLE.PA",  "EL.PA",   "SAF.PA",  "CS.PA",   "VIE.PA",  "CAP.PA",
    "LR.PA",   "DSY.PA",  "PUB.PA",  "RI.PA",   "SGO.PA",  "AC.PA",
    "ENGI.PA", "STM.PA",  "ORA.PA",  "TEP.PA",  "EN.PA",   "CA.PA",
    "BIM.PA",  "RNO.PA",  "ML.PA",   "ERF.PA",  "STLAM.MI","SW.PA",
]

# Fallback lists for when Wikipedia scraping fails
_SP500_FALLBACK: list[str] = [
    "AAPL",  "MSFT",  "NVDA",  "AMZN",  "GOOGL", "META",  "BRK-B", "LLY",
    "TSLA",  "AVGO",  "WMT",   "JPM",   "V",     "UNH",   "XOM",   "ORCL",
    "MA",    "COST",  "HD",    "PG",    "NFLX",  "JNJ",   "BAC",   "ABBV",
    "MRK",   "CVX",   "CRM",   "KO",    "AMD",   "CSCO",  "PEP",   "ACN",
    "TMO",   "LIN",   "MCD",   "ADBE",  "NKE",   "DIS",   "TXN",   "NEE",
]

_NASDAQ100_FALLBACK: list[str] = [
    "AAPL",  "MSFT",  "NVDA",  "AMZN",  "META",  "GOOGL", "GOOG",  "TSLA",
    "AVGO",  "ADBE",  "CSCO",  "NFLX",  "PEP",   "COST",  "TMUS",  "AMD",
    "QCOM",  "TXN",   "AMAT",  "INTU",  "HON",   "SBUX",  "ISRG",  "VRTX",
    "REGN",  "LRCX",  "KLAC",  "MRVL",  "MU",    "PANW",  "SNPS",  "CDNS",
    "ASML",  "MELI",  "ADP",   "ABNB",  "PYPL",  "GILD",  "ADI",   "CRWD",
]

# ──────────────────────────────────────────────────────────────────────────────
# Market metadata
# ──────────────────────────────────────────────────────────────────────────────

#: Maps index name → static metadata used for filtering and scoring
_MARKET_META: dict[str, dict] = {
    "DAX":         {"country": "DE", "exchange": "XETRA",   "type": "blue_chip", "priority": 0.2},
    "MDAX":        {"country": "DE", "exchange": "XETRA",   "type": "mid_cap",   "priority": 0.2},
    "SDAX":        {"country": "DE", "exchange": "XETRA",   "type": "small_cap", "priority": 0.2},
    "TecDAX":      {"country": "DE", "exchange": "XETRA",   "type": "mid_cap",   "priority": 0.2},
    "SP500":       {"country": "US", "exchange": "NYSE/NASDAQ", "type": "blue_chip", "priority": 0.0},
    "NASDAQ100":   {"country": "US", "exchange": "NASDAQ",  "type": "blue_chip", "priority": 0.0},
    "EUROSTOXX50": {"country": "EU", "exchange": "Various", "type": "blue_chip", "priority": -0.1},
    "FTSE100":     {"country": "GB", "exchange": "LSE",     "type": "blue_chip", "priority": -0.1},
    "CAC40":       {"country": "FR", "exchange": "Euronext","type": "blue_chip", "priority": -0.1},
}

# ──────────────────────────────────────────────────────────────────────────────
# Filter thresholds (per market type)
# ──────────────────────────────────────────────────────────────────────────────

_FILTERS: dict[str, dict] = {
    "blue_chip": {
        "min_volume_ratio":  2.0,
        "min_price_change":  3.0,   # absolute % move
        "min_avg_volume":    0,     # no minimum for blue chips
    },
    "mid_cap": {
        "min_volume_ratio":  1.5,
        "min_price_change":  4.0,
        "min_avg_volume":    100_000,
    },
    "small_cap": {
        "min_volume_ratio":  1.5,
        "min_price_change":  4.0,
        "min_avg_volume":    100_000,
    },
}

# Maps market group code → list of index names
_MARKET_GROUPS: dict[str, list[str]] = {
    "DE": ["DAX", "MDAX", "SDAX", "TecDAX"],
    "US": ["SP500", "NASDAQ100"],
    "EU": ["EUROSTOXX50", "FTSE100", "CAC40"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class ScreenerAgent(BaseAgent):
    """
    Scans multi-market universes and returns the top hot-stock candidates.

    Example::

        agent = ScreenerAgent()
        result = agent.run(markets=["US", "DE"], focus_market="DE", top=40)
        for c in result["candidates"]:
            print(c["ticker"], c["hotness"])
    """

    _PRICE_CACHE_TTL  = 300    # seconds — price data stale after 5 min
    _LIST_CACHE_TTL   = 86_400 # seconds — constituent lists stale after 24 h
    _BATCH_SIZE       = 50     # tickers per yfinance batch download
    _HISTORY_PERIOD   = "1mo"  # enough for 20-day volume average + RSI-14
    _RSI_WINDOW       = 14
    _VOL_AVG_WINDOW   = 20
    _HOTNESS_SCALE    = 10.0   # multiply normalised score → 0–10 display range

    # Caps for normalisation
    _VOL_RATIO_CAP    = 5.0    # volume ratios beyond 5× capped at 1.0
    _PRICE_CHG_CAP    = 10.0   # price moves beyond 10% capped at 1.0

    def __init__(self, db: Database | None = None) -> None:
        self._db          = db or Database()
        self._price_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._list_cache:  dict[str, tuple[list[str], float]]    = {}

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ScreenerAgent"

    def run(
        self,
        markets: list[str] | None = None,
        focus_market: str | None = "DE",
        top: int = 40,
        **kwargs: Any,
    ) -> dict:
        """
        Screen the requested markets and return the top *top* candidates.

        Args:
            markets:      List of market codes to scan.  Accepts "US", "DE",
                          "EU" (default: all three).
            focus_market: Market code that receives a priority weighting bonus.
                          Use None to disable (default: "DE").
            top:          Maximum number of candidates to return (default: 40).

        Returns:
            dict with keys:
                run_at          (str):        ISO-8601 timestamp.
                markets_scanned (list[str]):  Markets requested.
                focus_market    (str|None):   Prioritised market.
                universe_size   (int):        Total tickers scanned.
                screened        (int):        Tickers that passed filters.
                candidates      (list[dict]): Top *top* ranked results.
        """
        if markets is None:
            markets = ["US", "DE", "EU"]
        markets = [m.upper() for m in markets]
        if focus_market:
            focus_market = focus_market.upper()

        run_at = datetime.now(timezone.utc).isoformat()
        log.info("ScreenerAgent run  markets=%s  focus=%s  top=%d", markets, focus_market, top)

        # ── 1. Build universe ──────────────────────────────────────────────
        universe = self._build_universe(markets)   # {index: [tickers]}
        all_tickers_map: dict[str, str] = {}       # ticker → index (first wins)
        for index, tickers in universe.items():
            for t in tickers:
                if t not in all_tickers_map:
                    all_tickers_map[t] = index

        all_tickers = list(all_tickers_map.keys())
        log.info("Universe: %d unique tickers across %d indices",
                 len(all_tickers), len(universe))

        # ── 2. Batch-fetch price history ───────────────────────────────────
        history = self._fetch_batch_history(all_tickers)
        log.info("Fetched history for %d / %d tickers", len(history), len(all_tickers))

        # ── 3. Compute metrics and apply filters ───────────────────────────
        candidates: list[dict] = []
        for ticker, index in all_tickers_map.items():
            df = history.get(ticker)
            if df is None or df.empty:
                continue
            try:
                meta     = _MARKET_META[index]
                metrics  = self._compute_metrics(ticker, df)
                if metrics is None:
                    continue
                if not self._passes_filter(metrics, meta["type"]):
                    continue
                hotness  = self._compute_hotness(metrics, meta["priority"], focus_market or "")
                candidates.append({
                    "ticker":       ticker,
                    "name":         metrics.get("name", ""),
                    "market":       index,
                    "exchange":     meta["exchange"],
                    "country":      meta["country"],
                    "hotness":      round(hotness, 2),
                    "price_change": round(metrics["price_change"], 2),
                    "volume_ratio": round(metrics["volume_ratio"], 2),
                    "rsi":          round(metrics["rsi"], 1) if metrics.get("rsi") else None,
                    "market_cap":   metrics.get("market_cap"),
                    "avg_volume":   round(metrics["avg_volume"]),
                    "price":        round(metrics["price"], 4),
                })
            except Exception as exc:
                log.debug("Skipping %s (%s): %s", ticker, index, exc)

        # ── 4. Rank globally by hotness ────────────────────────────────────
        candidates.sort(key=lambda c: c["hotness"], reverse=True)

        # Guarantee DE representation when focus is DE
        top_candidates = self._enforce_focus_quota(candidates, focus_market, top)

        log.info("Screener complete: %d passed filters → %d returned",
                 len(candidates), len(top_candidates))

        # ── 5. Persist ────────────────────────────────────────────────────
        if top_candidates:
            try:
                self._db.log_screener_results(run_at, top_candidates)
            except Exception as exc:
                log.warning("Could not persist screener results: %s", exc)

        return {
            "run_at":          run_at,
            "markets_scanned": markets,
            "focus_market":    focus_market,
            "universe_size":   len(all_tickers),
            "screened":        len(candidates),
            "candidates":      top_candidates,
        }

    # ------------------------------------------------------------------
    # Universe building
    # ------------------------------------------------------------------

    def _build_universe(self, markets: list[str]) -> dict[str, list[str]]:
        """Return {index_name: [tickers]} for all requested market groups."""
        result: dict[str, list[str]] = {}
        for group in markets:
            indices = _MARKET_GROUPS.get(group, [])
            for idx in indices:
                result[idx] = self._get_index_tickers(idx)
        return result

    def _get_index_tickers(self, index: str) -> list[str]:
        """Return the ticker list for an index, with caching."""
        cached = self._list_cache.get(index)
        if cached and (time.monotonic() - cached[1]) < self._LIST_CACHE_TTL:
            return cached[0]

        tickers = self._fetch_index_tickers(index)
        self._list_cache[index] = (tickers, time.monotonic())
        return tickers

    def _fetch_index_tickers(self, index: str) -> list[str]:
        """Fetch or return the curated ticker list for an index."""
        if index == "DAX":         return list(_DAX_40)
        if index == "MDAX":        return list(_MDAX)
        if index == "SDAX":        return list(_SDAX)
        if index == "TecDAX":      return list(_TECDAX)
        if index == "EUROSTOXX50": return list(_EURO_STOXX_50)
        if index == "FTSE100":     return list(_FTSE_100)
        if index == "CAC40":       return list(_CAC_40)
        if index == "SP500":       return self._scrape_sp500()
        if index == "NASDAQ100":   return self._scrape_nasdaq100()
        return []

    def _scrape_sp500(self) -> list[str]:
        """Scrape S&P 500 constituents from Wikipedia; falls back to curated list."""
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                attrs={"id": "constituents"},
            )
            tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            log.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
            return tickers
        except Exception as exc:
            log.warning("S&P 500 Wikipedia scrape failed (%s) — using fallback list", exc)
            return list(_SP500_FALLBACK)

    def _scrape_nasdaq100(self) -> list[str]:
        """Scrape NASDAQ 100 constituents from Wikipedia; falls back to curated list."""
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                attrs={"id": "constituents"},
            )
            tickers = tables[0]["Ticker"].tolist()
            log.info("Fetched %d NASDAQ 100 tickers from Wikipedia", len(tickers))
            return tickers
        except Exception as exc:
            log.warning("NASDAQ 100 Wikipedia scrape failed (%s) — using fallback list", exc)
            return list(_NASDAQ100_FALLBACK)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _download_with_retry(
        self, chunk: list[str], max_retries: int = 2
    ) -> pd.DataFrame:
        """Download OHLCV for *chunk* with exponential-backoff retries."""
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(max_retries + 1):
            try:
                return yf.download(
                    chunk,
                    period=self._HISTORY_PERIOD,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt          # 1 s, 2 s …
                    log.debug(
                        "Download attempt %d/%d failed (%s) — retrying in %ds",
                        attempt + 1, max_retries + 1, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc

    def _fetch_batch_history(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Batch-download 1-month daily OHLCV for all tickers using the 5-min cache.

        Splits the ticker list into chunks of ``_BATCH_SIZE`` to avoid yfinance
        timeouts.  Returns {ticker: DataFrame}.
        """
        now_ts = time.monotonic()
        result: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for t in tickers:
            cached = self._price_cache.get(t)
            if cached and (now_ts - cached[1]) < self._PRICE_CACHE_TTL:
                result[t] = cached[0]
            else:
                to_fetch.append(t)

        total_chunks = math.ceil(len(to_fetch) / self._BATCH_SIZE) or 1
        for i in range(0, len(to_fetch), self._BATCH_SIZE):
            chunk     = to_fetch[i : i + self._BATCH_SIZE]
            chunk_idx = i // self._BATCH_SIZE + 1
            log.info("Scanning chunk %d/%d (%d tickers)…", chunk_idx, total_chunks, len(chunk))
            try:
                raw = self._download_with_retry(chunk)
                if raw.empty:
                    continue

                fetched = self._split_multiindex(raw, chunk)
                for t, df in fetched.items():
                    if not df.empty:
                        self._price_cache[t] = (df, time.monotonic())
                        result[t] = df

            except Exception as exc:
                log.warning("Batch download failed (chunk %d/%d): %s",
                            chunk_idx, total_chunks, exc)

            # Brief pause between chunks to avoid hammering the API
            if i + self._BATCH_SIZE < len(to_fetch):
                time.sleep(0.1)

        return result

    @staticmethod
    def _split_multiindex(raw: pd.DataFrame, chunk: list[str]) -> dict[str, pd.DataFrame]:
        """
        Convert a yfinance multi-ticker download into a per-ticker dict.

        yfinance returns a MultiIndex DataFrame when ``group_by='ticker'``:
            columns = MultiIndex[(ticker, field), ...]

        A single-ticker download has plain columns.
        """
        if len(chunk) == 1:
            # Single ticker: columns are plain field names
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            return {chunk[0]: df}

        out: dict[str, pd.DataFrame] = {}
        if not isinstance(raw.columns, pd.MultiIndex):
            return out

        # Determine which level holds the ticker symbol
        lvl0_vals = set(raw.columns.get_level_values(0))
        lvl1_vals = set(raw.columns.get_level_values(1))
        # Ticker is on the level whose values match our chunk
        chunk_set = set(chunk)
        ticker_level = 0 if chunk_set & lvl0_vals else 1

        for ticker in chunk:
            try:
                if ticker_level == 0:
                    df_t = raw[ticker].copy()
                else:
                    # Rearrange: swap levels, then select by ticker
                    df_t = raw.xs(ticker, axis=1, level=1).copy()
                df_t = df_t.dropna(how="all")
                if not df_t.empty:
                    # Normalise column names to title-case
                    df_t.columns = [c.capitalize() for c in df_t.columns]
                    out[ticker] = df_t
            except (KeyError, TypeError):
                pass
        return out

    def _get_market_cap(self, ticker: str) -> float | None:
        """Fetch market cap from yfinance fast_info (used only for finalists)."""
        try:
            fi = yf.Ticker(ticker).fast_info
            mc = getattr(fi, "market_cap", None)
            return float(mc) if mc else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, ticker: str, df: pd.DataFrame) -> dict | None:
        """
        Compute screening metrics from a daily OHLCV DataFrame.

        Returns None when data is insufficient (< 15 rows).

        Args:
            ticker: Ticker symbol (used only for logging).
            df:     Daily OHLCV DataFrame with at least 'Close' and 'Volume'.

        Returns:
            dict with keys: price, price_change, volume_ratio, avg_volume, rsi,
            or None when the DataFrame is too short.
        """
        if len(df) < self._RSI_WINDOW + 1:
            return None

        # Normalise column names
        col_map = {c.lower(): c for c in df.columns}
        close_col  = col_map.get("close")
        volume_col = col_map.get("volume")
        if not close_col or not volume_col:
            return None

        close  = df[close_col].squeeze().dropna()
        volume = df[volume_col].squeeze().dropna()

        if len(close) < 2 or len(volume) < 2:
            return None

        # Price change: today vs. yesterday (%)
        price_today    = float(close.iloc[-1])
        price_prev     = float(close.iloc[-2])
        price_change   = (price_today - price_prev) / price_prev * 100.0

        # Volume: today's vs. 20-day rolling average
        avg_vol        = float(volume.iloc[:-1].tail(self._VOL_AVG_WINDOW).mean())
        today_vol      = float(volume.iloc[-1])
        volume_ratio   = (today_vol / avg_vol) if avg_vol > 0 else 1.0

        # RSI-14
        rsi_val: float | None = None
        if len(close) >= self._RSI_WINDOW + 1:
            try:
                rsi_series = ta.momentum.RSIIndicator(
                    close=close, window=self._RSI_WINDOW
                ).rsi()
                clean = rsi_series.dropna()
                if not clean.empty:
                    rsi_val = float(clean.iloc[-1])
            except Exception:
                pass

        return {
            "price":        price_today,
            "price_change": price_change,
            "volume_ratio": volume_ratio,
            "avg_volume":   avg_vol,
            "rsi":          rsi_val,
            "name":         "",   # populated by market-cap fetch if needed
            "market_cap":   None, # populated lazily for top candidates
        }

    def _compute_hotness(
        self,
        metrics: dict,
        market_priority: float,
        focus_market: str,
    ) -> float:
        """
        Compute the hotness score on a 0–10 scale.

        Formula::

            hotness = (vol_ratio_norm × 0.30)
                    + (price_chg_norm × 0.30)
                    + (rsi_extreme    × 0.20)
                    + (liquidity_norm × 0.10)
                    + (market_priority_norm × 0.10)
            × 10

        All components are clamped to [0, 1] before weighting.
        """
        vol_ratio  = metrics.get("volume_ratio", 1.0) or 1.0
        price_chg  = abs(metrics.get("price_change", 0.0) or 0.0)
        rsi        = metrics.get("rsi")
        avg_volume = metrics.get("avg_volume", 0.0) or 0.0

        # Normalise volume ratio: cap at 5×
        vol_norm   = min(vol_ratio / self._VOL_RATIO_CAP, 1.0)

        # Normalise price change: cap at 10%
        chg_norm   = min(price_chg / self._PRICE_CHG_CAP, 1.0)

        # RSI extremeness: distance from neutral (50), scaled to [0, 1]
        rsi_extreme = abs((rsi or 50.0) - 50.0) / 50.0

        # Liquidity proxy: log10(avg_volume) normalised against 10M reference
        if avg_volume > 0:
            liq_norm = min(math.log10(avg_volume) / math.log10(10_000_000), 1.0)
        else:
            liq_norm = 0.0

        # Market priority: map raw priority value to [0, 1]
        # Raw range is [-0.1, 0.2] → normalise so 0.2 → 1.0, -0.1 → 0.0
        prio_norm  = min(max((market_priority + 0.1) / 0.3, 0.0), 1.0)

        score = (
            vol_norm   * 0.30
            + chg_norm   * 0.30
            + rsi_extreme * 0.20
            + liq_norm   * 0.10
            + prio_norm  * 0.10
        )

        return score * self._HOTNESS_SCALE

    def _passes_filter(self, metrics: dict, market_type: str) -> bool:
        """
        Return True if the ticker's metrics pass the market-type thresholds.

        Args:
            metrics:     Output of ``_compute_metrics``.
            market_type: "blue_chip" | "mid_cap" | "small_cap".
        """
        thresholds = _FILTERS.get(market_type, _FILTERS["blue_chip"])

        if metrics["volume_ratio"] < thresholds["min_volume_ratio"]:
            return False
        if abs(metrics["price_change"]) < thresholds["min_price_change"]:
            return False
        if thresholds["min_avg_volume"] > 0:
            if metrics["avg_volume"] < thresholds["min_avg_volume"]:
                return False
        return True

    # ------------------------------------------------------------------
    # Focus-market quota enforcement
    # ------------------------------------------------------------------

    def _enforce_focus_quota(
        self,
        ranked: list[dict],
        focus_market: str | None,
        top: int,
    ) -> list[dict]:
        """
        Return up to *top* candidates, guaranteeing German quota when focus='DE'.

        Rules:
        - If focus_market == "DE": reserve 10 spots for DE (if available),
          fill the rest from the global ranking.
        - Otherwise: pure global ranking.
        """
        if focus_market != "DE":
            return ranked[:top]

        de_markets = {"DAX", "MDAX", "SDAX", "TecDAX"}
        de_quota   = max(10, top // 3)           # ~10–13 DE stocks minimum
        other_cap  = top - de_quota

        de_stocks    = [c for c in ranked if c["market"] in de_markets]
        other_stocks = [c for c in ranked if c["market"] not in de_markets]

        # Top DE (already sorted by hotness)
        de_top    = de_stocks[:de_quota]
        other_top = other_stocks[:other_cap]

        # Merge and re-sort by hotness
        combined = sorted(de_top + other_top, key=lambda c: c["hotness"], reverse=True)

        # If DE quota not met, allow extras from global pool
        if len(combined) < top:
            seen = {c["ticker"] for c in combined}
            extras = [c for c in ranked if c["ticker"] not in seen]
            combined.extend(extras[: top - len(combined)])
            combined.sort(key=lambda c: c["hotness"], reverse=True)

        return combined[:top]

    # ------------------------------------------------------------------
    # Pretty printer (used by CLI)
    # ------------------------------------------------------------------

    @staticmethod
    def print_results(result: dict) -> None:
        """Print a formatted screener report to stdout."""
        r = result
        print(f"\n{'=' * 70}")
        print(f"  ScreenerAgent — {r['run_at'][:19].replace('T', ' ')} UTC")
        print(f"  Markets: {', '.join(r['markets_scanned'])}  |  "
              f"Focus: {r['focus_market'] or '—'}")
        print(f"  Universe: {r['universe_size']} tickers  |  "
              f"Passed filters: {r['screened']}  |  "
              f"Returned: {len(r['candidates'])}")
        print(f"{'=' * 70}")
        print(f"  {'#':>3}  {'Ticker':<10} {'Market':<12} {'Hotness':>7} "
              f"{'Chg%':>7} {'VolRatio':>8} {'RSI':>6}")
        print(f"  {'-' * 65}")
        for i, c in enumerate(r["candidates"], 1):
            rsi_str = f"{c['rsi']:.1f}" if c["rsi"] is not None else "  N/A"
            print(
                f"  {i:>3}  {c['ticker']:<10} {c['market']:<12} "
                f"{c['hotness']:>7.2f} "
                f"{c['price_change']:>+7.2f}% "
                f"{c['volume_ratio']:>7.1f}× "
                f"{rsi_str:>6}"
            )
        print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScreenerAgent — scan multi-market stock universes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["US", "DE", "EU"],
        metavar="MKT",
        help="Market groups to scan: US DE EU (default: all three).",
    )
    parser.add_argument(
        "--focus",
        default="DE",
        metavar="MKT",
        help="Focus market for priority weighting and quota (default: DE).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        metavar="N",
        help="Number of candidates to return (default: 40).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database logging (useful for quick ad-hoc runs).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    db = None if args.no_db else Database()
    agent = ScreenerAgent(db=db)

    result = agent.run(
        markets=args.markets,
        focus_market=args.focus or None,
        top=args.top,
    )

    ScreenerAgent.print_results(result)


if __name__ == "__main__":
    main()
