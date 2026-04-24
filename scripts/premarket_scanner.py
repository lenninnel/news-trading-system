"""
D9: Pre-market scanner — expand US session universe beyond core watchlist.

Runs once daily at 13:00 UTC (before US_PRE at 13:15). Filters the S&P 500
universe down to 20-30 high-opportunity candidates based on:

  1. Liquidity          (avg_volume > 500k shares)
  2. Earnings catalyst  (earnings in [-3d, +5d])
  3. Price momentum     (|5d price change| > 3%)
  4. Volume spike       (today's volume > 1.5× 20d average)
  5. Sector contagion   (peers of flagged tickers via sector_map.json)

The scanner intentionally uses ONLY cheap, free signals (yfinance prices +
volumes). News/FinBERT/Claude sentiment runs downstream in US_PRE on just
the top candidates — running it across 100+ tickers here was both slow and
cost ~2.5k API calls/day for negligible ranking value.

The final list is the union of:
  - Top-N ranked candidates (by earnings*3 + momentum*2 + volume*1)
  - Existing core watchlist (18 tickers — always included as a safety floor)
  - Sector peers of flagged tickers (via storage.database.get_peers)

Output is written to $SCANNER_OUTPUT_PATH (default
/home/trading/trading-data/scanner_output.json) and each selected ticker is
logged to signal_events with strategy="PreMarketScanner" for attribution.

Disabled by default — set ENABLE_PREMARKET_SCANNER=true to activate.
US_PRE reads the output file via ``load_scanner_output`` and falls back to the
core 18-ticker watchlist when the scanner is disabled, the file is missing, or
the file is stale (not today's date).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_OUTPUT_PATH = Path("/home/trading/trading-data/scanner_output.json")

# Lightweight top-candidates cache consumed by the daemon at US_PRE /
# US_OPEN start. Distinct from scanner_output.json: only the high-
# confidence picks (conf ≥ 0.8) end up here, so the session tickers can
# be extended without pulling in every sector peer.
DEFAULT_CANDIDATES_PATH = Path("/tmp/premarket_candidates.json")
CANDIDATE_CONF_THRESHOLD = 0.8

MIN_AVG_VOLUME = 500_000
MOMENTUM_LOOKBACK_DAYS = 5
MOMENTUM_THRESHOLD_PCT = 3.0
EARNINGS_WINDOW_FORWARD = 5
EARNINGS_WINDOW_BACKWARD = 3

# Volume-spike signal — proxy for unusual activity without paying for news.
VOLUME_LOOKBACK_DAYS = 20
VOLUME_SPIKE_RATIO_MIN = 1.5

# Per-criterion score weights (higher = more important for ranking)
SCORE_EARNINGS = 3
SCORE_MOMENTUM = 2
SCORE_VOLUME = 1

DEFAULT_TOP_N = 20

# Chunk size for yfinance batch downloads. 500 tickers in a single call tends
# to time out; 100 is a safe middle ground.
BATCH_CHUNK = 100

# Download window must cover both the momentum lookback AND the 20d volume
# baseline with a weekend/holiday buffer. "2mo" (~42 trading days) is plenty.
BATCH_PERIOD = "2mo"


# ── Paths & I/O helpers ───────────────────────────────────────────────────


def _output_path() -> Path:
    """Where to write scanner_output.json. Overridable via env for tests."""
    override = os.environ.get("SCANNER_OUTPUT_PATH")
    if override:
        return Path(override)
    return DEFAULT_OUTPUT_PATH


def _candidates_path() -> Path:
    """Where to write premarket_candidates.json. Overridable via env."""
    override = os.environ.get("PREMARKET_CANDIDATES_PATH")
    if override:
        return Path(override)
    return DEFAULT_CANDIDATES_PATH


def _is_scanner_enabled() -> bool:
    return os.environ.get("ENABLE_PREMARKET_SCANNER", "false").strip().lower() in (
        "1", "true", "yes",
    )


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


# ── Step 1: universe ──────────────────────────────────────────────────────


def fetch_sp500_universe() -> list[str]:
    """Scrape the S&P 500 constituent list from Wikipedia.

    Uses requests + BeautifulSoup with Python's built-in ``html.parser``
    to avoid a hard lxml/html5lib dependency. Parses the first ``wikitable``
    on the page — the Symbol column is always the first column.

    S&P 500 inclusion implies a market-cap floor well above $2B, so the
    task's explicit `market_cap > $2B` filter is implicitly satisfied by
    membership — we skip the per-ticker fast_info lookup that would cost
    500 HTTP calls.

    Returns an empty list on failure; callers should treat that as
    "scanner cannot run, fall back to core watchlist".
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            SP500_WIKI_URL,
            headers={"User-Agent": "news-trading-system/d9-scanner"},
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"}) or soup.find(
            "table", class_="wikitable"
        )
        if table is None:
            raise ValueError("Wikipedia page had no constituents table")

        symbols: list[str] = []
        for row in table.find_all("tr")[1:]:  # skip header
            cells = row.find_all("td")
            if not cells:
                continue
            # First cell is the Symbol; inner <a> text is cleanest
            text = (cells[0].get_text() or "").strip().upper()
            if text:
                symbols.append(text)

        # yfinance uses dashes for class shares (BRK-B, not BRK.B)
        cleaned = [s.replace(".", "-") for s in symbols if s]
        log.info("S&P 500 universe: %d tickers", len(cleaned))
        return cleaned
    except Exception as exc:
        log.warning("S&P 500 universe fetch failed: %s", exc)
        return []


# ── Step 2: liquidity + momentum + volume-spike (one batch call) ──────────


def fetch_liquidity_and_momentum(tickers: list[str]) -> dict[str, dict]:
    """Batch-download ~2 months of OHLCV for every ticker in one shot.

    Returns {ticker: {
        "pct_change_5d":  float,  # last 5 trading days
        "avg_volume":     float,  # full-window mean (used for liquidity floor)
        "avg_volume_20d": float,  # 20 days ending *yesterday* (spike baseline)
        "volume_today":   float,  # last bar's volume (most recent session)
        "last_close":     float,
    }}.

    Tickers with insufficient data are silently dropped. Network failures are
    logged and partial results returned.
    """
    if not tickers:
        return {}

    import yfinance as yf  # local import so unit tests can monkeypatch

    stats: dict[str, dict] = {}
    for start in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[start : start + BATCH_CHUNK]
        try:
            df = yf.download(
                tickers=" ".join(chunk),
                period=BATCH_PERIOD,
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            log.warning("yfinance batch download failed (%d tickers): %s", len(chunk), exc)
            continue

        for t in chunk:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t not in df.columns.get_level_values(0):
                        continue
                    sub = df[t]
                else:
                    # Single-ticker fallback (df has flat columns)
                    sub = df
                close = sub["Close"].dropna() if "Close" in sub else pd.Series(dtype=float)
                volume = sub["Volume"].dropna() if "Volume" in sub else pd.Series(dtype=float)
                if len(close) < MOMENTUM_LOOKBACK_DAYS + 1 or len(volume) < 2:
                    continue

                # 5-day % change: compare last close to the close 5 bars back.
                ref_close = float(close.iloc[-(MOMENTUM_LOOKBACK_DAYS + 1)])
                last_close = float(close.iloc[-1])
                if ref_close <= 0:
                    continue

                # 20d volume baseline excludes the most recent bar so the spike
                # ratio doesn't self-reference. If we don't have 20 prior bars
                # yet, None disables the volume signal for this ticker today.
                prior = volume.iloc[:-1]
                avg_volume_20d = (
                    float(prior.tail(VOLUME_LOOKBACK_DAYS).mean())
                    if len(prior) >= VOLUME_LOOKBACK_DAYS else None
                )

                stats[t] = {
                    "pct_change_5d": (last_close - ref_close) / ref_close * 100.0,
                    "avg_volume": float(volume.mean()),
                    "avg_volume_20d": avg_volume_20d,
                    "volume_today": float(volume.iloc[-1]),
                    "last_close": last_close,
                }
            except Exception:
                continue

    log.info("Batch stats fetched for %d/%d tickers", len(stats), len(tickers))
    return stats


def filter_liquid(stats: dict[str, dict]) -> list[str]:
    """Return tickers with avg_volume above the liquidity floor."""
    return [t for t, s in stats.items() if s.get("avg_volume", 0) >= MIN_AVG_VOLUME]


# ── Step 3: earnings catalyst ─────────────────────────────────────────────


def find_earnings_candidates(
    tickers: Iterable[str],
    *,
    today: date | None = None,
) -> set[str]:
    """Return tickers with earnings in [today-3d, today+5d].

    Uses the existing data.events_feed.get_earnings_date() helper which is
    cached 4 hours. Runs sequentially — parallelising is pointless because
    the helper already has a TTL cache and network calls are short.
    """
    from data.events_feed import get_earnings_date

    today = today or _today_utc()
    window_start = today - timedelta(days=EARNINGS_WINDOW_BACKWARD)
    window_end = today + timedelta(days=EARNINGS_WINDOW_FORWARD)

    flagged: set[str] = set()
    for t in tickers:
        try:
            edate = get_earnings_date(t)
        except Exception:
            continue
        if edate is not None and window_start <= edate <= window_end:
            flagged.add(t)
    log.info("Earnings candidates (window %s..%s): %d", window_start, window_end, len(flagged))
    return flagged


# ── Step 4: price momentum ────────────────────────────────────────────────


def find_momentum_candidates(stats: dict[str, dict]) -> set[str]:
    """Return tickers whose 5-day |pct_change| exceeds the threshold."""
    flagged = {
        t
        for t, s in stats.items()
        if abs(s.get("pct_change_5d", 0.0)) >= MOMENTUM_THRESHOLD_PCT
    }
    log.info("Momentum candidates (|Δ5d| ≥ %.1f%%): %d", MOMENTUM_THRESHOLD_PCT, len(flagged))
    return flagged


# ── Step 5: volume-spike catalyst (free proxy for "unusual activity") ────


def find_volume_spike_candidates(stats: dict[str, dict]) -> dict[str, float]:
    """Return {ticker: spike_ratio} for tickers where volume_today / 20d-avg
    exceeds VOLUME_SPIKE_RATIO_MIN.

    Pure function — reads already-fetched batch stats, makes no network calls.
    This replaces the old news/FinBERT sentiment pass, which was both slow and
    expensive (~2500 Alpaca News + Claude calls per scan).
    """
    scores: dict[str, float] = {}
    for t, s in stats.items():
        baseline = s.get("avg_volume_20d")
        today = s.get("volume_today")
        if not baseline or baseline <= 0 or today is None:
            continue
        ratio = today / baseline
        if ratio >= VOLUME_SPIKE_RATIO_MIN:
            scores[t] = ratio
    log.info("Volume-spike candidates (ratio ≥ %.1fx): %d", VOLUME_SPIKE_RATIO_MIN, len(scores))
    return scores


# ── Step 6: sector contagion ──────────────────────────────────────────────


def expand_with_peers(tickers: Iterable[str]) -> set[str]:
    """Return the union of input tickers and their sector peers.

    Uses storage.database.Database.get_peers() which reads config/sector_map.json.
    """
    from storage.database import Database

    db = Database()
    expanded = set(tickers)
    for t in list(expanded):
        try:
            peers = db.get_peers(t) or []
        except Exception:
            peers = []
        expanded.update(p.upper() for p in peers)
    return expanded


# ── Step 7: ranking ───────────────────────────────────────────────────────


def rank_candidates(
    *,
    earnings_flag: set[str],
    momentum_flag: set[str],
    volume_scores: dict[str, float],
    stats: dict[str, dict],
) -> list[dict]:
    """Score each flagged ticker and return a ranked list of dicts.

    Score = earnings*3 + momentum*2 + volume*1, with ties broken by the
    absolute 5-day price move (higher = more interesting).
    """
    universe = earnings_flag | momentum_flag | set(volume_scores.keys())

    ranked: list[dict] = []
    for t in universe:
        score = 0
        reasons: list[str] = []
        if t in earnings_flag:
            score += SCORE_EARNINGS
            reasons.append("earnings")
        if t in momentum_flag:
            score += SCORE_MOMENTUM
            reasons.append("momentum")
        if t in volume_scores:
            score += SCORE_VOLUME
            reasons.append("volume")

        ranked.append(
            {
                "ticker": t,
                "score": score,
                "reasons": reasons,
                "pct_change_5d": stats.get(t, {}).get("pct_change_5d"),
                "volume_ratio": volume_scores.get(t),
            }
        )

    ranked.sort(
        key=lambda r: (r["score"], abs(r["pct_change_5d"] or 0)),
        reverse=True,
    )
    return ranked


# ── Step 8: orchestration ─────────────────────────────────────────────────


def run_scanner(
    *,
    core_watchlist: list[str],
    top_n: int = DEFAULT_TOP_N,
    today: date | None = None,
) -> dict:
    """Run the full pre-market scan and return the output dict.

    The returned dict is ALSO the JSON written to ``scanner_output.json``::

        {
            "date": "YYYY-MM-DD",
            "generated_at": ISO-8601 UTC,
            "tickers": ["META", "AAPL", ...],           # dedup'd final list
            "ranked": [ {ticker, score, reasons, ...} ],
            "core_watchlist": [...],                    # always-included floor
            "peers_added": [...],                       # sector contagion
            "stats": {
                "universe_size": int,
                "post_liquidity": int,
                "earnings_flagged": int,
                "momentum_flagged": int,
                "volume_flagged": int,
                "final_count": int,
            }
        }
    """
    today = today or _today_utc()

    # 1. Universe
    universe = fetch_sp500_universe()

    # 2. Liquidity + momentum + volume baseline (single batch call)
    stats = fetch_liquidity_and_momentum(universe) if universe else {}
    liquid = filter_liquid(stats)
    liquid_stats = {t: stats[t] for t in liquid if t in stats}

    # 3. Earnings catalyst (bounded by liquid universe)
    earnings = find_earnings_candidates(liquid, today=today)

    # 4. Momentum catalyst
    momentum = find_momentum_candidates(liquid_stats)

    # 5. Volume-spike catalyst — pure function over already-fetched stats.
    volume = find_volume_spike_candidates(liquid_stats)

    # 6. Rank and take top-N
    ranked = rank_candidates(
        earnings_flag=earnings,
        momentum_flag=momentum,
        volume_scores=volume,
        stats=stats,
    )
    top = ranked[:top_n]
    top_tickers = [r["ticker"] for r in top]

    # 7. Sector contagion — pull in peers of top-ranked tickers
    expanded = expand_with_peers(top_tickers)
    peers_added = sorted(expanded - set(top_tickers))

    # 8. Always include core watchlist (safety floor)
    core_set = {t.upper() for t in core_watchlist}
    final_set = core_set | expanded

    output = {
        "date": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": sorted(final_set),
        "ranked": top,
        "core_watchlist": sorted(core_set),
        "peers_added": peers_added,
        "stats": {
            "universe_size": len(universe),
            "post_liquidity": len(liquid),
            "earnings_flagged": len(earnings),
            "momentum_flagged": len(momentum),
            "volume_flagged": len(volume),
            "final_count": len(final_set),
        },
    }
    return output


# ── Step 9: persistence ───────────────────────────────────────────────────


def save_output(output: dict, path: Path | None = None) -> Path:
    """Write the scanner output dict as JSON. Creates parent dirs as needed."""
    target = path or _output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True))
    log.info("Scanner output written to %s", target)
    return target


def save_premarket_candidates(
    output: dict,
    *,
    conf_threshold: float = CANDIDATE_CONF_THRESHOLD,
    path: Path | None = None,
) -> Path:
    """Write the filtered top candidates to premarket_candidates.json.

    Keeps only tickers from ``output["ranked"]`` whose normalised confidence
    (score / max_score) is at or above ``conf_threshold``. The daemon reads
    this at US_PRE / US_OPEN start to extend the core watchlist without
    pulling in every sector peer from ``scanner_output.json``.
    """
    max_score = SCORE_EARNINGS + SCORE_MOMENTUM + SCORE_VOLUME
    candidates: list[dict] = []
    for r in output.get("ranked", []):
        score = r.get("score", 0)
        confidence = score / max_score if max_score else 0.0
        if confidence < conf_threshold:
            continue
        candidates.append(
            {
                "ticker": r["ticker"],
                "confidence": round(confidence, 3),
                "reasons": list(r.get("reasons") or []),
            }
        )

    payload = {
        "date": output.get("date") or _today_utc().isoformat(),
        "generated_at": output.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
    }
    target = path or _candidates_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("Wrote %d pre-market candidates (conf ≥ %.2f) to %s",
             len(candidates), conf_threshold, target)
    return target


def load_premarket_candidates(path: Path | None = None) -> list[dict] | None:
    """Read premarket_candidates.json; return candidates list or None.

    Returns None when the file is missing, unparseable, or stale (date !=
    today UTC). Each candidate is ``{ticker, confidence, reasons}``.
    """
    target = path or _candidates_path()
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text())
    except Exception as exc:
        log.warning("Failed to parse candidates cache at %s: %s", target, exc)
        return None
    if data.get("date") != _today_utc().isoformat():
        log.info("Candidates cache stale (date=%s, today=%s)",
                 data.get("date"), _today_utc().isoformat())
        return None
    return list(data.get("candidates") or [])


def load_scanner_output(path: Path | None = None) -> dict | None:
    """Read scanner_output.json and return the dict, or None if missing/stale.

    "Stale" means the ``date`` field does not match today (UTC). US_PRE uses
    this to decide whether to layer the scanner on top of the core watchlist.
    """
    target = path or _output_path()
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text())
    except Exception as exc:
        log.warning("Failed to parse scanner output at %s: %s", target, exc)
        return None
    if data.get("date") != _today_utc().isoformat():
        log.info("Scanner output is stale (date=%s, today=%s)",
                 data.get("date"), _today_utc().isoformat())
        return None
    return data


def resolve_us_tickers(core_watchlist: list[str]) -> list[str]:
    """Return the effective US ticker list for today's sessions.

    If the scanner is disabled or its output is missing/stale, returns the
    core watchlist unchanged — preserving legacy behaviour. When enabled and
    a fresh output exists, returns the union of scanner picks and the core
    watchlist.
    """
    if not _is_scanner_enabled():
        return list(core_watchlist)
    data = load_scanner_output()
    if data is None:
        return list(core_watchlist)
    tickers = data.get("tickers") or []
    if not tickers:
        return list(core_watchlist)
    return sorted(set(tickers) | {t.upper() for t in core_watchlist})


# ── Step 10: signal_events attribution ────────────────────────────────────


def log_to_signal_events(output: dict, *, session: str = "PREMARKET_SCAN") -> None:
    """Log one row per selected ticker with strategy='PreMarketScanner'.

    Fire-and-forget — never raises. Lets analytics answer "which tickers did
    the scanner surface yesterday and how did they do?".
    """
    try:
        from analytics.signal_logger import SignalLogger
    except Exception as exc:
        log.warning("SignalLogger import failed: %s", exc)
        return

    logger = SignalLogger()
    ranked_by_ticker = {r["ticker"]: r for r in output.get("ranked", [])}
    max_score = SCORE_EARNINGS + SCORE_MOMENTUM + SCORE_VOLUME
    for ticker in output.get("tickers", []):
        r = ranked_by_ticker.get(ticker)
        reasons = r.get("reasons") if r else ["core_watchlist"]
        confidence = (r.get("score", 0) / max_score) if r else 0.0
        try:
            logger.log(
                {
                    "session": session,
                    "ticker": ticker,
                    "strategy": "PreMarketScanner",
                    "signal": "WATCH",
                    "confidence": round(confidence, 3),
                    # Scanner no longer computes sentiment; FinBERT/Claude
                    # scoring happens in US_PRE on just the top picks.
                    "sentiment_score": None,
                    "debate_outcome": ",".join(reasons) if reasons else None,
                }
            )
        except Exception as exc:
            log.warning("Failed to log scanner signal for %s: %s", ticker, exc)


# ── CLI entry point ───────────────────────────────────────────────────────


def main(core_watchlist: list[str] | None = None, *, top_n: int = DEFAULT_TOP_N) -> dict:
    """Run the scanner and persist output. Returns the output dict."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if core_watchlist is None:
        # Import lazily so unit tests can pass their own core list.
        from scheduler.daily_runner import DailyScheduler
        core_watchlist = DailyScheduler._load_us_tickers()

    output = run_scanner(core_watchlist=core_watchlist, top_n=top_n)
    save_output(output)
    save_premarket_candidates(output)
    log_to_signal_events(output)

    stats = output["stats"]
    print(
        f"[scanner] universe={stats['universe_size']} "
        f"liquid={stats['post_liquidity']} "
        f"earnings={stats['earnings_flagged']} "
        f"momentum={stats['momentum_flagged']} "
        f"volume={stats['volume_flagged']} "
        f"→ final={stats['final_count']}",
        flush=True,
    )
    return output


if __name__ == "__main__":
    main()
