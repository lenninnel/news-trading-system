"""Shared helpers for the cross-sectional factor test (offline analysis only).

Nothing here touches the NTS live path. The only external dependency is the
Polygon REST API (read-only), and every response is cached on disk under
``cache/`` so a second run never hits Polygon again.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
GROUPED_DIR = CACHE / "grouped"
GROUPED_ADJ_DIR = CACHE / "grouped_adj"
SNAP_DIR = CACHE / "snapshots"
REF_DIR = CACHE / "ref"
for _d in (GROUPED_DIR, GROUPED_ADJ_DIR, SNAP_DIR, REF_DIR, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# Polygon Starter entitlement: rolling 5-year history. Probed 2026-09-07:
# 2021-09-07 -> NOT_AUTHORIZED, 2021-09-08 -> OK.
DATA_START = date(2021, 9, 8)
DATA_END = date(2026, 9, 4)  # last completed US session before 2026-09-07 (Labor Day)

BASE = "https://api.polygon.io"

log = logging.getLogger("xsec")


def api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        env = HERE.parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POLYGON_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("POLYGON_API_KEY not set (env or repo .env)")
    return key


class Polygon:
    """Minimal paced client: timeouts, retries, backoff on 429 / hangs."""

    def __init__(self, pace_s: float = 0.25, timeout_s: int = 40):
        self.key = api_key()
        self.s = requests.Session()
        self.pace = pace_s
        self.timeout = timeout_s
        self._last = 0.0
        self.calls = 0

    def get(self, url: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.key
        for attempt in range(8):
            wait = self.pace - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.monotonic()
                r = self.s.get(url, params=params, timeout=self.timeout)
                self.calls += 1
                if r.status_code == 429:
                    log.warning("429 rate-limited, sleeping 20s (attempt %d)", attempt)
                    time.sleep(20)
                    self.pace = min(self.pace * 1.5, 5.0)
                    continue
                if r.status_code >= 500:
                    log.warning("HTTP %d, retry in %ds", r.status_code, 5 * (attempt + 1))
                    time.sleep(5 * (attempt + 1))
                    continue
                j = r.json()
                if j.get("status") == "NOT_AUTHORIZED":
                    return j  # caller decides (history entitlement boundary)
                if r.status_code != 200:
                    log.warning("HTTP %d body=%s", r.status_code, str(j)[:200])
                    time.sleep(5 * (attempt + 1))
                    continue
                return j
            except (requests.Timeout, requests.ConnectionError) as e:
                log.warning("%s on %s (attempt %d) -> backoff", type(e).__name__, url[-60:], attempt)
                time.sleep(10 * (attempt + 1))
                self.pace = min(self.pace * 1.5, 5.0)
        raise RuntimeError(f"giving up on {url}")

    def paginate(self, url: str, params: dict) -> list[dict]:
        out: list[dict] = []
        j = self.get(url, params)
        while True:
            out.extend(j.get("results", []) or [])
            nxt = j.get("next_url")
            if not nxt:
                return out
            j = self.get(nxt)
