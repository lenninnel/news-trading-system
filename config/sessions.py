"""Shared session schedule — single source of truth.

Both scheduler/daily_runner.py and api/main.py import SCHEDULE from
here. Adding/changing a session means updating this file ONLY; no
other file should redefine the schedule.

Times are UTC.
"""
from __future__ import annotations

from typing import TypedDict


class SessionSpec(TypedDict, total=False):
    name: str
    hour: int
    minute: int


# Authoritative session schedule. Maintain in this file ONLY.
# scheduler/daily_runner.py keeps full session metadata (tickers,
# workers, session_type) elsewhere; this constant captures only
# the time-of-day used by both the scheduler and the API.
SCHEDULE: list[SessionSpec] = [
    {"name": "XETRA_PRE",      "hour": 6,  "minute": 45},
    {"name": "XETRA_OPEN",     "hour": 7,  "minute": 0},
    {"name": "PREMARKET_SCAN", "hour": 13, "minute": 0},
    {"name": "US_PRE",         "hour": 13, "minute": 15},
    {"name": "PEAD_OPEN",      "hour": 13, "minute": 45},
    {"name": "US_OPEN",        "hour": 14, "minute": 30},
    {"name": "MIDDAY",         "hour": 18, "minute": 0},
    {"name": "EOD",            "hour": 22, "minute": 15},
]


__all__ = ["SCHEDULE", "SessionSpec"]
