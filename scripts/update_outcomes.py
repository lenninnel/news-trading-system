#!/usr/bin/env python3
"""
Standalone entry point for the signal outcome tracker.

Meant to run as a nightly cron (23:00 UTC) so 3d/5d/10d outcome
columns on ``signal_events`` get filled regardless of whether the
EOD trading session completed successfully.

Usage::

    python3 scripts/update_outcomes.py            # one-shot backfill
    python3 scripts/update_outcomes.py --backfill # same, explicit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as a bare script
# (e.g. from cron) rather than via ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics.outcome_tracker import run_outcome_tracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill 3d/5d/10d price outcomes on signal_events.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill all pending outcomes (default behaviour — flag kept "
             "for CLI clarity / cron scripts).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs; only print the summary line.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_outcome_tracker()
    total = sum(result.values())
    print(f"update_outcomes: updated {total} rows {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
