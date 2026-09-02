#!/usr/bin/env python3
"""
OnFailure handler: Telegram message for a failed systemd *user* unit.

Wired via ``OnFailure=nts-alert@%n.service`` (deployment/systemd/
nts-alert@.service). Standard library only, plain-text Telegram, and it
always exits 0 so a broken alert path never compounds the original
failure — a delivery problem is written to the journal instead.

Usage: nts_alert.py <unit-name>
"""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from notifications.telegram_plain import credentials_from_env, send_plain_text  # noqa: E402

_PROPS = "Result,ExecMainStatus,ExecMainCode,NRestarts,ActiveState,InactiveEnterTimestamp"


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        out = proc.stdout.strip()
        if proc.returncode != 0 and not out:
            return f"({' '.join(cmd[:2])} rc={proc.returncode}: {proc.stderr.strip()[:200]})"
        return out
    except Exception as exc:
        return f"({' '.join(cmd[:2])} failed: {exc})"


def build_message(unit: str, *, show_output: str, journal_tail: str,
                  host: str, now: datetime) -> str:
    props = {}
    for line in show_output.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k.strip()] = v.strip()
    header = [
        f"❌ {unit} FAILED on {host} — {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Result={props.get('Result', '?')} exit={props.get('ExecMainStatus', '?')} "
        f"state={props.get('ActiveState', '?')} NRestarts={props.get('NRestarts', '?')}",
        "--- journal tail ---",
    ]
    return "\n".join(header + [journal_tail or "(no journal lines)"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    unit = argv[0] if argv else "unknown.service"
    show = _run(["systemctl", "--user", "show", unit, "--no-pager", "-p", _PROPS])
    journal = _run(["journalctl", "--user", "-u", unit, "-n", "15", "--no-pager",
                    "-o", "short-iso"])
    message = build_message(
        unit, show_output=show, journal_tail=journal,
        host=socket.gethostname(), now=datetime.now(timezone.utc),
    )
    creds = credentials_from_env()
    if creds is None:
        print(f"nts-alert: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; message:\n{message}",
              file=sys.stderr)
        return 0
    ok, detail = send_plain_text(creds[0], creds[1], message)
    if not ok:
        print(f"nts-alert: delivery failed ({detail}); message:\n{message}", file=sys.stderr)
    else:
        print(f"nts-alert: delivered alert for {unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
