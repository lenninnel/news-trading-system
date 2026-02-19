"""
Email notifier for the News Trading System.

Sends daily summaries and critical-error alerts via SMTP (Gmail-compatible).

Configuration (watchlist.yaml)
-------------------------------
    email:
      enabled: true
      smtp_host: smtp.gmail.com
      smtp_port: 587          # STARTTLS
      from_address: you@gmail.com
      to_address: you@gmail.com
      # Password via env: EMAIL_PASSWORD or SMTP_PASSWORD

Usage
-----
    from notifications.email_notifier import EmailNotifier

    notifier = EmailNotifier.from_config(cfg)
    if notifier:
        notifier.send_daily_summary(...)
        notifier.send_alert("Subject", "Body text")
"""

from __future__ import annotations

import logging
import os
import smtplib
import textwrap
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

log = logging.getLogger(__name__)


class EmailNotifier:
    """
    SMTP email notifier (STARTTLS, Gmail-compatible).

    Args:
        smtp_host:    SMTP server hostname.
        smtp_port:    SMTP port (587 for STARTTLS).
        from_address: Sender email address.
        to_address:   Recipient email address.
        password:     SMTP password (app password for Gmail).
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_address: str,
        to_address: str,
        password: str,
    ) -> None:
        self._host     = smtp_host
        self._port     = smtp_port
        self._from     = from_address
        self._to       = to_address
        self._password = password

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "EmailNotifier | None":
        """
        Build from watchlist.yaml cfg dict.

        Returns None if email is disabled or required fields are missing.
        """
        email_cfg = cfg.get("email", {})
        if not email_cfg.get("enabled"):
            return None

        # Support both field name styles
        from_addr = email_cfg.get("from_address") or email_cfg.get("from", "")
        to_addr   = email_cfg.get("to_address")   or email_cfg.get("to",   "")
        host      = email_cfg.get("smtp_host", "smtp.gmail.com")
        port      = int(email_cfg.get("smtp_port", 587))
        password  = (
            os.environ.get("EMAIL_PASSWORD")
            or os.environ.get("SMTP_PASSWORD")
            or ""
        )

        if not all([from_addr, to_addr, password]):
            log.warning(
                "Email enabled but missing from_address, to_address, or "
                "EMAIL_PASSWORD env var — notifications disabled."
            )
            return None

        return cls(host, port, from_addr, to_addr, password)

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def _send(self, subject: str, text_body: str, html_body: str | None = None) -> None:
        """Send an email. Raises on SMTP error."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self._from
        msg["To"]      = self._to

        msg.attach(MIMEText(text_body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(self._host, self._port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self._from, self._password)
            smtp.sendmail(self._from, [self._to], msg.as_string())

        log.info("Email sent: %s → %s", subject, self._to)

    def send_alert(self, subject: str, body: str) -> None:
        """Send a plain-text alert email. Silently logs on failure."""
        try:
            self._send(subject, body)
        except Exception as exc:
            log.error("Email send failed: %s", exc)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def send_daily_summary(
        self,
        signals_count: int,
        trades_count: int,
        portfolio_value: float,
        results: list[dict],
        errors: list[str],
        status: str,
        health_ok: bool = True,
        health_details: dict | None = None,
    ) -> None:
        """
        Build and send the daily market-close summary email.

        Args:
            signals_count:   Total signals generated this cycle.
            trades_count:    Total paper trades executed.
            portfolio_value: Current open portfolio value in USD.
            results:         Per-ticker result dicts from the scheduler.
            errors:          List of error strings encountered.
            status:          "success" | "partial" | "failed"
            health_ok:       Whether all system health checks passed.
            health_details:  Dict of {check_name: bool} from HealthMonitor.
        """
        now_str   = datetime.now().strftime("%Y-%m-%d %H:%M")
        date_str  = datetime.now().strftime("%Y-%m-%d")
        status_emoji = {"success": "✅", "partial": "⚠️", "failed": "❌"}.get(status, "ℹ️")
        subject = (
            f"[Trading] {status_emoji} Daily Summary {date_str} — "
            f"{signals_count} signals, {trades_count} trades"
        )

        text = self._build_text(
            now_str, signals_count, trades_count, portfolio_value,
            results, errors, status, health_ok, health_details,
        )
        html = self._build_html(
            now_str, signals_count, trades_count, portfolio_value,
            results, errors, status, health_ok, health_details,
        )

        try:
            self._send(subject, text, html)
        except Exception as exc:
            log.error("Daily summary email failed: %s", exc)

    # ------------------------------------------------------------------
    # Body builders
    # ------------------------------------------------------------------

    def _build_text(
        self,
        now_str: str,
        signals_count: int,
        trades_count: int,
        portfolio_value: float,
        results: list[dict],
        errors: list[str],
        status: str,
        health_ok: bool,
        health_details: dict | None,
    ) -> str:
        lines = [
            f"News Trading System — Daily Summary",
            f"Generated: {now_str}",
            "=" * 50,
            f"Status          : {status.upper()}",
            f"Signals         : {signals_count}",
            f"Trades executed : {trades_count}",
            f"Portfolio value : ${portfolio_value:,.2f}",
            f"System health   : {'OK' if health_ok else 'DEGRADED'}",
            "",
        ]

        if health_details:
            lines.append("Health Checks:")
            for k, v in health_details.items():
                lines.append(f"  {'✓' if v else '✗'} {k}")
            lines.append("")

        if results:
            lines.append("Signal Breakdown:")
            for r in results:
                traded = "▶ TRADE" if r.get("traded") else "· "
                lines.append(
                    f"  {traded}  {r['ticker']:<6}  {r.get('signal',''):<14}  "
                    f"{r.get('conf', 0):.0%}"
                )
            lines.append("")

        if errors:
            lines.append(f"Errors ({len(errors)}):")
            for e in errors:
                lines.append(f"  • {e}")
            lines.append("")

        lines.append("—")
        lines.append("This is an automated message from your News Trading System.")
        return "\n".join(lines)

    def _build_html(
        self,
        now_str: str,
        signals_count: int,
        trades_count: int,
        portfolio_value: float,
        results: list[dict],
        errors: list[str],
        status: str,
        health_ok: bool,
        health_details: dict | None,
    ) -> str:
        status_color = {"success": "#22c55e", "partial": "#f59e0b", "failed": "#ef4444"}.get(
            status, "#6b7280"
        )
        health_color = "#22c55e" if health_ok else "#ef4444"

        rows_html = ""
        for r in results:
            sig = r.get("signal", "")
            sig_color = "#22c55e" if "BUY" in sig else ("#ef4444" if "SELL" in sig else "#6b7280")
            traded = "✅" if r.get("traded") else ""
            rows_html += (
                f"<tr>"
                f"<td style='padding:4px 8px'>{r['ticker']}</td>"
                f"<td style='padding:4px 8px;color:{sig_color};font-weight:bold'>{sig}</td>"
                f"<td style='padding:4px 8px'>{r.get('conf', 0):.0%}</td>"
                f"<td style='padding:4px 8px;text-align:center'>{traded}</td>"
                f"</tr>"
            )

        health_rows = ""
        if health_details:
            for k, v in health_details.items():
                color = "#22c55e" if v else "#ef4444"
                icon  = "✓" if v else "✗"
                health_rows += (
                    f"<tr>"
                    f"<td style='padding:4px 8px'>{k}</td>"
                    f"<td style='padding:4px 8px;color:{color};font-weight:bold'>{icon}</td>"
                    f"</tr>"
                )

        error_html = ""
        if errors:
            items = "".join(f"<li>{e}</li>" for e in errors)
            error_html = (
                f"<h3 style='color:#ef4444'>Errors ({len(errors)})</h3>"
                f"<ul>{items}</ul>"
            )

        return textwrap.dedent(f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#1f2937">
          <h2 style="border-bottom:2px solid #e5e7eb;padding-bottom:8px">
            📊 Daily Trading Summary
          </h2>
          <p style="color:#6b7280;font-size:14px">Generated: {now_str}</p>

          <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
            <tr>
              <td style="padding:8px;background:#f9fafb;font-weight:bold">Status</td>
              <td style="padding:8px;color:{status_color};font-weight:bold">{status.upper()}</td>
            </tr>
            <tr>
              <td style="padding:8px;background:#f9fafb;font-weight:bold">Signals</td>
              <td style="padding:8px">{signals_count}</td>
            </tr>
            <tr>
              <td style="padding:8px;background:#f9fafb;font-weight:bold">Trades Executed</td>
              <td style="padding:8px">{trades_count}</td>
            </tr>
            <tr>
              <td style="padding:8px;background:#f9fafb;font-weight:bold">Portfolio Value</td>
              <td style="padding:8px">${portfolio_value:,.2f}</td>
            </tr>
            <tr>
              <td style="padding:8px;background:#f9fafb;font-weight:bold">System Health</td>
              <td style="padding:8px;color:{health_color};font-weight:bold">
                {'OK' if health_ok else 'DEGRADED'}</td>
            </tr>
          </table>

          {'<h3>Health Checks</h3><table style="border-collapse:collapse">' + health_rows + '</table>' if health_rows else ''}

          {'<h3>Signal Breakdown</h3><table style="border-collapse:collapse;width:100%"><tr style="background:#f9fafb"><th style="padding:4px 8px;text-align:left">Ticker</th><th style="padding:4px 8px;text-align:left">Signal</th><th style="padding:4px 8px;text-align:left">Conf</th><th style="padding:4px 8px">Traded</th></tr>' + rows_html + '</table>' if rows_html else ''}

          {error_html}

          <hr style="border:none;border-top:1px solid #e5e7eb;margin-top:20px"/>
          <p style="color:#9ca3af;font-size:12px">
            Automated message from your News Trading System.
          </p>
        </body>
        </html>
        """).strip()
