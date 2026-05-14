"""Email dispatch for AI News Pipeline — Gmail SMTP with App Password.

Provides the pipeline with success-email formatting (daily brief + per-theme
deliverables) and failure-alert formatting (stage name, error, traceback, log
tail).  Internal SMTP send retries transient errors up to 2 times with a 10 s
fixed backoff.
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from email.mime.text import MIMEText

from .db import Database
from .models import Config

logger = logging.getLogger(__name__)

# Number of *additional* SMTP attempts after the initial try
_SMTP_MAX_RETRIES = 2
_SMTP_RETRY_BACKOFF_SECONDS = 10


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class EmailError(Exception):
    """Raised when an email cannot be sent (SMTP / connection error)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_failure_alert(
    config: Config,
    stage_name: str,
    error_message: str,
    traceback_str: str,
    log_tail: str,
) -> None:
    """Send a single failure-alert email.

    Called by the orchestrator when a stage exhausts all retries.

    Parameters
    ----------
    config:
        Pipeline configuration (holds SMTP credentials).
    stage_name:
        Name of the failed stage (e.g. ``"scrape"``).
    error_message:
        Human-readable error description.
    traceback_str:
        Full traceback string from the exception.
    log_tail:
        Last ~100 lines of the pipeline log file.
    """
    from datetime import datetime

    date = datetime.now().strftime("%Y-%m-%d")
    subject = f"AI Pipeline FAILURE — {stage_name} — {date}"
    body = (
        f"Stage: {stage_name}\n"
        f"Error: {error_message}\n"
        f"\n"
        f"Traceback:\n{traceback_str}\n"
        f"\n"
        f"Recent logs:\n{log_tail}"
    )
    _send_email(config, subject, body)
    logger.info(
        "Failure alert email sent — stage=%s date=%s",
        stage_name,
        date,
    )


def run(run_id: int, db: Database, config: Config) -> None:
    """Send success emails: one daily brief + one per approved theme.

    Parameters
    ----------
    run_id:
        The pipeline run whose results should be emailed.
    db:
        Database handle used to read deliverables.
    config:
        Pipeline configuration.
    """
    run = db.get_pipeline_run(run_id)
    if run is None:
        raise EmailError(f"Pipeline run {run_id} not found in database")
    run_date = run["run_date"]

    emails_sent = 0

    # --- daily brief ----------------------------------------------------------
    brief_row = db._conn.execute(
        "SELECT * FROM daily_briefs WHERE pipeline_run_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()

    if brief_row is not None:
        brief = dict(brief_row)
        _send_email(config, f"AI Daily Brief — {run_date}", brief["content"])
        emails_sent += 1

    # --- per-theme deliverables -----------------------------------------------
    theme_rows = db._conn.execute(
        "SELECT * FROM themes WHERE pipeline_run_id = ? "
        "AND status IN ('approved', 'auto_approved') "
        "ORDER BY order_index",
        (run_id,),
    ).fetchall()

    for row in theme_rows:
        theme = dict(row)
        deliverables = db.get_latest_deliverables(theme["id"])
        if not deliverables:
            continue

        parts: list[str] = []
        for dtype, heading in [
            ("summary_en", "=== ENGLISH SUMMARY ==="),
            ("script_en", "=== ENGLISH SCRIPT ==="),
            ("script_de", "=== GERMAN SCRIPT ==="),
        ]:
            if dtype in deliverables:
                parts.append(f"{heading}\n{deliverables[dtype]['content']}")

        if not parts:
            continue

        body = "\n\n".join(parts)
        _send_email(config, f"AI Theme: {theme['title']} — {run_date}", body)
        emails_sent += 1

    logger.info(
        "Email stage complete — sent %d email(s) for run %d",
        emails_sent,
        run_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _send_email(config: Config, subject: str, body: str) -> None:
    """Connect to Gmail SMTP, login with App Password, and send a plain-text email.

    Retries transient SMTP errors up to :data:`_SMTP_MAX_RETRIES` times with a
    fixed :data:`_SMTP_RETRY_BACKOFF_SECONDS` backoff.

    Raises
    ------
    EmailError
        If all retries are exhausted or the SMTP password env var is missing.
    """
    password = os.environ.get(config.email.smtp_password_env)
    if not password:
        raise EmailError(
            f"Environment variable '{config.email.smtp_password_env}' is not set"
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = config.email.sender
    msg["To"] = config.email.recipient

    last_exc: Exception | None = None

    for attempt in range(_SMTP_MAX_RETRIES + 1):  # 1 initial + N retries
        try:
            with smtplib.SMTP(config.email.smtp_host, config.email.smtp_port) as server:
                server.starttls()
                server.login(config.email.smtp_user, password)
                server.send_message(msg)
            logger.info("Email sent — subject=%r", subject)
            return
        except (smtplib.SMTPException, OSError) as exc:
            last_exc = exc
            if attempt < _SMTP_MAX_RETRIES:
                logger.warning(
                    "SMTP error (attempt %d/%d) — retrying in %ds: %s",
                    attempt + 1,
                    _SMTP_MAX_RETRIES + 1,
                    _SMTP_RETRY_BACKOFF_SECONDS,
                    exc,
                )
                time.sleep(_SMTP_RETRY_BACKOFF_SECONDS)

    raise EmailError(
        f"Failed to send email after {_SMTP_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc
