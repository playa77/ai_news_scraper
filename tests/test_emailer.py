"""Tests for the emailer module — SMTP dispatch, retry logic, and formatting.

Uses mocked SMTP to avoid real network calls and an in-memory SQLite DB
for ``run()`` integration tests.
"""

import os
import smtplib
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from src.db import Database
from src.models import (
    Config,
    DatabaseConfig,
    EmailConfig,
    FeedDef,
    FeedsConfig,
    InterestConfig,
    ModelDef,
    ModelsConfig,
    OpenRouterConfig,
    PipelineConfig,
)


# ===================================================================
# Helper
# ===================================================================


def _make_smtp_mock():
    """Return a MagicMock configured as a context manager standing in for SMTP.

    ``with smtplib.SMTP(...) as server:`` calls ``__enter__()`` on the mock
    returned by the patched ``smtplib.SMTP``.  We configure ``__enter__`` to
    return *itself* so that ``starttls``, ``login``, ``send_message`` etc. are
    recorded on the single mock object tests can assert against.
    """
    m = MagicMock()
    m.__enter__.return_value = m
    return m


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def config():
    """Create a minimal Config with fixed email settings for tests."""
    return Config(
        feeds=FeedsConfig(
            news=[FeedDef(name="test", url="https://example.com/rss")],
            commentators=[FeedDef(name="test2", url="https://example.com/atom")],
        ),
        models=ModelsConfig(
            strong=ModelDef(id="deepseek/deepseek-v4-pro", temperature=0.7),
            weak=ModelDef(id="deepseek/deepseek-v4-flash", temperature=0.7),
        ),
        pipeline=PipelineConfig(),
        email=EmailConfig(
            recipient="to@example.com",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_user="from@example.com",
            smtp_password_env="GMAIL_APP_PASSWORD",
        ),
        database=DatabaseConfig(path=":memory:"),
        openrouter=OpenRouterConfig(
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
        ),
    )


@pytest.fixture
def db():
    """In-memory database with schema initialized."""
    database = Database(":memory:")
    database.initialize_schema()
    yield database
    database.close()


@pytest.fixture
def seeded_db(db, config):
    """In-memory database with a pipeline run, themes, briefs, and deliverables.

    Returns a dict with the Database, config, and run_id so tests can inspect.
    """
    run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")

    # Insert a daily brief
    db.insert_daily_brief(run_id, "This is the daily brief content.", 42)

    # Insert two approved themes with deliverables
    theme_a_id = db.insert_theme(
        run_id, "AI Safety", "Safety desc", [1], "breakthrough", 0,
    )
    db.update_theme_status(theme_a_id, "approved")
    db.insert_deliverable(theme_a_id, "summary_en", "English summary for Safety", 1)
    db.insert_deliverable(theme_a_id, "script_en", "English script for Safety", 1)
    db.insert_deliverable(theme_a_id, "script_de", "German script for Safety", 1)

    theme_b_id = db.insert_theme(
        run_id, "LLM Advances", "Advances desc", [2], "emerging", 1,
    )
    db.update_theme_status(theme_b_id, "auto_approved")
    db.insert_deliverable(theme_b_id, "summary_en", "English summary for LLM", 1)
    db.insert_deliverable(theme_b_id, "script_de", "German script for LLM", 1)

    # A pending (not approved) theme — should be skipped
    theme_c_id = db.insert_theme(
        run_id, "Pending Theme", "Pending desc", [3], "emerging", 2,
    )
    db.update_theme_status(theme_c_id, "pending")

    return {"db": db, "config": config, "run_id": run_id}


# ===================================================================
# 1. Error hierarchy
# ===================================================================


class TestErrorHierarchy:
    """Verify ``EmailError`` is a proper exception subclass."""

    def test_is_exception_subclass(self):
        from src.emailer import EmailError

        assert issubclass(EmailError, Exception)

    def test_can_be_raised_and_caught(self):
        from src.emailer import EmailError

        with pytest.raises(EmailError):
            raise EmailError("test error")

    def test_preserves_message(self):
        from src.emailer import EmailError

        exc = EmailError("something broke")
        assert str(exc) == "something broke"


# ===================================================================
# 2. _send_email — basic SMTP interaction
# ===================================================================


class TestSendEmail:
    """Basic SMTP interaction: connect, starttls, login, send_message, quit."""

    def test_send_email_happy_path(self, config):
        """_send_email should call the full SMTP dance and return."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            _send_email(config, "Subject line", "Body text")

        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("from@example.com", "testpass")
        smtp.send_message.assert_called_once()

        # Verify the MIMEText message
        msg = smtp.send_message.call_args[0][0]
        assert msg["Subject"] == "Subject line"
        assert msg["From"] == "from@example.com"
        assert msg["To"] == "to@example.com"
        assert msg.get_payload(decode=True).decode("utf-8") == "Body text"

    def test_send_email_subject_special_chars(self, config):
        """Subject lines with dashes/unicode should pass through cleanly."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            _send_email(config, "AI Daily Brief — 2026-05-14", "content")

        msg = smtp.send_message.call_args[0][0]
        assert msg["Subject"] == "AI Daily Brief — 2026-05-14"

    def test_send_email_passes_host_and_port(self, config):
        """smtplib.SMTP is called with the configured host and port."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp) as mock_smtp_class, patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            _send_email(config, "Subject", "Body")

        mock_smtp_class.assert_called_once_with("smtp.gmail.com", 587)


# ===================================================================
# 3. _send_email — retry on transient errors
# ===================================================================


class TestSendEmailRetry:
    """_send_email retries on SMTPException/OSError up to 3 total attempts."""

    def test_retry_on_smtp_exception(self, config):
        """SMTPException causes a retry; succeeds on second attempt."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        smtp.starttls.side_effect = [
            smtplib.SMTPException("Transient error"),
            None,  # success on second call
        ]

        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep") as mock_sleep:
            _send_email(config, "Subject", "Body")

        # First attempt raised, second succeeded
        assert smtp.starttls.call_count == 2
        mock_sleep.assert_called_once_with(10)

    def test_retry_on_os_error(self, config):
        """OSError causes a retry; succeeds on second attempt."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        smtp.starttls.side_effect = [
            OSError("Connection reset"),
            None,
        ]

        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep") as mock_sleep:
            _send_email(config, "Subject", "Body")

        assert smtp.starttls.call_count == 2
        mock_sleep.assert_called_once_with(10)

    def test_exhaust_retries_raises_email_error(self, config):
        """All 3 attempts fail → EmailError is raised."""
        from src.emailer import EmailError, _send_email

        smtp = _make_smtp_mock()
        smtp.starttls.side_effect = smtplib.SMTPException("Always fails")

        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep") as mock_sleep:
            with pytest.raises(EmailError, match="Failed to send email"):
                _send_email(config, "Subject", "Body")

        # Should have tried 3 times total (1 initial + 2 retries)
        assert smtp.starttls.call_count == 3
        # Should have slept twice (after attempt 1 and attempt 2)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(10), call(10)])

    def test_retry_backoff_value(self, config):
        """Each retry uses 10-second backoff."""
        from src.emailer import _send_email

        smtp = _make_smtp_mock()
        smtp.starttls.side_effect = smtplib.SMTPException("Fail")

        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep") as mock_sleep:
            with pytest.raises(Exception):
                _send_email(config, "Subject", "Body")

        for sleep_call in mock_sleep.call_args_list:
            assert sleep_call == call(10)


# ===================================================================
# 4. _send_email — missing password env var
# ===================================================================


class TestSendEmailPasswordMissing:
    """_send_email raises EmailError when the SMTP password env var is unset."""

    def test_raises_email_error_when_env_var_missing(self, config):
        """No GMAIL_APP_PASSWORD → EmailError with descriptive message."""
        from src.emailer import EmailError, _send_email

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EmailError) as excinfo:
                _send_email(config, "Subject", "Body")

        assert "GMAIL_APP_PASSWORD" in str(excinfo.value)
        assert "not set" in str(excinfo.value)

    def test_raises_email_error_when_env_var_empty(self, config):
        """Empty GMAIL_APP_PASSWORD → EmailError."""
        from src.emailer import EmailError, _send_email

        with patch.dict(os.environ, {"GMAIL_APP_PASSWORD": ""}):
            with pytest.raises(EmailError) as excinfo:
                _send_email(config, "Subject", "Body")

        assert "GMAIL_APP_PASSWORD" in str(excinfo.value)

    def test_no_smtp_connection_attempted(self, config):
        """No SMTP class call when password is missing."""
        from src.emailer import _send_email

        with patch("smtplib.SMTP") as mock_smtp_class, patch.dict(
            os.environ, {}, clear=True
        ):
            with pytest.raises(Exception):
                _send_email(config, "Subject", "Body")

        mock_smtp_class.assert_not_called()


# ===================================================================
# 5. run() — full success email flow
# ===================================================================


class TestRunEmail:
    """run() sends brief + per-theme emails with correct subjects and bodies."""

    def test_sends_brief_email(self, seeded_db):
        """run() sends a brief email with subject 'AI Daily Brief — {run_date}'."""
        from src.emailer import run

        db = seeded_db["db"]
        config = seeded_db["config"]
        run_id = seeded_db["run_id"]

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        sent_subjects = [
            a[0][0]["Subject"] for a in smtp.send_message.call_args_list
        ]
        assert "AI Daily Brief — 2026-05-14" in sent_subjects

        # Verify brief body
        brief_msg = None
        for a in smtp.send_message.call_args_list:
            msg = a[0][0]
            if msg["Subject"] == "AI Daily Brief — 2026-05-14":
                brief_msg = msg
                break
        assert brief_msg is not None
        assert brief_msg.get_payload(decode=True).decode("utf-8") == "This is the daily brief content."

    def test_sends_theme_emails(self, seeded_db):
        """run() sends per-theme emails for approved/auto_approved themes."""
        from src.emailer import run

        db = seeded_db["db"]
        config = seeded_db["config"]
        run_id = seeded_db["run_id"]

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        sent_subjects = [
            a[0][0]["Subject"] for a in smtp.send_message.call_args_list
        ]

        # Both approved themes should be present
        assert "AI Theme: AI Safety — 2026-05-14" in sent_subjects
        assert "AI Theme: LLM Advances — 2026-05-14" in sent_subjects
        # Pending theme should NOT be present
        assert "AI Theme: Pending Theme — 2026-05-14" not in sent_subjects

    def test_theme_email_body_includes_all_deliverables(self, seeded_db):
        """run() includes summary_en, script_en, script_de in the theme body."""
        from src.emailer import run

        db = seeded_db["db"]
        config = seeded_db["config"]
        run_id = seeded_db["run_id"]

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        # Find the AI Safety theme email
        theme_msg = None
        for a in smtp.send_message.call_args_list:
            msg = a[0][0]
            if msg["Subject"] == "AI Theme: AI Safety — 2026-05-14":
                theme_msg = msg
                break
        assert theme_msg is not None

        body = theme_msg.get_payload(decode=True).decode("utf-8")
        assert "=== ENGLISH SUMMARY ===" in body
        assert "English summary for Safety" in body
        assert "=== ENGLISH SCRIPT ===" in body
        assert "English script for Safety" in body
        assert "=== GERMAN SCRIPT ===" in body
        assert "German script for Safety" in body

    def test_sends_correct_number_of_emails(self, seeded_db):
        """run() sends 1 brief + 2 theme emails = 3 total."""
        from src.emailer import run

        db = seeded_db["db"]
        config = seeded_db["config"]
        run_id = seeded_db["run_id"]

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        assert smtp.send_message.call_count == 3


# ===================================================================
# 6. run() — missing brief
# ===================================================================


class TestRunMissingBrief:
    """run() handles missing briefs gracefully — skips the brief email."""

    def test_skips_brief_when_none_exists(self, db, config):
        """No daily_briefs row → no brief email sent; theme emails still go out."""
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")

        # Add a theme with deliverables so we can verify at least something is sent
        theme_id = db.insert_theme(run_id, "Only Theme", "Desc", [1], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "summary_en", "Only summary", 1)

        from src.emailer import run

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        sent_subjects = [
            a[0][0]["Subject"] for a in smtp.send_message.call_args_list
        ]

        # No brief email
        assert "AI Daily Brief" not in " ".join(sent_subjects)
        # Theme email still sent
        assert "AI Theme: Only Theme — 2026-05-14" in sent_subjects
        assert smtp.send_message.call_count == 1


# ===================================================================
# 7. run() — themes with no deliverables
# ===================================================================


class TestRunNoDeliverables:
    """run() skips themes that have no deliverables."""

    def test_skips_theme_with_no_deliverables(self, db, config):
        """Approved theme with zero deliverables → skipped."""
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")
        db.insert_daily_brief(run_id, "Brief content", 10)

        theme_id = db.insert_theme(
            run_id, "Empty Theme", "No deliverables", [1], "emerging", 0,
        )
        db.update_theme_status(theme_id, "approved")
        # No deliverables inserted

        from src.emailer import run

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        sent_subjects = [
            a[0][0]["Subject"] for a in smtp.send_message.call_args_list
        ]

        # Only the brief should have been sent
        assert len(sent_subjects) == 1
        assert "AI Daily Brief — 2026-05-14" in sent_subjects
        assert "AI Theme: Empty Theme" not in " ".join(sent_subjects)

    def test_skips_theme_with_only_unknown_deliverable_types(self, db, config):
        """Theme with deliverables of types not in the heading list → skipped.

        The code iterates over known types (summary_en, script_en, script_de).
        If none match, ``parts`` stays empty and the theme is skipped.
        """
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")
        db.insert_daily_brief(run_id, "Brief", 10)

        theme_id = db.insert_theme(
            run_id, "Odd Theme", "Deliverables of unexpected type", [1], "emerging", 0,
        )
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "social_post", "Some social content", 1)

        from src.emailer import run

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ), patch("time.sleep"):
            run(run_id, db, config, InterestConfig(name="AI", id=1))

        sent_subjects = [
            a[0][0]["Subject"] for a in smtp.send_message.call_args_list
        ]

        assert len(sent_subjects) == 1
        assert "AI Daily Brief — 2026-05-14" in sent_subjects
        assert "AI Theme: Odd Theme" not in " ".join(sent_subjects)


# ===================================================================
# 8. run() — missing pipeline run
# ===================================================================


class TestRunMissingRun:
    """run() raises EmailError when the pipeline run does not exist."""

    def test_raises_email_error_for_nonexistent_run(self, db, config):
        """Calling run() with a run_id that doesn't exist → EmailError."""
        from src.emailer import EmailError, run

        with pytest.raises(EmailError) as excinfo:
            run(9999, db, config, InterestConfig(name="AI", id=1))

        assert "9999" in str(excinfo.value)
        assert "not found" in str(excinfo.value)

    def test_no_emails_sent_for_nonexistent_run(self, db, config):
        """When run_id is invalid, no SMTP calls are made."""
        from src.emailer import run

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            with pytest.raises(Exception):
                run(9999, db, config, InterestConfig(name="AI", id=1))

        smtp.send_message.assert_not_called()


# ===================================================================
# 9. send_failure_alert
# ===================================================================


class TestFailureAlert:
    """send_failure_alert sends a properly formatted failure email."""

    def test_subject_format(self, config):
        """Subject is 'AI Pipeline FAILURE — {stage_name} — {date}'."""
        from src.emailer import send_failure_alert

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            send_failure_alert(
                config, "AI", "scrape", "Connection refused", "Traceback...",
            )

        msg = smtp.send_message.call_args[0][0]
        today = datetime.now().strftime("%Y-%m-%d")
        assert msg["Subject"] == f"AI Pipeline FAILURE — scrape — {today}"

    def test_body_includes_all_sections(self, config):
        """Body contains stage name, error, traceback, and log tail."""
        from src.emailer import send_failure_alert

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            send_failure_alert(
                config,
                "AI",
                "extract",
                "TimeoutError: LLM did not respond",
                "Traceback (most recent call last):\n  File ...",
            )

        body = smtp.send_message.call_args[0][0].get_payload(decode=True).decode("utf-8")
        assert "Stage: extract" in body
        assert "TimeoutError: LLM did not respond" in body
        assert "Traceback (most recent call last):" in body

    def test_sends_single_email(self, config):
        """send_failure_alert sends exactly one email."""
        from src.emailer import send_failure_alert

        smtp = _make_smtp_mock()
        with patch("smtplib.SMTP", return_value=smtp), patch.dict(
            os.environ, {"GMAIL_APP_PASSWORD": "testpass"}
        ):
            send_failure_alert(
                config, "AI", "scrape", "error", "traceback",
            )

        assert smtp.send_message.call_count == 1
