"""AI News Pipeline Orchestrator — entry point invoked by systemd timer.

Executes stages sequentially: Scrape → Analyze → Generate+Evaluate (per theme)
→ Brief → Email. Each stage is wrapped in retry logic. On unrecoverable failure
a failure alert email is dispatched and the process exits with code 1.

Run with --help for full CLI documentation including flag descriptions,
pipeline logic outline, configuration file expectations, and usage examples.
"""

from __future__ import annotations

import argparse
import contextvars
import io
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from . import analyzer, brief, emailer, evaluator, generator, scraper
from .config import ConfigError, from_yaml
from .db import Database
from .llm import LLMClient
from .models import InterestConfig

# ---------------------------------------------------------------------------
# Context variables for structured-logging enrichment
# ---------------------------------------------------------------------------

_current_run_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "run_id", default=None
)
_current_stage_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "stage", default=None
)
_current_theme_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "theme_id", default=None
)

# Full path of the *current* log file, set once during ``setup_logging``.
_log_file_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class StageFailedError(Exception):
    """Raised when a stage exhausts all retries."""

    def __init__(self, stage_name: str, cause: Exception) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"Stage '{stage_name}' failed: {cause}")


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class _PipelineJsonFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object.

    Enriches with ``pipeline_run_id``, ``stage``, and ``theme_id`` drawn from
    the current :mod:`contextvars` state.
    """

    def format(self, record: logging.LogRecord) -> str:
        berlin_tz = ZoneInfo("Europe/Berlin")
        now = datetime.now(berlin_tz).isoformat()

        log_entry: dict[str, Any] = {
            "timestamp": now,
            "level": record.levelname,
            "pipeline_run_id": _current_run_id.get(),
            "stage": _current_stage_name.get(),
            "theme_id": _current_theme_id.get(),
            "message": record.getMessage(),
        }

        # Attach caller location if the record has it
        if record.name:
            log_entry["logger"] = record.name
        if record.filename:
            log_entry["source"] = f"{record.filename}:{record.lineno}"

        # Extra structured data attached via ``logger.log(..., extra={})``
        # ends up on the record's ``__dict__``.  We fold it into ``extra``.
        extra_data: dict[str, Any] = {}
        for key in ("extra_data",):
            val = getattr(record, key, None)
            if val is not None and isinstance(val, dict):
                extra_data.update(val)
        if extra_data:
            log_entry["extra"] = extra_data

        # Always include the exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = traceback.format_exception_only(
                record.exc_info[0], record.exc_info[1]
            )[0].strip()
            # Full traceback on ERROR
            if record.levelno >= logging.ERROR:
                log_entry["traceback"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure structured JSON logging to stdout *and* an optional file.

    Parameters
    ----------
    log_file:
        Path for the JSON log file.  When ``None`` (or falsy), file output is
        skipped (stdout-only, suitable for systemd/journald).
    """
    global _log_file_path  # noqa: PLW0603
    _log_file_path = log_file

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = _PipelineJsonFormatter()

    # Stream handler → stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Context managers for logging enrichment
# ---------------------------------------------------------------------------


class _StageContext:
    """Context manager that sets ``_current_stage_name`` for the duration of the block."""

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> None:
        self._token = _current_stage_name.set(self._stage)

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _current_stage_name.reset(self._token)


class _ThemeContext:
    """Context manager that sets ``_current_theme_id`` for the duration of the block."""

    def __init__(self, theme_id: int) -> None:
        self._theme_id = theme_id
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> None:
        self._token = _current_theme_id.set(self._theme_id)

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _current_theme_id.reset(self._token)


# ---------------------------------------------------------------------------
# Log-file tail helper (used for failure alerts)
# ---------------------------------------------------------------------------


def _read_log_tail(lines: int = 100) -> str:
    """Read the last *lines* lines from the log file, or return ``"(no log file)"``."""
    if not _log_file_path or not os.path.isfile(_log_file_path):
        return "(no log file configured)"

    try:
        with open(_log_file_path, "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
            return "".join(all_lines[-lines:])
    except OSError:
        return "(could not read log file)"


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def retry_wrapper(
    stage_name: str,
    stage_fn: Callable[..., None],
    max_retries: int,
    backoff_seconds: int,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Execute *stage_fn* wrapped in retry logic.

    Parameters
    ----------
    stage_name:
        Human-readable stage name used in log messages.
    stage_fn:
        The callable implementing the stage (e.g. ``scraper.run``).
    max_retries:
        Number of *additional* attempts after the first (so ``max_retries=2``
        means 3 total attempts).
    backoff_seconds:
        Fixed delay (in seconds) between retry attempts.
    *args, **kwargs:
        Forwarded to *stage_fn* on each invocation.

    Raises
    ------
    StageFailedError
        If all attempts (1 + *max_retries*) fail.
    """
    logger = logging.getLogger(__name__)
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            with _StageContext(stage_name):
                logger.info(
                    "Stage '%s' started (attempt %d/%d)",
                    stage_name,
                    attempt + 1,
                    max_retries + 1,
                )
                stage_fn(*args, **kwargs)
                logger.info("Stage '%s' completed", stage_name)
                return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Stage '%s' failed (attempt %d/%d): %s — retrying in %ds",
                    stage_name,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
            else:
                logger.error(
                    "Stage '%s' failed after %d attempts: %s",
                    stage_name,
                    max_retries + 1,
                    exc,
                )

    raise StageFailedError(stage_name, last_exc)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Generate+Evaluate loop (per theme)
# ---------------------------------------------------------------------------


def _run_generate_evaluate(
    run_id: int,
    db: Database,
    config: Any,
    llm_client: LLMClient,
    interest: InterestConfig,
) -> None:
    """Generate deliverables for all pending themes, then evaluate each with a
    refinement loop.

    This is the *Generate+Evaluate* stage from the architecture — it is wrapped
    as a single retryable unit by the orchestrator's ``retry_wrapper``.
    """
    logger = logging.getLogger(__name__)

    # --- Generate initial deliverables for all pending themes ---
    db.update_pipeline_run(run_id, current_stage="generate")
    generator.run(run_id, db, config, llm_client, interest)

    # --- Evaluate each theme with refinement loop ---
    db.update_pipeline_run(run_id, current_stage="evaluate")
    themes = db.get_themes_for_run(run_id)

    for theme in themes:
        if theme["status"] != "pending":
            continue

        theme_id: int = theme["id"]
        with _ThemeContext(theme_id):
            while True:
                result = evaluator.run(
                    run_id, db, config, llm_client, theme_id, interest
                )
                if result == "approved":
                    break

                # result == "needs_refinement" — get feedback and refine
                latest_eval = db.get_latest_evaluation(theme_id)
                feedback = _build_feedback_from_eval(latest_eval)
                if feedback:
                    generator.refine(
                        run_id, db, config, llm_client, theme_id, feedback, interest
                    )
                else:
                    logger.warning(
                        "No feedback extracted from evaluation of theme %d — "
                        "attempting refine with generic prompt",
                        theme_id,
                    )
                    generator.refine(
                        run_id,
                        db,
                        config,
                        llm_client,
                        theme_id,
                        "Please improve the deliverables based on general quality standards.",
                        interest,
                    )

        logger.info(
            "Theme %d (%s) completed — status=%s",
            theme_id,
            theme.get("title", "?"),
            db.get_themes_for_run(run_id)[0]["status"] if False else "see DB",
        )


def _build_feedback_from_eval(eval_round: Optional[dict]) -> str:
    """Reconstruct a combined feedback string from a stored evaluation round.

    Mirrors the logic of ``evaluator._build_combined_feedback`` but works from
    the already-persisted JSON blobs.
    """
    if not eval_round:
        return ""

    parts: list[str] = []

    # --- Quality feedback -------------------------------------------------------
    quality_raw = eval_round.get("quality_feedback", "")
    if quality_raw:
        try:
            quality_data = json.loads(quality_raw)
        except (json.JSONDecodeError, TypeError):
            quality_data = {}

        parts.append("=== QUALITY FEEDBACK ===")
        for dtype in ("summary_en", "script_en", "script_de"):
            item = quality_data.get(dtype)
            if isinstance(item, dict):
                parts.append(
                    f"\n{dtype}: {'PASS' if item.get('pass') else 'FAIL'}"
                )
                parts.append(f"  {item.get('feedback', 'No feedback')}")
            else:
                parts.append(f"\n{dtype}: UNKNOWN — {item}")

    # --- Adversarial feedback ---------------------------------------------------
    adv_raw = eval_round.get("adversarial_feedback", "")
    if adv_raw:
        try:
            adv_data = json.loads(adv_raw)
        except (json.JSONDecodeError, TypeError):
            adv_data = {}

        parts.append("\n=== ADVERSARIAL FEEDBACK ===")
        parts.append(
            f"\nOverall: {'PASS' if adv_data.get('pass') else 'FAIL'}"
        )
        parts.append(f"  {adv_data.get('feedback', 'No feedback')}")
        issues = adv_data.get("issues", [])
        if issues:
            parts.append("\nIssues found:")
            for issue in issues:
                if isinstance(issue, dict):
                    parts.append(
                        f"  - [{issue.get('deliverable', '?')}] "
                        f"{issue.get('problem', '?')}: "
                        f"{issue.get('claim', '?')}"
                    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, load config, run the pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 1. Load configuration ------------------------------------------------
    try:
        config = from_yaml(args.config)
    except ConfigError as exc:
        # Exit code 2 = configuration error (cannot start)
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    # 2. Set up structured logging -----------------------------------------
    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # 3. Database initialisation -------------------------------------------
    db = Database(config.database.path)
    db.initialize_schema()

    if args.init_db:
        logger.info("Database schema initialized at %s — exiting", config.database.path)
        db.close()
        print(f"Database schema initialized at {config.database.path}")
        sys.exit(0)

    # 3.5 Seed feeds from YAML config into DB (for imports from old config) --
    ai_id = db.get_interest_by_name("AI")["id"]
    if config.feeds:
        for feed in config.feeds.news:
            db.upsert_feed(ai_id, feed.url, feed.name, "news")
        for feed in config.feeds.commentators:
            db.upsert_feed(ai_id, feed.url, feed.name, "commentators")

    # 4. Create pipeline run record ----------------------------------------
    berlin = ZoneInfo(config.pipeline.timezone)
    now_berlin = datetime.now(berlin)
    run_date = now_berlin.strftime("%Y-%m-%d")
    started_at = now_berlin.isoformat()

    run_id = db.create_pipeline_run(ai_id, run_date, started_at)
    _current_run_id.set(run_id)

    logger.info(
        "Pipeline started — run_id=%d date=%s timezone=%s",
        run_id,
        run_date,
        config.pipeline.timezone,
    )

    # 5. LLM client (reused across all stages) -----------------------------
    api_key = os.environ.get(config.openrouter.api_key_env, "")
    if not api_key:
        logger.error(
            "Environment variable '%s' is not set",
            config.openrouter.api_key_env,
        )
        db.update_pipeline_run(run_id, status="failed", error_message="Missing API key")
        db.close()
        sys.exit(2)

    llm_client = LLMClient(
        base_url=config.openrouter.base_url,
        api_key=api_key,
        timeout=config.pipeline.llm_request_timeout_seconds,
    )

    # Load the AI interest
    interest_row = db.get_interest_by_name("AI")
    interest = InterestConfig(
        id=interest_row["id"],
        name=interest_row["name"],
        start_time=interest_row["start_time"],
        interval_hours=interest_row["interval_hours"],
        input_data_length_mode=interest_row["input_data_length_mode"],
        input_word_count=interest_row.get("input_word_count"),
        target_summary_words=interest_row["target_summary_words"],
        target_script_en_words=interest_row["target_script_en_words"],
        target_script_de_words=interest_row["target_script_de_words"],
        target_brief_words=interest_row["target_brief_words"],
        enable_summary=bool(interest_row["enable_summary"]),
        enable_script_en=bool(interest_row["enable_script_en"]),
        enable_script_de=bool(interest_row["enable_script_de"]),
        enable_brief=bool(interest_row["enable_brief"]),
    )

    max_retries = config.pipeline.max_retries
    backoff = config.pipeline.retry_backoff_seconds

    # 6. Execute stages -----------------------------------------------------
    try:
        # --- Scrape ---
        retry_wrapper(
            "scrape",
            scraper.run,
            max_retries,
            backoff,
            run_id,
            db,
            config,
            interest,
        )

        # --- Analyze ---
        retry_wrapper(
            "analyze",
            analyzer.run,
            max_retries,
            backoff,
            run_id,
            db,
            config,
            llm_client,
        )

        # --- Generate + Evaluate (per theme) ---
        retry_wrapper(
            "generate_evaluate",
            _run_generate_evaluate,
            max_retries,
            backoff,
            run_id,
            db,
            config,
            llm_client,
            interest,
        )

        # --- Brief ---
        retry_wrapper(
            "brief",
            brief.run,
            max_retries,
            backoff,
            run_id,
            db,
            config,
            llm_client,
            interest,
        )

        # --- Email ---
        retry_wrapper(
            "email",
            emailer.run,
            max_retries,
            backoff,
            run_id,
            db,
            config,
            interest,
        )

    except StageFailedError as exc:
        # Unrecoverable stage failure
        error_msg = str(exc.cause)
        tb_str = "".join(
            traceback.format_exception(type(exc.cause), exc.cause, exc.cause.__traceback__)
        )

        db.update_pipeline_run(
            run_id,
            status="failed",
            error_message=f"Stage '{exc.stage_name}' failed: {error_msg}",
            completed_at=datetime.now(berlin).isoformat(),
        )

        logger.error(
            "Pipeline FAILED at stage '%s' — sending failure alert",
            exc.stage_name,
        )

        # Send failure alert (do NOT let alert failure mask the original error)
        try:
            log_tail = _read_log_tail(100)
            emailer.send_failure_alert(
                config, exc.stage_name, error_msg, tb_str, log_tail
            )
        except Exception as alert_exc:
            # Last resort — log to stdout/journald
            logger.error("Failed to send failure alert: %s", alert_exc)

        llm_client.close()
        db.close()
        sys.exit(1)

    # 7. Success ------------------------------------------------------------
    db.update_pipeline_run(
        run_id,
        status="completed",
        completed_at=datetime.now(berlin).isoformat(),
    )

    logger.info("Pipeline completed successfully — run_id=%d", run_id)

    llm_client.close()
    db.close()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


DESCRIPTION = (
    "AI News Pipeline — nightly batch scraper and content generator.\n"
    "\n"
    "Primary command logic (executed when invoked without --init-db):\n"
    "  1. Load YAML configuration from the directory specified by --config, validate\n"
    "     against Pydantic models, and check required environment variables.\n"
    "  2. Initialise the SQLite database schema (creates tables if missing).\n"
    "  3. Create a pipeline_run record keyed to the current date in the configured\n"
    "     timezone.\n"
    "  4. Execute stages sequentially, each with configurable retry logic:\n"
    "       a) Scrape   — Fetch articles from RSS/Atom feeds listed in\n"
    "                     feeds.yaml; extract full text via HTTP.\n"
    "       b) Analyze  — Identify 1–5 themes across scraped articles using an LLM.\n"
    "       c) Generate — For each theme produce deliverables (eng audio script,\n"
    "                     de audio script, de summary) using the strong model, then\n"
    "                     evaluate quality with the weak model and refine iteratively.\n"
    "       d) Brief    — Synthesise a daily pipeline brief from all themes.\n"
    "       e) Email    — Send the brief and per-theme deliverables via Gmail SMTP.\n"
    "  5. On unrecoverable failure at any stage, send an alert email with the\n"
    "     error details and recent log output, then exit with code 1.\n"
    "  6. On success, exit with code 0.\n"
    "\n"
    "Configuration files expected under --config directory:\n"
    "  pipeline.yaml    — Retry counts, timeouts, theme limits, timezone.\n"
    "  feeds.yaml       — Lists of news and commentator RSS/Atom feeds.\n"
    "  models.yaml      — LLM model IDs and temperatures (strong + weak).\n"
    "  database.yaml    — Path to the SQLite database file.\n"
    "  openrouter.yaml  — OpenRouter API base URL and env-var name for the API key.\n"
    "  email.yaml       — SMTP credentials and recipient address.\n"
    "\n"
    "Required environment variable:\n"
    "  OPENROUTER_API_KEY — API key for OpenRouter LLM access (name configurable,\n"
    "                        default name shown here).\n"
)

EPILOG = (
    "Examples:\n"
    "  # Run the full nightly pipeline:\n"
    "  python -m src.main --config /opt/ai-news-pipeline/config/\n"
    "\n"
    "  # Run with a custom log location:\n"
    "  python -m src.main --config ./config/ --log-file /var/log/pipeline.jsonl\n"
    "\n"
    "  # Initialise (or re-initialise) the database schema only:\n"
    "  python -m src.main --config ./config/ --init-db\n"
    "\n"
    "Exit codes:\n"
    "  0  Pipeline completed successfully, or --init-db finished.\n"
    "  1  Unrecoverable pipeline stage failure (alert email sent).\n"
    "  2  Configuration error or missing required environment variable.\n"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="DIR",
        help=(
            "Path to the directory containing the YAML configuration files "
            "(pipeline.yaml, feeds.yaml, models.yaml, database.yaml, "
            "openrouter.yaml, email.yaml). The .env file is also loaded from "
            "this directory, its parent, or the current working directory."
        ),
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help=(
            "Create or re-create the SQLite database schema (tables and indexes) "
            "at the path specified in database.yaml, then exit with code 0. "
            "Use this to bootstrap the database before the first pipeline run "
            "or after a manual wipe."
        ),
    )
    parser.add_argument(
        "--log-file",
        default="pipeline.log",
        metavar="PATH",
        help=(
            "File path for the structured JSON Lines log (one JSON object per "
            "line). Logs are always emitted to stdout regardless of this flag. "
            "Default: %(default)s."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
