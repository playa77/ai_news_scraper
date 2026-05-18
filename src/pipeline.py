"""Per-interest pipeline orchestration.

Provides ``run_interest_pipeline`` which executes all pipeline stages for a
single interest, respecting the interest's deliverable toggles, data-length
policy, and word-count targets.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from . import analyzer, brief, emailer, evaluator, generator, scraper
from .db import Database
from .llm import LLMClient
from .models import Config, InterestConfig

logger = logging.getLogger(__name__)


class StageFailedError(Exception):
    """Raised when a stage exhausts all retries."""

    def __init__(self, stage_name: str, cause: Exception) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"Stage '{stage_name}' failed: {cause}")


def _interest_from_db(db: Database, interest_id: int) -> InterestConfig:
    """Load an interest from the database as an InterestConfig."""
    row = db.get_interest(interest_id)
    if not row:
        raise ValueError(f"Interest {interest_id} not found")
    return InterestConfig(
        id=row["id"],
        name=row["name"],
        start_time=row["start_time"],
        interval_hours=row["interval_hours"],
        input_data_length_mode=row["input_data_length_mode"],
        input_word_count=row.get("input_word_count"),
        target_summary_words=row["target_summary_words"],
        target_script_en_words=row["target_script_en_words"],
        target_script_de_words=row["target_script_de_words"],
        target_brief_words=row["target_brief_words"],
        enable_summary=bool(row["enable_summary"]),
        enable_script_en=bool(row["enable_script_en"]),
        enable_script_de=bool(row["enable_script_de"]),
        enable_brief=bool(row["enable_brief"]),
    )


def retry_wrapper(
    stage_name: str,
    stage_fn: Callable[..., None],
    max_retries: int,
    backoff_seconds: int,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Execute *stage_fn* wrapped in retry logic."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Stage '%s' started (attempt %d/%d)",
                stage_name, attempt + 1, max_retries + 1,
            )
            stage_fn(*args, **kwargs)
            logger.info("Stage '%s' completed", stage_name)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Stage '%s' failed (attempt %d/%d): %s — retrying in %ds",
                    stage_name, attempt + 1, max_retries + 1, exc, backoff_seconds,
                )
                time.sleep(backoff_seconds)
            else:
                logger.error(
                    "Stage '%s' failed after %d attempts: %s",
                    stage_name, max_retries + 1, exc,
                )
    raise StageFailedError(stage_name, last_exc)  # type: ignore[arg-type]


def _build_feedback_from_eval(eval_round: Optional[dict]) -> str:
    """Reconstruct a combined feedback string from a stored evaluation round."""
    if not eval_round:
        return ""
    parts: list[str] = []
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
                parts.append(f"\n{dtype}: {'PASS' if item.get('pass') else 'FAIL'}")
                parts.append(f"  {item.get('feedback', 'No feedback')}")
            else:
                parts.append(f"\n{dtype}: UNKNOWN — {item}")
    adv_raw = eval_round.get("adversarial_feedback", "")
    if adv_raw:
        try:
            adv_data = json.loads(adv_raw)
        except (json.JSONDecodeError, TypeError):
            adv_data = {}
        parts.append("\n=== ADVERSARIAL FEEDBACK ===")
        parts.append(f"\nOverall: {'PASS' if adv_data.get('pass') else 'FAIL'}")
        parts.append(f"  {adv_data.get('feedback', 'No feedback')}")
        issues = adv_data.get("issues", [])
        if issues:
            parts.append("\nIssues found:")
            for issue in issues:
                if isinstance(issue, dict):
                    parts.append(
                        f"  - [{issue.get('deliverable', '?')}] "
                        f"{issue.get('problem', '?')}: {issue.get('claim', '?')}"
                    )
    return "\n".join(parts)


def _run_generate_evaluate(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    interest: InterestConfig,
) -> None:
    """Generate deliverables for all pending themes, then evaluate each with a
    refinement loop. Respects deliverable toggles from the interest config."""
    db.update_pipeline_run(run_id, current_stage="generate")
    generator.run(run_id, db, config, llm_client, interest)

    db.update_pipeline_run(run_id, current_stage="evaluate")
    themes = db.get_themes_for_run(run_id)

    for theme in themes:
        if theme["status"] != "pending":
            continue
        theme_id: int = theme["id"]
        while True:
            result = evaluator.run(run_id, db, config, llm_client, theme_id, interest)
            if result == "approved":
                break
            latest_eval = db.get_latest_evaluation(theme_id)
            feedback = _build_feedback_from_eval(latest_eval)
            if feedback:
                generator.refine(run_id, db, config, llm_client, theme_id, feedback, interest)
            else:
                logger.warning(
                    "No feedback extracted from evaluation of theme %d — "
                    "attempting refine with generic prompt", theme_id,
                )
                generator.refine(
                    run_id, db, config, llm_client, theme_id,
                    "Please improve the deliverables based on general quality standards.",
                    interest,
                )


def run_interest_pipeline(
    interest_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
) -> None:
    """Execute a full pipeline run for a single interest.

    Parameters
    ----------
    interest_id:
        The ``interests.id`` to run the pipeline for.
    db:
        Open :class:`Database` instance.
    config:
        Parsed global pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    """
    interest = _interest_from_db(db, interest_id)

    if interest.is_paused:
        logger.info("Interest '%s' is paused — skipping run", interest.name)
        return

    berlin = ZoneInfo(config.pipeline.timezone)
    now_berlin = datetime.now(berlin)
    run_date = now_berlin.strftime("%Y-%m-%d")
    started_at = now_berlin.isoformat()

    run_id = db.create_pipeline_run(interest_id, run_date, started_at)

    logger.info(
        "Pipeline started for interest '%s' — run_id=%d date=%s timezone=%s",
        interest.name, run_id, run_date, config.pipeline.timezone,
    )

    max_retries = config.pipeline.max_retries
    backoff = config.pipeline.retry_backoff_seconds

    try:
        # --- Scrape ---
        retry_wrapper(
            "scrape", scraper.run, max_retries, backoff,
            run_id, db, config, interest,
        )

        # --- Analyze ---
        retry_wrapper(
            "analyze", analyzer.run, max_retries, backoff,
            run_id, db, config, llm_client,
        )

        # --- Generate + Evaluate (per theme) ---
        retry_wrapper(
            "generate_evaluate", _run_generate_evaluate, max_retries, backoff,
            run_id, db, config, llm_client, interest,
        )

        # --- Brief (skip if disabled) ---
        if interest.enable_brief:
            retry_wrapper(
                "brief", brief.run, max_retries, backoff,
                run_id, db, config, llm_client, interest,
            )
        else:
            logger.info("Brief disabled for interest '%s' — skipping", interest.name)

        # --- Email (skip if no deliverables enabled) ---
        if interest.any_deliverable_enabled:
            retry_wrapper(
                "email", emailer.run, max_retries, backoff,
                run_id, db, config, interest,
            )
        else:
            logger.info("No deliverables enabled for interest '%s' — skipping email", interest.name)

    except StageFailedError as exc:
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
            "Pipeline FAILED for interest '%s' at stage '%s' — sending failure alert",
            interest.name, exc.stage_name,
        )
        try:
            emailer.send_failure_alert(config, interest.name, exc.stage_name, error_msg, tb_str)
        except Exception as alert_exc:
            logger.error("Failed to send failure alert: %s", alert_exc)
        raise

    db.update_pipeline_run(
        run_id,
        status="completed",
        completed_at=datetime.now(berlin).isoformat(),
    )
    logger.info("Pipeline completed successfully for interest '%s' — run_id=%d", interest.name, run_id)
