"""Brief generator stage — synthesize a daily brief from approved themes.

Calls the strong LLM model to produce a ~350-word English daily brief
summarizing all approved themes from the current pipeline run.

The brief is stored in the ``daily_briefs`` table and retrieved by
tomorrow's analyzer stage for novelty comparison.
"""

from __future__ import annotations

import logging
import pathlib

from .config import Config
from .db import Database
from .llm import LLMClient
from .models import InterestConfig

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


class BriefError(Exception):
    """Raised when daily brief generation fails."""


def run(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    interest: InterestConfig,
) -> None:
    """Execute the brief generation stage.

    Parameters
    ----------
    run_id:
        The ``pipeline_runs.id`` this brief belongs to.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    interest:
        The ``InterestConfig`` for this pipeline run.
    """
    db.update_pipeline_run(run_id, current_stage="brief")

    # Get approved and auto-approved themes
    themes = db.get_themes_for_run(run_id)
    approved_themes = [
        t for t in themes if t["status"] in ("approved", "auto_approved")
    ]

    if not approved_themes:
        logger.info("No approved themes for run %d — generating empty brief", run_id)
        brief_content = (
            f"No new {interest.name} themes were identified for this run ({run_id}). "
            "Check back tomorrow for updates."
        )
        word_count = _word_count(brief_content)
        db.insert_daily_brief(run_id, brief_content, word_count)
        logger.info("Daily brief generated — %d words (no themes)", word_count)
        return

    # Build the themes section for the prompt
    themes_section = _build_themes_section(approved_themes, db, interest)

    # Load and render the prompt template
    prompt_template = (_PROMPTS_DIR / "brief.txt").read_text(encoding="utf-8")
    parts = prompt_template.split("=== USER ===")
    if len(parts) != 2:
        raise BriefError("brief.txt prompt template is malformed")

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    user_prompt = parts[1].strip().format(
        themes_section=themes_section, target_words=interest.target_brief_words
    )

    try:
        brief_content = llm_client.complete(
            model_id=config.models.strong.id,
            temperature=config.models.strong.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise BriefError(f"LLM call for daily brief failed: {exc}") from exc

    word_count = _word_count(brief_content)
    db.insert_daily_brief(run_id, brief_content, word_count)

    logger.info(
        "Daily brief generated — %d words, %d themes included",
        word_count,
        len(approved_themes),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_themes_section(
    themes: list[dict], db: Database, interest: InterestConfig
) -> str:
    """Build the themes section of the brief prompt from approved themes."""
    sections: list[str] = []
    for theme in themes:
        sections.append(f"THEME: {theme['title']}")
        sections.append(f"Type: {theme['novelty_type']}")
        if interest.enable_summary:
            # Get the latest summary_en for this theme
            deliverables = db.get_latest_deliverables(theme["id"])
            summary_text = ""
            if "summary_en" in deliverables:
                summary_text = deliverables["summary_en"]["content"]
            sections.append(f"Summary: {summary_text}")
        sections.append("")

    return "\n".join(sections)


def _word_count(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())
