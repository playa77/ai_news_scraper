"""Analyzer stage — LLM-based theme identification from articles.

Calls the strong LLM model to identify 1–5 units of meaning (themes) from
the current pipeline run's articles.  If a previous daily brief exists, the
LLM is instructed to differentiate novel vs. continuation themes.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from typing import Optional

from .config import Config
from .db import Database
from .llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


class AnalysisParseError(Exception):
    """Raised when the LLM response cannot be parsed into valid themes."""


def run(run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None:
    """Execute the analyze stage.

    Parameters
    ----------
    run_id:
        The ``pipeline_runs.id`` this analysis belongs to.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    """
    db.update_pipeline_run(run_id, current_stage="analyze")

    articles = db.get_articles_for_run(run_id)
    if not articles:
        logger.info("No articles found for run %d — skipping analysis", run_id)
        return

    max_themes = config.pipeline.max_themes
    article_count = len(articles)

    # Build the articles section for the prompt
    articles_section = _build_articles_section(articles)

    # Build the previous-brief section (if available)
    run_record = db.get_pipeline_run(run_id)
    run_date = run_record["run_date"] if run_record else ""
    previous_brief = db.get_previous_daily_brief(run_date)
    previous_brief_section = _build_previous_brief_section(previous_brief)

    # Load and render the prompt template
    prompt_template = (_PROMPTS_DIR / "analyze.txt").read_text(encoding="utf-8")
    parts = prompt_template.split("=== USER ===")
    if len(parts) != 2:
        raise AnalysisParseError("analyze.txt prompt template is malformed")

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    system_prompt = system_prompt.format(max_themes=max_themes)
    user_prompt = parts[1].strip().format(
        previous_brief_section=previous_brief_section,
        articles_section=articles_section,
        max_themes=max_themes,
        article_count=article_count,
    )

    # Call LLM and parse response
    try:
        raw_response = llm_client.complete(
            model_id=config.models.strong.id,
            temperature=config.models.strong.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise AnalysisParseError(f"LLM call failed: {exc}") from exc

    themes = _parse_themes_response(raw_response, len(articles), max_themes)

    # Store themes
    novelty_distribution: dict[str, int] = {}
    for idx, theme in enumerate(themes, start=1):
        source_ids = [articles[i]["id"] for i in theme["source_article_indices"]]
        novelty_type = theme["novelty_type"]

        db.insert_theme(
            pipeline_run_id=run_id,
            title=theme["title"],
            description=theme["description"],
            source_article_ids=source_ids,
            novelty_type=novelty_type,
            order_index=idx,
        )

        novelty_distribution[novelty_type] = novelty_distribution.get(novelty_type, 0) + 1

    logger.info(
        "Analysis complete — %d themes identified, novelty distribution: %s",
        len(themes),
        novelty_distribution,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_articles_section(articles: list[dict]) -> str:
    """Format the article list for the LLM prompt."""
    lines: list[str] = []
    for idx, art in enumerate(articles):
        lines.append(f"[{idx}] Title: {art['title']}")
        content = art.get("full_content") or art.get("rss_excerpt", "")
        # Truncate very long articles to avoid excessive token usage
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"
        lines.append(f"    Content: {content}")
        lines.append("")
    return "\n".join(lines)


def _build_previous_brief_section(previous_brief: Optional[dict]) -> str:
    """Build the section of the prompt that includes the previous daily brief."""
    if previous_brief:
        return (
            "PREVIOUS DAILY BRIEF (for novelty comparison):\n\n"
            f"{previous_brief['content']}\n\n"
            "When identifying themes above, classify each as:\n"
            '- "novel" — a new topic not covered in the previous brief.\n'
            '- "continuation" — a meaningful update to a topic already mentioned '
            "in the previous brief."
        )
    return (
        "No previous daily brief is available (this is the first run, or no prior "
        "completed runs exist). Classify all themes as \"novel\".\n"
    )


def _parse_themes_response(raw: str, article_count: int, max_themes: int = 10) -> list[dict]:
    """Parse the LLM response into validated theme objects.

    Raises
    ------
    AnalysisParseError
        If the response cannot be parsed or fails validation.
    """
    # Strip markdown code fences if present
    json_text = raw.strip()
    json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
    json_text = re.sub(r"\s*```$", "", json_text)

    try:
        themes = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse analyzer response as JSON. Raw: %s", raw[:500])
        raise AnalysisParseError(f"Invalid JSON from analyzer: {exc}") from exc

    if not isinstance(themes, list):
        raise AnalysisParseError(f"Expected a JSON array, got {type(themes).__name__}")

    if len(themes) < 1 or len(themes) > max_themes:
        raise AnalysisParseError(
            f"Expected 1–{max_themes} themes, got {len(themes)}"
        )

    validated: list[dict] = []
    for theme in themes:
        if not isinstance(theme, dict):
            raise AnalysisParseError(f"Theme is not a dict: {theme!r}")

        required_fields = {"title", "description", "novelty_type", "source_article_indices"}
        missing = required_fields - set(theme.keys())
        if missing:
            raise AnalysisParseError(f"Theme missing required fields: {missing}")

        if theme["novelty_type"] not in ("novel", "continuation"):
            raise AnalysisParseError(
                f"Invalid novelty_type {theme['novelty_type']!r} — must be novel or continuation"
            )

        indices = theme["source_article_indices"]
        if not isinstance(indices, list) or not indices:
            raise AnalysisParseError("source_article_indices must be a non-empty list")
        for i in indices:
            if not isinstance(i, int) or i < 0 or i >= article_count:
                raise AnalysisParseError(f"Invalid article index: {i}")

        validated.append(theme)

    return validated
