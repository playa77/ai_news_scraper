"""Generator stage — LLM-based content generation for themes.

Produces three deliverables per theme:
1. ``summary_en`` — ~750 word English summary
2. ``script_en`` — ~1000–1500 word English YouTube script
3. ``script_de`` — ~1000–1500 word German YouTube script (natively written)

The German script is generated from source articles + English summary ONLY,
**not** from the English script, enforcing native generation.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from .config import Config
from .db import Database
from .llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


class GeneratorError(Exception):
    """Raised when content generation fails."""


def run(run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None:
    """Generate deliverables for all pending themes in this pipeline run.

    Called once by the orchestrator. Iterates over all themes for the run
    that have ``status = 'pending'``.

    Parameters
    ----------
    run_id:
        The ``pipeline_runs.id`` this generation belongs to.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    """
    themes = db.get_themes_for_run(run_id)

    for theme in themes:
        if theme["status"] != "pending":
            continue

        # Resolve source articles
        source_article_ids = _parse_article_ids(theme["source_article_ids"])
        articles = _get_articles(db, source_article_ids)

        _generate_theme_deliverables(
            run_id=run_id,
            db=db,
            config=config,
            llm_client=llm_client,
            theme=theme,
            articles=articles,
            version=1,
        )


def refine(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    theme_id: int,
    evaluation_feedback: str,
) -> None:
    """Refine deliverables for a theme based on evaluation feedback.

    Creates new versions (incremented from previous) for all three
    deliverable types.

    Parameters
    ----------
    run_id:
        The pipeline run ID.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    theme_id:
        The theme whose deliverables need refinement.
    evaluation_feedback:
        Concatenated feedback from quality and adversarial evaluators.
    """
    latest = db.get_latest_deliverables(theme_id)
    if not latest:
        logger.warning("No existing deliverables for theme %d — cannot refine", theme_id)
        return

    # Get the current version from any deliverable
    current_version = max(v["version"] for v in latest.values())

    # Get theme and articles
    themes = db.get_themes_for_run(run_id)
    theme = next((t for t in themes if t["id"] == theme_id), None)
    if not theme:
        raise GeneratorError(f"Theme {theme_id} not found")

    source_article_ids = _parse_article_ids(theme["source_article_ids"])
    articles = _get_articles(db, source_article_ids)

    articles_text = _build_articles_text(articles)
    refine_template = (_PROMPTS_DIR / "refine.txt").read_text(encoding="utf-8")
    parts = refine_template.split("=== USER ===")
    if len(parts) != 2:
        raise GeneratorError("refine.txt prompt template is malformed")
    refine_system = parts[0].replace("=== SYSTEM ===\n", "").strip()
    refine_user_template = parts[1].strip()

    for dtype in ("summary_en", "script_en", "script_de"):
        if dtype not in latest:
            logger.warning("No %s deliverable for theme %d — skipping refine", dtype, theme_id)
            continue

        old_content = latest[dtype]["content"]
        user_prompt = refine_user_template.format(
            deliverable_type=dtype,
            current_content=old_content,
            evaluation_feedback=evaluation_feedback,
            articles_text=articles_text,
        )

        try:
            new_content = llm_client.complete(
                model_id=config.models.strong.id,
                temperature=config.models.strong.temperature,
                system_prompt=refine_system,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            raise GeneratorError(f"Refinement LLM call failed for {dtype}: {exc}") from exc

        new_version = current_version + 1
        db.insert_deliverable(theme_id, dtype, new_content, new_version)
        logger.info(
            "Refined %s for theme %d — version %d (%d words)",
            dtype,
            theme_id,
            new_version,
            _word_count(new_content),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_theme_deliverables(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    theme: dict,
    articles: list[dict],
    version: int,
) -> None:
    """Generate all three deliverables for a single theme."""
    theme_id = theme["id"]
    theme_title = theme["title"]
    theme_description = theme["description"]
    articles_text = _build_articles_text(articles)

    # ---- summary_en ----
    summary_en = _generate_one(
        llm_client=llm_client,
        config=config,
        prompt_file="summary_en.txt",
        fmt_kwargs={
            "theme_title": theme_title,
            "theme_description": theme_description,
            "articles_text": articles_text,
        },
        deliverable_type="summary_en",
        theme_id=theme_id,
    )
    db.insert_deliverable(theme_id, "summary_en", summary_en, version)
    logger.info(
        "Generated summary_en for theme %d — version %d (%d words)",
        theme_id, version, _word_count(summary_en),
    )

    # ---- script_en ----
    script_en = _generate_one(
        llm_client=llm_client,
        config=config,
        prompt_file="script_en.txt",
        fmt_kwargs={
            "theme_title": theme_title,
            "theme_description": theme_description,
            "summary_en": summary_en,
            "articles_text": articles_text,
        },
        deliverable_type="script_en",
        theme_id=theme_id,
    )
    db.insert_deliverable(theme_id, "script_en", script_en, version)
    logger.info(
        "Generated script_en for theme %d — version %d (%d words)",
        theme_id, version, _word_count(script_en),
    )

    # ---- script_de (German — NO English script input) ----
    script_de = _generate_one(
        llm_client=llm_client,
        config=config,
        prompt_file="script_de.txt",
        fmt_kwargs={
            "theme_title": theme_title,
            "theme_description": theme_description,
            "summary_en": summary_en,
            "articles_text": articles_text,
        },
        deliverable_type="script_de",
        theme_id=theme_id,
    )
    db.insert_deliverable(theme_id, "script_de", script_de, version)
    logger.info(
        "Generated script_de for theme %d — version %d (%d words)",
        theme_id, version, _word_count(script_de),
    )


def _generate_one(
    llm_client: LLMClient,
    config: Config,
    prompt_file: str,
    fmt_kwargs: dict,
    deliverable_type: str,
    theme_id: int,
) -> str:
    """Load a prompt template, render it, call the LLM, and return the result."""
    template = (_PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    parts = template.split("=== USER ===")
    if len(parts) != 2:
        raise GeneratorError(f"{prompt_file} prompt template is malformed")

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    user_prompt = parts[1].strip().format(**fmt_kwargs)

    try:
        return llm_client.complete(
            model_id=config.models.strong.id,
            temperature=config.models.strong.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise GeneratorError(
            f"LLM call failed for {deliverable_type} (theme {theme_id}): {exc}"
        ) from exc


def _build_articles_text(articles: list[dict]) -> str:
    """Format article contents for a prompt."""
    lines: list[str] = []
    for idx, art in enumerate(articles):
        lines.append(f"--- Article {idx + 1} ---")
        lines.append(f"Title: {art['title']}")
        lines.append("")
        content = art.get("full_content") or art.get("rss_excerpt", "")
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def _parse_article_ids(source_article_ids: str) -> list[int]:
    """Parse the JSON-encoded ``source_article_ids`` field from the themes table."""
    import json

    return json.loads(source_article_ids)


def _get_articles(db: Database, article_ids: list[int]) -> list[dict]:
    """Fetch articles by ID from the database."""
    articles: list[dict] = []
    for aid in article_ids:
        article = db.get_article_by_id(aid)
        if article:
            articles.append(article)
    return articles


def _word_count(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())
