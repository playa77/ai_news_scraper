"""Generator stage — LLM-based content generation for themes.

Produces three deliverables per theme:
1. ``summary_en`` — ~375 word English summary
2. ``script_en`` — ~500–750 word English YouTube script
3. ``script_de`` — ~500–750 word German YouTube script (natively written)

The German script is generated from source articles + English summary ONLY,
**not** from the English script, enforcing native generation.

LLM calls are parallelized:
- Themes are processed concurrently (ThreadPoolExecutor, 3 workers).
- Within each theme, ``script_en`` and ``script_de`` run in parallel after
  ``summary_en`` completes.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .config import Config
from .db import Database
from .llm import LLMClient
from .models import InterestConfig

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"

# Protect SQLite writes (WAL allows concurrent reads, but only one writer).
_db_write_lock = threading.Lock()


class GeneratorError(Exception):
    """Raised when content generation fails."""


def run(run_id: int, db: Database, config: Config, llm_client: LLMClient, interest: InterestConfig) -> None:
    """Generate deliverables for all pending themes in this pipeline run.

    Called once by the orchestrator.  Processes pending themes concurrently
    using a thread pool to reduce total runtime.

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
    interest:
        Per-interest configuration controlling deliverable toggles and word counts.
    """
    themes = db.get_themes_for_run(run_id)
    pending = [t for t in themes if t["status"] == "pending"]

    if not pending:
        return

    def _process(theme: dict) -> None:
        source_article_ids = _parse_article_ids(theme["source_article_ids"])
        articles = _get_articles(db, source_article_ids)
        _generate_theme_deliverables(
            run_id=run_id,
            db=db,
            config=config,
            llm_client=llm_client,
            interest=interest,
            theme=theme,
            articles=articles,
            version=1,
        )

    max_workers = min(3, len(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, t): t for t in pending}
        for future in as_completed(futures):
            theme = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(
                    "Theme %d (%s) generation failed: %s",
                    theme["id"],
                    theme.get("title", "?"),
                    exc,
                )
                raise


def refine(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    theme_id: int,
    evaluation_feedback: str,
    interest: InterestConfig,
) -> None:
    """Refine deliverables for a theme based on evaluation feedback.

    Creates new versions (incremented from previous) for enabled
    deliverable types — ``script_en`` and ``script_de`` are refined
    in parallel.

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
    interest:
        Per-interest configuration controlling deliverable toggles and word counts.
    """
    latest = db.get_latest_deliverables(theme_id)
    if not latest:
        logger.warning("No existing deliverables for theme %d — cannot refine", theme_id)
        return

    current_version = max(v["version"] for v in latest.values())

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

    # Refine summary_en first (scripts depend on it for context)
    summary_content = None
    if interest.enable_summary and "summary_en" in latest:
        summary_content = _refine_one(
            llm_client, config, refine_system, refine_user_template,
            theme, articles_text, current_version, evaluation_feedback,
            "summary_en", latest["summary_en"]["content"],
            target_words=interest.target_summary_words,
        )
        new_version = current_version + 1
        with _db_write_lock:
            db.insert_deliverable(theme_id, "summary_en", summary_content, new_version)
        logger.info(
            "Refined summary_en for theme %d — version %d (%d words)",
            theme_id, new_version, _word_count(summary_content),
        )

    # Refine script_en and script_de in parallel
    def _refine_script(dtype: str, target_words: int) -> None:
        if dtype not in latest:
            logger.warning("No %s deliverable for theme %d — skipping refine", dtype, theme_id)
            return
        old = latest[dtype]["content"]
        new = _refine_one(
            llm_client, config, refine_system, refine_user_template,
            theme, articles_text, current_version, evaluation_feedback,
            dtype, old,
            target_words=target_words,
        )
        new_version = current_version + 1
        with _db_write_lock:
            db.insert_deliverable(theme_id, dtype, new, new_version)
        logger.info(
            "Refined %s for theme %d — version %d (%d words)",
            dtype, theme_id, new_version, _word_count(new),
        )

    script_types = []
    if interest.enable_script_en and "script_en" in latest:
        script_types.append(("script_en", interest.target_script_en_words))
    if interest.enable_script_de and "script_de" in latest:
        script_types.append(("script_de", interest.target_script_de_words))

    if len(script_types) == 2:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_refine_script, dtype, tw) for dtype, tw in script_types]
            for future in futures:
                future.result()
    elif len(script_types) == 1:
        _refine_script(script_types[0][0], script_types[0][1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_theme_deliverables(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    interest: InterestConfig,
    theme: dict,
    articles: list[dict],
    version: int,
) -> None:
    """Generate deliverables for a single theme, respecting interest toggles.

    ``summary_en`` is generated first (if needed); ``script_en`` and
    ``script_de`` are then generated in parallel (if enabled).

    If at least one script is enabled but ``summary_en`` is disabled,
    ``summary_en`` is still generated internally (scripts depend on it
    for prompt context). Only when all deliverables are disabled is
    ``summary_en`` truly skipped.
    """
    theme_id = theme["id"]
    theme_title = theme["title"]
    theme_description = theme["description"]
    articles_text = _build_articles_text(articles)

    # Determine which deliverables to generate
    generate_summary = interest.enable_summary
    generate_script_en = interest.enable_script_en
    generate_script_de = interest.enable_script_de

    # If at least one script is enabled, summary is always needed as context
    if (generate_script_en or generate_script_de) and not generate_summary:
        generate_summary = True

    # ---- summary_en (must complete before scripts) ----
    summary_en = None
    if generate_summary:
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
            target_words=interest.target_summary_words,
        )
        with _db_write_lock:
            db.insert_deliverable(theme_id, "summary_en", summary_en, version)
        logger.info(
            "Generated summary_en for theme %d — version %d (%d words)",
            theme_id, version, _word_count(summary_en),
        )
    else:
        logger.info(
            "Skipping summary_en for theme %d — disabled by interest config",
            theme_id,
        )

    # ---- script_en + script_de in parallel ----
    def _gen_script(dtype: str, prompt_file: str, target_words: int) -> str:
        return _generate_one(
            llm_client=llm_client,
            config=config,
            prompt_file=prompt_file,
            fmt_kwargs={
                "theme_title": theme_title,
                "theme_description": theme_description,
                "summary_en": summary_en or "",
                "articles_text": articles_text,
            },
            deliverable_type=dtype,
            theme_id=theme_id,
            target_words=target_words,
        )

    script_tasks = []
    if generate_script_en:
        script_tasks.append(("script_en", "script_en.txt", interest.target_script_en_words))
    if generate_script_de:
        script_tasks.append(("script_de", "script_de.txt", interest.target_script_de_words))

    if script_tasks:
        with ThreadPoolExecutor(max_workers=min(2, len(script_tasks))) as executor:
            futures = {}
            for dtype, prompt_file, tw in script_tasks:
                future = executor.submit(_gen_script, dtype, prompt_file, tw)
                futures[future] = (dtype, prompt_file)
            for future in as_completed(futures):
                dtype, _ = futures[future]
                content = future.result()
                with _db_write_lock:
                    db.insert_deliverable(theme_id, dtype, content, version)
                logger.info(
                    "Generated %s for theme %d — version %d (%d words)",
                    dtype, theme_id, version, _word_count(content),
                )
    else:
        logger.info(
            "Skipping script deliverables for theme %d — disabled by interest config",
            theme_id,
        )


def _refine_one(
    llm_client: LLMClient,
    config: Config,
    refine_system: str,
    refine_user_template: str,
    theme: dict,
    articles_text: str,
    current_version: int,
    evaluation_feedback: str,
    deliverable_type: str,
    old_content: str,
    target_words: int | None = None,
) -> str:
    user_prompt = refine_user_template.format(
        deliverable_type=deliverable_type,
        current_content=old_content,
        evaluation_feedback=evaluation_feedback,
        articles_text=articles_text,
        target_words=target_words,
    )
    try:
        return llm_client.complete(
            model_id=config.models.strong.id,
            temperature=config.models.strong.temperature,
            system_prompt=refine_system,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        raise GeneratorError(
            f"Refinement LLM call failed for {deliverable_type}: {exc}"
        ) from exc


def _generate_one(
    llm_client: LLMClient,
    config: Config,
    prompt_file: str,
    fmt_kwargs: dict,
    deliverable_type: str,
    theme_id: int,
    target_words: int | None = None,
) -> str:
    """Load a prompt template, render it, call the LLM, and return the result."""
    template = (_PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    parts = template.split("=== USER ===")
    if len(parts) != 2:
        raise GeneratorError(f"{prompt_file} prompt template is malformed")

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    fmt_kwargs["target_words"] = target_words
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
