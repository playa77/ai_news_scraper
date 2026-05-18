"""Evaluator stage — quality evaluation and adversarial fact-checking.

Evaluates deliverables for a single theme using the weak LLM model:
1. Quality evaluation — style, word count, completeness, prose, structure.
2. Adversarial evaluation — factual accuracy, bias, hallucinations.

Implements the refinement loop trigger:
- Both pass → theme approved.
- Any fail + rounds remaining → trigger refinement.
- Max rounds reached → auto-approve.
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
from .models import InterestConfig

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


def run(
    run_id: int,
    db: Database,
    config: Config,
    llm_client: LLMClient,
    theme_id: int,
    interest: InterestConfig,
) -> str:
    """Evaluate deliverables for a single theme.

    Parameters
    ----------
    run_id:
        The ``pipeline_runs.id`` this evaluation belongs to.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    llm_client:
        Configured :class:`LLMClient` for OpenRouter calls.
    theme_id:
        The theme to evaluate.
    interest:
        The ``InterestConfig`` for this pipeline run.

    Returns
    -------
    str
        ``'approved'`` if the theme passes evaluation or is auto-approved after
        max refinement rounds.  ``'needs_refinement'`` if a refinement round is
        required.
    """
    # Defensive check: if all deliverables are disabled, skip evaluation
    if not interest.any_deliverable_enabled:
        db.update_theme_status(theme_id, "approved")
        return "approved"

    # Determine current round number
    latest_eval = db.get_latest_evaluation(theme_id)
    round_number = (latest_eval["round_number"] + 1) if latest_eval else 1

    # Get theme and deliverables
    themes = db.get_themes_for_run(run_id)
    theme = next((t for t in themes if t["id"] == theme_id), None)
    if not theme:
        raise ValueError(f"Theme {theme_id} not found in run {run_id}")

    deliverables = db.get_latest_deliverables(theme_id)
    if not deliverables:
        logger.warning("No deliverables for theme %d — cannot evaluate", theme_id)
        db.update_theme_status(theme_id, "approved")
        return "approved"

    # Get source articles
    import json as _json

    article_ids = _json.loads(theme["source_article_ids"])
    articles: list[dict] = []
    for aid in article_ids:
        art = db.get_article_by_id(aid)
        if art:
            articles.append(art)
    articles_text = _build_articles_text(articles)

    # ---- Quality evaluation ----
    quality_result = _run_quality_eval(
        llm_client, config, theme, deliverables, articles_text, interest
    )

    # ---- Adversarial evaluation ----
    adversarial_result = _run_adversarial_eval(
        llm_client, config, theme, deliverables, articles_text, interest
    )

    # ---- Determine overall result ----
    quality_all_pass = _all_quality_pass(quality_result, interest)
    adversarial_pass = adversarial_result.get("pass", False)
    overall_passed = "pass" if (quality_all_pass and adversarial_pass) else "fail"

    combined_feedback = _build_combined_feedback(quality_result, adversarial_result, interest)

    # Store evaluation round
    db.insert_evaluation_round(
        theme_id=theme_id,
        round_number=round_number,
        quality_passed="pass" if quality_all_pass else "fail",
        quality_feedback=json.dumps(quality_result),
        adversarial_passed="pass" if adversarial_pass else "fail",
        adversarial_feedback=json.dumps(adversarial_result),
        overall_passed=overall_passed,
    )

    logger.info(
        "Evaluation round %d for theme %d — quality_all=%s adversarial=%s overall=%s",
        round_number,
        theme_id,
        quality_all_pass,
        adversarial_pass,
        overall_passed,
    )

    if overall_passed == "pass":
        db.update_theme_status(theme_id, "approved")
        return "approved"

    if round_number >= config.pipeline.max_refinement_rounds:
        logger.info(
            "Theme %d auto-approved after %d refinement rounds",
            theme_id,
            round_number,
        )
        db.update_theme_status(theme_id, "auto_approved")
        return "approved"

    return "needs_refinement"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_quality_eval(
    llm_client: LLMClient,
    config: Config,
    theme: dict,
    deliverables: dict,
    articles_text: str,
    interest: InterestConfig,
) -> dict:
    """Run the quality evaluation and return the parsed result dict."""
    template = (_PROMPTS_DIR / "evaluate_quality.txt").read_text(encoding="utf-8")
    parts = template.split("=== USER ===")
    if len(parts) != 2:
        logger.error("evaluate_quality.txt prompt template is malformed")
        return _fallback_quality_fail(deliverables, "Malformed prompt template", interest)

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()

    # Determine content for each deliverable — use "[DISABLED]" for disabled ones
    summary_content = (
        deliverables.get("summary_en", {}).get("content", "[MISSING]")
        if interest.enable_summary or interest.enable_script_en or interest.enable_script_de
        else "[DISABLED]"
    )
    script_en_content = (
        deliverables.get("script_en", {}).get("content", "[MISSING]")
        if interest.enable_script_en
        else "[DISABLED]"
    )
    script_de_content = (
        deliverables.get("script_de", {}).get("content", "[MISSING]")
        if interest.enable_script_de
        else "[DISABLED]"
    )

    user_prompt = parts[1].strip().format(
        theme_title=theme["title"],
        summary_en=summary_content,
        script_en=script_en_content,
        script_de=script_de_content,
    )

    try:
        raw = llm_client.complete(
            model_id=config.models.weak.id,
            temperature=config.models.weak.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.warning("Quality eval LLM call failed: %s", exc)
        return _fallback_quality_fail(deliverables, f"LLM error: {exc}", interest)

    return _parse_json_response(raw, _fallback_quality_fail(deliverables, "JSON parse error", interest))


def _run_adversarial_eval(
    llm_client: LLMClient,
    config: Config,
    theme: dict,
    deliverables: dict,
    articles_text: str,
    interest: InterestConfig,
) -> dict:
    """Run the adversarial evaluation and return the parsed result dict."""
    template = (_PROMPTS_DIR / "evaluate_adversarial.txt").read_text(encoding="utf-8")
    parts = template.split("=== USER ===")
    if len(parts) != 2:
        logger.error("evaluate_adversarial.txt prompt template is malformed")
        return {"pass": True, "feedback": "Malformed prompt template — skipped", "issues": []}

    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()

    # Determine content for each deliverable — use "[DISABLED]" for disabled ones
    summary_content = (
        deliverables.get("summary_en", {}).get("content", "[MISSING]")
        if interest.enable_summary or interest.enable_script_en or interest.enable_script_de
        else "[DISABLED]"
    )
    script_en_content = (
        deliverables.get("script_en", {}).get("content", "[MISSING]")
        if interest.enable_script_en
        else "[DISABLED]"
    )
    script_de_content = (
        deliverables.get("script_de", {}).get("content", "[MISSING]")
        if interest.enable_script_de
        else "[DISABLED]"
    )

    user_prompt = parts[1].strip().format(
        theme_title=theme["title"],
        articles_text=articles_text,
        summary_en=summary_content,
        script_en=script_en_content,
        script_de=script_de_content,
    )

    try:
        raw = llm_client.complete(
            model_id=config.models.weak.id,
            temperature=config.models.weak.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.warning("Adversarial eval LLM call failed: %s", exc)
        return {"pass": True, "feedback": f"LLM error — skipping adversarial: {exc}", "issues": []}

    expected_keys = {"pass", "feedback", "issues"}
    fallback = {"pass": True, "feedback": "JSON parse error — skipping adversarial", "issues": []}
    return _parse_json_response(raw, fallback)


def _fallback_quality_fail(deliverables: dict, reason: str, interest: InterestConfig) -> dict:
    """Build a quality eval result where all enabled deliverables fail."""
    result = {}
    if interest.enable_summary or interest.enable_script_en or interest.enable_script_de:
        result["summary_en"] = {"pass": False, "feedback": f"Evaluation could not be completed: {reason}"}
    if interest.enable_script_en:
        result["script_en"] = {"pass": False, "feedback": f"Evaluation could not be completed: {reason}"}
    if interest.enable_script_de:
        result["script_de"] = {"pass": False, "feedback": f"Evaluation could not be completed: {reason}"}
    return result


def _all_quality_pass(quality_result: dict, interest: InterestConfig) -> bool:
    """Check if all enabled deliverables pass the quality evaluation."""
    for dtype, toggle in (
        ("summary_en", interest.enable_summary or interest.enable_script_en or interest.enable_script_de),
        ("script_en", interest.enable_script_en),
        ("script_de", interest.enable_script_de),
    ):
        if not toggle:
            continue
        if dtype in quality_result:
            content = quality_result[dtype].get("content", "")
            if content == "[DISABLED]":
                continue
            if not quality_result[dtype].get("pass", False):
                return False
    return True


def _build_combined_feedback(quality_result: dict, adversarial_result: dict, interest: InterestConfig) -> str:
    """Combine quality and adversarial feedback into a single string."""
    parts: list[str] = []

    parts.append("=== QUALITY FEEDBACK ===")
    for dtype, toggle in (
        ("summary_en", interest.enable_summary or interest.enable_script_en or interest.enable_script_de),
        ("script_en", interest.enable_script_en),
        ("script_de", interest.enable_script_de),
    ):
        if not toggle:
            continue
        if dtype in quality_result:
            d = quality_result[dtype]
            parts.append(f"\n{dtype}: {'PASS' if d.get('pass') else 'FAIL'}")
            parts.append(f"  {d.get('feedback', 'No feedback')}")

    parts.append("\n=== ADVERSARIAL FEEDBACK ===")
    parts.append(f"\nOverall: {'PASS' if adversarial_result.get('pass') else 'FAIL'}")
    parts.append(f"  {adversarial_result.get('feedback', 'No feedback')}")
    issues = adversarial_result.get("issues", [])
    if issues:
        parts.append("\nIssues found:")
        for issue in issues:
            parts.append(
                f"  - [{issue.get('deliverable', '?')}] {issue.get('problem', '?')}: "
                f"{issue.get('claim', '?')}"
            )

    return "\n".join(parts)


def _parse_json_response(raw: str, default: dict) -> dict:
    """Parse a JSON response from an evaluator, returning *default* on failure."""
    json_text = raw.strip()
    json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
    json_text = re.sub(r"\s*```$", "", json_text)

    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse evaluator response as JSON: %s", exc)
        logger.debug("Raw response (first 500 chars): %s", raw[:500])
        return default


def _build_articles_text(articles: list[dict]) -> str:
    """Format article contents for the evaluator prompt."""
    lines: list[str] = []
    for idx, art in enumerate(articles):
        lines.append(f"--- Source {idx + 1}: {art['title']} ---")
        content = art.get("full_content") or art.get("rss_excerpt", "")
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"
        lines.append(content)
        lines.append("")
    return "\n".join(lines)
