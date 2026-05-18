"""Tests for the evaluator module — quality evaluation, adversarial fact-checking,
and refinement loop control.

Tests cover JSON parsing helpers, fallback logic, quality pass checks,
combined feedback building, the full ``run()`` integration (with in-memory DB
and mocked LLM client), and the internal eval functions.
"""

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from src.db import Database
from src.models import InterestConfig
from src.evaluator import (
    _all_quality_pass,
    _build_combined_feedback,
    _build_articles_text,
    _fallback_quality_fail,
    _parse_json_response,
    _run_adversarial_eval,
    _run_quality_eval,
    run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def db():
    """Create an in-memory database with schema initialized."""
    database = Database(":memory:")
    database.initialize_schema()
    yield database
    database.close()


@pytest.fixture
def sample_articles():
    """Load the sample articles fixture as a list of dicts."""
    with open(_FIXTURES_DIR / "sample_articles.json") as f:
        return json.load(f)


@pytest.fixture
def mock_llm():
    """Return a MagicMock that acts as a minimal LLMClient."""
    return MagicMock()


@pytest.fixture
def mock_config():
    """Return a mock config with weak model and pipeline settings."""
    config = MagicMock()
    config.models.weak.id = "deepseek/deepseek-chat"
    config.models.weak.temperature = 0.3
    config.pipeline.max_refinement_rounds = 3
    return config


@pytest.fixture
def seeded_evaluator_db(db, sample_articles):
    """``db`` with a pipeline run, feed, articles, theme, and deliverables pre-inserted.

    Returns a dict with ``db``, ``run_id``, ``theme_id``, and ``article_ids``
    so tests can reference the stored IDs.
    """
    run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")
    feed_id = db.upsert_feed(db.get_interest_by_name("AI")["id"], "https://example.com/rss", "Example Feed", "news")

    article_ids = []
    for art in sample_articles:
        aid = db.insert_article(
            feed_id=feed_id,
            url=art["url"],
            title=art["title"],
            author=art.get("author"),
            published_at=art["published_at"],
            scraped_at=art["scraped_at"],
            rss_excerpt=art.get("rss_excerpt"),
            full_content=art.get("full_content"),
            content_status=art.get("content_status", "full"),
            pipeline_run_id=run_id,
        )
        article_ids.append(aid)

    # Insert a theme referencing the first two articles
    theme_id = db.insert_theme(
        pipeline_run_id=run_id,
        title="Test Theme: AI Breakthroughs",
        description="A test theme about recent AI breakthroughs.",
        source_article_ids=[article_ids[0], article_ids[1]],
        novelty_type="novel",
        order_index=1,
    )

    # Insert deliverables for the theme
    db.insert_deliverable(theme_id, "summary_en", "English summary content here.", 1)
    db.insert_deliverable(theme_id, "script_en", "English script content here.", 1)
    db.insert_deliverable(theme_id, "script_de", "German script content here.", 1)

    return {
        "db": db,
        "run_id": run_id,
        "feed_id": feed_id,
        "theme_id": theme_id,
        "article_ids": article_ids,
    }


# ===================================================================
# 1. _parse_json_response
# ===================================================================


class TestParseJsonResponse:
    """Parsing evaluator JSON responses."""

    @pytest.fixture
    def default(self):
        return {"summary_en": {"pass": False, "feedback": "DEFAULT"}}

    def test_valid_json_returns_parsed_dict(self, default):
        """Valid JSON string is parsed and returned."""
        raw = '{"summary_en": {"pass": true, "feedback": "Good"}}'
        result = _parse_json_response(raw, default)
        assert result["summary_en"]["pass"] is True
        assert result["summary_en"]["feedback"] == "Good"

    def test_json_with_markdown_fence_json_lang(self, default):
        """JSON wrapped in ```json ... ``` fences is stripped and parsed."""
        raw = '```json\n{"summary_en": {"pass": true, "feedback": "Good"}}\n```'
        result = _parse_json_response(raw, default)
        assert result["summary_en"]["pass"] is True

    def test_json_with_markdown_fence_no_lang(self, default):
        """JSON wrapped in ``` ... ``` (no lang hint) is stripped and parsed."""
        raw = '```\n{"summary_en": {"pass": true, "feedback": "Good"}}\n```'
        result = _parse_json_response(raw, default)
        assert result["summary_en"]["pass"] is True

    def test_markdown_fence_with_whitespace(self, default):
        """Leading/trailing whitespace around fences is handled."""
        raw = '  \n  ```json\n{"a": 1}\n  ```  \n'
        result = _parse_json_response(raw, default)
        assert result["a"] == 1

    def test_invalid_json_returns_default(self, default):
        """Malformed JSON returns the default value."""
        raw = '{"summary_en": {"pass": true, "feedback": "Broken'
        result = _parse_json_response(raw, default)
        assert result == default

    def test_non_json_string_returns_default(self, default):
        """Plain text that is not JSON returns the default value."""
        raw = "This is not JSON at all."
        result = _parse_json_response(raw, default)
        assert result == default

    def test_empty_string_returns_default(self, default):
        """Empty string returns the default value."""
        result = _parse_json_response("", default)
        assert result == default

    def test_none_raw_raises_attribute_error(self):
        """None as raw input raises AttributeError (not a str)."""
        with pytest.raises(AttributeError):
            _parse_json_response(None, {"default": True})

    def test_json_array_returns_parsed_list(self, default):
        """A JSON array (valid) is returned as-is."""
        raw = '[{"a": 1}, {"b": 2}]'
        result = _parse_json_response(raw, default)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_fence_only_no_content(self, default):
        """Fences with no JSON content return default."""
        raw = "```json\n\n```"
        result = _parse_json_response(raw, default)
        assert result == default


# ===================================================================
# 2. _fallback_quality_fail
# ===================================================================


class TestFallbackQualityFail:
    """Building a quality fail result when evaluation cannot proceed."""

    def test_returns_all_three_deliverable_types(self):
        """Result contains summary_en, script_en, and script_de."""
        deliverables = {
            "summary_en": {"content": "x", "version": 1},
            "script_en": {"content": "y", "version": 1},
            "script_de": {"content": "z", "version": 1},
        }
        result = _fallback_quality_fail(deliverables, "LLM error", InterestConfig(name="test"))
        assert "summary_en" in result
        assert "script_en" in result
        assert "script_de" in result

    def test_all_marked_fail(self):
        """Every deliverable has pass=False."""
        deliverables = {
            "summary_en": {"content": "x", "version": 1},
            "script_en": {"content": "y", "version": 1},
            "script_de": {"content": "z", "version": 1},
        }
        result = _fallback_quality_fail(deliverables, "Some reason", InterestConfig(name="test"))
        for dtype in ("summary_en", "script_en", "script_de"):
            assert result[dtype]["pass"] is False

    def test_reason_included_in_feedback(self):
        """The reason string is embedded in each deliverable's feedback."""
        deliverables = {
            "summary_en": {"content": "x", "version": 1},
            "script_en": {"content": "y", "version": 1},
            "script_de": {"content": "z", "version": 1},
        }
        reason = "LLM API unreachable"
        result = _fallback_quality_fail(deliverables, reason, InterestConfig(name="test"))
        for dtype in ("summary_en", "script_en", "script_de"):
            assert reason in result[dtype]["feedback"]

    def test_handles_missing_deliverable_keys_gracefully(self):
        """Missing deliverable types are omitted from the result."""
        result = _fallback_quality_fail({}, "No deliverables", InterestConfig(name="test"))
        assert result == {
            "summary_en": {"pass": False, "feedback": "Evaluation could not be completed: No deliverables"},
            "script_en": {"pass": False, "feedback": "Evaluation could not be completed: No deliverables"},
            "script_de": {"pass": False, "feedback": "Evaluation could not be completed: No deliverables"},
        }


# ===================================================================
# 3. _all_quality_pass
# ===================================================================


class TestAllQualityPass:
    """Checking whether all quality evaluator deliverables pass."""

    def test_all_pass_returns_true(self):
        """All three deliverable types set to pass returns True."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        assert _all_quality_pass(quality_result, InterestConfig(name="test")) is True

    def test_any_single_fail_returns_false(self):
        """One deliverable failing returns False."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": False, "feedback": "Too short"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        assert _all_quality_pass(quality_result, InterestConfig(name="test")) is False

        # Try each position
        for failing_dtype in ("summary_en", "script_en", "script_de"):
            qr = {d: {"pass": True, "feedback": ""} for d in ("summary_en", "script_en", "script_de")}
            qr[failing_dtype]["pass"] = False
            assert _all_quality_pass(qr, InterestConfig(name="test")) is False, f"Expected False when {failing_dtype} fails"

    def test_all_fail_returns_false(self):
        """All three deliverables failing returns False."""
        quality_result = {
            "summary_en": {"pass": False, "feedback": "Bad"},
            "script_en": {"pass": False, "feedback": "Bad"},
            "script_de": {"pass": False, "feedback": "Bad"},
        }
        assert _all_quality_pass(quality_result, InterestConfig(name="test")) is False

    def test_missing_deliverable_type_is_skipped(self):
        """A missing deliverable type is skipped (only present types are checked)."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            # script_de is missing
        }
        assert _all_quality_pass(quality_result, InterestConfig(name="test")) is True

    def test_empty_dict_returns_true(self):
        """An empty quality result returns True (nothing to fail)."""
        assert _all_quality_pass({}, InterestConfig(name="test")) is True

    def test_nested_pass_key_missing_returns_false(self):
        """A deliverable with no 'pass' key defaults to False."""
        quality_result = {
            "summary_en": {"feedback": "No pass key"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        assert _all_quality_pass(quality_result, InterestConfig(name="test")) is False


# ===================================================================
# 4. _build_combined_feedback
# ===================================================================


class TestBuildCombinedFeedback:
    """Combining quality and adversarial feedback into a single string."""

    def test_includes_quality_section(self):
        """The combined feedback contains the quality feedback header and per-deliverable results."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Well written"},
            "script_en": {"pass": False, "feedback": "Too long"},
            "script_de": {"pass": True, "feedback": "Gut geschrieben"},
        }
        adversarial_result = {"pass": True, "feedback": "No issues", "issues": []}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "=== QUALITY FEEDBACK ===" in result
        assert "summary_en: PASS" in result
        assert "script_en: FAIL" in result
        assert "script_de: PASS" in result
        assert "Well written" in result
        assert "Too long" in result
        assert "Gut geschrieben" in result

    def test_includes_adversarial_section(self):
        """The combined feedback contains the adversarial header, pass/fail, and feedback."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {
            "pass": False,
            "feedback": "Found factual errors",
            "issues": [],
        }
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "=== ADVERSARIAL FEEDBACK ===" in result
        assert "Overall: FAIL" in result
        assert "Found factual errors" in result

    def test_adversarial_pass_label(self):
        """A passing adversarial result shows 'PASS'."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "All accurate", "issues": []}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "Overall: PASS" in result

    def test_includes_issues_list(self):
        """Issues from the adversarial result are listed individually."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {
            "pass": False,
            "feedback": "Issues found",
            "issues": [
                {"deliverable": "summary_en", "problem": "hallucination", "claim": "GPT-5 has 1T params"},
                {"deliverable": "script_en", "problem": "bias", "claim": "AI will replace all jobs"},
            ],
        }
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "Issues found:" in result
        assert "[summary_en] hallucination: GPT-5 has 1T params" in result
        assert "[script_en] bias: AI will replace all jobs" in result

    def test_no_issues_skips_issues_section(self):
        """When issues list is empty, the 'Issues found:' section is not included."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Clean", "issues": []}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "Issues found:" not in result

    def test_missing_issues_key_skips_issues_section(self):
        """When issues key is absent, no issues section is shown."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Clean"}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "Issues found:" not in result

    def test_missing_deliverables_in_quality(self):
        """Missing deliverable types in quality result are skipped."""
        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            # script_en and script_de missing
        }
        adversarial_result = {"pass": True, "feedback": "Clean", "issues": []}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "summary_en: PASS" in result
        assert "script_en:" not in result
        assert "script_de:" not in result

    def test_fallback_feedback_when_missing(self):
        """Missing feedback text defaults to 'No feedback'."""
        quality_result = {
            "summary_en": {"pass": True},
            "script_en": {"pass": True},
            "script_de": {"pass": True},
        }
        adversarial_result = {"pass": True}
        result = _build_combined_feedback(quality_result, adversarial_result, InterestConfig(name="test"))

        assert "No feedback" in result


# ===================================================================
# 5. _build_articles_text
# ===================================================================


class TestBuildArticlesText:
    """Formatting article contents for evaluator prompts."""

    def test_formats_with_title_and_content(self, sample_articles):
        """Articles are formatted with source numbering, title, and content."""
        articles = sample_articles[:2]
        result = _build_articles_text(articles)

        assert "--- Source 1: OpenAI Releases GPT-5" in result
        assert "--- Source 2: DeepMind's AlphaFold 3" in result
        assert "OpenAI has officially released GPT-5" in result
        assert "DeepMind's latest publication in Nature" in result

    def test_uses_rss_excerpt_when_no_full_content(self, sample_articles):
        """Fall back to rss_excerpt if full_content is missing."""
        article = dict(sample_articles[2])
        article.pop("full_content", None)
        result = _build_articles_text([article])
        assert "The European Commission published the final" in result

    def test_truncates_content_at_5000_chars(self):
        """Content longer than 5000 chars is truncated with an indicator."""
        long_article = {
            "title": "Very Long Article",
            "full_content": "x" * 6000,
        }
        result = _build_articles_text([long_article])
        assert "... [truncated]" in result

    def test_empty_article_list(self):
        """An empty list produces an empty string."""
        assert _build_articles_text([]) == ""

    def test_short_content_not_truncated(self):
        """Content under 5000 chars is passed through as-is."""
        article = {
            "title": "Short Article",
            "full_content": "Short content here.",
        }
        result = _build_articles_text([article])
        assert "Short content here." in result
        assert "... [truncated]" not in result


# ===================================================================
# 6. run() — approval flow
# ===================================================================


class TestRunApprovalFlow:
    """Full ``run()`` execution — approval outcomes with patched eval functions.

    We patch ``_run_quality_eval`` and ``_run_adversarial_eval`` directly
    to avoid needing real LLM responses.
    """

    def test_both_pass_returns_approved(self, seeded_evaluator_db, mock_config, mock_llm):
        """When both quality and adversarial pass, returns 'approved'."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "approved"

    def test_both_pass_sets_theme_approved(self, seeded_evaluator_db, mock_config, mock_llm):
        """Theme status is set to 'approved' on pass."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        theme_row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        assert theme_row["status"] == "approved"

    def test_both_pass_stores_overall_passed_pass(self, seeded_evaluator_db, mock_config, mock_llm):
        """Evaluation round stores overall_passed='pass'."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 1
        assert evals[0]["overall_passed"] == "pass"
        assert evals[0]["quality_passed"] == "pass"
        assert evals[0]["adversarial_passed"] == "pass"

    def test_quality_fails_adversarial_passes_returns_needs_refinement(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Quality fail + adversarial pass → needs_refinement (when rounds remain)."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Too short"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_quality_passes_adversarial_fails_returns_needs_refinement(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Quality pass + adversarial fail → needs_refinement (when rounds remain)."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": False, "feedback": "Found hallucination", "issues": [{"deliverable": "script_en", "problem": "hallucination", "claim": "Made up stat"}]}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_both_fail_returns_needs_refinement(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Both evaluations fail → needs_refinement (when rounds remain)."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Bad"},
            "script_en": {"pass": False, "feedback": "Bad"},
            "script_de": {"pass": False, "feedback": "Bad"},
        }
        adversarial_result = {"pass": False, "feedback": "Errors everywhere", "issues": [{"deliverable": "summary_en", "problem": "bias", "claim": "Biased claim"}]}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_overall_fail_eval_round_stores_fail(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When overall result is fail, the evaluation round stores overall_passed='fail'."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Bad"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 1
        assert evals[0]["overall_passed"] == "fail"
        assert evals[0]["quality_passed"] == "fail"
        assert evals[0]["adversarial_passed"] == "pass"


# ===================================================================
# 7. run() — refinement loop
# ===================================================================


class TestRunRefinementLoop:
    """Round tracking and max-refinement auto-approve behavior."""

    def test_first_eval_is_round_1(self, seeded_evaluator_db, mock_config, mock_llm):
        """First evaluation stores round_number=1."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert evals[0]["round_number"] == 1

    def test_second_eval_is_round_2(self, seeded_evaluator_db, mock_config, mock_llm):
        """Second evaluation (after one previous) stores round_number=2."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Insert a previous evaluation round
        db.insert_evaluation_round(
            theme_id=theme_id,
            round_number=1,
            quality_passed="fail",
            quality_feedback="{}",
            adversarial_passed="pass",
            adversarial_feedback="{}",
            overall_passed="fail",
        )

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 2
        assert evals[1]["round_number"] == 2

    def test_third_eval_is_round_3(self, seeded_evaluator_db, mock_config, mock_llm):
        """Third evaluation stores round_number=3."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Insert two previous evaluation rounds
        db.insert_evaluation_round(theme_id, 1, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 2, "fail", "{}", "pass", "{}", "fail")

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 3
        assert evals[2]["round_number"] == 3

    def test_fail_at_max_rounds_auto_approved_returns_approved(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When max_refinement_rounds is reached, a fail result auto-approves."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Insert two previous evaluation rounds
        db.insert_evaluation_round(theme_id, 1, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 2, "fail", "{}", "pass", "{}", "fail")

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Still bad"},
            "script_en": {"pass": False, "feedback": "Still bad"},
            "script_de": {"pass": False, "feedback": "Still bad"},
        }
        adversarial_result = {"pass": False, "feedback": "Still wrong", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "approved"

    def test_fail_at_max_rounds_sets_auto_approved_status(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Theme status is set to 'auto_approved' when auto-approved."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        db.insert_evaluation_round(theme_id, 1, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 2, "fail", "{}", "pass", "{}", "fail")

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Bad"},
            "script_en": {"pass": False, "feedback": "Bad"},
            "script_de": {"pass": False, "feedback": "Bad"},
        }
        adversarial_result = {"pass": False, "feedback": "Bad", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        theme_row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        assert theme_row["status"] == "auto_approved"

    def test_auto_approved_stores_overall_passed_fail_in_db(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Even when auto-approved, the last eval round stores the actual fail result."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        db.insert_evaluation_round(theme_id, 1, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 2, "fail", "{}", "pass", "{}", "fail")

        quality_result = {
            "summary_en": {"pass": False, "feedback": "Bad"},
            "script_en": {"pass": False, "feedback": "Bad"},
            "script_de": {"pass": False, "feedback": "Bad"},
        }
        adversarial_result = {"pass": False, "feedback": "Bad", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 3
        assert evals[2]["overall_passed"] == "fail"


# ===================================================================
# 8. run() — edge cases
# ===================================================================


class TestRunEdgeCases:
    """Edge case and error handling in ``run()``."""

    def test_no_deliverables_approved_immediately(self, seeded_evaluator_db, mock_config, mock_llm):
        """When no deliverables exist for the theme, it is approved immediately."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Delete the existing deliverables
        db._conn.execute("DELETE FROM deliverables WHERE theme_id = ?", (theme_id,))
        db._conn.commit()

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "approved"
        theme_row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        assert theme_row["status"] == "approved"

    def test_no_deliverables_skips_llm_calls(self, seeded_evaluator_db, mock_config, mock_llm):
        """When no deliverables exist, LLM should not be called."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        db._conn.execute("DELETE FROM deliverables WHERE theme_id = ?", (theme_id,))
        db._conn.commit()

        run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))
        mock_llm.complete.assert_not_called()

    def test_theme_not_found_raises_value_error(self, seeded_evaluator_db, mock_config, mock_llm):
        """When the theme_id doesn't belong to the run, ValueError is raised."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        # Use a non-existent theme_id
        with pytest.raises(ValueError, match="Theme 9999 not found in run"):
            run(run_id, db, mock_config, mock_llm, 9999, InterestConfig(name="AI", id=1))

    def test_llm_error_in_quality_eval_falls_back_and_overall_fails(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When quality eval LLM raises an exception, fallback is used and overall may fail."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Mock the quality eval to simulate an LLM error
        with (
            patch(
                "src.evaluator._run_quality_eval",
                return_value=_fallback_quality_fail(
                    {
                        "summary_en": {"content": "x", "version": 1},
                        "script_en": {"content": "y", "version": 1},
                        "script_de": {"content": "z", "version": 1},
                    },
                    "LLM error: API unreachable",
                    InterestConfig(name="test"),
                ),
            ),
            patch("src.evaluator._run_adversarial_eval", return_value={"pass": True, "feedback": "OK", "issues": []}),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_llm_error_in_adversarial_eval_skips_and_uses_pass(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When adversarial eval LLM raises, it's skipped and treated as pass."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        # Simulate adversarial LLM error by returning pass result
        adversarial_fallback = {"pass": True, "feedback": "LLM error — skipping adversarial: timeout", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_fallback),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        # Quality passes + adversarial skip-pass = overall pass
        assert result == "approved"

    def test_quality_parse_failure_falls_back_to_fail(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When quality eval returns unparseable JSON, fallback fail is used."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        adversarial_result = {"pass": True, "feedback": "OK", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=_fallback_quality_fail(
                {
                    "summary_en": {"content": "x", "version": 1},
                    "script_en": {"content": "y", "version": 1},
                    "script_de": {"content": "z", "version": 1},
                },
                "JSON parse error",
                InterestConfig(name="test"),
            )),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_adversarial_parse_failure_skips_and_passes(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When adversarial eval returns unparseable JSON, it is skipped as pass."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        # Simulate parse failure → pass with skip message
        adversarial_skip = {"pass": True, "feedback": "JSON parse error — skipping adversarial", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_skip),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "approved"

    def test_round_number_tracking_with_existing_evaluations(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Round number increments correctly from existing evals stored out of order."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Insert evaluations out of insertion order (should pick max round_number)
        db.insert_evaluation_round(theme_id, 1, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 3, "fail", "{}", "pass", "{}", "fail")
        db.insert_evaluation_round(theme_id, 2, "fail", "{}", "pass", "{}", "fail")

        quality_result = {
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        }
        adversarial_result = {"pass": True, "feedback": "Accurate", "issues": []}

        with (
            patch("src.evaluator._run_quality_eval", return_value=quality_result),
            patch("src.evaluator._run_adversarial_eval", return_value=adversarial_result),
        ):
            result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        # Next round should be 4
        evals = db.get_evaluation_rounds(theme_id)
        assert len(evals) == 4
        assert evals[3]["round_number"] == 4
        assert result == "approved"


# ===================================================================
# 9. _run_quality_eval — with mocked LLM
# ===================================================================


class TestRunQualityEval:
    """Testing the quality eval function with a mocked LLM client."""

    def test_returns_parsed_quality_result(self, seeded_evaluator_db, mock_config, mock_llm):
        """Valid LLM JSON response is parsed and returned."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)
        articles_text = "Mock article text"

        mock_llm.complete.return_value = json.dumps({
            "summary_en": {"pass": True, "feedback": "Good quality"},
            "script_en": {"pass": True, "feedback": "Well written"},
            "script_de": {"pass": False, "feedback": "Too short"},
        })

        result = _run_quality_eval(mock_llm, mock_config, theme, deliverables, articles_text, InterestConfig(name="test"))

        assert result["summary_en"]["pass"] is True
        assert result["script_en"]["pass"] is True
        assert result["script_de"]["pass"] is False
        assert result["summary_en"]["feedback"] == "Good quality"

    def test_llm_error_returns_fallback_fail(self, seeded_evaluator_db, mock_config, mock_llm):
        """When LLM raises, fallback quality fail is returned."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.side_effect = RuntimeError("API timeout")

        result = _run_quality_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        for dtype in ("summary_en", "script_en", "script_de"):
            assert result[dtype]["pass"] is False
            assert "LLM error" in result[dtype]["feedback"]

    def test_parse_failure_returns_fallback_fail(self, seeded_evaluator_db, mock_config, mock_llm):
        """When LLM returns non-JSON, fallback quality fail is returned."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = "not json"

        result = _run_quality_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        for dtype in ("summary_en", "script_en", "script_de"):
            assert result[dtype]["pass"] is False
            assert "JSON parse error" in result[dtype]["feedback"]

    def test_calls_llm_with_weak_model(self, seeded_evaluator_db, mock_config, mock_llm):
        """LLM is called with the weak model's id and temperature."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "summary_en": {"pass": True, "feedback": "OK"},
            "script_en": {"pass": True, "feedback": "OK"},
            "script_de": {"pass": True, "feedback": "OK"},
        })

        _run_quality_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["model_id"] == "deepseek/deepseek-chat"
        assert call_kwargs["temperature"] == 0.3

    def test_system_prompt_includes_quality_rubric(self, seeded_evaluator_db, mock_config, mock_llm):
        """System prompt mentions quality evaluation criteria."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "summary_en": {"pass": True, "feedback": "OK"},
            "script_en": {"pass": True, "feedback": "OK"},
            "script_de": {"pass": True, "feedback": "OK"},
        })

        _run_quality_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        call_kwargs = mock_llm.complete.call_args[1]
        system_prompt = call_kwargs["system_prompt"]
        assert "senior editor" in system_prompt.lower() or "evaluating the quality" in system_prompt.lower()

    def test_user_prompt_includes_theme_and_deliverables(self, seeded_evaluator_db, mock_config, mock_llm):
        """User prompt contains the theme title and deliverable content."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "summary_en": {"pass": True, "feedback": "OK"},
            "script_en": {"pass": True, "feedback": "OK"},
            "script_de": {"pass": True, "feedback": "OK"},
        })

        _run_quality_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]
        assert theme["title"] in user_prompt
        assert "English summary content here." in user_prompt
        assert "English script content here." in user_prompt
        assert "German script content here." in user_prompt


# ===================================================================
# 10. _run_adversarial_eval — with mocked LLM
# ===================================================================


class TestRunAdversarialEval:
    """Testing the adversarial eval function with a mocked LLM client."""

    def test_returns_parsed_adversarial_result(self, seeded_evaluator_db, mock_config, mock_llm):
        """Valid LLM JSON response is parsed and returned."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "pass": True,
            "feedback": "All claims verified",
            "issues": [],
        })

        result = _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        assert result["pass"] is True
        assert result["feedback"] == "All claims verified"
        assert result["issues"] == []

    def test_returns_parsed_with_issues(self, seeded_evaluator_db, mock_config, mock_llm):
        """Adversarial result with issues is parsed correctly."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "pass": False,
            "feedback": "Found issues",
            "issues": [
                {"deliverable": "summary_en", "claim": "AI will replace all jobs", "problem": "bias"},
            ],
        })

        result = _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        assert result["pass"] is False
        assert len(result["issues"]) == 1

    def test_llm_error_returns_pass_with_skip_message(self, seeded_evaluator_db, mock_config, mock_llm):
        """When LLM raises, result is pass with a skip message."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.side_effect = ConnectionError("Network error")

        result = _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        assert result["pass"] is True
        assert "LLM error" in result["feedback"] or "skipping" in result["feedback"]
        assert result["issues"] == []

    def test_parse_failure_returns_pass_with_skip_message(self, seeded_evaluator_db, mock_config, mock_llm):
        """When LLM returns non-JSON, result is pass with skip message."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = "not json at all"

        result = _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        assert result["pass"] is True
        assert "JSON parse error" in result["feedback"]
        assert result["issues"] == []

    def test_calls_llm_with_weak_model(self, seeded_evaluator_db, mock_config, mock_llm):
        """LLM is called with the weak model's id and temperature."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "pass": True, "feedback": "OK", "issues": [],
        })

        _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["model_id"] == "deepseek/deepseek-chat"
        assert call_kwargs["temperature"] == 0.3

    def test_system_prompt_includes_adversarial_criteria(self, seeded_evaluator_db, mock_config, mock_llm):
        """System prompt mentions fact-checking criteria."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "pass": True, "feedback": "OK", "issues": [],
        })

        _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "articles", InterestConfig(name="test"))

        call_kwargs = mock_llm.complete.call_args[1]
        system_prompt = call_kwargs["system_prompt"]
        assert "fact-checker" in system_prompt.lower() or "Factual Accuracy" in system_prompt

    def test_user_prompt_includes_theme_and_articles(self, seeded_evaluator_db, mock_config, mock_llm):
        """User prompt contains the theme title and source articles text."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]

        themes = db.get_themes_for_run(ctx["run_id"])
        theme = [t for t in themes if t["id"] == theme_id][0]
        deliverables = db.get_latest_deliverables(theme_id)

        mock_llm.complete.return_value = json.dumps({
            "pass": True, "feedback": "OK", "issues": [],
        })

        _run_adversarial_eval(mock_llm, mock_config, theme, deliverables, "Test article text for adversarial", InterestConfig(name="test"))

        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]
        assert theme["title"] in user_prompt
        assert "Test article text for adversarial" in user_prompt


# ===================================================================
# 11. run() — LLM integration (real eval functions, mocked LLM client)
# ===================================================================


class TestRunLLMIntegration:
    """Full ``run()`` with actual ``_run_quality_eval`` / ``_run_adversarial_eval``
    functions but a mocked LLM client.  This validates that the two LLM calls
    are made with the correct arguments and their responses are handled
    correctly end-to-end."""

    def test_both_llm_calls_made(self, seeded_evaluator_db, mock_config, mock_llm):
        """run() makes exactly two LLM calls (quality + adversarial)."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_response = json.dumps({
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        })
        adversarial_response = json.dumps({
            "pass": True, "feedback": "Accurate", "issues": [],
        })

        mock_llm.complete.side_effect = [quality_response, adversarial_response]

        run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert mock_llm.complete.call_count == 2

    def test_quality_fail_adversarial_pass_with_real_functions(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Quality returns fail, adversarial returns pass → needs_refinement."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_response = json.dumps({
            "summary_en": {"pass": False, "feedback": "Too short"},
            "script_en": {"pass": True, "feedback": "OK"},
            "script_de": {"pass": True, "feedback": "OK"},
        })
        adversarial_response = json.dumps({
            "pass": True, "feedback": "Accurate", "issues": [],
        })

        mock_llm.complete.side_effect = [quality_response, adversarial_response]

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_quality_pass_adversarial_fail_with_real_functions(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Quality passes, adversarial fails → needs_refinement."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_response = json.dumps({
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        })
        adversarial_response = json.dumps({
            "pass": False,
            "feedback": "Factual error",
            "issues": [{"deliverable": "summary_en", "claim": "Wrong stat", "problem": "hallucination"}],
        })

        mock_llm.complete.side_effect = [quality_response, adversarial_response]

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_both_pass_with_real_functions_returns_approved(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """Both evaluations pass → approved."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_response = json.dumps({
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        })
        adversarial_response = json.dumps({
            "pass": True, "feedback": "All accurate", "issues": [],
        })

        mock_llm.complete.side_effect = [quality_response, adversarial_response]

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "approved"

    def test_quality_llm_error_real_functions(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When quality LLM errors, fallback fail + adversarial pass → needs_refinement."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        # Quality LLM errors, adversarial returns pass
        mock_llm.complete.side_effect = [RuntimeError("API error"), json.dumps({
            "pass": True, "feedback": "Accurate", "issues": [],
        })]

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        assert result == "needs_refinement"

    def test_adversarial_llm_error_real_functions(
        self, seeded_evaluator_db, mock_config, mock_llm
    ):
        """When adversarial LLM errors, skip+pass is used."""
        ctx = seeded_evaluator_db
        db = ctx["db"]
        theme_id = ctx["theme_id"]
        run_id = ctx["run_id"]

        quality_response = json.dumps({
            "summary_en": {"pass": True, "feedback": "Good"},
            "script_en": {"pass": True, "feedback": "Good"},
            "script_de": {"pass": True, "feedback": "Good"},
        })
        mock_llm.complete.side_effect = [quality_response, RuntimeError("Adversarial API error")]

        result = run(run_id, db, mock_config, mock_llm, theme_id, InterestConfig(name="AI", id=1))

        # Quality passes + adversarial skip → approved
        assert result == "approved"


# ===================================================================
# 12. Prompt template existence
# ===================================================================


class TestPromptTemplates:
    """Verify the evaluator prompt templates exist and are well-formed."""

    @pytest.fixture
    def prompts_dir(self):
        """Absolute path to the prompts directory."""
        return pathlib.Path(__file__).parent.parent / "prompts"

    def test_quality_template_exists(self, prompts_dir):
        """evaluate_quality.txt must exist."""
        path = prompts_dir / "evaluate_quality.txt"
        assert path.exists(), f"Template not found at {path}"
        assert path.is_file()

    def test_adversarial_template_exists(self, prompts_dir):
        """evaluate_adversarial.txt must exist."""
        path = prompts_dir / "evaluate_adversarial.txt"
        assert path.exists(), f"Template not found at {path}"
        assert path.is_file()

    def test_quality_template_has_user_separator(self, prompts_dir):
        """Quality template contains the === USER === separator."""
        text = (prompts_dir / "evaluate_quality.txt").read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        assert len(parts) == 2, "Expected exactly one '=== USER ===' separator"

    def test_adversarial_template_has_user_separator(self, prompts_dir):
        """Adversarial template contains the === USER === separator."""
        text = (prompts_dir / "evaluate_adversarial.txt").read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        assert len(parts) == 2, "Expected exactly one '=== USER ===' separator"

    def test_quality_template_has_required_placeholders(self, prompts_dir):
        """Quality template has all required format placeholders."""
        text = (prompts_dir / "evaluate_quality.txt").read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        user_part = parts[1].strip()
        assert "{theme_title}" in user_part
        assert "{summary_en}" in user_part
        assert "{script_en}" in user_part
        assert "{script_de}" in user_part

    def test_adversarial_template_has_required_placeholders(self, prompts_dir):
        """Adversarial template has all required format placeholders."""
        text = (prompts_dir / "evaluate_adversarial.txt").read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        user_part = parts[1].strip()
        assert "{theme_title}" in user_part
        assert "{articles_text}" in user_part
        assert "{summary_en}" in user_part
        assert "{script_en}" in user_part
        assert "{script_de}" in user_part

    def test_quality_template_has_system_marker(self, prompts_dir):
        """Quality template starts with === SYSTEM ===."""
        text = (prompts_dir / "evaluate_quality.txt").read_text(encoding="utf-8")
        assert text.startswith("=== SYSTEM ===")

    def test_adversarial_template_has_system_marker(self, prompts_dir):
        """Adversarial template starts with === SYSTEM ===."""
        text = (prompts_dir / "evaluate_adversarial.txt").read_text(encoding="utf-8")
        assert text.startswith("=== SYSTEM ===")
