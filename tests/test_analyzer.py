"""Tests for the analyzer module — LLM-based theme identification.

Tests cover prompt building helpers, LLM response parsing/validation, the
full ``run()`` integration (with an in-memory DB and mocked LLM client),
and prompt template loading.
"""

import json
import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import (
    AnalysisParseError,
    _build_articles_section,
    _build_previous_brief_section,
    _parse_themes_response,
    run,
)
from src.db import Database

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
def sample_themes():
    """Load the sample themes fixture as a list of dicts.

    Note: the fixture includes an ``order_index`` field that is **not** part
    of the LLM response schema; the LLM response only contains ``title``,
    ``description``, ``novelty_type``, and ``source_article_indices``.
    """
    with open(_FIXTURES_DIR / "sample_themes.json") as f:
        return json.load(f)


@pytest.fixture
def mock_llm():
    """Return a MagicMock that acts as a minimal LLMClient.

    The default ``.complete()`` return value is a valid JSON theme array.
    Tests that need a different response should reassign ``.complete.return_value``
    or ``.complete.side_effect``.
    """
    client = MagicMock()
    client.complete.return_value = json.dumps(
        [
            {
                "title": "GPT-5 Release and Next-Generation AI Models",
                "description": (
                    "OpenAI's GPT-5 brings significant reasoning improvements. "
                    "Paired with NVIDIA's new Rubin architecture."
                ),
                "novelty_type": "novel",
                "source_article_indices": [0, 4],
            },
            {
                "title": "AI Safety and Alignment Concerns",
                "description": "A study claims Claude 4 shows deceptive alignment.",
                "novelty_type": "novel",
                "source_article_indices": [3],
            },
            {
                "title": "EU AI Act Implementation Framework",
                "description": "The EU published final AI Act guidelines.",
                "novelty_type": "novel",
                "source_article_indices": [2],
            },
        ]
    )
    return client


@pytest.fixture
def mock_config():
    """Return a minimal mock config with a strong model definition."""
    config = MagicMock()
    config.models.strong.id = "deepseek/deepseek-v4-pro"
    config.models.strong.temperature = 0.5
    config.pipeline.max_themes = 5
    return config


@pytest.fixture
def seeded_articles_db(db, sample_articles):
    """``db`` with a pipeline run, a feed, and articles pre-inserted.

    Returns a dict with ``db``, ``run_id``, ``feed_id``, and ``article_ids``
    so tests can reference the stored IDs.
    """
    ai_id = db.get_interest_by_name("AI")["id"]
    run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
    feed_id = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")

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

    return {"db": db, "run_id": run_id, "feed_id": feed_id, "article_ids": article_ids}


# ===================================================================
# 1. _build_articles_section
# ===================================================================


class TestBuildArticlesSection:
    """Formatting the article list for the LLM prompt."""

    def test_formats_with_indices_titles_and_content(self, sample_articles):
        """Each article appears with its index, title, and content."""
        result = _build_articles_section(sample_articles[:2])

        assert "[0] Title: OpenAI Releases GPT-5" in result
        assert "[1] Title: DeepMind's AlphaFold 3" in result
        assert "OpenAI has officially released GPT-5" in result
        assert "DeepMind's latest publication in Nature" in result

    def test_includes_blank_line_between_articles(self, sample_articles):
        """Articles are separated by a blank line."""
        result = _build_articles_section(sample_articles[:2])
        # Each article ends with a blank line
        assert result.count("\n\n") >= 2

    def test_uses_rss_excerpt_when_no_full_content(self, sample_articles):
        """Fall back to rss_excerpt if full_content is missing."""
        article = dict(sample_articles[2])  # has rss_excerpt and full_content
        article.pop("full_content", None)
        result = _build_articles_section([article])
        assert "The European Commission published the final" in result

    def test_truncates_content_at_5000_chars(self):
        """Content longer than 5000 chars is truncated with an indicator."""
        long_article = {
            "id": 99,
            "title": "Very Long Article",
            "full_content": "x" * 6000,
        }
        result = _build_articles_section([long_article])
        assert "... [truncated]" in result
        # Verify content part (after "Content: ") is at most 5000 + len("[truncated]")
        content_line = [l for l in result.split("\n") if l.startswith("    Content:")][0]
        content_text = content_line.replace("    Content: ", "")
        # 5000 original chars + len("... [truncated]") = 5015
        assert len(content_text) <= 5015
        assert content_text.endswith("... [truncated]")

    def test_empty_article_list(self):
        """An empty list produces an empty string."""
        assert _build_articles_section([]) == ""

    def test_short_content_not_truncated(self):
        """Content under 5000 chars is passed through as-is."""
        article = {
            "id": 1,
            "title": "Short Article",
            "full_content": "Short content here.",
        }
        result = _build_articles_section([article])
        assert "Short content here." in result
        assert "... [truncated]" not in result


# ===================================================================
# 2. _build_previous_brief_section
# ===================================================================


class TestBuildPreviousBriefSection:
    """Building the 'previous brief' section of the LLM prompt."""

    def test_with_brief_includes_content_and_instructions(self):
        """When a previous brief is provided, include its content and novelty classification instructions."""
        brief = {"id": 1, "content": "Yesterday's AI news summary.\nKey themes: GPT-4 updates."}
        result = _build_previous_brief_section(brief)

        assert "PREVIOUS DAILY BRIEF" in result
        assert "Yesterday's AI news summary." in result
        assert '"novel"' in result
        assert '"continuation"' in result

    def test_without_brief_indicates_no_brief_and_classify_all_novel(self):
        """When no previous brief exists, indicate so and instruct to classify all as novel."""
        result = _build_previous_brief_section(None)

        assert "No previous daily brief is available" in result
        assert "Classify all themes as \"novel\"" in result
        assert "novel" in result
        # Should NOT mention "continuation"
        assert "continuation" not in result

    def test_with_brief_mentions_both_novelty_types(self):
        """When a brief exists, both 'novel' and 'continuation' are explained."""
        brief = {"id": 2, "content": "Some brief content."}
        result = _build_previous_brief_section(brief)

        assert 'classify each as:' in result
        assert '- "novel"' in result
        assert '- "continuation"' in result


# ===================================================================
# 3. _parse_themes_response
# ===================================================================


class TestParseThemesResponse:
    """Parsing and validating the raw string returned by the LLM."""

    def test_valid_json_array(self, sample_themes):
        """A straight JSON array with 3 valid themes is parsed correctly.

        Note: the fixture includes ``source_article_ids`` (DB column name) and
        ``order_index`` as extra fields; the LLM response schema requires
        ``source_article_indices``.  We remap here to test the parser.
        """
        # Remap source_article_ids -> source_article_indices for LLM response format
        llm_themes = []
        for t in sample_themes:
            llm_t = dict(t)
            llm_t["source_article_indices"] = llm_t.pop("source_article_ids")
            llm_themes.append(llm_t)

        raw = json.dumps(llm_themes)
        result = _parse_themes_response(raw, article_count=10)
        assert len(result) == 3

    def test_valid_json_minimal_fields(self):
        """Only required fields are needed; extra fields (like order_index) are ignored."""
        raw = json.dumps([
            {"title": "T1", "description": "D1", "novelty_type": "novel", "source_article_indices": [0]},
            {"title": "T2", "description": "D2", "novelty_type": "continuation", "source_article_indices": [1, 2]},
        ])
        result = _parse_themes_response(raw, article_count=5)
        assert len(result) == 2
        assert result[0]["title"] == "T1"
        assert result[1]["novelty_type"] == "continuation"

    def test_markdown_code_fence_json(self):
        """JSON wrapped in ```json ... ``` fences is stripped and parsed."""
        inner = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [0]},
        ])
        raw = f"```json\n{inner}\n```"
        result = _parse_themes_response(raw, article_count=3)
        assert len(result) == 1

    def test_markdown_code_fence_no_lang(self):
        """JSON wrapped in plain ``` ... ``` (no language hint) is also stripped."""
        inner = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [0]},
        ])
        raw = f"```\n{inner}\n```"
        result = _parse_themes_response(raw, article_count=3)
        assert len(result) == 1

    def test_strips_whitespace_around_fences(self):
        """Leading/trailing whitespace around fences is handled."""
        raw = "  \n  ```json\n[{\"title\": \"T\", \"description\": \"D\", \"novelty_type\": \"novel\", \"source_article_indices\": [0]}]\n  ```  \n"
        result = _parse_themes_response(raw, article_count=3)
        assert len(result) == 1

    def test_zero_themes_raises_error(self):
        """0 themes should raise AnalysisParseError."""
        raw = json.dumps([])
        with pytest.raises(AnalysisParseError, match="Expected 1.*themes"):
            _parse_themes_response(raw, article_count=5)

    def test_six_themes_raises_error(self):
        """Themes exceeding the explicit max should raise AnalysisParseError."""
        raw = json.dumps([
            {"title": f"T{i}", "description": "D", "novelty_type": "novel", "source_article_indices": [0]}
            for i in range(6)
        ])
        with pytest.raises(AnalysisParseError, match="Expected 1.*5 themes"):
            _parse_themes_response(raw, article_count=5, max_themes=5)

    def test_missing_required_fields_raises_error(self):
        """A theme missing 'title' or 'description' etc. raises AnalysisParseError."""
        raw = json.dumps([
            {"title": "Only title", "novelty_type": "novel", "source_article_indices": [0]},
        ])
        with pytest.raises(AnalysisParseError, match="missing required fields"):
            _parse_themes_response(raw, article_count=5)

    def test_missing_multiple_fields_raises_error(self):
        """Missing several required fields reports all of them."""
        raw = json.dumps([
            {"title": "Only title"},
        ])
        with pytest.raises(AnalysisParseError, match="missing required fields"):
            _parse_themes_response(raw, article_count=5)

    def test_invalid_novelty_type_raises_error(self):
        """novelty_type that is not 'novel' or 'continuation' raises AnalysisParseError."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "invalid", "source_article_indices": [0]},
        ])
        with pytest.raises(AnalysisParseError, match="Invalid novelty_type"):
            _parse_themes_response(raw, article_count=5)

    def test_negative_article_index_raises_error(self):
        """A negative article index is invalid."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [-1]},
        ])
        with pytest.raises(AnalysisParseError, match="Invalid article index"):
            _parse_themes_response(raw, article_count=5)

    def test_article_index_out_of_range_raises_error(self):
        """An index >= article_count is invalid."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [5]},
        ])
        with pytest.raises(AnalysisParseError, match="Invalid article index"):
            _parse_themes_response(raw, article_count=5)

    def test_empty_source_article_indices_raises_error(self):
        """An empty source_article_indices list is invalid."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": []},
        ])
        with pytest.raises(AnalysisParseError, match="non-empty"):
            _parse_themes_response(raw, article_count=5)

    def test_non_json_response_raises_error(self):
        """Non-JSON text raises AnalysisParseError."""
        raw = "This is not JSON at all."
        with pytest.raises(AnalysisParseError, match="Invalid JSON"):
            _parse_themes_response(raw, article_count=5)

    def test_response_is_not_a_list_raises_error(self):
        """A JSON object (not array) raises AnalysisParseError."""
        raw = json.dumps({"title": "T", "description": "D"})
        with pytest.raises(AnalysisParseError, match="Expected a JSON array"):
            _parse_themes_response(raw, article_count=5)

    def test_theme_is_not_a_dict_raises_error(self):
        """A theme entry that is not a dict (e.g. string) raises AnalysisParseError."""
        raw = json.dumps(["not a dict"])
        with pytest.raises(AnalysisParseError, match="Theme is not a dict"):
            _parse_themes_response(raw, article_count=5)

    def test_single_theme_is_valid(self):
        """Exactly 1 theme is at the lower boundary and should succeed."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [0]},
        ])
        result = _parse_themes_response(raw, article_count=5)
        assert len(result) == 1

    def test_five_themes_are_valid(self):
        """Exactly 5 themes is at the upper boundary and should succeed."""
        raw = json.dumps([
            {"title": f"T{i}", "description": "D", "novelty_type": "novel", "source_article_indices": [0]}
            for i in range(5)
        ])
        result = _parse_themes_response(raw, article_count=5)
        assert len(result) == 5

    def test_continuation_novelty_type_is_valid(self):
        """"continuation" novelty_type is accepted."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "continuation", "source_article_indices": [0]},
        ])
        result = _parse_themes_response(raw, article_count=5)
        assert result[0]["novelty_type"] == "continuation"

    def test_source_article_indices_are_not_mutated(self):
        """The parsed list should retain the exact indices from the response."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [2, 4]},
        ])
        result = _parse_themes_response(raw, article_count=10)
        assert result[0]["source_article_indices"] == [2, 4]

    def test_article_count_boundary_exact_max_index(self):
        """index == article_count - 1 is valid (last article)."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [4]},
        ])
        result = _parse_themes_response(raw, article_count=5)
        assert len(result) == 1

    def test_article_count_boundary_exact_max_index_plus_one(self):
        """index == article_count is invalid (one past last)."""
        raw = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "novel", "source_article_indices": [5]},
        ])
        with pytest.raises(AnalysisParseError, match="Invalid article index"):
            _parse_themes_response(raw, article_count=5)


# ===================================================================
# 4. run() integration tests
# ===================================================================


class TestRunIntegration:
    """Full ``run()`` execution with in-memory DB and mocked LLM client."""

    # ------------------------------------------------------------------
    # 4a. Successful analysis
    # ------------------------------------------------------------------

    def test_successful_analysis_inserts_themes(self, seeded_articles_db, mock_llm, mock_config):
        """Themes returned by the LLM are stored in the database."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        themes = db.get_themes_for_run(run_id)
        assert len(themes) == 3

        # Verify theme titles
        titles = [t["title"] for t in themes]
        assert "GPT-5 Release and Next-Generation AI Models" in titles
        assert "AI Safety and Alignment Concerns" in titles
        assert "EU AI Act Implementation Framework" in titles

        # Verify order_index
        ordered = [t["order_index"] for t in themes]
        assert ordered == [1, 2, 3]

    def test_successful_analysis_stores_source_article_ids(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """source_article_ids stored as JSON matches the article primary keys."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]
        article_ids = ctx["article_ids"]

        run(run_id, db, mock_config, mock_llm)

        themes = db.get_themes_for_run(run_id)
        # Theme referencing articles at indices [0, 4] in the sample array
        gpt_theme = [t for t in themes if "GPT-5" in t["title"]][0]
        stored_ids = json.loads(gpt_theme["source_article_ids"])
        assert article_ids[0] in stored_ids  # article at index 0
        assert article_ids[4] in stored_ids  # article at index 4

    def test_successful_analysis_novelty_types_stored(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """novelty_type is persisted for each theme."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        themes = db.get_themes_for_run(run_id)
        for t in themes:
            assert t["novelty_type"] in ("novel", "continuation")

    # ------------------------------------------------------------------
    # 4b. Edge cases: no articles, previous brief
    # ------------------------------------------------------------------

    def test_no_articles_skips_analysis(self, db, mock_llm, mock_config):
        """When there are no articles for the run, analysis is skipped and no themes are stored."""
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")

        run(run_id, db, mock_config, mock_llm)

        themes = db.get_themes_for_run(run_id)
        assert themes == []
        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()

    def test_with_previous_brief_prompt_includes_brief(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """When a previous brief exists, the user prompt includes it."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        # Create a previous completed run with a daily brief
        prev_run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-13", "2026-05-13T06:00:00")
        db.update_pipeline_run(prev_run_id, status="completed", completed_at="2026-05-13T06:30:00")
        db.insert_daily_brief(prev_run_id, "Yesterday's AI summary.", 15)

        run(run_id, db, mock_config, mock_llm)

        # Verify the LLM was called with a prompt containing the brief content
        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]
        assert "PREVIOUS DAILY BRIEF" in user_prompt
        assert "Yesterday's AI summary." in user_prompt

    def test_without_previous_brief_prompt_indicates_no_brief(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """When no previous brief exists, the prompt says so."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]
        # No previous run or brief in the DB

        run(run_id, db, mock_config, mock_llm)

        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]
        assert "No previous daily brief is available" in user_prompt

    # ------------------------------------------------------------------
    # 4c. Error handling
    # ------------------------------------------------------------------

    def test_llm_client_error_is_raised(self, seeded_articles_db, mock_llm, mock_config):
        """When the LLM client raises an exception, AnalysisParseError is raised."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        mock_llm.complete.side_effect = RuntimeError("API unreachable")

        with pytest.raises(AnalysisParseError, match="LLM call failed"):
            run(run_id, db, mock_config, mock_llm)

    def test_analysis_parse_error_on_unparseable_response(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """When the LLM returns unparseable text, AnalysisParseError propagates."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        mock_llm.complete.return_value = "not json at all"

        with pytest.raises(AnalysisParseError, match="Invalid JSON"):
            run(run_id, db, mock_config, mock_llm)

    def test_validation_error_in_run_propagates(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """When parsed themes fail validation, AnalysisParseError is raised."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        # Return value with invalid novelty_type
        mock_llm.complete.return_value = json.dumps([
            {"title": "T", "description": "D", "novelty_type": "bogus", "source_article_indices": [0]},
        ])

        with pytest.raises(AnalysisParseError, match="Invalid novelty_type"):
            run(run_id, db, mock_config, mock_llm)

    def test_themes_not_inserted_on_error(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """When parsing/validation fails, no themes are stored in the DB."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        mock_llm.complete.return_value = "invalid"

        with pytest.raises(AnalysisParseError):
            run(run_id, db, mock_config, mock_llm)

        themes = db.get_themes_for_run(run_id)
        assert themes == []

    # ------------------------------------------------------------------
    # 4d. Stage tracking
    # ------------------------------------------------------------------

    def test_current_stage_updated_to_analyze(self, seeded_articles_db, mock_llm, mock_config):
        """The pipeline run's current_stage is set to 'analyze' at the start."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run_record_before = db.get_pipeline_run(run_id)
        assert run_record_before["current_stage"] is None

        run(run_id, db, mock_config, mock_llm)

        run_record_after = db.get_pipeline_run(run_id)
        assert run_record_after["current_stage"] == "analyze"

    def test_current_stage_updated_even_when_no_articles(
        self, db, mock_llm, mock_config
    ):
        """current_stage is set to 'analyze' even when there are no articles."""
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")

        run(run_id, db, mock_config, mock_llm)

        run_record = db.get_pipeline_run(run_id)
        assert run_record["current_stage"] == "analyze"

    # ------------------------------------------------------------------
    # 4e. LLM call arguments
    # ------------------------------------------------------------------

    def test_llm_called_with_correct_model_and_temperature(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """The LLM client is invoked with the strong model ID and temperature."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["model_id"] == "deepseek/deepseek-v4-pro"
        assert call_kwargs["temperature"] == 0.5

    def test_llm_called_with_system_prompt(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """The LLM client receives a system prompt from the template."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        call_kwargs = mock_llm.complete.call_args[1]
        system_prompt = call_kwargs["system_prompt"]
        assert "expert ai news analyst" in system_prompt.lower()

    def test_llm_called_with_user_prompt_containing_articles(
        self, seeded_articles_db, mock_llm, mock_config
    ):
        """The user prompt includes the formatted article section."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]
        assert "[0] Title: OpenAI Releases GPT-5" in user_prompt
        assert "[1] Title: DeepMind's AlphaFold 3" in user_prompt


# ===================================================================
# 5. Prompt template loading
# ===================================================================


class TestPromptTemplate:
    """Verifying the analyze.txt prompt template is loaded and structured correctly."""

    @pytest.fixture
    def template_path(self):
        """Absolute path to the analyze.txt prompt template."""
        return pathlib.Path(__file__).parent.parent / "prompts" / "analyze.txt"

    def test_template_file_exists(self, template_path):
        """The prompt template file must exist on disk."""
        assert template_path.exists(), f"Template not found at {template_path}"
        assert template_path.is_file()

    def test_template_is_split_by_user_marker(self, template_path):
        """The template contains '=== USER ===' marker for system/user separation."""
        text = template_path.read_text(encoding="utf-8")
        assert "=== USER ===" in text

        parts = text.split("=== USER ===")
        assert len(parts) == 2, "Expected exactly one '=== USER ===' separator"

    def test_template_has_system_marker(self, template_path):
        """The system section begins with '=== SYSTEM ==='."""
        text = template_path.read_text(encoding="utf-8")
        assert "=== SYSTEM ===" in text

    def test_system_prompt_contains_instructions(self, template_path):
        """The system prompt must contain key instructions for the LLM."""
        text = template_path.read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        system_part = parts[0].replace("=== SYSTEM ===\n", "").strip()

        assert "expert ai news analyst" in system_part.lower()
        assert "valid JSON array" in system_part
        assert "source_article_indices" in system_part

    def test_user_prompt_contains_format_placeholders(self, template_path):
        """The user prompt includes the {previous_brief_section} and {articles_section} placeholders."""
        text = template_path.read_text(encoding="utf-8")
        parts = text.split("=== USER ===")
        user_part = parts[1].strip()

        assert "{previous_brief_section}" in user_part
        assert "{articles_section}" in user_part
        assert "{max_themes}" in user_part
        assert "{article_count}" in user_part

    def test_user_prompt_formatting(self, seeded_articles_db, mock_config, mock_llm):
        """Verify that run() successfully formats the user prompt with actual values."""
        ctx = seeded_articles_db
        db = ctx["db"]
        run_id = ctx["run_id"]

        run(run_id, db, mock_config, mock_llm)

        call_kwargs = mock_llm.complete.call_args[1]
        user_prompt = call_kwargs["user_prompt"]

        # Placeholders should have been replaced
        assert "{previous_brief_section}" not in user_prompt
        assert "{articles_section}" not in user_prompt
        # Real content should be present
        assert "Below is the list of" in user_prompt
        assert "articles for analysis" in user_prompt

    def test_system_prompt_has_novelty_classification_rules(self, template_path):
        """System prompt explains novel vs continuation classification."""
        text = template_path.read_text(encoding="utf-8")
        assert '"novel"' in text
        assert '"continuation"' in text
