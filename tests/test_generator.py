"""Tests for the generator module — LLM-based content generation for themes.

Covers ``run()``, ``refine()``, and all internal helpers.  Uses an in-memory
SQLite database and a mocked LLM client throughout.
"""

import json
import logging

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.db import Database
from src.generator import (
    GeneratorError,
    run,
    refine,
    _generate_one,
    _generate_theme_deliverables,
    _build_articles_text,
    _parse_article_ids,
    _get_articles,
    _word_count,
)
from src.models import InterestConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Create an in-memory database with schema initialized."""
    database = Database(":memory:")
    database.initialize_schema()
    yield database
    database.close()


@pytest.fixture
def mock_llm():
    """Return a MagicMock that behaves like an LLMClient."""
    llm = MagicMock()
    llm.complete.return_value = "Generated content here"
    return llm


@pytest.fixture
def mock_config():
    """Return a mock Config object with a strong model definition."""
    config = MagicMock()
    config.models.strong.id = "test-model"
    config.models.strong.temperature = 0.7
    return config


@pytest.fixture
def ai_id(db):
    """Return the ID of the default 'AI' interest created by schema initialization."""
    return db.get_interest_by_name("AI")["id"]


@pytest.fixture
def seeded_run(db, ai_id):
    """Create a pipeline run and return its ID."""
    return db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")


@pytest.fixture
def interest(db):
    """Return a default InterestConfig for the AI interest."""
    return InterestConfig(id=1, name="AI")


@pytest.fixture
def seeded_feed(db, seeded_run, ai_id):
    """Create a feed and return its ID."""
    feed_id = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
    return feed_id


def _insert_article(db, feed_id, run_id, article_id, title, full_content=None, rss_excerpt=None):
    """Insert an article with a specific ID by manipulating the DB directly."""
    import datetime
    # We need to allow inserting at a specific ID; use raw SQL
    url = f"https://example.com/article-{article_id}"
    now = datetime.datetime.now().isoformat()
    db._conn.execute(
        "INSERT INTO articles (id, feed_id, url, title, author, published_at, "
        "scraped_at, rss_excerpt, full_content, content_status, pipeline_run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            article_id, feed_id, url, title, None,
            now, now, rss_excerpt, full_content, "full", run_id,
        ),
    )
    db._conn.commit()
    return article_id


@pytest.fixture
def seeded_articles(db, seeded_run, seeded_feed):
    """Insert sample articles and return their IDs."""
    ids = []
    articles_data = [
        {
            "title": "AI Makes Breakthrough in Protein Folding",
            "full_content": "DeepMind's AlphaFold has achieved a major breakthrough in protein folding prediction. "
            "This will accelerate drug discovery and our understanding of diseases. "
            "The AI system can predict protein structures with atomic accuracy. "
            "Researchers are excited about the implications for medicine. "
            "The technology has been made freely available to the scientific community.",
        },
        {
            "title": "OpenAI Releases GPT-5",
            "full_content": "OpenAI has released GPT-5 with unprecedented reasoning capabilities. "
            "The model scores highly on benchmarks including math, coding, and scientific reasoning. "
            "Early testers report significant improvements over GPT-4. "
            "The model is available through API access and ChatGPT Plus.",
            "rss_excerpt": "GPT-5 brings major improvements in reasoning.",
        },
        {
            "title": "EU Passes Comprehensive AI Regulation",
            "full_content": "The European Union has passed the AI Act, a comprehensive regulatory framework. "
            "The act categorizes AI systems by risk level. High-risk systems face strict requirements. "
            "Some transparency requirements apply to all AI systems. "
            "The act includes provisions for generative AI and foundation models.",
        },
    ]
    for i, data in enumerate(articles_data, start=1):
        aid = _insert_article(
            db, seeded_feed, seeded_run, i,
            data["title"],
            full_content=data.get("full_content"),
            rss_excerpt=data.get("rss_excerpt"),
        )
        ids.append(aid)
    return ids


@pytest.fixture
def seeded_theme_pending(db, seeded_run, seeded_articles):
    """Insert a pending theme for the seeded run. Returns the theme dict."""
    theme_id = db.insert_theme(
        pipeline_run_id=seeded_run,
        title="AI Regulation and Breakthroughs",
        description="Recent developments in AI regulation and breakthrough technologies",
        source_article_ids=seeded_articles,
        novelty_type="emerging",
        order_index=0,
    )
    theme = db.get_themes_for_run(seeded_run)[0]
    return theme


@pytest.fixture
def seeded_theme_approved(db, seeded_run, seeded_articles):
    """Insert a non-pending (approved) theme."""
    theme_id = db.insert_theme(
        pipeline_run_id=seeded_run,
        title="Approved Theme",
        description="This theme is already approved",
        source_article_ids=[seeded_articles[0]],
        novelty_type="established",
        order_index=1,
    )
    db.update_theme_status(theme_id, "approved")
    return db.get_themes_for_run(seeded_run)[1]  # second theme


@pytest.fixture
def seeded_theme_auto_approved(db, seeded_run, seeded_articles):
    """Insert an auto_approved theme."""
    theme_id = db.insert_theme(
        pipeline_run_id=seeded_run,
        title="Auto Approved Theme",
        description="This theme is auto-approved",
        source_article_ids=[seeded_articles[1]],
        novelty_type="emerging",
        order_index=2,
    )
    db.update_theme_status(theme_id, "auto_approved")
    return db.get_themes_for_run(seeded_run)[2]


@pytest.fixture
def seeded_theme_no_articles(db, seeded_run):
    """Insert a pending theme whose source article IDs don't match any articles."""
    theme_id = db.insert_theme(
        pipeline_run_id=seeded_run,
        title="Theme with Missing Articles",
        description="Articles were deleted",
        source_article_ids=[999, 998],
        novelty_type="emerging",
        order_index=0,
    )
    # Fetch the theme back — since it's the only theme, it's at index 0
    themes = db.get_themes_for_run(seeded_run)
    return themes[0]


@pytest.fixture
def full_seeded_env(db, seeded_run, seeded_feed, seeded_articles, seeded_theme_pending):
    """Convenience fixture returning all seeded context as a dict."""
    return {
        "db": db,
        "run_id": seeded_run,
        "feed_id": seeded_feed,
        "article_ids": seeded_articles,
        "theme": seeded_theme_pending,
    }


# ---------------------------------------------------------------------------
# German script isolation (CRITICAL)
# ---------------------------------------------------------------------------


class TestGermanScriptIsolation:
    """Verify the German script is generated natively without English script input.

    The ``script_de.txt`` prompt template MUST NOT contain a ``{script_en}``
    placeholder, and ``_generate_theme_deliverables`` MUST NOT pass ``script_en``
    in the ``fmt_kwargs`` when generating the German script.
    """

    def test_script_de_prompt_has_no_script_en_placeholder(self):
        """script_de.txt must NOT reference {script_en} in its template."""
        prompt_path = (
            __file__.rsplit("/", 2)[0] + "/prompts/script_de.txt"
            if "/tests/" in __file__
            else "prompts/script_de.txt"
        )
        import pathlib
        alt_path = pathlib.Path(__file__).parent.parent / "prompts" / "script_de.txt"
        content = alt_path.read_text(encoding="utf-8")
        assert "{script_en}" not in content, (
            "script_de.txt must NOT contain {script_en}. "
            "The German script must be generated natively from source articles "
            "and the English summary only."
        )

    def test_script_de_prompt_has_correct_placeholders(self):
        """script_de.txt must only have: theme_title, theme_description, summary_en, articles_text."""
        import pathlib
        alt_path = pathlib.Path(__file__).parent.parent / "prompts" / "script_de.txt"
        content = alt_path.read_text(encoding="utf-8")
        # Extract all {placeholder} names
        import re
        placeholders = set(re.findall(r"\{(\w+)\}", content))
        expected = {"theme_title", "theme_description", "summary_en", "articles_text"}
        assert placeholders == expected, (
            f"script_de.txt has placeholders {placeholders}, expected {expected}. "
            "The German script must NOT receive the English script as input."
        )

    def test_generate_theme_deliverables_omits_script_en_for_script_de(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Verify that _generate_theme_deliverables does NOT pass script_en in fmt_kwargs
        when generating script_de."""
        env = full_seeded_env
        theme = env["theme"]
        articles = _get_articles(env["db"], [1, 2, 3])

        # We'll spy on _generate_one to capture the fmt_kwargs for each call
        original_generate_one = _generate_one

        captured_kwargs = []

        def tracking_generate_one(llm_client, config, prompt_file, fmt_kwargs, deliverable_type, theme_id, target_words=None):
            captured_kwargs.append({
                "prompt_file": prompt_file,
                "deliverable_type": deliverable_type,
                "fmt_kwargs_keys": set(fmt_kwargs.keys()),
                "has_script_en": "script_en" in fmt_kwargs,
            })
            return original_generate_one(llm_client, config, prompt_file, fmt_kwargs, deliverable_type, theme_id, target_words=target_words)

        with patch("src.generator._generate_one", side_effect=tracking_generate_one):
            _generate_theme_deliverables(
                run_id=env["run_id"],
                db=env["db"],
                config=mock_config,
                llm_client=mock_llm,
                interest=interest,
                theme=theme,
                articles=articles,
                version=1,
            )

        # Find the script_de call
        script_de_call = next(
            (c for c in captured_kwargs if c["deliverable_type"] == "script_de"),
            None,
        )
        assert script_de_call is not None, "No script_de generation call was made"
        assert not script_de_call["has_script_en"], (
            "script_de fmt_kwargs must NOT contain 'script_en'. "
            "The German script should only receive theme_title, theme_description, "
            "summary_en, and articles_text."
        )
        assert "script_en" not in script_de_call["fmt_kwargs_keys"], (
            "fmt_kwargs for script_de must not include 'script_en' key"
        )
        expected_keys = {"theme_title", "theme_description", "summary_en", "articles_text"}
        assert script_de_call["fmt_kwargs_keys"] == expected_keys, (
            f"script_de fmt_kwargs keys are {script_de_call['fmt_kwargs_keys']}, "
            f"expected {expected_keys}"
        )


# ---------------------------------------------------------------------------
# _word_count
# ---------------------------------------------------------------------------


class TestWordCount:
    """Count words in text strings."""

    def test_simple_text(self):
        assert _word_count("hello world") == 2

    def test_multiple_spaces(self):
        assert _word_count("hello   world") == 2

    def test_leading_trailing_spaces(self):
        assert _word_count("  hello world  ") == 2

    def test_newlines_and_tabs(self):
        assert _word_count("hello\nworld\tfoo") == 3

    def test_empty_string(self):
        assert _word_count("") == 0

    def test_only_whitespace(self):
        assert _word_count("   \n\t  ") == 0

    def test_single_word(self):
        assert _word_count("hello") == 1

    def test_punctuation(self):
        assert _word_count("hello, world! how's it?") == 4


# ---------------------------------------------------------------------------
# _parse_article_ids
# ---------------------------------------------------------------------------


class TestParseArticleIds:
    """Parse JSON-encoded article ID arrays."""

    def test_simple_array(self):
        result = _parse_article_ids('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_single_element(self):
        result = _parse_article_ids('[42]')
        assert result == [42]

    def test_empty_array(self):
        result = _parse_article_ids('[]')
        assert result == []

    def test_no_spaces(self):
        result = _parse_article_ids('[1,2,3]')
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _build_articles_text
# ---------------------------------------------------------------------------


class TestBuildArticlesText:
    """Format article contents for LLM prompts."""

    def test_single_article_with_full_content(self):
        articles = [
            {"title": "Test Article", "full_content": "Some content here", "rss_excerpt": "Excerpt"},
        ]
        result = _build_articles_text(articles)
        assert "--- Article 1 ---" in result
        assert "Title: Test Article" in result
        assert "Some content here" in result
        assert "Excerpt" not in result  # full_content takes precedence

    def test_multiple_articles(self):
        articles = [
            {"title": "Article One", "full_content": "Content one"},
            {"title": "Article Two", "full_content": "Content two"},
        ]
        result = _build_articles_text(articles)
        assert "--- Article 1 ---" in result
        assert "Title: Article One" in result
        assert "--- Article 2 ---" in result
        assert "Title: Article Two" in result

    def test_fallback_to_rss_excerpt(self):
        articles = [
            {"title": "No Full Content", "full_content": None, "rss_excerpt": "RSS excerpt here"},
        ]
        result = _build_articles_text(articles)
        assert "RSS excerpt here" in result

    def test_fallback_when_full_content_empty_string(self):
        articles = [
            {"title": "Empty Full", "full_content": "", "rss_excerpt": "RSS fallback"},
        ]
        result = _build_articles_text(articles)
        assert "RSS fallback" in result

    def test_truncates_content_at_5000_chars(self):
        long_content = "a" * 6000
        articles = [
            {"title": "Long Article", "full_content": long_content},
        ]
        result = _build_articles_text(articles)
        assert len(result) < 6000 + 200  # truncated + header
        assert "... [truncated]" in result

    def test_boundary_5000_chars_not_truncated(self):
        content = "a" * 5000
        articles = [
            {"title": "Boundary", "full_content": content},
        ]
        result = _build_articles_text(articles)
        assert "... [truncated]" not in result

    def test_boundary_5001_chars_truncated(self):
        content = "a" * 5001
        articles = [
            {"title": "Just Over", "full_content": content},
        ]
        result = _build_articles_text(articles)
        assert "... [truncated]" in result

    def test_missing_both_content_and_excerpt(self):
        articles = [
            {"title": "No Content", "full_content": None},
        ]
        # When neither full_content nor rss_excerpt is available, the content
        # section is empty. If rss_excerpt key is absent, .get() returns "".
        result = _build_articles_text(articles)
        assert "Title: No Content" in result


# ---------------------------------------------------------------------------
# _get_articles
# ---------------------------------------------------------------------------


class TestGetArticles:
    """Fetch articles from the database by ID."""

    def test_fetches_existing_articles(self, db, seeded_run, seeded_feed):
        _insert_article(db, seeded_feed, seeded_run, 1, "Article 1", full_content="Content 1")
        _insert_article(db, seeded_feed, seeded_run, 2, "Article 2", full_content="Content 2")
        result = _get_articles(db, [1, 2])
        assert len(result) == 2
        assert result[0]["title"] == "Article 1"
        assert result[1]["title"] == "Article 2"

    def test_skips_missing_articles(self, db):
        result = _get_articles(db, [1, 2])
        assert result == []

    def test_mixed_existing_and_missing(self, db, seeded_run, seeded_feed):
        _insert_article(db, seeded_feed, seeded_run, 1, "Existing Article", full_content="Exists")
        result = _get_articles(db, [1, 999])
        assert len(result) == 1
        assert result[0]["title"] == "Existing Article"

    def test_empty_input_list(self, db):
        result = _get_articles(db, [])
        assert result == []


# ---------------------------------------------------------------------------
# run() integration tests
# ---------------------------------------------------------------------------


class TestRun:
    """Integration tests for the top-level ``run()`` function."""

    def test_generates_three_deliverables_per_theme(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Each pending theme should get summary_en, script_en, and script_de."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        theme_id = env["theme"]["id"]
        deliverables = env["db"].get_latest_deliverables(theme_id)
        assert set(deliverables.keys()) == {"summary_en", "script_en", "script_de"}

    def test_all_deliverables_have_version_1(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """First-generation deliverables should all be version 1."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        theme_id = env["theme"]["id"]
        deliverables = env["db"].get_latest_deliverables(theme_id)
        for dtype in ("summary_en", "script_en", "script_de"):
            assert deliverables[dtype]["version"] == 1, f"{dtype} should be version 1"

    def test_multiple_pending_themes_are_processed(self, db, mock_llm, mock_config, seeded_run, seeded_feed, seeded_articles, interest):
        """Multiple pending themes are all processed by run()."""
        # Insert two pending themes
        db.insert_theme(
            seeded_run, "Theme Alpha", "First theme",
            [seeded_articles[0]], "emerging", 0,
        )
        db.insert_theme(
            seeded_run, "Theme Beta", "Second theme",
            [seeded_articles[1]], "established", 1,
        )
        themes = db.get_themes_for_run(seeded_run)
        assert len(themes) == 2

        run(seeded_run, db, mock_config, mock_llm, interest)

        for theme in themes:
            deliverables = db.get_latest_deliverables(theme["id"])
            assert len(deliverables) == 3, f"Theme '{theme['title']}' missing deliverables"

    def test_skips_approved_themes(self, db, mock_llm, mock_config, seeded_run, seeded_feed, seeded_articles, interest):
        """Themes with status 'approved' should be skipped."""
        db.insert_theme(
            seeded_run, "Pending Theme", "Will be processed",
            [seeded_articles[0]], "emerging", 0,
        )
        theme_id_approved = db.insert_theme(
            seeded_run, "Approved Theme", "Will be skipped",
            [seeded_articles[1]], "established", 1,
        )
        db.update_theme_status(theme_id_approved, "approved")

        run(seeded_run, db, mock_config, mock_llm, interest)

        themes = db.get_themes_for_run(seeded_run)
        pending_deliverables = db.get_latest_deliverables(themes[0]["id"])
        assert len(pending_deliverables) == 3

        approved_deliverables = db.get_latest_deliverables(theme_id_approved)
        assert approved_deliverables == {}

    def test_skips_auto_approved_themes(self, db, mock_llm, mock_config, seeded_run, seeded_feed, seeded_articles, interest):
        """Themes with status 'auto_approved' should be skipped."""
        db.insert_theme(
            seeded_run, "Pending Theme", "Will be processed",
            [seeded_articles[0]], "emerging", 0,
        )
        theme_id_auto = db.insert_theme(
            seeded_run, "Auto Approved", "Will be skipped",
            [seeded_articles[1]], "emerging", 1,
        )
        db.update_theme_status(theme_id_auto, "auto_approved")

        run(seeded_run, db, mock_config, mock_llm, interest)

        auto_deliverables = db.get_latest_deliverables(theme_id_auto)
        assert auto_deliverables == {}

    def test_handles_missing_source_articles_gracefully(self, db, mock_llm, mock_config, seeded_run, seeded_theme_no_articles, interest):
        """A theme whose source articles don't exist in the DB should still
        generate deliverables (with empty articles text)."""
        run(seeded_run, db, mock_config, mock_llm, interest)
        deliverables = db.get_latest_deliverables(seeded_theme_no_articles["id"])
        # Should generate with empty articles_text
        assert set(deliverables.keys()) == {"summary_en", "script_en", "script_de"}

    def test_llm_called_with_correct_prompts(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Verify the LLM is called three times with correct prompt types."""
        env = full_seeded_env
        mock_llm.complete.return_value = "Generated content"

        run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        # Should have 3 calls to complete()
        assert mock_llm.complete.call_count == 3

        # Verify model and temperature from config are passed
        for call_args in mock_llm.complete.call_args_list:
            kwargs = call_args[1]
            assert kwargs["model_id"] == "test-model"
            assert kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# refine() tests
# ---------------------------------------------------------------------------


class TestRefine:
    """Tests for the ``refine()`` function."""

    def test_creates_new_versions_for_all_deliverable_types(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Refine should create version 2 for all 3 deliverable types."""
        env = full_seeded_env
        # First generate version 1
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        theme_id = env["theme"]["id"]

        # Reset mock call count for refine
        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Refined content here"

        # Refine
        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Fix the intro, add more context", interest)

        deliverables = env["db"].get_latest_deliverables(theme_id)
        for dtype in ("summary_en", "script_en", "script_de"):
            assert deliverables[dtype]["version"] == 2, f"{dtype} should be version 2"
            assert deliverables[dtype]["content"] == "Refined content here"

    def test_versions_increment_from_previous(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Refining multiple times should increment versions."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        # Refine once -> version 2
        mock_llm.complete.return_value = "Refined v2"
        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback 1", interest)

        deliverables = env["db"].get_latest_deliverables(theme_id)
        for dtype in ("summary_en", "script_en", "script_de"):
            assert deliverables[dtype]["version"] == 2

        # Refine again -> version 3
        mock_llm.complete.return_value = "Refined v3"
        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback 2", interest)

        deliverables = env["db"].get_latest_deliverables(theme_id)
        for dtype in ("summary_en", "script_en", "script_de"):
            assert deliverables[dtype]["version"] == 3

    def test_uses_refine_prompt_template(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Verify the refine call uses the refine.txt prompt."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Refined"

        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Test feedback", interest)

        # Each refine call should use the system prompt from refine.txt
        for call_args in mock_llm.complete.call_args_list:
            system_prompt = call_args[1]["system_prompt"]
            assert "revising scriptwriter" in system_prompt.lower() or "revise" in system_prompt.lower()

    def test_passes_evaluation_feedback_to_llm(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """The evaluation_feedback string should appear in the user prompt."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Refined"
        feedback = "The content lacks depth. Add more technical details and citations."

        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, feedback, interest)

        for call_args in mock_llm.complete.call_args_list:
            user_prompt = call_args[1]["user_prompt"]
            assert feedback in user_prompt, "evaluation_feedback must be passed to the LLM"

    def test_handles_missing_deliverable_type(self, db, mock_llm, mock_config, full_seeded_env, caplog, interest):
        """If a deliverable type is missing from latest, it is silently skipped
        (no warning needed since interest toggles already guard against this)."""
        env = full_seeded_env
        # Generate version 1 for all
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        # Delete the script_de deliverable to simulate missing type
        env["db"]._conn.execute("DELETE FROM deliverables WHERE theme_id = ? AND deliverable_type = 'script_de'", (theme_id,))
        env["db"]._conn.commit()

        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Refined"

        with caplog.at_level(logging.WARNING):
            refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback", interest)

        # Only 2 refine LLM calls should have been made (summary_en, script_en)
        assert mock_llm.complete.call_count == 2

        # script_de should still be absent from latest
        latest = env["db"].get_latest_deliverables(theme_id)
        assert "script_de" not in latest

    def test_no_existing_deliverables_logs_warning(self, db, mock_llm, mock_config, full_seeded_env, caplog, interest):
        """If no existing deliverables exist, refine logs a warning and returns."""
        env = full_seeded_env
        theme_id = env["theme"]["id"]

        with caplog.at_level(logging.WARNING):
            refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback", interest)

        assert any("No existing deliverables" in msg for msg in caplog.messages)
        assert mock_llm.complete.call_count == 0

    def test_refine_raises_for_unknown_theme(self, db, mock_llm, mock_config, seeded_run, interest):
        """Refining a theme not in the current run should raise GeneratorError."""
        # Create a theme in a different run with deliverables — the theme_id
        # exists in the DB but is not part of seeded_run's themes
        other_run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-13", "2026-05-13T06:00:00")
        theme_id = db.insert_theme(
            other_run_id, "Other Run Theme", "Desc", [1], "emerging", 0,
        )
        db.insert_deliverable(theme_id, "summary_en", "Some content", 1)

        with pytest.raises(GeneratorError, match=f"Theme {theme_id} not found"):
            refine(seeded_run, db, mock_config, mock_llm, theme_id, "Feedback", interest)

    def test_passes_correct_config_to_llm(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Refine calls should pass the same config (model, temperature) as normal generation."""
        env = full_seeded_env
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Refined"

        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback", interest)

        for call_args in mock_llm.complete.call_args_list:
            kwargs = call_args[1]
            assert kwargs["model_id"] == "test-model"
            assert kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """LLM client errors should be wrapped in GeneratorError."""

    def test_llm_error_during_generation_raises_generator_error(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """If the LLM client raises during generation, GeneratorError should be raised."""
        env = full_seeded_env
        mock_llm.complete.side_effect = RuntimeError("API connection failed")

        with pytest.raises(GeneratorError, match="LLM call failed for summary_en"):
            run(env["run_id"], env["db"], mock_config, mock_llm, interest)

    def test_llm_error_during_refinement_raises_generator_error(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """If the LLM client raises during refinement, GeneratorError should be raised."""
        env = full_seeded_env
        # First generate deliverables
        mock_llm.complete.return_value = "Generated content"
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)
        theme_id = env["theme"]["id"]

        # Now make the LLM fail during refine
        mock_llm.complete.side_effect = RuntimeError("Refinement API error")

        with pytest.raises(GeneratorError, match="Refinement LLM call failed for summary_en"):
            refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Feedback", interest)

    def test_llm_error_does_not_corrupt_existing_deliverables(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """If the LLM fails during generation of script_en, the summary_en should still be in DB."""
        env = full_seeded_env
        # First call succeeds, second call fails
        mock_llm.complete.side_effect = [
            "Summary content",  # summary_en succeeds
            RuntimeError("Script generation failed"),  # script_en fails
        ]

        with pytest.raises(GeneratorError):
            run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        # summary_en should still be inserted even though script_en failed
        theme_id = env["theme"]["id"]
        history = env["db"].get_deliverable_history(theme_id, "summary_en")
        assert len(history) == 1
        assert history[0]["content"] == "Summary content"

    def test_no_deliverables_on_first_call_failure(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """If the LLM fails on the first call (summary_en), no deliverables are inserted."""
        env = full_seeded_env
        mock_llm.complete.side_effect = RuntimeError("Immediate failure")

        with pytest.raises(GeneratorError):
            run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        # No deliverables should exist
        theme_id = env["theme"]["id"]
        latest = env["db"].get_latest_deliverables(theme_id)
        assert latest == {}


# ---------------------------------------------------------------------------
# _generate_one edge cases
# ---------------------------------------------------------------------------


class TestGenerateOne:
    """Edge cases for the internal _generate_one helper."""

    def test_malformed_template_raises_error(self, mock_llm, mock_config):
        """A prompt template without === USER === separator should raise GeneratorError."""
        import tempfile
        import pathlib

        # Create a temporary malformed template
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_prompt = tmpdir + "/bad.txt"
            with open(bad_prompt, "w") as f:
                f.write("=== SYSTEM ===\nJust a system prompt, no user section")

            with patch("src.generator._PROMPTS_DIR", pathlib.Path(tmpdir)):
                with pytest.raises(GeneratorError, match="bad.txt prompt template is malformed"):
                    _generate_one(
                        llm_client=mock_llm,
                        config=mock_config,
                        prompt_file="bad.txt",
                        fmt_kwargs={"theme_title": "Test"},
                        deliverable_type="summary_en",
                        theme_id=1,
                    )


# ---------------------------------------------------------------------------
# Full integration: run + refine pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of generate + refine workflow."""

    def test_generate_then_refine_workflow(self, db, mock_llm, mock_config, full_seeded_env, interest):
        """Simulate the full generate -> evaluate -> refine cycle."""
        env = full_seeded_env
        theme_id = env["theme"]["id"]

        # Phase 1: Generate
        mock_llm.complete.return_value = "Version 1 content"
        run(env["run_id"], env["db"], mock_config, mock_llm, interest)

        v1 = env["db"].get_latest_deliverables(theme_id)
        assert all(v["version"] == 1 for v in v1.values())

        # Phase 2: Refine
        mock_llm.complete.reset_mock()
        mock_llm.complete.return_value = "Version 2 content"
        refine(env["run_id"], env["db"], mock_config, mock_llm, theme_id, "Needs more examples", interest)

        v2 = env["db"].get_latest_deliverables(theme_id)
        assert all(v["version"] == 2 for v in v2.values())
        assert all(v["content"] == "Version 2 content" for v in v2.values())

        # Verify version 1 content is preserved in history
        for dtype in ("summary_en", "script_en", "script_de"):
            history = env["db"].get_deliverable_history(theme_id, dtype)
            assert len(history) == 2
            assert history[0]["version"] == 1
            assert history[1]["version"] == 2
