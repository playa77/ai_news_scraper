"""Tests for the brief generator module.

Covers ``_word_count``, ``_build_themes_section``, and the ``run()``
orchestrator function with an in-memory SQLite database and a mocked
LLM client.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.brief import BriefError, _build_themes_section, _word_count, run
from src.db import Database
from src.models import InterestConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_BRIEF = (
    "This is a daily brief about AI news. It covers multiple themes "
    "including recent advances in large language models and their impact "
    "on enterprise workflows. Companies continue to invest heavily in "
    "infrastructure while researchers push the boundaries of reasoning."
)


@pytest.fixture
def db():
    """Create an in-memory database with schema initialized."""
    database = Database(":memory:")
    database.initialize_schema()
    yield database
    database.close()


@pytest.fixture
def mock_llm():
    """Return a MagicMock that behaves like LLMClient.complete()."""
    client = MagicMock()
    client.complete.return_value = _FAKE_BRIEF
    return client


@pytest.fixture
def mock_config():
    """Return a minimal mock of the Config object with .models.strong."""
    cfg = MagicMock()
    cfg.models.strong.id = "deepseek/deepseek-v4-pro"
    cfg.models.strong.temperature = 0.3
    return cfg


@pytest.fixture
def seeded_run(db):
    """Create a single pipeline run and return its id."""
    return db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")


@pytest.fixture
def seeded_theme(db, seeded_run):
    """Create a pipeline run with an approved theme that has a summary_en
    deliverable.  Returns a dict with keys ``db, run_id, theme_id``."""
    theme_id = db.insert_theme(
        seeded_run, "LLM Advances", "Advances in large language models",
        [1], "emerging", 0,
    )
    db.update_theme_status(theme_id, "approved")
    db.insert_deliverable(theme_id, "summary_en",
                          "LLMs are improving rapidly with new architectures.", 1)
    return {"db": db, "run_id": seeded_run, "theme_id": theme_id}


@pytest.fixture
def multi_theme_db(db, seeded_run):
    """Seed the database with three themes having different statuses and
    deliverables, plus a feed and articles for realism.

    Themes:
      * Theme A — approved, has summary_en
      * Theme B — auto_approved, has summary_en
      * Theme C — pending (should be excluded from brief)
    """
    # Insert a feed + articles so foreign keys are happy
    feed_id = db.upsert_feed(db.get_interest_by_name("AI")["id"], "https://example.com/rss", "AI News", "news")
    a1 = db.insert_article(feed_id, "https://example.com/art1", "Article 1",
                           None, "2026-05-14T07:00:00", "2026-05-14T07:05:00",
                           None, None, "full", seeded_run)
    a2 = db.insert_article(feed_id, "https://example.com/art2", "Article 2",
                           None, "2026-05-14T07:10:00", "2026-05-14T07:15:00",
                           None, None, "full", seeded_run)
    a3 = db.insert_article(feed_id, "https://example.com/art3", "Article 3",
                           None, "2026-05-14T07:20:00", "2026-05-14T07:25:00",
                           None, None, "full", seeded_run)

    # --- Theme A: approved with summary_en ---
    t1 = db.insert_theme(seeded_run, "LLM Advances",
                         "Advances in large language models",
                         [a1, a2], "emerging", 0)
    db.update_theme_status(t1, "approved")
    db.insert_deliverable(t1, "summary_en",
                          "LLMs are improving rapidly with new architectures "
                          "enabling longer context windows and better reasoning.", 1)

    # --- Theme B: auto_approved with summary_en ---
    t2 = db.insert_theme(seeded_run, "AI Regulation",
                         "New regulations for AI safety",
                         [a3], "established", 1)
    db.update_theme_status(t2, "auto_approved")
    db.insert_deliverable(t2, "summary_en",
                          "Governments worldwide are drafting frameworks "
                          "for AI safety and accountability.", 1)

    # --- Theme C: pending (excluded) ---
    t3 = db.insert_theme(seeded_run, "Quantum ML",
                         "Quantum machine learning breakthroughs",
                         [], "emerging", 2)
    # status remains 'pending' by default

    return {
        "db": db,
        "run_id": seeded_run,
        "feed_id": feed_id,
        "theme_approved": t1,
        "theme_auto_approved": t2,
        "theme_pending": t3,
    }


# ---------------------------------------------------------------------------
# _word_count
# ---------------------------------------------------------------------------

class TestWordCount:
    """Unit tests for the internal ``_word_count`` helper."""

    def test_counts_simple_text(self):
        """Standard sentence should return correct word count."""
        assert _word_count("hello world this is a test") == 6

    def test_returns_zero_for_empty_string(self):
        """Empty string should yield 0."""
        assert _word_count("") == 0

    def test_handles_multiple_spaces(self):
        """Extra whitespace between words should not inflate count."""
        assert _word_count("hello    world   test") == 3

    def test_handles_newlines(self):
        """Newlines and tabs should be treated as whitespace."""
        assert _word_count("hello\nworld\tfoo\n\nbar") == 4

    def test_single_word(self):
        """A single word returns count 1."""
        assert _word_count("hello") == 1

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace should be ignored."""
        assert _word_count("   hello world   ") == 2


# ---------------------------------------------------------------------------
# _build_themes_section
# ---------------------------------------------------------------------------

class TestBuildThemesSection:
    """Unit tests for the ``_build_themes_section`` helper."""

    def test_formats_themes_with_titles_and_summary(self, seeded_theme):
        """Builds section text containing theme title, type, and summary."""
        db = seeded_theme["db"]
        theme_id = seeded_theme["theme_id"]

        themes = db.get_themes_for_run(seeded_theme["run_id"])
        section = _build_themes_section(themes, db, InterestConfig(name="AI", id=1))

        assert "THEME: LLM Advances" in section
        assert "Type: emerging" in section
        assert "Summary: LLMs are improving rapidly" in section

    def test_includes_all_themes(self, multi_theme_db):
        """All themes passed in should appear in the output."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        themes = db.get_themes_for_run(run_id)
        # Only approved/auto_approved — but _build_themes_section doesn't filter,
        # it formats whatever it receives.
        section = _build_themes_section(themes, db, InterestConfig(name="AI", id=1))

        assert "THEME: LLM Advances" in section
        assert "THEME: AI Regulation" in section
        assert "THEME: Quantum ML" in section

    def test_handles_missing_summary_en(self, db, seeded_run):
        """Theme without summary_en deliverable gets empty Summary field."""
        theme_id = db.insert_theme(seeded_run, "Orphan Theme",
                                   "No deliverables yet",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")

        theme = db.get_themes_for_run(seeded_run)[0]
        section = _build_themes_section([theme], db, InterestConfig(name="AI", id=1))

        assert "THEME: Orphan Theme" in section
        assert "Summary: " in section  # present but empty

    def test_empty_themes_list(self, db):
        """Empty themes list produces empty string."""
        assert _build_themes_section([], db, InterestConfig(name="AI", id=1)) == ""

    def test_summary_uses_latest_version(self, db, seeded_run):
        """When multiple versions exist, the highest version is used."""
        theme_id = db.insert_theme(seeded_run, "Versioned Theme",
                                   "Testing version selection",
                                   [], "established", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "summary_en", "old version", 1)
        db.insert_deliverable(theme_id, "summary_en", "latest version", 2)

        theme = db.get_themes_for_run(seeded_run)[0]
        section = _build_themes_section([theme], db, InterestConfig(name="AI", id=1))

        assert "Summary: latest version" in section
        assert "Summary: old version" not in section

    def test_ignores_non_summary_en_deliverables(self, db, seeded_run):
        """Deliverables that are not summary_en type are ignored."""
        theme_id = db.insert_theme(seeded_run, "Social Theme",
                                   "Has only social posts",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "social_post", "some social content", 1)

        theme = db.get_themes_for_run(seeded_run)[0]
        section = _build_themes_section([theme], db, InterestConfig(name="AI", id=1))

        assert "THEME: Social Theme" in section
        assert "Summary: " in section  # empty because no summary_en


# ---------------------------------------------------------------------------
# run() — success cases
# ---------------------------------------------------------------------------

class TestRunSuccess:
    """Happy-path tests for the ``run()`` function."""

    def test_generates_brief_from_approved_themes(self, multi_theme_db, mock_llm,
                                                  mock_config):
        """Brief is generated when approved and auto_approved themes exist."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["model_id"] == "deepseek/deepseek-v4-pro"
        assert call_kwargs["temperature"] == 0.3
        assert "THEME: LLM Advances" in call_kwargs["user_prompt"]
        assert "THEME: AI Regulation" in call_kwargs["user_prompt"]
        # Pending theme should NOT appear in the prompt
        assert "THEME: Quantum ML" not in call_kwargs["user_prompt"]

    def test_brief_stored_with_correct_word_count(self, multi_theme_db, mock_llm,
                                                  mock_config):
        """The daily_briefs row stores the content and accurate word count."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        # Retrieve the brief that was just inserted
        # (there should be exactly one brief for this run)
        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row["content"] == _FAKE_BRIEF
        expected_wc = len(_FAKE_BRIEF.split())
        assert row["word_count"] == expected_wc

    def test_pipeline_run_stage_updated_to_brief(self, multi_theme_db, mock_llm,
                                                 mock_config):
        """The pipeline run's current_stage is updated to 'brief'."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        pr = db.get_pipeline_run(run_id)
        assert pr["current_stage"] == "brief"

    def test_only_approved_and_auto_approved_included(self, multi_theme_db, mock_llm,
                                                      mock_config):
        """Pending themes are excluded from the prompt."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        call_kwargs = mock_llm.complete.call_args[1]
        assert "THEME: Quantum ML" not in call_kwargs["user_prompt"]

    def test_multiple_approved_themes_all_included(self, multi_theme_db, mock_llm,
                                                   mock_config):
        """Both approved and auto_approved themes appear in the prompt."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        call_kwargs = mock_llm.complete.call_args[1]
        assert "THEME: LLM Advances" in call_kwargs["user_prompt"]
        assert "THEME: AI Regulation" in call_kwargs["user_prompt"]

    def test_single_approved_theme(self, seeded_theme, mock_llm, mock_config):
        """A single approved theme still generates a brief."""
        db = seeded_theme["db"]
        run_id = seeded_theme["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert "THEME: LLM Advances" in call_kwargs["user_prompt"]

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row["content"] == _FAKE_BRIEF

    def test_brief_content_not_empty(self, multi_theme_db, mock_llm, mock_config):
        """Brief stored in DB is not an empty string."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert len(row["content"]) > 0

    def test_word_count_matches_content(self, multi_theme_db, mock_llm, mock_config):
        """Verify word_count column accurately reflects content word count."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchone()
        actual_count = len(row["content"].split())
        assert row["word_count"] == actual_count

    def test_daily_brief_row_has_correct_pipeline_run_id(self, multi_theme_db,
                                                         mock_llm, mock_config):
        """The inserted daily_brief references the correct pipeline_run_id."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchone()
        assert row["pipeline_run_id"] == run_id


# ---------------------------------------------------------------------------
# run() — no approved themes
# ---------------------------------------------------------------------------

class TestRunNoApprovedThemes:
    """Behaviour when no themes are approved / auto_approved."""

    def test_no_approved_themes_generates_placeholder(self, db, seeded_run,
                                                      mock_llm, mock_config):
        """When all themes are pending, a placeholder brief is generated
        without calling the LLM."""
        theme_id = db.insert_theme(seeded_run, "Pending Theme",
                                   "Not approved yet",
                                   [], "emerging", 0)
        # explicitly pending — db.update_theme_status not called, stays 'pending'

        run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (seeded_run,)
        ).fetchone()
        assert row is not None
        assert "No new AI themes were identified" in row["content"]
        assert str(seeded_run) in row["content"]

    def test_no_themes_at_all_generates_placeholder(self, db, seeded_run,
                                                    mock_llm, mock_config):
        """When there are no themes at all, a placeholder brief is generated."""
        run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        mock_llm.complete.assert_not_called()

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (seeded_run,)
        ).fetchone()
        assert row is not None
        assert "No new AI themes were identified" in row["content"]

    def test_placeholder_word_count_is_accurate(self, db, seeded_run,
                                                mock_llm, mock_config):
        """Placeholder brief has correct word_count stored."""
        run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (seeded_run,)
        ).fetchone()
        expected = len(row["content"].split())
        assert row["word_count"] == expected

    def test_placeholder_stage_still_updated(self, db, seeded_run,
                                             mock_llm, mock_config):
        """Even with no themes, current_stage is still set to 'brief'."""
        run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        pr = db.get_pipeline_run(seeded_run)
        assert pr["current_stage"] == "brief"


# ---------------------------------------------------------------------------
# run() — error / edge cases
# ---------------------------------------------------------------------------

class TestRunErrors:
    """Error handling and edge cases in ``run()``."""

    def test_llm_error_raises_brief_error(self, db, seeded_run, mock_config):
        """When LLM.complete() raises, BriefError is propagated."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("API timeout")

        # Insert an approved theme so we hit the LLM path
        theme_id = db.insert_theme(seeded_run, "Test Theme", "Desc",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "summary_en", "Some content", 1)

        with pytest.raises(BriefError, match="LLM call for daily brief failed"):
            run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

    @patch("src.brief._PROMPTS_DIR")
    def test_malformed_prompt_template_raises_brief_error(self, mock_prompts_dir,
                                                          db, seeded_run,
                                                          mock_config, mock_llm):
        """If the prompt template lacks the '=== USER ===' delimiter,
        BriefError is raised."""
        # Create a fake prompts dir with a malformed brief.txt
        import pathlib
        import tempfile

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        malformed = tmpdir / "brief.txt"
        malformed.write_text(
            "=== SYSTEM ===\nSome system prompt without the USER delimiter\n",
            encoding="utf-8",
        )
        mock_prompts_dir.__truediv__.return_value = malformed

        theme_id = db.insert_theme(seeded_run, "Test Theme", "Desc",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "summary_en", "Some content", 1)

        with pytest.raises(BriefError, match="brief.txt prompt template is malformed"):
            run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

    def test_missing_deliverables_still_generates_brief(self, db, seeded_run,
                                                       mock_llm, mock_config):
        """An approved theme without any deliverables should still produce
        a brief (summary section will be empty)."""
        theme_id = db.insert_theme(seeded_run, "Bare Theme",
                                   "No deliverables at all",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        # No deliverables inserted

        run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args[1]
        assert "THEME: Bare Theme" in call_kwargs["user_prompt"]
        assert "Summary: " in call_kwargs["user_prompt"]

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (seeded_run,)
        ).fetchone()
        assert row is not None
        assert row["content"] == _FAKE_BRIEF

    def test_llm_client_error_is_wrapped(self, db, seeded_run, mock_config):
        """An LLMClientError from the LLM is wrapped in BriefError."""
        from src.llm import LLMClientError

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = LLMClientError("Rate limit exceeded")

        theme_id = db.insert_theme(seeded_run, "Test Theme", "Desc",
                                   [], "emerging", 0)
        db.update_theme_status(theme_id, "approved")
        db.insert_deliverable(theme_id, "summary_en", "Content", 1)

        with pytest.raises(BriefError, match="LLM call for daily brief failed"):
            run(seeded_run, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))


# ---------------------------------------------------------------------------
# run() — integration verification
# ---------------------------------------------------------------------------

class TestRunVerification:
    """Higher-level verification of the brief generation outcome."""

    def test_run_is_idempotent_for_stage_update(self, multi_theme_db, mock_llm,
                                                mock_config):
        """Calling run() twice updates stage to 'brief' both times (no crash)."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))
        # Reset mock call count for second invocation
        mock_llm.complete.reset_mock()
        # Need to re-set return_value since reset_mock clears it
        mock_llm.complete.return_value = _FAKE_BRIEF

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        pr = db.get_pipeline_run(run_id)
        assert pr["current_stage"] == "brief"

        # Two briefs should now exist for this run (one per call)
        rows = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = ?", (run_id,)
        ).fetchall()
        assert len(rows) == 2

    def test_system_prompt_contains_briefing_instructions(self, multi_theme_db,
                                                          mock_llm, mock_config):
        """The system prompt sent to the LLM includes briefing requirements."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        call_kwargs = mock_llm.complete.call_args[1]
        system = call_kwargs["system_prompt"]
        assert "executive briefing writer" in system.lower()
        assert "approximately 350 words" in system

    def test_user_prompt_contains_themes_section(self, multi_theme_db,
                                                 mock_llm, mock_config):
        """The user prompt includes the rendered themes section."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        call_kwargs = mock_llm.complete.call_args[1]
        user = call_kwargs["user_prompt"]
        assert "THEME: LLM Advances" in user
        assert "THEME: AI Regulation" in user

    def test_auto_approved_themes_included(self, multi_theme_db, mock_llm,
                                           mock_config):
        """auto_approved themes should be treated the same as approved."""
        db = multi_theme_db["db"]
        run_id = multi_theme_db["run_id"]

        run(run_id, db, mock_config, mock_llm, InterestConfig(name="AI", id=1))

        call_kwargs = mock_llm.complete.call_args[1]
        assert "THEME: AI Regulation" in call_kwargs["user_prompt"]
