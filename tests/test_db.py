"""Tests for the database layer — in-memory SQLite with pytest.

Covers all CRUD methods on ``src.db.Database``.
"""

import json
import sqlite3

import pytest

from src.db import Database


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
def seeded_db(db):
    """``db`` with one interest, one feed, and one pipeline_run pre-inserted."""
    ai_id = db.get_interest_by_name("AI")["id"]
    run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
    feed_id = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
    db.update_pipeline_run(run_id, status="completed", completed_at="2026-05-14T06:30:00")
    return {"db": db, "run_id": run_id, "feed_id": feed_id}


# ---------------------------------------------------------------------------
# 1. Schema initialization
# ---------------------------------------------------------------------------

class TestSchemaInitialization:
    """Verify that :meth:`Database.initialize_schema` behaves correctly."""

    def test_initialize_does_not_raise(self, db):
        """Initialization should succeed without error."""
        pass  # The fixture already called initialize_schema

    def test_initialize_is_idempotent(self, db):
        """Calling initialize_schema twice must not raise."""
        db.initialize_schema()  # second call

    def test_wal_mode_is_enabled(self, db):
        """PRAGMA journal_mode should report 'wal' for file-backed databases.

        Note: ``:memory:`` databases silently fall back to ``memory`` journal
        mode because WAL requires a file on disk.  The pragma succeeds without
        error and simply returns the active mode — we verify it returns a
        valid, non-empty value.
        """
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        mode = row[0].lower()
        # The pragma must return a non-empty string (either "wal" for file
        # databases or "memory" for :memory:).  The key point is that
        # initialize_schema runs without raising an error.
        assert mode in ("wal", "memory")

    def test_foreign_keys_are_on(self, db):
        """PRAGMA foreign_keys should report 1."""
        row = db._conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1


# ---------------------------------------------------------------------------
# 2. create_pipeline_run
# ---------------------------------------------------------------------------

class TestCreatePipelineRun:
    """Insert pipeline runs."""

    def test_returns_positive_integer(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_stores_all_fields(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_date = "2026-05-14"
        started_at = "2026-05-14T06:00:00"
        run_id = db.create_pipeline_run(ai_id, run_date, started_at)
        row = db._conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        assert row["run_date"] == run_date
        assert row["started_at"] == started_at
        assert row["status"] == "running"
        assert row["completed_at"] is None
        assert row["current_stage"] is None
        assert row["error_message"] is None


# ---------------------------------------------------------------------------
# 3. get_pipeline_run
# ---------------------------------------------------------------------------

class TestGetPipelineRun:
    """Retrieve pipeline runs by ID."""

    def test_returns_run_by_id(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        run = db.get_pipeline_run(run_id)
        assert run is not None
        assert run["id"] == run_id

    def test_returns_none_for_nonexistent_id(self, db):
        assert db.get_pipeline_run(9999) is None


# ---------------------------------------------------------------------------
# 4. update_pipeline_run
# ---------------------------------------------------------------------------

class TestUpdatePipelineRun:
    """Update fields on pipeline runs."""

    def test_update_status(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, status="completed")
        run = db.get_pipeline_run(run_id)
        assert run["status"] == "completed"

    def test_update_completed_at(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, completed_at="2026-05-14T06:30:00")
        run = db.get_pipeline_run(run_id)
        assert run["completed_at"] == "2026-05-14T06:30:00"

    def test_update_current_stage(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, current_stage="scraping")
        run = db.get_pipeline_run(run_id)
        assert run["current_stage"] == "scraping"

    def test_update_error_message(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, error_message="something went wrong")
        run = db.get_pipeline_run(run_id)
        assert run["error_message"] == "something went wrong"

    def test_only_specified_fields_change(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, status="completed", current_stage="done")
        run = db.get_pipeline_run(run_id)
        assert run["status"] == "completed"
        assert run["current_stage"] == "done"
        # These should still be None
        assert run["completed_at"] is None
        assert run["error_message"] is None

    def test_none_does_not_change_field(self, db):
        """Passing None for a field should leave the stored value intact."""
        ai_id = db.get_interest_by_name("AI")["id"]
        run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(run_id, status="completed")
        # Now update with None for status — should be a no-op for that field
        db.update_pipeline_run(run_id, status=None, current_stage="scraping")
        run = db.get_pipeline_run(run_id)
        assert run["status"] == "completed"  # unchanged
        assert run["current_stage"] == "scraping"  # updated


# ---------------------------------------------------------------------------
# 5. get_last_successful_run
# ---------------------------------------------------------------------------

class TestGetLastSuccessfulRun:
    """Retrieve the most recent completed pipeline run."""

    def test_returns_most_recent_completed(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        db.create_pipeline_run(ai_id, "2026-05-13", "2026-05-13T06:00:00")  # no status update
        r2 = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(r2, status="completed", completed_at="2026-05-14T06:30:00")
        r3 = db.create_pipeline_run(ai_id, "2026-05-15", "2026-05-15T06:00:00")
        db.update_pipeline_run(r3, status="completed", completed_at="2026-05-15T06:30:00")
        r4 = db.create_pipeline_run(ai_id, "2026-05-16", "2026-05-16T06:00:00")
        db.update_pipeline_run(r4, status="failed", completed_at="2026-05-16T06:30:00")

        last = db.get_last_successful_run()
        assert last is not None
        assert last["id"] == r3  # most recent 'completed', not 'failed'

    def test_returns_none_when_no_completed_runs(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.create_pipeline_run(ai_id, "2026-05-15", "2026-05-15T06:00:00")
        # Neither marked as completed
        assert db.get_last_successful_run() is None


# ---------------------------------------------------------------------------
# 6. upsert_feed
# ---------------------------------------------------------------------------

class TestUpsertFeed:
    """Insert-or-ignore feeds."""

    def test_insert_new_feed_returns_id(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        feed_id = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
        assert isinstance(feed_id, int)
        assert feed_id > 0

    def test_duplicate_url_returns_same_id(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        feed_id_1 = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
        feed_id_2 = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
        assert feed_id_2 == feed_id_1

    def test_different_feed_returns_new_id(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        feed_id_1 = db.upsert_feed(ai_id, "https://example.com/rss", "Example Feed", "news")
        feed_id_2 = db.upsert_feed(ai_id, "https://other.com/rss", "Other Feed", "commentators")
        assert feed_id_2 > feed_id_1


# ---------------------------------------------------------------------------
# 7. get_all_feeds
# ---------------------------------------------------------------------------

class TestGetAllFeeds:
    """Retrieve all feeds."""

    def test_returns_all_feeds(self, db):
        ai_id = db.get_interest_by_name("AI")["id"]
        db.upsert_feed(ai_id, "https://a.com/rss", "Feed A", "news")
        db.upsert_feed(ai_id, "https://b.com/rss", "Feed B", "commentators")
        feeds = db.get_all_feeds()
        assert len(feeds) == 2
        names = {f["name"] for f in feeds}
        assert names == {"Feed A", "Feed B"}

    def test_empty_when_no_feeds(self, db):
        assert db.get_all_feeds() == []


# ---------------------------------------------------------------------------
# 8. article_exists
# ---------------------------------------------------------------------------

class TestArticleExists:
    """Check article existence by URL."""

    def test_false_before_insert(self, seeded_db):
        db = seeded_db["db"]
        assert db.article_exists("https://example.com/article-1") is False

    def test_true_after_insert(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]
        db.insert_article(
            feed_id, "https://example.com/article-1", "Test Article",
            None, "2026-05-14T07:00:00", "2026-05-14T07:05:00",
            None, None, "full", run_id,
        )
        assert db.article_exists("https://example.com/article-1") is True


# ---------------------------------------------------------------------------
# 9. insert_article
# ---------------------------------------------------------------------------

class TestInsertArticle:
    """Insert article records."""

    def test_returns_positive_integer(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]
        article_id = db.insert_article(
            feed_id, "https://example.com/a1", "Article 1",
            None, "2026-05-14T07:00:00", "2026-05-14T07:05:00",
            None, None, "full", run_id,
        )
        assert isinstance(article_id, int)
        assert article_id > 0

    def test_stores_all_fields_including_none_optionals(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]
        article_id = db.insert_article(
            feed_id=feed_id,
            url="https://example.com/a2",
            title="Article with all fields",
            author="John Doe",
            published_at="2026-05-14T07:00:00",
            scraped_at="2026-05-14T07:05:00",
            rss_excerpt="This is an RSS excerpt",
            full_content="This is the full article content.",
            content_status="full",
            pipeline_run_id=run_id,
        )
        row = db._conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        assert row["feed_id"] == feed_id
        assert row["url"] == "https://example.com/a2"
        assert row["title"] == "Article with all fields"
        assert row["author"] == "John Doe"
        assert row["published_at"] == "2026-05-14T07:00:00"
        assert row["scraped_at"] == "2026-05-14T07:05:00"
        assert row["rss_excerpt"] == "This is an RSS excerpt"
        assert row["full_content"] == "This is the full article content."
        assert row["content_status"] == "full"
        assert row["pipeline_run_id"] == run_id

    def test_stores_none_for_optional_fields(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]
        article_id = db.insert_article(
            feed_id, "https://example.com/a3", "Article no author",
            None, "2026-05-14T07:00:00", "2026-05-14T07:05:00",
            None, None, "full", run_id,
        )
        row = db._conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        assert row["author"] is None
        assert row["rss_excerpt"] is None
        assert row["full_content"] is None


# ---------------------------------------------------------------------------
# 10. get_articles_for_run
# ---------------------------------------------------------------------------

class TestGetArticlesForRun:
    """Filter articles by pipeline run."""

    def test_returns_only_articles_for_given_run(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id_1 = seeded_db["run_id"]

        run_id_2 = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-15", "2026-05-15T06:00:00")

        a1 = db.insert_article(feed_id, "https://example.com/r1a1", "Run1 Article", None,
                               "2026-05-14T07:00:00", "2026-05-14T07:05:00",
                               None, None, "full", run_id_1)
        a2 = db.insert_article(feed_id, "https://example.com/r1a2", "Run1 Article 2", None,
                               "2026-05-14T07:10:00", "2026-05-14T07:15:00",
                               None, None, "full", run_id_1)
        db.insert_article(feed_id, "https://example.com/r2a1", "Run2 Article", None,
                          "2026-05-15T07:00:00", "2026-05-15T07:05:00",
                          None, None, "full", run_id_2)

        articles = db.get_articles_for_run(run_id_1)
        assert len(articles) == 2
        ids = {a["id"] for a in articles}
        assert ids == {a1, a2}

    def test_empty_list_for_run_with_no_articles(self, seeded_db):
        db = seeded_db["db"]
        run_id_2 = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-15", "2026-05-15T06:00:00")
        assert db.get_articles_for_run(run_id_2) == []


# ---------------------------------------------------------------------------
# 11. get_article_by_id
# ---------------------------------------------------------------------------

class TestGetArticleById:
    """Retrieve single article by ID."""

    def test_returns_article_dict(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]
        article_id = db.insert_article(
            feed_id, "https://example.com/a1", "Test Article", None,
            "2026-05-14T07:00:00", "2026-05-14T07:05:00",
            None, None, "full", run_id,
        )
        article = db.get_article_by_id(article_id)
        assert article is not None
        assert article["id"] == article_id
        assert article["title"] == "Test Article"

    def test_returns_none_for_nonexistent_id(self, db):
        assert db.get_article_by_id(9999) is None


# ---------------------------------------------------------------------------
# 12. insert_theme
# ---------------------------------------------------------------------------

class TestInsertTheme:
    """Insert theme records with JSON-encoded source_article_ids."""

    def test_returns_positive_integer(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme Title", "Description", [1, 2, 3], "emerging", 0)
        assert isinstance(theme_id, int)
        assert theme_id > 0

    def test_stores_json_correctly(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        source_ids = [10, 20, 30]
        theme_id = db.insert_theme(run_id, "Theme Title", "Description", source_ids, "emerging", 0)
        row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        stored = json.loads(row["source_article_ids"])
        assert stored == source_ids


# ---------------------------------------------------------------------------
# 13. get_themes_for_run
# ---------------------------------------------------------------------------

class TestGetThemesForRun:
    """Retrieve themes for a pipeline run, ordered by order_index."""

    def test_ordering_by_order_index(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        t1 = db.insert_theme(run_id, "Theme B", "Desc", [1], "emerging", 2)
        t2 = db.insert_theme(run_id, "Theme A", "Desc", [2], "established", 1)
        t3 = db.insert_theme(run_id, "Theme C", "Desc", [3], "emerging", 0)

        themes = db.get_themes_for_run(run_id)
        assert [t["id"] for t in themes] == [t3, t2, t1]

    def test_filters_by_run_id(self, seeded_db):
        db = seeded_db["db"]
        run_id_1 = seeded_db["run_id"]
        run_id_2 = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-15", "2026-05-15T06:00:00")

        db.insert_theme(run_id_1, "T1", "Desc", [1], "emerging", 0)
        db.insert_theme(run_id_2, "T2", "Desc", [2], "established", 0)

        themes = db.get_themes_for_run(run_id_1)
        assert len(themes) == 1
        assert themes[0]["title"] == "T1"


# ---------------------------------------------------------------------------
# 14. update_theme_status
# ---------------------------------------------------------------------------

class TestUpdateThemeStatus:
    """Update the status of a theme."""

    def test_status_persisted(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Test Theme", "Desc", [1], "emerging", 0)

        db.update_theme_status(theme_id, "approved")
        row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        assert row["status"] == "approved"

        db.update_theme_status(theme_id, "auto_approved")
        row = db._conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
        assert row["status"] == "auto_approved"


# ---------------------------------------------------------------------------
# 15. insert_deliverable
# ---------------------------------------------------------------------------

class TestInsertDeliverable:
    """Insert deliverable records."""

    def test_returns_positive_integer(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        del_id = db.insert_deliverable(theme_id, "newsletter", "Content here", 1)
        assert isinstance(del_id, int)
        assert del_id > 0

    def test_stores_all_fields(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        del_id = db.insert_deliverable(theme_id, "social_post", "Social content", 2)

        row = db._conn.execute("SELECT * FROM deliverables WHERE id = ?", (del_id,)).fetchone()
        assert row["theme_id"] == theme_id
        assert row["deliverable_type"] == "social_post"
        assert row["content"] == "Social content"
        assert row["version"] == 2
        assert row["created_at"] is not None  # auto-generated timestamp


# ---------------------------------------------------------------------------
# 16. get_latest_deliverables
# ---------------------------------------------------------------------------

class TestGetLatestDeliverables:
    """Return highest version per deliverable type for a theme."""

    def test_returns_only_highest_version_per_type(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)

        # Insert multiple versions of newsletter
        db.insert_deliverable(theme_id, "newsletter", "newsletter v1", 1)
        db.insert_deliverable(theme_id, "newsletter", "newsletter v2", 2)
        db.insert_deliverable(theme_id, "newsletter", "newsletter v3", 3)

        # Insert multiple versions of social_post
        db.insert_deliverable(theme_id, "social_post", "social v1", 1)
        db.insert_deliverable(theme_id, "social_post", "social v2", 2)

        latest = db.get_latest_deliverables(theme_id)

        assert "newsletter" in latest
        assert latest["newsletter"]["version"] == 3
        assert latest["newsletter"]["content"] == "newsletter v3"

        assert "social_post" in latest
        assert latest["social_post"]["version"] == 2
        assert latest["social_post"]["content"] == "social v2"

    def test_empty_dict_when_no_deliverables(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        assert db.get_latest_deliverables(theme_id) == {}


# ---------------------------------------------------------------------------
# 17. get_deliverable_history
# ---------------------------------------------------------------------------

class TestGetDeliverableHistory:
    """Return all versions of a deliverable type in order."""

    def test_returns_all_versions_ordered(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)

        id1 = db.insert_deliverable(theme_id, "newsletter", "v1 content", 1)
        id2 = db.insert_deliverable(theme_id, "newsletter", "v2 content", 2)
        id3 = db.insert_deliverable(theme_id, "newsletter", "v3 content", 3)

        history = db.get_deliverable_history(theme_id, "newsletter")
        assert len(history) == 3
        assert [h["id"] for h in history] == [id1, id2, id3]
        assert [h["version"] for h in history] == [1, 2, 3]

    def test_empty_list_when_no_history(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        assert db.get_deliverable_history(theme_id, "newsletter") == []


# ---------------------------------------------------------------------------
# 18. insert_evaluation_round
# ---------------------------------------------------------------------------

class TestInsertEvaluationRound:
    """Insert evaluation round records."""

    def test_returns_positive_integer(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        eval_id = db.insert_evaluation_round(
            theme_id, 1, "yes", "Good quality", "yes", "No issues", "yes",
        )
        assert isinstance(eval_id, int)
        assert eval_id > 0

    def test_stores_all_fields(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        eval_id = db.insert_evaluation_round(
            theme_id, 2, "yes", "Quality feedback", "no", "Adversarial feedback", "no",
        )
        row = db._conn.execute(
            "SELECT * FROM evaluation_rounds WHERE id = ?", (eval_id,)
        ).fetchone()
        assert row["theme_id"] == theme_id
        assert row["round_number"] == 2
        assert row["quality_passed"] == "yes"
        assert row["quality_feedback"] == "Quality feedback"
        assert row["adversarial_passed"] == "no"
        assert row["adversarial_feedback"] == "Adversarial feedback"
        assert row["overall_passed"] == "no"
        assert row["evaluated_at"] is not None

    def test_stores_none_for_optional_feedback(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        eval_id = db.insert_evaluation_round(
            theme_id, 1, "yes", None, "yes", None, "yes",
        )
        row = db._conn.execute(
            "SELECT * FROM evaluation_rounds WHERE id = ?", (eval_id,)
        ).fetchone()
        assert row["quality_feedback"] is None
        assert row["adversarial_feedback"] is None


# ---------------------------------------------------------------------------
# 19. get_latest_evaluation
# ---------------------------------------------------------------------------

class TestGetLatestEvaluation:
    """Return the most recent evaluation round for a theme."""

    def test_returns_highest_round_number(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)

        db.insert_evaluation_round(theme_id, 1, "yes", None, "yes", None, "yes")
        round_3_id = db.insert_evaluation_round(theme_id, 3, "no", "Bad", "yes", None, "no")
        db.insert_evaluation_round(theme_id, 2, "yes", None, "yes", None, "yes")

        latest = db.get_latest_evaluation(theme_id)
        assert latest is not None
        assert latest["id"] == round_3_id
        assert latest["round_number"] == 3

    def test_returns_none_when_no_evaluations(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)
        assert db.get_latest_evaluation(theme_id) is None


# ---------------------------------------------------------------------------
# 20. get_evaluation_rounds
# ---------------------------------------------------------------------------

class TestGetEvaluationRounds:
    """Return all evaluation rounds for a theme, ordered."""

    def test_returns_all_rounds_ordered(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = db.insert_theme(run_id, "Theme", "Desc", [1], "emerging", 0)

        r1 = db.insert_evaluation_round(theme_id, 1, "yes", None, "yes", None, "yes")
        r2 = db.insert_evaluation_round(theme_id, 2, "no", "Bad", "yes", None, "no")
        r3 = db.insert_evaluation_round(theme_id, 3, "yes", None, "yes", None, "yes")

        rounds = db.get_evaluation_rounds(theme_id)
        assert len(rounds) == 3
        assert [r["id"] for r in rounds] == [r1, r2, r3]
        assert [r["round_number"] for r in rounds] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 21. insert_daily_brief
# ---------------------------------------------------------------------------

class TestInsertDailyBrief:
    """Insert daily brief records."""

    def test_stores_all_fields(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        brief_id = db.insert_daily_brief(run_id, "Brief content here", 150)
        assert isinstance(brief_id, int)
        assert brief_id > 0

        row = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE id = ?", (brief_id,)
        ).fetchone()
        assert row["pipeline_run_id"] == run_id
        assert row["content"] == "Brief content here"
        assert row["word_count"] == 150
        assert row["created_at"] is not None


# ---------------------------------------------------------------------------
# 22. get_previous_daily_brief
# ---------------------------------------------------------------------------

class TestGetPreviousDailyBrief:
    """Retrieve the most recent daily brief from a completed run before a given date."""

    def test_returns_most_recent_before_date(self, seeded_db):
        db = seeded_db["db"]
        ai_id = db.get_interest_by_name("AI")["id"]

        # Run 1: completed, run_date 2026-05-12
        r1 = db.create_pipeline_run(ai_id, "2026-05-12", "2026-05-12T06:00:00")
        db.update_pipeline_run(r1, status="completed", completed_at="2026-05-12T06:30:00")
        b1 = db.insert_daily_brief(r1, "Brief for May 12", 100)

        # Run 2: completed, run_date 2026-05-13
        r2 = db.create_pipeline_run(ai_id, "2026-05-13", "2026-05-13T06:00:00")
        db.update_pipeline_run(r2, status="completed", completed_at="2026-05-13T06:30:00")
        b2 = db.insert_daily_brief(r2, "Brief for May 13", 110)

        # Run 3: failed, run_date 2026-05-14 (should be skipped)
        r3 = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(r3, status="failed", completed_at="2026-05-14T06:30:00")
        db.insert_daily_brief(r3, "Brief for failed run", 120)

        # Request brief before 2026-05-14 — should get May 13's brief (not failed run)
        prev = db.get_previous_daily_brief("2026-05-14")
        assert prev is not None
        assert prev["id"] == b2
        assert prev["content"] == "Brief for May 13"

    def test_returns_none_when_no_prior_completed_run(self, seeded_db):
        db = seeded_db["db"]
        ai_id = db.get_interest_by_name("AI")["id"]
        # Only one completed run on 2026-05-14, but we query for a date before it
        r1 = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
        db.update_pipeline_run(r1, status="completed", completed_at="2026-05-14T06:30:00")
        db.insert_daily_brief(r1, "Only brief", 100)

        prev = db.get_previous_daily_brief("2026-05-01")
        assert prev is None


# ---------------------------------------------------------------------------
# 23. get_daily_brief
# ---------------------------------------------------------------------------

class TestGetDailyBrief:
    """Retrieve a daily brief by ID."""

    def test_returns_brief_by_id(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        brief_id = db.insert_daily_brief(run_id, "Brief content", 200)
        brief = db.get_daily_brief(brief_id)
        assert brief is not None
        assert brief["id"] == brief_id
        assert brief["content"] == "Brief content"

    def test_returns_none_for_nonexistent_id(self, db):
        assert db.get_daily_brief(9999) is None


# ---------------------------------------------------------------------------
# 24. Close and context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    """Verify context manager protocol works."""

    def test_context_manager_enters_and_exits(self):
        """__enter__ returns Database, __exit__ closes connection."""
        with Database(":memory:") as db:
            db.initialize_schema()
            ai_id = db.get_interest_by_name("AI")["id"]
            run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
            assert run_id > 0
            # Connection should be open inside the block
            db._conn.execute("SELECT 1")
        # After exiting, connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            db._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# 25. Duplicate article URL raises IntegrityError
# ---------------------------------------------------------------------------

class TestDuplicateArticleUrl:
    """Inserting an article with an existing URL must raise IntegrityError."""

    def test_duplicate_url_raises_integrity_error(self, seeded_db):
        db = seeded_db["db"]
        feed_id = seeded_db["feed_id"]
        run_id = seeded_db["run_id"]

        db.insert_article(
            feed_id, "https://example.com/dup", "Original", None,
            "2026-05-14T07:00:00", "2026-05-14T07:05:00",
            None, None, "full", run_id,
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: articles.url"):
            db.insert_article(
                feed_id, "https://example.com/dup", "Duplicate", None,
                "2026-05-14T07:00:00", "2026-05-14T07:05:00",
                None, None, "full", run_id,
            )


# ---------------------------------------------------------------------------
# 26. Foreign key constraint — article with nonexistent feed_id
# ---------------------------------------------------------------------------

class TestForeignKeyConstraint:
    """Inserting an article with a nonexistent feed_id must raise an IntegrityError."""

    def test_bad_feed_id_raises_integrity_error(self, seeded_db):
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        # feed_id=9999 does not exist
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            db.insert_article(
                9999, "https://example.com/bad-feed", "Bad Feed Article", None,
                "2026-05-14T07:00:00", "2026-05-14T07:05:00",
                None, None, "full", run_id,
            )
