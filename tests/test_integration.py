"""Integration test — end-to-end pipeline with all external dependencies mocked.

Covers Phase 5 requirement: "Run the full pipeline with mocked external APIs to
verify DB state and email output."

Mocks:
  - ``feedparser.parse`` → returns 5 sample RSS entries
  - ``httpx.get`` + ``trafilatura.extract`` → returns full article content
  - ``LLMClient`` → returns predetermined responses for all stages
  - ``smtplib.SMTP`` → captures sent emails without real network I/O

Verifies:
  - Pipeline exits with code 0
  - Pipeline run status = ``'completed'``
  - Articles, themes, deliverables, evaluation rounds, daily brief stored in DB
  - Emails sent for brief + per-theme deliverables
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
import time
import yaml
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Fixed time near the mock feed entry timestamps (2026-05-14)
_FIXED_NOW = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _make_dt_mock() -> MagicMock:
    """Return a MagicMock that replaces ``datetime`` with time fixed to May 14, 2026."""
    m = MagicMock()
    m.now.return_value = _FIXED_NOW
    m.fromisoformat = datetime.fromisoformat
    m.fromtimestamp = datetime.fromtimestamp
    m.timedelta = timedelta
    m.timezone = timezone
    return m


# ===================================================================
# Helpers
# ===================================================================


def _make_config_dir(tmp_path, db_path: str | None = None) -> str:
    """Write domain-specific YAML config files into a subdirectory and return its path."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    if db_path is None:
        db_path = str(tmp_path / "integration_test.db")

    # Each top-level section gets its own file
    sections = {
        "feeds": {
            "news": [{"name": "AI News", "url": "https://example.com/ai-news"}],
            "commentators": [
                {"name": "AI Commentators", "url": "https://example.com/ai-commentators"}
            ],
        },
        "models": {
            "strong": {"id": "deepseek/deepseek-v4-pro", "temperature": 0.7},
            "weak": {"id": "deepseek/deepseek-v4-flash", "temperature": 0.7},
        },
        "pipeline": {
            "max_retries": 0,
            "max_refinement_rounds": 3,
            "retry_backoff_seconds": 1,
            "article_fetch_timeout_seconds": 15,
        },
        "email": {
            "recipient": "to@example.com",
            "sender": "from@example.com",
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "from@example.com",
            "smtp_password_env": "GMAIL_APP_PASSWORD",
        },
        "database": {"path": db_path},
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
        },
    }

    for filename, content in sections.items():
        with open(cfg_dir / f"{filename}.yaml", "w") as f:
            yaml.dump(content, f)

    return str(cfg_dir)


def _make_mock_feed() -> MagicMock:
    """Return a feedparser-like mock with 5 sample entries (all recent)."""
    feed = MagicMock()
    feed.bozo = 0
    now_ts = time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0))
    feed.entries = [
        {
            "title": f"AI News Article {i}",
            "link": f"https://example.com/article-{i}",
            "author": f"Author {i}",
            "summary": f"RSS excerpt for article {i}.",
            "published_parsed": now_ts,
        }
        for i in range(1, 6)
    ]
    return feed


def _make_smtp_mock():
    """Return a MagicMock acting as an SMTP context manager."""
    m = MagicMock()
    m.__enter__.return_value = m
    return m


# Pre-built LLM responses for each stage call.
# With 5 articles → 3 themes → 9 generator calls (3×3) → 6 evaluator
# calls (3×2, all passing) → 1 brief call = 17 total LLM calls.

_ANALYZER_RESPONSE = json.dumps(
    [
        {
            "title": "AI Breakthroughs in 2026",
            "description": "Major advances in AI reasoning and drug discovery.",
            "novelty_type": "novel",
            "source_article_indices": [0, 1],
        },
        {
            "title": "AI Regulation and Safety",
            "description": "New frameworks for AI governance and alignment research.",
            "novelty_type": "novel",
            "source_article_indices": [2, 3],
        },
        {
            "title": "AI Hardware Evolution",
            "description": "Next-gen AI chips and infrastructure scaling.",
            "novelty_type": "novel",
            "source_article_indices": [4],
        },
    ]
)

_GENERATOR_RESPONSE = (
    "This is generated content for the AI News Pipeline integration test. "
    "It contains enough words to pass word count thresholds. "
    "The content covers key developments in artificial intelligence, "
    "including model releases, regulatory changes, and hardware advances. "
    "Companies continue to push boundaries while regulators work to establish "
    "appropriate frameworks for safe and responsible AI deployment. "
    "More content here to ensure sufficient length for quality evaluation. "
    "Researchers across academia and industry are collaborating on new "
    "benchmarks and evaluation methods to better understand model capabilities. "
    "The rapid pace of innovation shows no signs of slowing down."
)

_QUALITY_PASS_RESPONSE = json.dumps(
    {
        "summary_en": {"pass": True, "feedback": "Well-written and comprehensive."},
        "script_en": {"pass": True, "feedback": "Engaging and well-structured."},
        "script_de": {"pass": True, "feedback": "Gut geschrieben, native quality."},
    }
)

_ADVERSARIAL_PASS_RESPONSE = json.dumps(
    {
        "pass": True,
        "feedback": "All factual claims verified against source articles.",
        "issues": [],
    }
)

_BRIEF_RESPONSE = (
    "This is the daily AI news brief for May 14, 2026. "
    "Today's AI landscape features three major developments. "
    "First, breakthroughs in AI reasoning and drug discovery continue "
    "to accelerate. Second, regulatory frameworks are taking shape with "
    "new governance models emerging globally. Third, hardware innovation "
    "is enabling the next generation of AI infrastructure. "
    "These trends signal a maturing AI ecosystem where capability advances "
    "are increasingly matched by governance and infrastructure developments."
)


def _infinite_llm_responses() -> list[str]:
    """Return enough LLM responses for all calls (with generous padding)."""
    return (
        [_ANALYZER_RESPONSE]  # 1
        + [_GENERATOR_RESPONSE] * 9  # 2–10
        + [_QUALITY_PASS_RESPONSE, _ADVERSARIAL_PASS_RESPONSE] * 5  # 11–20 (eval)
        + [_BRIEF_RESPONSE] * 2  # 21–22
    )



# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def smtp_mock():
    """Return a fresh SMTP mock for each test."""


@pytest.fixture
def mock_feed():
    """Return a feedparser mock with 5 sample entries."""
    return _make_mock_feed()


# ===================================================================
# Integration tests
# ===================================================================


class TestFullPipelineIntegration:
    """End-to-end pipeline run with all external services mocked."""

    def test_pipeline_completes_successfully(self, tmp_path):
        """Full pipeline runs and exits with code 0."""
        config_path = _make_config_dir(tmp_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        # Patch datetime to match entry timestamps (May 14, 2026)
        mock_dt = MagicMock()
        mock_dt.now.return_value = _FIXED_NOW
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.fromtimestamp = datetime.fromtimestamp
        mock_dt.timedelta = timedelta
        mock_dt.timezone = timezone

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full article content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full article content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 0

    def test_pipeline_run_status_completed(self, tmp_path):
        """After successful pipeline run, status is 'completed'."""
        config_path = _make_config_dir(tmp_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        # We need to intercept the Database to use a known path
        db_path = str(tmp_path / "test.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        # Open the DB and verify
        from src.db import Database

        db = Database(db_path)
        db.initialize_schema()
        runs = db._conn.execute("SELECT * FROM pipeline_runs ORDER BY id DESC").fetchall()
        run = dict(runs[0])
        assert run["status"] == "completed"
        assert run["completed_at"] is not None
        assert run["error_message"] is None
        db.close()

    def test_articles_stored_in_db(self, tmp_path):
        """All scraped articles are stored with correct fields."""
        db_path = str(tmp_path / "test_articles.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()

        # We need each article to have a unique normalized URL, so use
        # different links per entry.
        feed_news = MagicMock()
        feed_news.bozo = 0
        now_ts = time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0))
        feed_news.entries = [
            {
                "title": f"Article {i}",
                "link": f"https://example.com/a{i}",
                "author": f"Author {i}",
                "summary": f"RSS excerpt {i}.",
                "published_parsed": now_ts,
            }
            for i in range(1, 6)
        ]

        feed_comment = MagicMock()
        feed_comment.bozo = 0
        feed_comment.entries = [
            {
                "title": f"Commentary {i}",
                "link": f"https://example.com/c{i}",
                "author": f"Commentator {i}",
                "summary": f"Commentary excerpt {i}.",
                "published_parsed": now_ts,
            }
            for i in range(1, 3)
        ]

        # Patch datetime to match entry timestamps
        mock_dt = MagicMock()
        mock_dt.now.return_value = _FIXED_NOW
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.fromtimestamp = datetime.fromtimestamp
        mock_dt.timedelta = timedelta
        mock_dt.timezone = timezone

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch(
                "src.scraper.feedparser.parse",
                side_effect=[feed_news, feed_comment],
            ),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", mock_dt),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        from src.db import Database
        db = Database(db_path)
        articles = db.get_articles_for_run(1)
        assert len(articles) == 7  # 5 news + 2 commentator
        for art in articles:
            assert art["title"] is not None
            assert art["url"].startswith("https://example.com/")
            assert art["published_at"] is not None
            assert art["scraped_at"] is not None
            assert art["content_status"] == "full"
            assert art["full_content"] is not None
        db.close()

    def test_themes_and_deliverables_stored(self, tmp_path):
        """Themes identified by analyzer + deliverables generated are stored."""
        db_path = str(tmp_path / "test_themes.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        from src.db import Database
        db = Database(db_path)

        # Check themes
        themes = db.get_themes_for_run(1)
        assert len(themes) == 3
        for theme in themes:
            assert theme["title"]
            assert theme["description"]
            assert theme["novelty_type"] in ("novel", "continuation")
            assert theme["status"] in ("approved",)

        # Check deliverables — 3 per theme
        for theme in themes:
            deliverables = db.get_latest_deliverables(theme["id"])
            assert set(deliverables.keys()) == {"summary_en", "script_en", "script_de"}
            for dtype in ("summary_en", "script_en", "script_de"):
                assert len(deliverables[dtype]["content"]) > 0
                assert deliverables[dtype]["version"] == 1

        db.close()

    def test_daily_brief_stored(self, tmp_path):
        """Daily brief is generated and stored with correct word count."""
        db_path = str(tmp_path / "test_brief.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        from src.db import Database
        db = Database(db_path)

        rows = db._conn.execute(
            "SELECT * FROM daily_briefs WHERE pipeline_run_id = 1"
        ).fetchall()
        assert len(rows) == 1
        brief = dict(rows[0])
        assert brief["pipeline_run_id"] == 1
        assert len(brief["content"]) > 0
        expected_wc = len(brief["content"].split())
        assert brief["word_count"] == expected_wc
        db.close()

    def test_emails_sent_for_brief_and_themes(self, tmp_path):
        """Correct number of emails sent: 1 brief + N themes."""
        db_path = str(tmp_path / "test_emails.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        # Verify emails: 1 brief + 3 theme emails = 4
        assert smtp.send_message.call_count == 4

        subjects = [a[0][0]["Subject"] for a in smtp.send_message.call_args_list]
        brief_subjects = [s for s in subjects if "AI Daily Brief" in s]
        assert len(brief_subjects) == 1
        theme_subjects = [s for s in subjects if "AI Theme:" in s]
        assert len(theme_subjects) == 3

    def test_pipeline_idempotent_no_duplicate_articles(self, tmp_path):
        """Running the pipeline twice does not insert duplicate articles."""
        db_path = str(tmp_path / "test_idempotent.db")
        config_path = _make_config_dir(tmp_path, db_path=db_path)
        llm_mock = MagicMock()
        llm_mock.complete.side_effect = _infinite_llm_responses() + _infinite_llm_responses()
        llm_mock.close = MagicMock()
        smtp = _make_smtp_mock()
        mock_feed = _make_mock_feed()

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            # First run
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        # Second run — articles should be the same set
        # mock_feed has the same entries, but they were already stored.
        # The second run should skip them because the first run is now
        # a completed run, and the cutoff is `started_at` from that run.
        # But wait: all mock entries have `published_parsed` set to 2026-05-14,
        # and the first run's started_at was also around 2026-05-14.
        # So new entries won't be added. Let's verify.

        with (
            patch("src.llm.LLMClient", return_value=llm_mock),  # patch at source level
            patch("src.main_old.LLMClient", return_value=llm_mock),
            patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "sk-test",
                "GMAIL_APP_PASSWORD": "test-password",
            }),
            patch("src.main_old.setup_logging"),
            patch("src.scraper.feedparser.parse", return_value=mock_feed),
            patch("src.scraper.trafilatura.extract", return_value="Full content. " * 30),
            patch("src.scraper.httpx.get", return_value=_make_httpx_response("Full content. " * 30)),
            patch("src.scraper.datetime", _make_dt_mock()),
            patch("smtplib.SMTP", return_value=smtp),
            patch("time.sleep"),
            patch.object(sys, "argv", ["main.py", "--config", config_path]),
        ):
            from src.main_old import main

            with pytest.raises(SystemExit):
                main()

        from src.db import Database
        db = Database(db_path)

        # All articles should be associated with run 1 only
        run1_articles = db.get_articles_for_run(1)
        run2_articles = db.get_articles_for_run(2)
        assert len(run1_articles) > 0
        # Run 2 should have 0 new articles (all duplicates from run 1)
        assert len(run2_articles) == 0, (
            f"Expected 0 new articles in run 2, got {len(run2_articles)}. "
            f"Articles were not properly deduplicated across runs."
        )
        db.close()


# ===================================================================
# Helper: httpx mock response
# ===================================================================


def _make_httpx_response(text: str):
    """Build a mock httpx.Response with content."""
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    return resp


# ===================================================================
# Helper: feedparser mock with unique entries
# ===================================================================


def _make_feed_with_unique_entries(base_url: str, prefix: str, count: int) -> MagicMock:
    """Return a feedparser mock with *count* unique entries."""
    feed = MagicMock()
    feed.bozo = 0
    now_ts = time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0))
    feed.entries = [
        {
            "title": f"{prefix} {i}",
            "link": f"{base_url}/{i}",
            "author": f"Author {prefix} {i}",
            "summary": f"RSS excerpt {prefix} {i}.",
            "published_parsed": now_ts,
        }
        for i in range(1, count + 1)
    ]
    return feed
