"""Tests for the RSS scraper — feed parsing, article extraction, deduplication.

Covers all helper functions (``_normalize_url``, ``_extract_published``,
``_parse_iso8601``, ``_extract_article``) and the main ``run()`` entry point.
"""

from __future__ import annotations

import time
from datetime import datetime as real_datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.db import Database
from src.models import (
    Config,
    DatabaseConfig,
    EmailConfig,
    FeedDef,
    FeedsConfig,
    InterestConfig,
    ModelDef,
    ModelsConfig,
    OpenRouterConfig,
    PipelineConfig,
)
from src.scraper import (
    _extract_article,
    _extract_published,
    _normalize_url,
    _parse_iso8601,
    run,
)


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
def config():
    """Return a minimal valid :class:`Config` with one news and one commentator feed."""
    return Config(
        feeds=FeedsConfig(
            news=[FeedDef(name="AI News", url="https://example.com/ai-news")],
            commentators=[
                FeedDef(name="AI Commentators", url="https://example.com/ai-commentators")
            ],
        ),
        models=ModelsConfig(
            strong=ModelDef(id="gpt-4", temperature=0.7),
            weak=ModelDef(id="gpt-3.5", temperature=0.5),
        ),
        pipeline=PipelineConfig(article_fetch_timeout_seconds=15),
        email=EmailConfig(
            recipient="t@t.com",
            sender="s@s.com",
            smtp_host="smtp.t.com",
            smtp_port=587,
            smtp_user="u",
            smtp_password_env="P",
        ),
        database=DatabaseConfig(path=":memory:"),
        openrouter=OpenRouterConfig(
            api_key_env="K",
            base_url="https://api.example.com/v1",
        ),
    )


@pytest.fixture
def mock_feed_entry():
    """Return a standard feedparser-like entry dict (one item from an RSS feed)."""
    return {
        "title": "Test Article",
        "link": "https://example.com/article?utm_source=test#ref",
        "author": "Test Author",
        "summary": "This is an RSS excerpt for testing purposes.",
        "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
    }


@pytest.fixture
def mock_feed(mock_feed_entry):
    """Return a mock feedparser parse result with one healthy entry."""
    feed = MagicMock()
    feed.bozo = 0
    feed.entries = [mock_feed_entry]
    return feed


@pytest.fixture
def run_id(db):
    """Create and return a fresh pipeline run (no previous completed run)."""
    ai_id = db.get_interest_by_name("AI")["id"]
    return db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")


@pytest.fixture
def interest(run_id, db):
    """Return an InterestConfig for the AI interest used in run_id."""
    ai_id = db.get_interest_by_name("AI")["id"]
    # Upsert the config feeds so the scraper can find them by interest_id
    db.upsert_feed(ai_id, "https://example.com/ai-news", "AI News", "news")
    db.upsert_feed(ai_id, "https://example.com/ai-commentators", "AI Commentators", "commentators")
    return InterestConfig(id=ai_id, name="AI")


# ---------------------------------------------------------------------------
# 1. _normalize_url
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    """``_normalize_url`` strips query parameters and fragments."""

    def test_strips_query_params(self):
        """Query parameters after ``?`` must be removed."""
        result = _normalize_url("https://example.com/article?utm_source=twitter&id=123")
        assert result == "https://example.com/article"

    def test_strips_fragment(self):
        """Fragment after ``#`` must be removed."""
        result = _normalize_url("https://example.com/article#section-2")
        assert result == "https://example.com/article"

    def test_strips_both_query_and_fragment(self):
        """Both query string and fragment must be removed."""
        result = _normalize_url(
            "https://example.com/article?page=2&ref=news#comments"
        )
        assert result == "https://example.com/article"

    def test_empty_string_returns_empty(self):
        """Empty input must return an empty string."""
        assert _normalize_url("") == ""

    def test_url_without_params_or_fragment_unchanged(self):
        """URL without query params or fragment must be returned as-is."""
        url = "https://example.com/articles/some-story"
        assert _normalize_url(url) == url

    def test_fragment_before_query_is_stripped(self):
        """Fragment before query string (unusual but valid) — both are removed."""
        result = _normalize_url("https://example.com/page#frag?query=1")
        assert result == "https://example.com/page"


# ---------------------------------------------------------------------------
# 2. _extract_published
# ---------------------------------------------------------------------------


class TestExtractPublished:
    """``_extract_published`` extracts a timezone-aware datetime from feedparser entries."""

    def test_uses_published_parsed(self):
        """``published_parsed`` must be preferred over ``updated_parsed``."""
        entry = {
            "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
            "updated_parsed": time.struct_time((2026, 5, 13, 8, 0, 0, 2, 133, 0)),
        }
        dt = _extract_published(entry)
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 5
        assert dt.day == 14
        assert dt.hour == 10
        assert dt.tzinfo is not None

    def test_falls_back_to_updated_parsed(self):
        """When ``published_parsed`` is missing, ``updated_parsed`` must be used."""
        entry = {
            "updated_parsed": time.struct_time((2026, 5, 13, 8, 30, 0, 2, 133, 0)),
        }
        dt = _extract_published(entry)
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 5
        assert dt.day == 13
        assert dt.hour == 8

    def test_returns_none_when_no_date_fields(self):
        """Entry without ``published_parsed`` or ``updated_parsed`` must return ``None``."""
        entry = {"title": "No date"}
        assert _extract_published(entry) is None

    def test_returns_none_when_fields_are_none(self):
        """Entry with ``None`` date fields must return ``None``."""
        entry = {"published_parsed": None, "updated_parsed": None}
        assert _extract_published(entry) is None

    def test_returns_utc_aware_datetime(self):
        """Returned datetime must be timezone-aware (UTC)."""
        entry = {
        "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
        }
        dt = _extract_published(entry)
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) == timedelta(0)

    def test_handles_malformed_struct_time(self):
        """Malformed struct-like values that raise ``TypeError`` must be skipped."""
        entry = {"published_parsed": "not-a-tuple", "updated_parsed": None}
        assert _extract_published(entry) is None


# ---------------------------------------------------------------------------
# 3. _parse_iso8601
# ---------------------------------------------------------------------------


class TestParseIso8601:
    """``_parse_iso8601`` parses ISO 8601 strings to timezone-aware datetimes."""

    def test_timezone_aware_timestamp(self):
        """A timestamp with explicit ``+00:00`` offset must be preserved as UTC."""
        ts = "2026-05-14T06:00:00+00:00"
        dt = _parse_iso8601(ts)
        assert dt.year == 2026
        assert dt.month == 5
        assert dt.day == 14
        assert dt.hour == 6
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)

    def test_naive_timestamp_gets_utc(self):
        """A naive timestamp (no timezone) must be treated as UTC."""
        ts = "2026-05-14T06:00:00"
        dt = _parse_iso8601(ts)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)
        assert dt.hour == 6

    def test_non_utc_offset_converted_to_aware(self):
        """A timestamp with a non-UTC offset must have that offset preserved."""
        ts = "2026-05-14T08:00:00+02:00"
        dt = _parse_iso8601(ts)
        assert dt.utcoffset() == timedelta(hours=2)

    def test_with_z_suffix(self):
        """ISO 8601 with ``Z`` suffix must be parsed (3.11+ supports this)."""
        ts = "2026-05-14T06:00:00Z"
        dt = _parse_iso8601(ts)
        assert dt.hour == 6
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# 4. _extract_article
# ---------------------------------------------------------------------------


class TestExtractArticleFullContent:
    """``_extract_article`` returns ``full`` status when extraction > 200 chars."""

    def test_full_extraction_success(self):
        """Successful extraction with content > 200 chars must return (content, ``"full"``)."""
        long_content = "Long article content. " * 20  # > 200 chars
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>some page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", return_value=long_content):
            content, status = _extract_article("https://example.com/article", "excerpt", 15)

        assert status == "full"
        assert content == long_content


class TestExtractArticleExcerptOnly:
    """``_extract_article`` returns ``excerpt_only`` for short/empty extraction."""

    def test_short_extraction_returns_excerpt(self):
        """Extraction ≤ 200 chars must return the RSS excerpt with ``excerpt_only``."""
        short_content = "Short."
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", return_value=short_content):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_empty_extraction_returns_excerpt(self):
        """Extraction returning empty string must fall back to excerpt."""
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", return_value=""):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_none_extraction_returns_excerpt(self):
        """Extraction returning ``None`` must fall back to excerpt."""
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", return_value=None):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_whitespace_only_extraction_returns_excerpt(self):
        """Extraction returning only whitespace must fall back to excerpt."""
        rss_excerpt = "RSS fallback excerpt."

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", return_value="   \n  \t  "):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_http_404_returns_excerpt(self):
        """HTTP 404 must return excerpt without paywall flag."""
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=mock_resp
        )

        with patch("src.scraper.httpx.get", side_effect=mock_resp.raise_for_status.side_effect):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_http_500_returns_excerpt(self):
        """HTTP 500 must return excerpt without paywall flag."""
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_resp
        )

        with patch("src.scraper.httpx.get", side_effect=mock_resp.raise_for_status.side_effect):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_connection_error_returns_excerpt(self):
        """Connection error must return excerpt."""
        rss_excerpt = "RSS fallback excerpt."

        with patch("src.scraper.httpx.get", side_effect=ConnectionError("connection refused")):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_trafilatura_exception_returns_excerpt(self):
        """Exception from trafilatura.extract must return excerpt."""
        rss_excerpt = "RSS fallback excerpt."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp), \
             patch("src.scraper.trafilatura.extract", side_effect=ValueError("extraction failed")):
            content, status = _extract_article("https://example.com/article", rss_excerpt, 15)

        assert status == "excerpt_only"
        assert content == rss_excerpt

    def test_timeout_passed_to_httpx(self):
        """The timeout parameter must be forwarded to ``httpx.get``."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.text = "<html>page</html>"

        with patch("src.scraper.httpx.get", return_value=mock_resp) as mock_get, \
             patch("src.scraper.trafilatura.extract", return_value="content long enough for full extraction and then some more"):
            _extract_article("https://example.com/article", "excerpt", 42)

        mock_get.assert_called_once()
        _kwargs = mock_get.call_args.kwargs
        assert _kwargs["timeout"] == 42.0


class TestExtractArticlePaywall:
    """``_extract_article`` returns ``excerpt_paywall`` for HTTP 402/403."""

    @pytest.mark.parametrize("status_code", [402, 403])
    def test_paywall_status_codes(self, status_code):
        """HTTP 402 and 403 must both return ``excerpt_paywall``."""
        rss_excerpt = "RSS paywall fallback."
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status_code
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code} Forbidden", request=MagicMock(), response=mock_resp
        )

        with patch("src.scraper.httpx.get", side_effect=mock_resp.raise_for_status.side_effect):
            content, status = _extract_article(
                "https://example.com/paywall-article", rss_excerpt, 15
            )

        assert status == "excerpt_paywall"
        assert content == rss_excerpt


# ---------------------------------------------------------------------------
# 5. run() — Successful scrape
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunSuccessfulScrape:
    """``run()`` happy path: feeds parsed, articles inserted, pipeline updated."""

    def test_single_entry_processed(self, db, config, mock_feed, run_id, interest):
        """A single feed entry must be inserted as an article."""
        with patch("src.scraper.feedparser.parse", return_value=mock_feed), \
             patch("src.scraper._extract_article", return_value=("full content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["author"] == "Test Author"
        assert articles[0]["content_status"] == "full"

    def test_updates_pipeline_stage(self, db, config, mock_feed, run_id, interest):
        """``update_pipeline_run`` must be called with ``current_stage='scrape'``."""
        with patch("src.scraper.feedparser.parse", return_value=mock_feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        run_record = db.get_pipeline_run(run_id)
        assert run_record["current_stage"] == "scrape"

    def test_both_news_and_commentator_feeds(self, db, config, mock_feed_entry, run_id, interest):
        """Both feed categories (news + commentators) must be processed."""
        # Override config with one feed of each type
        cfg = Config(
            feeds=FeedsConfig(
                news=[FeedDef(name="News Feed", url="https://example.com/news")],
                commentators=[
                    FeedDef(name="Comment Feed", url="https://example.com/comments")
                ],
            ),
            models=config.models,
            pipeline=config.pipeline,
            email=config.email,
            database=config.database,
            openrouter=config.openrouter,
        )

        feed_news = MagicMock()
        feed_news.bozo = 0
        feed_news_entry = dict(mock_feed_entry)
        feed_news_entry["link"] = "https://example.com/news-article"
        feed_news.entries = [feed_news_entry]

        feed_comment = MagicMock()
        feed_comment.bozo = 0
        feed_comment_entry = dict(mock_feed_entry)
        feed_comment_entry["link"] = "https://example.com/comment-article"
        feed_comment.entries = [feed_comment_entry]

        with patch(
            "src.scraper.feedparser.parse",
            side_effect=[feed_news, feed_comment],
        ), patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, cfg, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 2
        urls = {a["url"] for a in articles}
        assert urls == {
            "https://example.com/news-article",
            "https://example.com/comment-article",
        }

    def test_article_has_required_fields(self, db, config, mock_feed, run_id, interest):
        """Inserted articles must have all required fields populated."""
        with patch("src.scraper.feedparser.parse", return_value=mock_feed), \
             patch("src.scraper._extract_article", return_value=("full content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        article = articles[0]
        assert article["feed_id"] > 0
        assert article["url"] == _normalize_url("https://example.com/article?utm_source=test#ref")
        assert article["title"] == "Test Article"
        assert article["author"] == "Test Author"
        assert article["published_at"] is not None
        assert article["scraped_at"] is not None
        assert article["rss_excerpt"] == "This is an RSS excerpt for testing purposes."
        assert article["full_content"] == "full content"
        assert article["content_status"] == "full"
        assert article["pipeline_run_id"] == run_id

    def test_feed_upserted(self, db, config, mock_feed, run_id, interest):
        """The feed must be upserted into the database."""
        with patch("src.scraper.feedparser.parse", return_value=mock_feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        feeds = db.get_all_feeds()
        assert len(feeds) >= 1
        feed_urls = {f["url"] for f in feeds}
        assert "https://example.com/ai-news" in feed_urls

    def test_multiple_entries_in_feed(self, db, config, mock_feed_entry, run_id, interest):
        """Multiple entries in a single feed must all be processed."""
        entry2 = dict(mock_feed_entry)
        entry2["title"] = "Second Article"
        entry2["link"] = "https://example.com/second"
        entry3 = dict(mock_feed_entry)
        entry3["title"] = "Third Article"
        entry3["link"] = "https://example.com/third"

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [mock_feed_entry, entry2, entry3]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch(
                 "src.scraper._extract_article",
                 return_value=("content", "full"),
             ):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 3
        titles = [a["title"] for a in articles]
        assert "Second Article" in titles
        assert "Third Article" in titles

    def test_article_without_author(self, db, config, run_id, interest):
        """An entry without an author must still be inserted (author = ``None``)."""
        entry = {
            "title": "No Author Article",
            "link": "https://example.com/no-author",
            "summary": "Some excerpt.",
            "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
        }
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["author"] is None

    def test_untitled_article_defaults(self, db, config, run_id, interest):
        """An entry without a title must default to ``'Untitled'``."""
        entry = {
            "link": "https://example.com/untitled",
            "summary": "Some excerpt.",
            "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
        }
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Untitled"


# ---------------------------------------------------------------------------
# 6. run() — Content status variants
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunContentStatus:
    """``run()`` correctly stores different content_status values."""

    def test_full_content(self, db, config, mock_feed, run_id, interest):
        """``full`` content_status from article extraction must be stored."""
        with patch("src.scraper.feedparser.parse", return_value=mock_feed), \
             patch("src.scraper._extract_article", return_value=("extracted content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert articles[0]["content_status"] == "full"
        assert articles[0]["full_content"] == "extracted content"

    def test_excerpt_only(self, db, config, mock_feed_entry, run_id, interest):
        """``excerpt_only`` content_status must use RSS excerpt as content."""
        entry = dict(mock_feed_entry)
        entry["link"] = "https://example.com/excerpt-only"
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch(
                 "src.scraper._extract_article",
                 return_value=("This is an RSS excerpt for testing purposes.", "excerpt_only"),
             ):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert articles[0]["content_status"] == "excerpt_only"
        assert articles[0]["full_content"] == "This is an RSS excerpt for testing purposes."

    def test_paywall(self, db, config, mock_feed_entry, run_id, interest):
        """``excerpt_paywall`` content_status must be stored."""
        entry = dict(mock_feed_entry)
        entry["link"] = "https://example.com/paywall"
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch(
                 "src.scraper._extract_article",
                 return_value=("RSS excerpt", "excerpt_paywall"),
             ):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert articles[0]["content_status"] == "excerpt_paywall"


# ---------------------------------------------------------------------------
# 7. run() — Deduplication
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunDeduplication:
    """``run()`` skips articles with URLs that already exist in the database."""

    def test_duplicate_url_skipped(self, db, config, mock_feed_entry, run_id, interest):
        """An article whose normalized URL already exists must be skipped."""
        # Pre-insert an article with the normalized URL
        normalized = _normalize_url(mock_feed_entry["link"])
        ai_id = db.get_interest_by_name("AI")["id"]
        feed_id = db.upsert_feed(ai_id, "https://example.com/ai-news", "AI News", "news")
        db.insert_article(
            feed_id=feed_id,
            url=normalized,
            title="Existing",
            author=None,
            published_at="2026-05-14T10:00:00",
            scraped_at="2026-05-14T10:05:00",
            rss_excerpt="",
            full_content=None,
            content_status="full",
            pipeline_run_id=run_id,
        )

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [mock_feed_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("new content", "full")):
            run(run_id, db, config, interest)

        # Only the pre-inserted article should exist (duplicate was skipped)
        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Existing"

    def test_different_urls_not_deduplicated(self, db, config, mock_feed_entry, run_id, interest):
        """Different URLs after normalization must both be inserted."""
        feed = MagicMock()
        feed.bozo = 0
        entry_a = dict(mock_feed_entry)
        entry_a["link"] = "https://example.com/article-a"
        entry_b = dict(mock_feed_entry)
        entry_b["link"] = "https://example.com/article-b"
        feed.entries = [entry_a, entry_b]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 2


# ---------------------------------------------------------------------------
# 8. run() — Cutoff (date filtering)
# ---------------------------------------------------------------------------


class TestRunCutoff:
    """``run()`` filters entries by publication date based on cutoff."""

    def test_no_previous_run_uses_24h_cutoff(self, db, config, run_id, interest):
        """When no previous successful run exists, a 24h cutoff must be used."""
        fixed_now = real_datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Entry older than 24h — should be skipped
        old_entry = {
            "title": "Old Article",
            "link": "https://example.com/old",
            "author": "Old Author",
            "summary": "Old excerpt.",
            "published_parsed": time.struct_time((2026, 5, 13, 10, 0, 0, 2, 133, 0)),
        }
        # Entry within 24h — should be processed
        recent_entry = {
            "title": "Recent Article",
            "link": "https://example.com/recent",
            "author": "Recent Author",
            "summary": "Recent excerpt.",
            "published_parsed": time.struct_time((2026, 5, 15, 10, 0, 0, 4, 135, 0)),
        }

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [old_entry, recent_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")), \
             patch("src.scraper.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = real_datetime.fromisoformat
            mock_dt.fromtimestamp = real_datetime.fromtimestamp
            mock_dt.timedelta = timedelta
            mock_dt.timezone = timezone

            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Recent Article"

    def test_previous_run_uses_last_started_at(self, db, config, mock_feed_entry, interest):
        """When a previous successful run exists, its ``started_at`` must be the cutoff."""
        ai_id = db.get_interest_by_name("AI")["id"]
        # Create and complete a previous run
        prev_run_at = "2026-05-14T10:00:00"
        prev_run_id = db.create_pipeline_run(ai_id, "2026-05-14", prev_run_at)
        db.update_pipeline_run(prev_run_id, status="completed", completed_at="2026-05-14T10:30:00")

        # Current run
        current_run_id = db.create_pipeline_run(ai_id, "2026-05-15", "2026-05-15T06:00:00")

        # Entry published before the previous run's started_at — should be skipped
        old_entry = dict(mock_feed_entry)
        old_entry["title"] = "Old Article"
        old_entry["link"] = "https://example.com/old-entry"
        old_entry["published_parsed"] = time.struct_time((2026, 5, 14, 8, 0, 0, 3, 134, 0))

        # Entry published after the previous run's started_at — should be processed
        new_entry = dict(mock_feed_entry)
        new_entry["title"] = "New Article"
        new_entry["link"] = "https://example.com/new-entry"
        new_entry["published_parsed"] = time.struct_time((2026, 5, 14, 12, 0, 0, 3, 134, 0))

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [old_entry, new_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(current_run_id, db, config, interest)

        articles = db.get_articles_for_run(current_run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "New Article"

    def test_entry_at_exact_cutoff_skipped(self, db, config, mock_feed_entry, interest):
        """An entry published at the exact cutoff time should be skipped (not > cutoff)."""
        ai_id = db.get_interest_by_name("AI")["id"]
        prev_run_at = "2026-05-14T10:00:00"
        prev_run_id = db.create_pipeline_run(ai_id, "2026-05-14", prev_run_at)
        db.update_pipeline_run(prev_run_id, status="completed", completed_at="2026-05-14T10:30:00")

        current_run_id = db.create_pipeline_run(ai_id, "2026-05-15", "2026-05-15T06:00:00")

        # Entry at exactly the cutoff time (10:00:00)
        entry = dict(mock_feed_entry)
        entry["link"] = "https://example.com/at-cutoff"
        entry["published_parsed"] = time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0))

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(current_run_id, db, config, interest)

        articles = db.get_articles_for_run(current_run_id)
        assert len(articles) == 0

    def test_entry_with_no_date_skipped(self, db, config, run_id, interest):
        """An entry with no parsed date must be skipped (published is ``None``)."""
        entry = {
            "title": "No Date Article",
            "link": "https://example.com/no-date",
            "summary": "Excerpt.",
        }
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 0


# ---------------------------------------------------------------------------
# 9. run() — Malformed / empty feeds
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunMalformedFeeds:
    """``run()`` handles malformed, empty, and failing feeds gracefully."""

    def test_bozo_feed_with_no_entries_skipped(self, db, config, run_id, interest):
        """A bozo feed with zero entries must be skipped (logged, not inserted)."""
        feed = MagicMock()
        feed.bozo = 1
        feed.entries = []

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article") as mock_extract:
            run(run_id, db, config, interest)

        # No articles should be inserted
        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 0
        mock_extract.assert_not_called()

    def test_bozo_feed_with_entries_still_processed(self, db, config, mock_feed_entry, run_id, interest):
        """A bozo feed that still has entries must be processed (bozo alone is not a skip)."""
        feed = MagicMock()
        feed.bozo = 1
        feed.entries = [mock_feed_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1

    def test_empty_feed_no_entries(self, db, config, run_id, interest):
        """A feed with no entries must produce zero articles."""
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = []

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article") as mock_extract:
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 0
        mock_extract.assert_not_called()

    def test_feed_parse_exception_skipped(self, db, config, run_id, interest):
        """When ``feedparser.parse`` raises an exception, the feed must be skipped."""
        with patch(
            "src.scraper.feedparser.parse",
            side_effect=ConnectionError("network error"),
        ), patch("src.scraper._extract_article") as mock_extract:
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 0
        mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# 10. run() — Entry with empty/normalized-to-empty URL
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunEmptyUrl:
    """``run()`` skips entries whose URL is empty or normalizes to empty."""

    def test_entry_with_empty_link_skipped(self, db, config, run_id, interest):
        """An entry with no ``link`` field must be skipped."""
        entry = {
            "title": "No Link",
            "summary": "Excerpt.",
            "published_parsed": time.struct_time((2026, 5, 14, 10, 0, 0, 3, 134, 0)),
        }
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article") as mock_extract:
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 0
        mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# 11. run() — Failed article insert does not crash
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunInsertFailure:
    """``run()`` handles a single article insertion failure without crashing."""

    def test_insert_exception_continues(self, db, config, mock_feed_entry, run_id, interest):
        """If one article insert fails, remaining entries must still be processed."""
        entry_good = dict(mock_feed_entry)
        entry_good["link"] = "https://example.com/good"

        entry_bad = dict(mock_feed_entry)
        entry_bad["link"] = "https://example.com/bad"

        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [entry_bad, entry_good]

        # Mock insert_article to fail on the first call, succeed on the second
        original_insert = db.insert_article

        def flaky_insert(**kwargs):
            if kwargs.get("url") == "https://example.com/bad":
                raise RuntimeError("DB failure")
            return original_insert(**kwargs)

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")), \
             patch.object(db, "insert_article", side_effect=flaky_insert):
            run(run_id, db, config, interest)

        # The good article should still be inserted
        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["url"] == "https://example.com/good"


# ---------------------------------------------------------------------------
# 12. run() — Normalized URL used for dedup check and insert
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_datetime_now")
class TestRunUrlNormalization:
    """``run()`` normalizes URLs before dedup check and storage."""

    def test_url_normalized_before_dedup(self, db, config, mock_feed_entry, run_id, interest):
        """URL must be normalized before checking ``article_exists``."""
        ai_id = db.get_interest_by_name("AI")["id"]
        feed_id = db.upsert_feed(ai_id, "https://example.com/ai-news", "AI News", "news")
        db.insert_article(
            feed_id=feed_id,
            url="https://example.com/article",  # normalized form
            title="Existing",
            author=None,
            published_at="2026-05-14T10:00:00",
            scraped_at="2026-05-14T10:05:00",
            rss_excerpt="",
            full_content=None,
            content_status="full",
            pipeline_run_id=run_id,
        )

        # Entry has query params and fragment — should normalize to the URL above
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [mock_feed_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        # No new articles added (duplicate was skipped)
        articles = db.get_articles_for_run(run_id)
        assert len(articles) == 1
        assert articles[0]["title"] == "Existing"

    def test_url_normalized_before_insert(self, db, config, mock_feed_entry, run_id, interest):
        """URL must be stored in its normalized form (without params/fragment)."""
        feed = MagicMock()
        feed.bozo = 0
        feed.entries = [mock_feed_entry]

        with patch("src.scraper.feedparser.parse", return_value=feed), \
             patch("src.scraper._extract_article", return_value=("content", "full")):
            run(run_id, db, config, interest)

        articles = db.get_articles_for_run(run_id)
        stored_url = articles[0]["url"]
        assert "?" not in stored_url
        assert "#" not in stored_url
        assert stored_url == "https://example.com/article"
