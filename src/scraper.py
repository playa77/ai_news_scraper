"""Scraper stage — RSS feed fetching and full-article extraction.

Implements the ``scrape`` stage of the AI News Pipeline:
- Reads feed list from YAML config.
- Fetches RSS via ``feedparser``, filters items newer than the last successful run.
- Normalizes URLs for deduplication.
- Extracts full article content via ``trafilatura``, with paywall/excerpt fallback.
- Stores articles in SQLite.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import feedparser
import httpx
import trafilatura

from .config import Config
from .db import Database
from .models import InterestConfig

logger = logging.getLogger(__name__)


def run(run_id: int, db: Database, config: Config, interest: InterestConfig) -> None:
    """Execute the scrape stage.

    Parameters
    ----------
    run_id:
        The ``pipeline_runs.id`` this scrape belongs to.
    db:
        Open :class:`Database` instance.
    config:
        Parsed pipeline configuration.
    interest:
        The ``InterestConfig`` for this pipeline run.
    """
    db.update_pipeline_run(run_id, current_stage="scrape")

    # Recover articles from failed runs that were never analyzed.
    recovered = db.recover_orphaned_articles(run_id)
    if recovered:
        logger.info(
            "Recovered %d orphaned article(s) from previous failed run(s) — "
            "reassigned to run %d",
            recovered,
            run_id,
        )

    # Determine the reference timestamp for filtering new articles
    last_run = db.get_last_successful_run(interest_id=interest.id)
    if last_run:
        cutoff = _parse_iso8601(last_run["started_at"])
    else:
        # No previous run — take articles from the last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    all_feeds = db.get_all_feeds(interest.id)
    total_new = 0
    status_counts: dict[str, int] = {}

    for feed_def in all_feeds:
        category = feed_def["category"]
        feed_id = feed_def["id"]

        try:
            parsed = feedparser.parse(feed_def["url"])
        except Exception as exc:
            logger.warning(
                "Failed to parse feed %r (name=%r): %s",
                feed_def["url"],
                feed_def["name"],
                exc,
            )
            continue

        if parsed.bozo and not parsed.entries:
            logger.warning(
                "Feed %r (name=%r) appears malformed with no entries: %s",
                feed_def["url"],
                feed_def["name"],
                parsed.bozo_exception if hasattr(parsed, "bozo_exception") else "unknown",
            )
            continue

        new_in_feed = 0
        for entry in parsed.entries:
            published = _extract_published(entry)
            if published is None or published <= cutoff:
                continue

            normalized_url = _normalize_url(entry.get("link", ""))
            if not normalized_url:
                continue

            if db.article_exists(normalized_url):
                continue

            title = entry.get("title", "Untitled")
            author = entry.get("author")
            rss_excerpt = entry.get("summary", "")

            # Attempt full article extraction (policy governed by interest config)
            if interest.input_data_length_mode == "headers_only":
                full_content, content_status = None, "headers_only"
            else:
                full_content, content_status = _extract_article(
                    entry.get("link", ""),
                    rss_excerpt,
                    config.pipeline.article_fetch_timeout_seconds,
                )
                if interest.input_data_length_mode == "word_count" and full_content:
                    n = interest.input_word_count or 256
                    full_content = " ".join(full_content.split()[:n])

            try:
                db.insert_article(
                    feed_id=feed_id,
                    url=normalized_url,
                    title=title,
                    author=author,
                    published_at=published.isoformat(),
                    scraped_at=datetime.now(timezone.utc).isoformat(),
                    rss_excerpt=rss_excerpt,
                    full_content=full_content,
                    content_status=content_status,
                    pipeline_run_id=run_id,
                )
            except Exception:
                logger.exception("Failed to insert article %r", normalized_url)
                continue

            new_in_feed += 1
            status_counts[content_status] = status_counts.get(content_status, 0) + 1

        total_new += new_in_feed
        logger.info(
            "Fetched %d new articles from feed %r (name=%r)",
            new_in_feed,
            feed_def["url"],
            feed_def["name"],
        )

    logger.info(
        "Scrape complete — %d total new articles, status distribution: %s",
        total_new,
        status_counts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Normalize an article URL for deduplication.

    Strips query parameters and fragments.  Over-deduplication is preferable
    to storing duplicate articles.
    """
    if not url:
        return ""
    url = re.sub(r"#.*$", "", url)  # strip fragment
    url = re.sub(r"\?.*$", "", url)  # strip query params
    return url


def _extract_published(entry: dict) -> Optional[datetime]:
    """Extract a timezone-aware datetime from a feedparser entry.

    Returns ``None`` if the published date cannot be parsed.
    """
    for field in ("published_parsed", "updated_parsed"):
        tp = entry.get(field)
        if tp is not None:
            try:
                from calendar import timegm

                ts = timegm(tp)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (TypeError, OSError, OverflowError):
                continue
    return None


def _extract_article(
    url: str,
    rss_excerpt: str,
    timeout: int,
) -> tuple[Optional[str], str]:
    """Fetch and extract the full article text.

    Returns
    -------
    (content, content_status)
        ``content_status`` is one of ``full``, ``excerpt_only``, or ``excerpt_paywall``.
    """
    try:
        resp = httpx.get(
            url,
            timeout=float(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "AI-News-Pipeline/1.0 (news-aggregation-bot; mailto:admin@example.com)"
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (402, 403):
            logger.info("Paywall detected for %r (HTTP %d)", url, exc.response.status_code)
            return rss_excerpt, "excerpt_paywall"
        logger.warning(
            "Failed to fetch %r (HTTP %d): using excerpt", url, exc.response.status_code
        )
        return rss_excerpt, "excerpt_only"
    except Exception as exc:
        logger.warning("Failed to fetch %r: %s — using excerpt", url, exc)
        return rss_excerpt, "excerpt_only"

    try:
        extracted = trafilatura.extract(
            resp.text,
            include_comments=False,
            favor_precision=True,
        )
    except Exception:
        logger.warning("trafilatura extraction failed for %r", url)
        return rss_excerpt, "excerpt_only"

    if not extracted or not extracted.strip():
        logger.warning("trafilatura returned empty content for %r", url)
        return rss_excerpt, "excerpt_only"

    if len(extracted) <= 200:
        logger.info("trafilatura returned short content (%d chars) for %r", len(extracted), url)
        return rss_excerpt, "excerpt_only"

    logger.info("Full article extracted for %r (%d chars)", url, len(extracted))
    return extracted, "full"


def _parse_iso8601(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
