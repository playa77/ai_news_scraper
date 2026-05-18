"""Shared fixtures for the test suite."""

from unittest.mock import patch

import pytest


@pytest.fixture
def ai_id(db):
    """Return the ID of the default 'AI' interest created by schema initialization."""
    return db.get_interest_by_name("AI")["id"]


@pytest.fixture
def patch_datetime_now(request):
    """Patch ``src.scraper.datetime.now`` to return a fixed UTC time.

    The fixed time (2026-05-14T12:00:00+00:00) is close to the mock feed
    entry timestamps used in scraper tests, preventing the 24-hour cutoff
    from filtering out test entries.
    """
    from datetime import datetime, timedelta, timezone

    fixed_now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)

    with patch("src.scraper.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.fromtimestamp = datetime.fromtimestamp
        mock_dt.timedelta = timedelta
        mock_dt.timezone = timezone
        yield
