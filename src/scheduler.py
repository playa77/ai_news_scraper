"""In-process scheduler for multi-interest pipeline runs.

Uses APScheduler to schedule per-interest pipeline runs based on each
interest's ``start_time`` and ``interval_hours``.  Supports:
- Computed run times from start_time + k*interval within each day.
- Skipping paused interests (all deliverables disabled).
- Catch-up: on startup, runs the most recently missed run per interest.
- "Run now" trigger from the web UI.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.job import Job

from .models import InterestConfig

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Raised on scheduler failures."""


class PipelineScheduler:
    """Manages per-interest pipeline scheduling via APScheduler.

    Parameters
    ----------
    get_interests:
        Callable that returns a ``list[InterestConfig]`` of all interests.
    is_running:
        Callable that returns ``True`` if a run is in progress for an interest_id.
    run_interest:
        Callable ``(interest_id: int) -> None`` that executes a pipeline run.
    timezone:
        IANA timezone string (e.g. ``"Europe/Berlin"``).
    """

    def __init__(
        self,
        get_interests: Callable[[], list[InterestConfig]],
        is_running: Callable[[int], bool],
        run_interest: Callable[[int], None],
        timezone: str = "Europe/Berlin",
    ) -> None:
        self._get_interests = get_interests
        self._is_running = is_running
        self._run_interest = run_interest
        self._tz = ZoneInfo(timezone)
        self._aps = BackgroundScheduler(timezone=timezone)
        self._running_jobs: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the scheduler and run initial catch-up checks."""
        self._schedule_all_interests()
        self._aps.start()
        logger.info("Scheduler started — timezone=%s", self._aps.timezone)
        self._run_catch_up()

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._aps.shutdown(wait=False)
        logger.info("Scheduler stopped")

    def trigger_now(self, interest_id: int, interest_name: str) -> str:
        """Trigger an immediate pipeline run for an interest.

        Returns an error message string if the run cannot be started,
        or an empty string on success.
        """
        interest = self._get_interest(interest_id)
        if interest is None:
            return f"Interest {interest_id} not found"

        if interest.is_paused:
            return f"Interest '{interest_name}' is paused — at least one deliverable must be enabled to run"

        if self._is_running(interest_id):
            return f"A pipeline run for '{interest_name}' is already in progress"

        with self._lock:
            if interest_id in self._running_jobs and self._running_jobs[interest_id].is_alive():
                return f"A pipeline run for '{interest_name}' is already in progress"

            thread = threading.Thread(
                target=self._run_interest_safe,
                args=(interest_id, interest_name),
                daemon=True,
            )
            self._running_jobs[interest_id] = thread
            thread.start()

        logger.info("Manual run triggered for interest '%s' (id=%d)", interest_name, interest_id)
        return ""

    def _get_interest(self, interest_id: int) -> Optional[InterestConfig]:
        """Get a single interest by ID."""
        for i in self._get_interests():
            if i.id == interest_id:
                return i
        return None

    def _schedule_all_interests(self) -> None:
        """Remove all existing jobs and reschedule based on current interest configs."""
        self._aps.remove_all_jobs()

        for interest in self._get_interests():
            if interest.is_paused:
                logger.info(
                    "Interest '%s' is paused — skipping schedule", interest.name
                )
                continue
            self._schedule_interest(interest)

    def reschedule_all(self) -> None:
        """Reschedule all interests from current config (called after config changes)."""
        self._schedule_all_interests()

    def _schedule_interest(self, interest: InterestConfig) -> None:
        """Schedule all daily run times for a single interest.

        Computes run times as: start_time + k*interval for all k≥0 within 24h.
        Uses APScheduler CronTrigger with ``hour`` and ``minute`` fields.
        """
        start_h, start_m = _parse_hhmm(interest.start_time)
        interval = max(1, interest.interval_hours)

        # Compute all hour:minute slots within a day
        run_hours = set()
        for k in range(0, 24, interval):
            hour = (start_h + k) % 24
            run_hours.add(hour)

        # Build cron-style hour list (comma-separated for readability in logs)
        hour_list = ",".join(str(h) for h in sorted(run_hours))

        trigger = CronTrigger(hour=hour_list, minute=str(start_m), timezone=str(self._tz))

        job_id = f"interest_{interest.id}"
        self._aps.add_job(
            func=self._scheduled_run,
            trigger=trigger,
            args=[interest.id, interest.name],
            id=job_id,
            name=f"Scheduled run for '{interest.name}'",
            replace_existing=True,
            misfire_grace_time=None,  # Allow any missed run to fire
        )

        logger.info(
            "Scheduled interest '%s' — start=%s interval=%dh → hours=[%s] minute=%d",
            interest.name, interest.start_time, interval, hour_list, start_m,
        )

    def _scheduled_run(self, interest_id: int, interest_name: str) -> None:
        """Called by APScheduler when a scheduled time arrives."""
        if self._is_running(interest_id):
            logger.info(
                "Scheduled run for '%s' skipped — already running", interest_name
            )
            return

        interest = self._get_interest(interest_id)
        if interest is None or interest.is_paused:
            logger.info(
                "Scheduled run for '%s' skipped — paused or deleted", interest_name
            )
            return

        with self._lock:
            if interest_id in self._running_jobs and self._running_jobs[interest_id].is_alive():
                return

            thread = threading.Thread(
                target=self._run_interest_safe,
                args=(interest_id, interest_name),
                daemon=True,
            )
            self._running_jobs[interest_id] = thread
            thread.start()

    def _run_interest_safe(self, interest_id: int, interest_name: str) -> None:
        """Execute a pipeline run, logging and cleaning up on completion."""
        try:
            logger.info("Pipeline run starting for interest '%s'", interest_name)
            self._run_interest(interest_id)
            logger.info("Pipeline run completed for interest '%s'", interest_name)
        except Exception as exc:
            logger.error(
                "Pipeline run failed for interest '%s': %s", interest_name, exc,
                exc_info=True,
            )
        finally:
            with self._lock:
                self._running_jobs.pop(interest_id, None)

    def _run_catch_up(self) -> None:
        """On startup, run the most recently missed scheduled run for each interest.

        Uses systemd-like ``Persistent=true`` semantics: only one missed run
        per interest, not the full backlog.
        """
        now = datetime.now(self._tz)

        for interest in self._get_interests():
            if interest.is_paused or interest.id is None:
                continue

            interest_id: int = interest.id

            # Find the most recent scheduled time that was missed
            missed_time = _find_most_recent_missed(interest, now)
            if missed_time is None:
                continue

            # Check if this interest already ran today (avoid duplicate)
            if self._is_running(interest_id):
                continue

            logger.info(
                "Catch-up: interest '%s' missed run at %s — executing now",
                interest.name,
                missed_time.strftime("%Y-%m-%d %H:%M"),
            )
            with self._lock:
                if interest_id in self._running_jobs and self._running_jobs[interest_id].is_alive():
                    continue
                thread = threading.Thread(
                    target=self._run_interest_safe,
                    args=(interest_id, interest.name),
                    daemon=True,
                )
                self._running_jobs[interest_id] = thread
                thread.start()


def _parse_hhmm(time_str: str) -> tuple[int, int]:
    """Parse HH:MM string into (hour, minute) integers."""
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 4, 0


def _find_most_recent_missed(
    interest: InterestConfig, now: datetime
) -> Optional[datetime]:
    """Find the most recent missed scheduled run time for an interest.

    Returns ``None`` if no runs were missed (i.e. the last scheduled time
    is in the future or is the current run time within a grace window).
    """
    start_h, start_m = _parse_hhmm(interest.start_time)
    interval = max(1, min(interest.interval_hours, 168))

    # Find the most recent k*interval time that's within the past (interval+1) hours
    # Look back up to interval + 1 hours from now to find a missed slot
    lookback = timedelta(hours=interval + 1)
    cutoff = now - lookback

    # Compute all possible run times today and yesterday
    best: Optional[datetime] = None

    # Check today and yesterday
    for day_offset in range(2):
        check_day = now.date() - timedelta(days=day_offset)
        for k in range(0, 24, interval):
            run_hour = (start_h + k) % 24
            if day_offset == 1 and run_hour > start_h:
                # For yesterday, only consider hours that wrap past midnight
                run_time = datetime(
                    check_day.year, check_day.month, check_day.day,
                    run_hour, start_m, tzinfo=now.tzinfo,
                )
            else:
                run_time = datetime(
                    check_day.year, check_day.month, check_day.day,
                    run_hour, start_m, tzinfo=now.tzinfo,
                )

            if run_time < now and run_time > cutoff:
                if best is None or run_time > best:
                    best = run_time

    if best is not None:
        # Only consider truly missed if it's more than 5 minutes in the past
        # (if the server just started within 5 min of scheduled time, skip catch-up)
        if now - best > timedelta(minutes=5):
            return best

    return None
