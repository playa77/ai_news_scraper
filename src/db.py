"""Database layer for AI News Pipeline — SQLite with WAL mode.

All SQLite operations are centralized here:
- Idempotent schema initialization (including auto-migration from v1)
- CRUD operations for all tables
- Connection management (single connection per pipeline run)
"""

import json
import sqlite3
from typing import Any, Optional

# The SQL function that checks for table existence to avoid re-creating
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS interests (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT    NOT NULL UNIQUE,
    start_time              TEXT    NOT NULL DEFAULT '04:00',
    interval_hours          INTEGER NOT NULL DEFAULT 24,
    input_data_length_mode  TEXT    NOT NULL DEFAULT 'full_article',
    input_word_count        INTEGER,
    target_summary_words    INTEGER NOT NULL DEFAULT 750,
    target_script_en_words  INTEGER NOT NULL DEFAULT 1250,
    target_script_de_words  INTEGER NOT NULL DEFAULT 1250,
    target_brief_words      INTEGER NOT NULL DEFAULT 700,
    enable_summary          INTEGER NOT NULL DEFAULT 1,
    enable_script_en        INTEGER NOT NULL DEFAULT 1,
    enable_script_de        INTEGER NOT NULL DEFAULT 1,
    enable_brief            INTEGER NOT NULL DEFAULT 1,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interest_id     INTEGER REFERENCES interests(id),
    run_date        TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    current_stage   TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS feeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interest_id     INTEGER REFERENCES interests(id),
    url             TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    UNIQUE(interest_id, url)
);

CREATE TABLE IF NOT EXISTS articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id         INTEGER NOT NULL REFERENCES feeds(id),
    url             TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    author          TEXT,
    published_at    TEXT    NOT NULL,
    scraped_at      TEXT    NOT NULL,
    rss_excerpt     TEXT,
    full_content    TEXT,
    content_status  TEXT    NOT NULL DEFAULT 'full',
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS themes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    source_article_ids TEXT NOT NULL,
    novelty_type    TEXT    NOT NULL,
    order_index     INTEGER NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS deliverables (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    deliverable_type TEXT   NOT NULL,
    content         TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    round_number    INTEGER NOT NULL DEFAULT 1,
    quality_passed  TEXT,
    quality_feedback TEXT,
    adversarial_passed TEXT,
    adversarial_feedback TEXT,
    overall_passed  TEXT    NOT NULL,
    evaluated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_briefs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    content         TEXT    NOT NULL,
    word_count      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_pipeline_run ON articles(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_themes_pipeline_run ON themes(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_deliverables_theme_type ON deliverables(theme_id, deliverable_type);
CREATE INDEX IF NOT EXISTS idx_evaluation_rounds_theme ON evaluation_rounds(theme_id);
CREATE INDEX IF NOT EXISTS idx_daily_briefs_pipeline_run ON daily_briefs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(run_date);
"""


class DatabaseError(Exception):
    """Raised on SQLite operational errors."""
    pass


class Database:
    """Centralized database access for the AI News Pipeline.

    Manages a single connection per instance. Call :meth:`close` when done.
    """

    def __init__(self, db_path: str) -> None:
        """Open the SQLite database connection.

        Args:
            db_path: Path to the SQLite database file, or ``:memory:`` for in-memory.
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize_schema(self) -> None:
        """Execute all CREATE TABLE and CREATE INDEX IF NOT EXISTS statements.

        This is idempotent — safe to call multiple times.
        Also runs auto-migration from v1 schema (adding columns if missing).
        """
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._migrate_v1_to_v2()

    def _migrate_v1_to_v2(self) -> None:
        """Auto-migrate a v1 database to v2 (multi-interest).

        Steps:
        1. Add ``interest_id`` columns to ``pipeline_runs`` and ``feeds`` if missing.
        2. Recreate ``feeds`` unique constraint to be per-interest if it was URL-only.
        3. Create a default 'AI' interest if none exists.
        4. Assign all existing feeds and pipeline_runs to the default interest.
        """
        # Check if this is a v1 database (no interests table was previously present,
        # but we just created it via _DDL — check if 'name' column exists)
        cursor = self._conn.execute("PRAGMA table_info(pipeline_runs)")
        cols = {row[1] for row in cursor.fetchall()}
        if "interest_id" not in cols:
            self._conn.execute(
                "ALTER TABLE pipeline_runs ADD COLUMN interest_id INTEGER REFERENCES interests(id)"
            )
            self._conn.commit()

        cursor = self._conn.execute("PRAGMA table_info(feeds)")
        cols = {row[1] for row in cursor.fetchall()}
        if "interest_id" not in cols:
            # SQLite doesn't support DROP CONSTRAINT, so we rebuild the table
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS feeds_new (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    interest_id INTEGER REFERENCES interests(id),
                    url         TEXT    NOT NULL,
                    name        TEXT    NOT NULL,
                    category    TEXT    NOT NULL,
                    UNIQUE(interest_id, url)
                );
                INSERT OR IGNORE INTO feeds_new (id, url, name, category)
                    SELECT id, url, name, category FROM feeds;
                DROP TABLE feeds;
                ALTER TABLE feeds_new RENAME TO feeds;
            """)
            self._conn.commit()

        # Create default 'AI' interest if none exists
        existing = self._conn.execute("SELECT COUNT(*) FROM interests").fetchone()
        if existing and existing[0] == 0:
            self._conn.execute(
                "INSERT INTO interests (name, start_time, interval_hours) VALUES (?, ?, ?)",
                ("AI", "04:00", 24),
            )
            self._conn.commit()
            ai_id = self._conn.execute(
                "SELECT id FROM interests WHERE name = 'AI'"
            ).fetchone()["id"]

            # Assign all existing feeds and runs to the AI interest
            self._conn.execute("UPDATE feeds SET interest_id = ? WHERE interest_id IS NULL", (ai_id,))
            self._conn.execute("UPDATE pipeline_runs SET interest_id = ? WHERE interest_id IS NULL", (ai_id,))
            self._conn.commit()

            # Create indexes if they don't exist yet (for fresh v1→v2 migration)
            self._conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_articles_pipeline_run ON articles(pipeline_run_id);
                CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
                CREATE INDEX IF NOT EXISTS idx_themes_pipeline_run ON themes(pipeline_run_id);
                CREATE INDEX IF NOT EXISTS idx_deliverables_theme_type ON deliverables(theme_id, deliverable_type);
                CREATE INDEX IF NOT EXISTS idx_evaluation_rounds_theme ON evaluation_rounds(theme_id);
                CREATE INDEX IF NOT EXISTS idx_daily_briefs_pipeline_run ON daily_briefs(pipeline_run_id);
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(run_date);
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # pipeline_runs
    # ------------------------------------------------------------------

    def create_pipeline_run(self, interest_id: int, run_date: str, started_at: str) -> int:
        """Insert a new pipeline run record.

        Returns:
            The ID of the newly created pipeline run.
        """
        cursor = self._conn.execute(
            "INSERT INTO pipeline_runs (interest_id, run_date, started_at) VALUES (?, ?, ?)",
            (interest_id, run_date, started_at),
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_pipeline_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        completed_at: Optional[str] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update fields on a pipeline run record.

        Only provided (non-None) fields are updated.
        """
        fields = []
        values: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at)
        if current_stage is not None:
            fields.append("current_stage = ?")
            values.append(current_stage)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        if not fields:
            return

        values.append(run_id)
        self._conn.execute(
            f"UPDATE pipeline_runs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        self._conn.commit()

    def get_pipeline_run(self, run_id: int) -> Optional[dict]:
        """Return a pipeline run by ID, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # feeds
    # ------------------------------------------------------------------

    def upsert_feed(self, interest_id: int, url: str, name: str, category: str) -> int:
        """Insert a feed if it does not exist for this interest, or return its existing ID.

        Returns:
            The feed ID (new or existing).
        """
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO feeds (interest_id, url, name, category) VALUES (?, ?, ?, ?)",
            (interest_id, url, name, category),
        )
        if cursor.lastrowid:
            self._conn.commit()
            return cursor.lastrowid
        # Already exists — fetch the ID
        row = self._conn.execute(
            "SELECT id FROM feeds WHERE interest_id = ? AND url = ?", (interest_id, url)
        ).fetchone()
        return row["id"] if row else 0

    def get_all_feeds(self, interest_id: Optional[int] = None) -> list[dict]:
        """Return all configured feeds, optionally filtered by interest."""
        if interest_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM feeds WHERE interest_id = ? ORDER BY id", (interest_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM feeds ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def update_feed(self, feed_id: int, name: str, url: str, category: str) -> bool:
        """Update a feed's name, url, and category. Returns True if a row was updated."""
        cursor = self._conn.execute(
            "UPDATE feeds SET name = ?, url = ?, category = ? WHERE id = ?",
            (name, url, category, feed_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_feed(self, feed_id: int) -> bool:
        """Delete a feed by ID. Returns True if a row was deleted."""
        cursor = self._conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_feed(self, feed_id: int) -> Optional[dict]:
        """Return a single feed by ID, or None."""
        row = self._conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # articles
    # ------------------------------------------------------------------

    def article_exists(self, url: str) -> bool:
        """Check whether an article with the given URL already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM articles WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def insert_article(
        self,
        feed_id: int,
        url: str,
        title: str,
        author: Optional[str],
        published_at: str,
        scraped_at: str,
        rss_excerpt: Optional[str],
        full_content: Optional[str],
        content_status: str,
        pipeline_run_id: int,
    ) -> int:
        """Insert a new article record.

        Returns:
            The ID of the newly created article.
        """
        cursor = self._conn.execute(
            "INSERT INTO articles (feed_id, url, title, author, published_at, "
            "scraped_at, rss_excerpt, full_content, content_status, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feed_id,
                url,
                title,
                author,
                published_at,
                scraped_at,
                rss_excerpt,
                full_content,
                content_status,
                pipeline_run_id,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_articles_for_run(self, pipeline_run_id: int) -> list[dict]:
        """Return all articles associated with a pipeline run."""
        rows = self._conn.execute(
            "SELECT * FROM articles WHERE pipeline_run_id = ? ORDER BY id",
            (pipeline_run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def recover_orphaned_articles(self, current_run_id: int) -> int:
        """Reassign articles from failed runs that were never analyzed to the current run.

        Articles are orphaned when a pipeline run fails after scraping but
        before analysis completes.  This method finds those articles and
        sets their ``pipeline_run_id`` to *current_run_id* so the analyzer
        will see them.

        Returns:
            Number of articles recovered.
        """
        cursor = self._conn.execute(
            """UPDATE articles SET pipeline_run_id = ?
               WHERE pipeline_run_id IN (
                   SELECT pr.id FROM pipeline_runs pr
                   WHERE pr.status = 'failed'
                     AND pr.id != ?
                     AND NOT EXISTS (
                         SELECT 1 FROM themes t
                         WHERE t.pipeline_run_id = pr.id
                     )
               )""",
            (current_run_id, current_run_id),
        )
        self._conn.commit()
        return cursor.rowcount

    def get_article_by_id(self, article_id: int) -> Optional[dict]:
        """Return a single article by ID, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # themes
    # ------------------------------------------------------------------

    def insert_theme(
        self,
        pipeline_run_id: int,
        title: str,
        description: str,
        source_article_ids: list[int],
        novelty_type: str,
        order_index: int,
    ) -> int:
        """Insert a new theme record.

        Args:
            source_article_ids: List of article IDs — stored as JSON array.
        """
        json_ids = json.dumps(source_article_ids)
        cursor = self._conn.execute(
            "INSERT INTO themes (pipeline_run_id, title, description, "
            "source_article_ids, novelty_type, order_index) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pipeline_run_id, title, description, json_ids, novelty_type, order_index),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_themes_for_run(self, pipeline_run_id: int) -> list[dict]:
        """Return all themes for a pipeline run, ordered by order_index."""
        rows = self._conn.execute(
            "SELECT * FROM themes WHERE pipeline_run_id = ? ORDER BY order_index",
            (pipeline_run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_theme_status(self, theme_id: int, status: str) -> None:
        """Update the status of a theme (e.g., 'approved', 'auto_approved')."""
        self._conn.execute(
            "UPDATE themes SET status = ? WHERE id = ?", (status, theme_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # deliverables
    # ------------------------------------------------------------------

    def insert_deliverable(
        self,
        theme_id: int,
        deliverable_type: str,
        content: str,
        version: int,
    ) -> int:
        """Insert a deliverable version.

        Returns:
            The ID of the new deliverable record.
        """
        import datetime

        now = datetime.datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO deliverables (theme_id, deliverable_type, content, "
            "version, created_at) VALUES (?, ?, ?, ?, ?)",
            (theme_id, deliverable_type, content, version, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_latest_deliverables(self, theme_id: int) -> dict[str, dict]:
        """Return the highest-version deliverable per type for a theme.

        Returns:
            A dict mapping deliverable_type to ``{content, version}``.
        """
        rows = self._conn.execute(
            "SELECT d.deliverable_type, d.content, d.version "
            "FROM deliverables d "
            "INNER JOIN ("
            "  SELECT deliverable_type, MAX(version) AS max_version "
            "  FROM deliverables WHERE theme_id = ? "
            "  GROUP BY deliverable_type"
            ") latest ON d.deliverable_type = latest.deliverable_type "
            "AND d.version = latest.max_version "
            "WHERE d.theme_id = ?",
            (theme_id, theme_id),
        ).fetchall()
        return {
            r["deliverable_type"]: {"content": r["content"], "version": r["version"]}
            for r in rows
        }

    def get_deliverable_history(
        self, theme_id: int, deliverable_type: str
    ) -> list[dict]:
        """Return all versions of a deliverable type for a theme, ordered by version."""
        rows = self._conn.execute(
            "SELECT * FROM deliverables WHERE theme_id = ? AND deliverable_type = ? "
            "ORDER BY version ASC",
            (theme_id, deliverable_type),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # evaluation_rounds
    # ------------------------------------------------------------------

    def insert_evaluation_round(
        self,
        theme_id: int,
        round_number: int,
        quality_passed: str,
        quality_feedback: Optional[str],
        adversarial_passed: str,
        adversarial_feedback: Optional[str],
        overall_passed: str,
    ) -> int:
        """Insert an evaluation round record.

        Returns:
            The ID of the new evaluation round.
        """
        import datetime

        now = datetime.datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO evaluation_rounds (theme_id, round_number, quality_passed, "
            "quality_feedback, adversarial_passed, adversarial_feedback, "
            "overall_passed, evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                theme_id,
                round_number,
                quality_passed,
                quality_feedback,
                adversarial_passed,
                adversarial_feedback,
                overall_passed,
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_latest_evaluation(self, theme_id: int) -> Optional[dict]:
        """Return the most recent evaluation round for a theme, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM evaluation_rounds WHERE theme_id = ? "
            "ORDER BY round_number DESC LIMIT 1",
            (theme_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_evaluation_rounds(self, theme_id: int) -> list[dict]:
        """Return all evaluation rounds for a theme, ordered by round_number."""
        rows = self._conn.execute(
            "SELECT * FROM evaluation_rounds WHERE theme_id = ? "
            "ORDER BY round_number ASC",
            (theme_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # daily_briefs
    # ------------------------------------------------------------------

    def insert_daily_brief(
        self, pipeline_run_id: int, content: str, word_count: int
    ) -> int:
        """Insert a daily brief record.

        Returns:
            The ID of the new daily brief.
        """
        import datetime

        now = datetime.datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO daily_briefs (pipeline_run_id, content, word_count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (pipeline_run_id, content, word_count, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_previous_daily_brief(self, current_run_date: str) -> Optional[dict]:
        """Return the most recent daily brief from a completed run before the given date.

        Joins daily_briefs with pipeline_runs to filter by completed status and date.
        """
        row = self._conn.execute(
            "SELECT db.* FROM daily_briefs db "
            "INNER JOIN pipeline_runs pr ON db.pipeline_run_id = pr.id "
            "WHERE pr.run_date < ? AND pr.status = 'completed' "
            "ORDER BY pr.run_date DESC LIMIT 1",
            (current_run_date,),
        ).fetchone()
        return dict(row) if row else None

    def get_daily_brief(self, brief_id: int) -> Optional[dict]:
        """Return a daily brief by ID, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM daily_briefs WHERE id = ?", (brief_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # interests
    # ------------------------------------------------------------------

    def create_interest(self, **kwargs) -> int:
        """Create a new interest. Accepts all interest table columns as kwargs."""
        fields = []
        placeholders = []
        values = []
        valid_cols = {
            "name", "start_time", "interval_hours", "input_data_length_mode",
            "input_word_count", "target_summary_words", "target_script_en_words",
            "target_script_de_words", "target_brief_words", "enable_summary",
            "enable_script_en", "enable_script_de", "enable_brief",
        }
        for col in valid_cols:
            if col in kwargs:
                fields.append(col)
                placeholders.append("?")
                values.append(kwargs[col])
        if "name" not in kwargs:
            raise DatabaseError("Interest 'name' is required")

        cursor = self._conn.execute(
            f"INSERT INTO interests ({', '.join(fields)}) VALUES ({', '.join(placeholders)})",
            values,
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_interest(self, interest_id: int) -> Optional[dict]:
        """Return an interest by ID, or None."""
        row = self._conn.execute(
            "SELECT * FROM interests WHERE id = ?", (interest_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_interest_by_name(self, name: str) -> Optional[dict]:
        """Return an interest by name, or None."""
        row = self._conn.execute(
            "SELECT * FROM interests WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_interests(self) -> list[dict]:
        """Return all interests ordered by name."""
        rows = self._conn.execute(
            "SELECT * FROM interests ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_interest(self, interest_id: int, **kwargs) -> None:
        """Update fields on an interest. Only provided fields are updated."""
        valid_cols = {
            "name", "start_time", "interval_hours", "input_data_length_mode",
            "input_word_count", "target_summary_words", "target_script_en_words",
            "target_script_de_words", "target_brief_words", "enable_summary",
            "enable_script_en", "enable_script_de", "enable_brief",
        }
        fields = []
        values = []
        for col in valid_cols:
            if col in kwargs:
                fields.append(f"{col} = ?")
                values.append(kwargs[col])
        if not fields:
            return
        fields.append("updated_at = datetime('now')")
        values.append(interest_id)
        self._conn.execute(
            f"UPDATE interests SET {', '.join(fields)} WHERE id = ?", values
        )
        self._conn.commit()

    def delete_interest(self, interest_id: int) -> None:
        """Delete an interest and all its associated feeds, pipeline runs cascade."""
        self._conn.execute("DELETE FROM feeds WHERE interest_id = ?", (interest_id,))
        # Pipeline runs cascade via foreign key
        self._conn.execute("DELETE FROM interests WHERE id = ?", (interest_id,))
        self._conn.commit()

    def get_last_successful_run(self, interest_id: Optional[int] = None) -> Optional[dict]:
        """Return the most recent successfully completed pipeline run, or ``None``."""
        if interest_id is not None:
            row = self._conn.execute(
                "SELECT * FROM pipeline_runs WHERE status = 'completed' AND interest_id = ? "
                "ORDER BY id DESC LIMIT 1", (interest_id,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM pipeline_runs WHERE status = 'completed' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_last_run_for_interest(self, interest_id: int) -> Optional[dict]:
        """Return the most recent pipeline run for an interest (any status)."""
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE interest_id = ? ORDER BY id DESC LIMIT 1",
            (interest_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_latest_run_status(self, interest_id: int) -> Optional[dict]:
        """Return the most recent pipeline run status for an interest."""
        row = self._conn.execute(
            "SELECT status, started_at, completed_at, error_message "
            "FROM pipeline_runs WHERE interest_id = ? ORDER BY id DESC LIMIT 1",
            (interest_id,),
        ).fetchone()
        return dict(row) if row else None

    def is_interest_running(self, interest_id: int) -> bool:
        """Check if a pipeline run is currently in progress for this interest."""
        row = self._conn.execute(
            "SELECT 1 FROM pipeline_runs WHERE interest_id = ? AND status = 'running' LIMIT 1",
            (interest_id,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
