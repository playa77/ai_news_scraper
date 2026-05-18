"""AI News Pipeline — entry point for the persistent web server + scheduler.

Replaces the old oneshot pipeline entry point.  Supports two modes:

1. ``--serve`` — Start the web server + in-process scheduler (persistent daemon).
2. ``--init-db`` — Initialize the database schema and exit.

The old single-interest batch mode is replaced by the scheduler which
runs ``pipeline.run_interest_pipeline`` per interest.

CLI flags:
    --config DIR        Path to config directory (required)
    --serve             Start the web server + scheduler (persistent)
    --init-db           Initialize schema and exit
    --port PORT         Override HTTPS port (default from server.yaml or 8443)
    --cert-dir DIR      Override certificate directory
    --admin-password PW Override admin password
    --log-file PATH     Path for structured JSON log
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from .config import ConfigError, from_yaml
from .db import Database
from .models import InterestConfig


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI News Pipeline — multi-interest news scraper and content generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="DIR",
        help="Path to the directory containing YAML configuration files.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create the SQLite database schema and exit.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the web server and in-process scheduler (persistent daemon).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="HTTPS listen port (default: 8443 or from server.yaml).",
    )
    parser.add_argument(
        "--cert-dir",
        default=None,
        metavar="DIR",
        help="Directory for TLS certificates (default: /opt/ai-news-pipeline).",
    )
    parser.add_argument(
        "--admin-password",
        default=None,
        metavar="PW",
        help="Admin password for HTTP Basic Auth (overrides server.yaml).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="File path for structured JSON log.",
    )
    return parser


def main() -> None:
    """Parse CLI args and either initialize DB, serve, or run a oneshot pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 1. Load configuration
    try:
        config = from_yaml(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    # 2. Set up structured logging
    from .main_old import setup_logging  # reuse existing logging
    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # 3. Database initialization
    db = Database(config.database.path)
    db.initialize_schema()

    if args.init_db:
        logger.info("Database schema initialized at %s — exiting", config.database.path)
        db.close()
        print(f"Database schema initialized at {config.database.path}")
        sys.exit(0)

    if not args.serve:
        parser.print_help()
        print("\nUse --serve to start the server or --init-db to initialize the database.", file=sys.stderr)
        sys.exit(2)

    # 4. Verify API key
    api_key = os.environ.get(config.openrouter.api_key_env, "")
    if not api_key:
        logger.warning(
            "Environment variable '%s' is not set — pipeline runs will fail",
            config.openrouter.api_key_env,
        )

    # 5. Start scheduler + web server
    logger.info("Starting AI News Pipeline server — config_dir=%s", args.config)

    from .pipeline import run_interest_pipeline
    from .llm import LLMClient
    from .scheduler import PipelineScheduler
    from .server import run_server as _run_server, _load_server_config

    # Load server config
    server_config = _load_server_config(args.config)
    cert_dir = args.cert_dir or server_config.cert_dir
    port = args.port or server_config.port

    # Create LLM client factory for pipeline runs
    def _make_llm_client() -> LLMClient:
        ak = os.environ.get(config.openrouter.api_key_env, "")
        return LLMClient(
            base_url=config.openrouter.base_url,
            api_key=ak,
            timeout=config.pipeline.llm_request_timeout_seconds,
        )

    def _get_interests() -> list[InterestConfig]:
        """Load all interests from DB as InterestConfig objects."""
        rows = db.get_all_interests()
        interests: list[InterestConfig] = []
        for row in rows:
            interests.append(InterestConfig(
                id=row["id"],
                name=row["name"],
                start_time=row["start_time"],
                interval_hours=row["interval_hours"],
                input_data_length_mode=row["input_data_length_mode"],
                input_word_count=row.get("input_word_count"),
                target_summary_words=row["target_summary_words"],
                target_script_en_words=row["target_script_en_words"],
                target_script_de_words=row["target_script_de_words"],
                target_brief_words=row["target_brief_words"],
                enable_summary=bool(row["enable_summary"]),
                enable_script_en=bool(row["enable_script_en"]),
                enable_script_de=bool(row["enable_script_de"]),
                enable_brief=bool(row["enable_brief"]),
            ))
        return interests

    def _is_running(interest_id: int) -> bool:
        return db.is_interest_running(interest_id)

    def _run_interest(interest_id: int) -> None:
        llm = _make_llm_client()
        try:
            run_interest_pipeline(interest_id, db, config, llm)
        finally:
            llm.close()

    # Create scheduler
    scheduler = PipelineScheduler(
        get_interests=_get_interests,
        is_running=_is_running,
        run_interest=_run_interest,
        timezone=config.pipeline.timezone,
    )
    scheduler.start()
    logger.info("Scheduler started with %d interests", len(_get_interests()))

    # Start web server (blocking)
    try:
        _run_server(
            config_dir=args.config,
            db=db,
            scheduler=scheduler,
            port=port,
            cert_dir=cert_dir,
            admin_password=args.admin_password,
            debug=False,
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down on signal")
    finally:
        scheduler.stop()
        db.close()
        logger.info("AI News Pipeline server stopped")


if __name__ == "__main__":
    main()
