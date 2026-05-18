"""Web server for the multi-interest AI News Pipeline.

Provides an authenticated HTTPS web UI for managing interests, feeds,
and global configuration.  Uses Flask with Werkzeug's built-in SSL support
and HTTP Basic Auth.

Routes:
    ``/``                          — Dashboard (GET)
    ``/interest/new``              — Create interest (GET + POST)
    ``/interest/<id>/edit``        — Edit interest (GET + POST)
    ``/interest/<id>/delete``      — Delete interest (POST)
    ``/interest/<id>/run``         — Run now (POST)
    ``/global-config``             — Global config editor (GET + POST)
    ``/interest/<id>/feed/add``    — Add feed (POST)
    ``/interest/<int_id>/feed/<feed_id>/edit``  — Edit feed (POST)
    ``/interest/<int_id>/feed/<feed_id>/delete`` — Delete feed (POST)
"""

from __future__ import annotations

import base64
import binascii as _binascii
import functools
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Callable, Optional

import flask
import yaml

from .certs import ensure_certificates
from .config import from_yaml
from .db import Database
from .models import (
    Config,
    InterestConfig,
    ServerConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PORT = 8443
DEFAULT_CERT_DIR = "/opt/ai-news-pipeline"
SERVER_CONFIG_PATH_ENV = "AI_NEWS_SERVER_CONFIG"

# Valid input data length modes
_LENGTH_MODES: list[tuple[str, str]] = [
    ("headers_only", "Headers Only — scrape RSS metadata only, skip full article fetching"),
    ("word_count", "Word Count — fetch full article, truncate to first N words"),
    ("full_article", "Full Article — fetch and use complete article text"),
]

# Deliverable toggle fields mapping
_DELIVERABLE_FIELDS = {
    "enable_summary": "English Summary",
    "enable_script_en": "English Script",
    "enable_script_de": "German Script",
    "enable_brief": "Daily Brief",
}

# Global config sections
_CONFIG_SECTIONS = {
    "pipeline": "pipeline.yaml",
    "models": "models.yaml",
    "email": "email.yaml",
    "database": "database.yaml",
    "openrouter": "openrouter.yaml",
}


class ServerError(Exception):
    """Raised on server startup or configuration errors."""


# ---------------------------------------------------------------------------
# ServerConfig loading
# ---------------------------------------------------------------------------


def _load_server_config(config_dir: str) -> ServerConfig:
    """Load server config from ``server.yaml`` in the config directory.
    
    If the file doesn't exist, return defaults.
    """
    server_yaml = Path(config_dir) / "server.yaml"
    if server_yaml.exists():
        with open(server_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    return ServerConfig(
        port=data.get("port", DEFAULT_PORT),
        admin_password=data.get("admin_password", ""),
        cert_dir=data.get("cert_dir", DEFAULT_CERT_DIR),
    )


def _save_server_config(config_dir: str, server: ServerConfig) -> None:
    """Save server config to ``server.yaml``."""
    server_yaml = Path(config_dir) / "server.yaml"
    data = {
        "port": server.port,
        "admin_password": server.admin_password,
        "cert_dir": server.cert_dir,
    }
    with open(server_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info("Server config saved to %s", server_yaml)


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------


def _check_auth(username: str, password: str) -> bool:
    """Check if the provided username/password match the admin credentials."""
    admin_password = flask.current_app.config.get("ADMIN_PASSWORD", "")
    if not admin_password:
        return False
    return secrets.compare_digest(username, "admin") and secrets.compare_digest(
        password, admin_password
    )


def _authenticate() -> flask.Response:
    """Send a 401 response with WWW-Authenticate header."""
    response = flask.make_response("Authentication required", 401)
    response.headers["WWW-Authenticate"] = 'Basic realm="AI News Pipeline Admin"'
    return response


def require_auth(f: Callable) -> Callable:
    """Decorator that enforces HTTP Basic Auth on a route."""

    @functools.wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        auth_header = flask.request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return _authenticate()

        try:
            encoded = auth_header.split(" ", 1)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (IndexError, ValueError, _binascii.Error):
            return _authenticate()

        if not _check_auth(username, password):
            return _authenticate()

        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config_dir: str, db: Database) -> flask.Flask:
    """Create and configure the Flask application.

    Args:
        config_dir: Path to the YAML config directory.
        db: Open :class:`Database` instance.

    Returns:
        Configured Flask application.
    """
    app = flask.Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / "templates"),
    )

    # Load server config
    server_config = _load_server_config(config_dir)
    app.config["CONFIG_DIR"] = config_dir
    app.config["ADMIN_PASSWORD"] = server_config.admin_password
    app.config["CERT_DIR"] = server_config.cert_dir
    app.config["DB"] = db
    app.secret_key = secrets.token_hex(32)

    # Register routes
    _register_routes(app, server_config)

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: flask.Flask, server_config: ServerConfig) -> None:
    """Register all route handlers on the Flask app."""

    def _db() -> Database:
        return app.config["DB"]

    def _config() -> Config:
        return from_yaml(app.config["CONFIG_DIR"])

    def _interest_from_db(interest_id: int) -> Optional[InterestConfig]:
        row = _db().get_interest(interest_id)
        if not row:
            return None
        return InterestConfig(
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
        )

    def _interests_list() -> list[dict]:
        """Return all interests with computed dashboard fields."""
        interests = []
        for row in _db().get_all_interests():
            interest_id = row["id"]
            interest = _interest_from_db(interest_id)
            if interest is None:
                continue
            last_run = _db().get_latest_run_status(interest_id)
            info = {
                "id": interest_id,
                "name": interest.name,
                "interval_hours": interest.interval_hours,
                "start_time": interest.start_time,
                "enable_summary": interest.enable_summary,
                "enable_script_en": interest.enable_script_en,
                "enable_script_de": interest.enable_script_de,
                "enable_brief": interest.enable_brief,
                "is_paused": interest.is_paused,
                "last_run_status": last_run["status"] if last_run else "never",
                "last_run_started": last_run["started_at"] if last_run else None,
                "error_message": last_run.get("error_message") if last_run else None,
                "is_running": _db().is_interest_running(interest_id),
            }
            # Compute next run time
            info["next_run"] = _compute_next_run(interest)
            interests.append(info)
        return interests

    # ---- Dashboard ----
    @app.route("/")
    @require_auth
    def dashboard() -> str:
        interests = _interests_list()
        return flask.render_template("dashboard.html", interests=interests)

    # ---- Interest Create ----
    @app.route("/interest/new", methods=["GET", "POST"])
    @require_auth
    def interest_create() -> Any:
        if flask.request.method == "GET":
            return flask.render_template(
                "interest_editor.html",
                interest=None,
                feeds=[],
                length_modes=_LENGTH_MODES,
            )

        # POST: create interest
        return _handle_interest_save(None)

    # ---- Interest Edit ----
    @app.route("/interest/<int:interest_id>/edit", methods=["GET", "POST"])
    @require_auth
    def interest_edit(interest_id: int) -> Any:
        interest = _interest_from_db(interest_id)
        if interest is None:
            flask.abort(404, description="Interest not found")

        if flask.request.method == "GET":
            feeds = _db().get_all_feeds(interest_id)
            return flask.render_template(
                "interest_editor.html",
                interest=interest,
                feeds=feeds,
                length_modes=_LENGTH_MODES,
            )

        # POST: update interest
        return _handle_interest_save(interest_id)

    # ---- Interest Delete ----
    @app.route("/interest/<int:interest_id>/delete", methods=["POST"])
    @require_auth
    def interest_delete(interest_id: int) -> Any:
        interest = _interest_from_db(interest_id)
        if interest is None:
            flask.abort(404, description="Interest not found")

        _db().delete_interest(interest_id)
        assert interest is not None  # abort above guarantees this
        logger.info("Interest '%s' deleted", interest.name)
        flask.flash(f"Interest '{interest.name}' deleted successfully", "success")
        return flask.redirect("/")

    # ---- Run Now ----
    @app.route("/interest/<int:interest_id>/run", methods=["POST"])
    @require_auth
    def run_now(interest_id: int) -> Any:
        interest = _interest_from_db(interest_id)
        if interest is None:
            flask.abort(404, description="Interest not found")

        # Get the scheduler from app config
        scheduler = app.config.get("SCHEDULER")
        if scheduler is None:
            flask.flash("Scheduler not available", "error")
            return flask.redirect("/")

        assert interest is not None  # abort above guarantees this
        error = scheduler.trigger_now(interest_id, interest.name)
        if error:
            flask.flash(error, "error")
        else:
            flask.flash(f"Pipeline run started for '{interest.name}'", "success")

        return flask.redirect("/")

    # ---- Feed Add ----
    @app.route("/interest/<int:interest_id>/feed/add", methods=["POST"])
    @require_auth
    def feed_add(interest_id: int) -> Any:
        interest = _interest_from_db(interest_id)
        if interest is None:
            flask.abort(404, description="Interest not found")

        name = flask.request.form.get("name", "").strip()
        url = flask.request.form.get("url", "").strip()
        category = flask.request.form.get("category", "news").strip()

        if not name or not url:
            flask.flash("Feed name and URL are required", "error")
            return flask.redirect(f"/interest/{interest_id}/edit")

        if not url.startswith(("http://", "https://")):
            flask.flash("Feed URL must start with http:// or https://", "error")
            return flask.redirect(f"/interest/{interest_id}/edit")

        _db().upsert_feed(interest_id, url, name, category)
        flask.flash(f"Feed '{name}' added successfully", "success")
        return flask.redirect(f"/interest/{interest_id}/edit")

    # ---- Feed Edit ----
    @app.route("/interest/<int:interest_id>/feed/<int:feed_id>/edit", methods=["POST"])
    @require_auth
    def feed_edit(interest_id: int, feed_id: int) -> Any:
        feed = _db().get_feed(feed_id)
        if feed is None or feed["interest_id"] != interest_id:
            flask.abort(404, description="Feed not found")

        name = flask.request.form.get("name", "").strip()
        url = flask.request.form.get("url", "").strip()
        category = flask.request.form.get("category", "news").strip()

        if not name or not url:
            flask.flash("Feed name and URL are required", "error")
            return flask.redirect(f"/interest/{interest_id}/edit")

        if not url.startswith(("http://", "https://")):
            flask.flash("Feed URL must start with http:// or https://", "error")
            return flask.redirect(f"/interest/{interest_id}/edit")

        _db().update_feed(feed_id, name, url, category)
        flask.flash(f"Feed '{name}' updated successfully", "success")
        return flask.redirect(f"/interest/{interest_id}/edit")

    # ---- Feed Delete ----
    @app.route("/interest/<int:interest_id>/feed/<int:feed_id>/delete", methods=["POST"])
    @require_auth
    def feed_delete(interest_id: int, feed_id: int) -> Any:
        feed = _db().get_feed(feed_id)
        if feed is None or feed["interest_id"] != interest_id:
            flask.abort(404, description="Feed not found")

        _db().delete_feed(feed_id)
        feed_name = feed["name"] if feed else "unknown"
        flask.flash(f"Feed '{feed_name}' deleted successfully", "success")
        return flask.redirect(f"/interest/{interest_id}/edit")

    # ---- Global Config ----
    @app.route("/global-config", methods=["GET", "POST"])
    @require_auth
    def global_config() -> Any:
        if flask.request.method == "GET":
            return _render_global_config_editor()

        # POST: save global config
        return _handle_global_config_save()

    def _render_global_config_editor() -> str:
        """Build the config editor page from YAML files."""
        config_dir = Path(app.config["CONFIG_DIR"])
        sections: list[dict[str, Any]] = []

        for section_key, yaml_file in _CONFIG_SECTIONS.items():
            file_path = config_dir / yaml_file
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}

            # For secret fields, only show env var names
            fields = []
            if section_key == "openrouter":
                fields.append({
                    "key": "api_key_env",
                    "label": "API Key Environment Variable",
                    "value": data.get("api_key_env", ""),
                    "type": "text",
                    "secret": True,
                    "help": "Environment variable name containing the OpenRouter API key. NOT the key itself.",
                })
                fields.append({
                    "key": "base_url",
                    "label": "Base URL",
                    "value": data.get("base_url", ""),
                    "type": "text",
                    "secret": False,
                })
            elif section_key == "email":
                for key, label, ftype in [
                    ("recipient", "Recipient Email", "text"),
                    ("sender", "Sender Email", "text"),
                    ("smtp_host", "SMTP Host", "text"),
                    ("smtp_port", "SMTP Port", "number"),
                    ("smtp_user", "SMTP Username", "text"),
                    ("smtp_password_env", "SMTP Password Env Variable", "text"),
                ]:
                    fields.append({
                        "key": key,
                        "label": label,
                        "value": data.get(key, ""),
                        "type": ftype,
                        "secret": key == "smtp_password_env",
                        "help": (
                            "Environment variable name containing the SMTP password/app password. NOT the password itself."
                            if key == "smtp_password_env" else ""
                        ),
                    })
            elif section_key == "models":
                for model_type in ("strong", "weak"):
                    model_data = data.get(model_type, {})
                    fields.append({
                        "key": f"{model_type}.id",
                        "label": f"{model_type.title()} Model ID",
                        "value": model_data.get("id", ""),
                        "type": "text",
                        "secret": False,
                    })
                    fields.append({
                        "key": f"{model_type}.temperature",
                        "label": f"{model_type.title()} Temperature",
                        "value": str(model_data.get("temperature", 0.7)),
                        "type": "number",
                        "secret": False,
                        "attrs": 'step="0.1" min="0" max="2"',
                    })
            elif section_key == "pipeline":
                for key, label, ftype, default in [
                    ("schedule", "Schedule (HH:MM)", "text", "04:00"),
                    ("timezone", "Timezone", "text", "Europe/Berlin"),
                    ("max_retries", "Max Retries", "number", "2"),
                    ("max_refinement_rounds", "Max Refinement Rounds", "number", "3"),
                    ("retry_backoff_seconds", "Retry Backoff (seconds)", "number", "30"),
                    ("article_fetch_timeout_seconds", "Article Fetch Timeout (seconds)", "number", "15"),
                    ("llm_request_timeout_seconds", "LLM Request Timeout (seconds)", "number", "120"),
                    ("max_themes", "Max Themes", "number", "10"),
                ]:
                    fields.append({
                        "key": key,
                        "label": label,
                        "value": str(data.get(key, default)),
                        "type": ftype,
                        "secret": False,
                    })
            elif section_key == "database":
                fields.append({
                    "key": "path",
                    "label": "Database Path",
                    "value": data.get("path", ""),
                    "type": "text",
                    "secret": False,
                })

            sections.append({
                "name": section_key,
                "file": yaml_file,
                "fields": fields,
            })

        return flask.render_template("global_config.html", sections=sections)

    def _handle_global_config_save() -> Any:
        """Parse form data and write back to YAML files."""
        config_dir = Path(app.config["CONFIG_DIR"])
        errors = []

        for section_key, yaml_file in _CONFIG_SECTIONS.items():
            file_path = config_dir / yaml_file
            try:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}

            # Update data from form fields
            if section_key == "openrouter":
                data["api_key_env"] = flask.request.form.get("openrouter.api_key_env", "")
                data["base_url"] = flask.request.form.get("openrouter.base_url", "")
            elif section_key == "email":
                for key in ("recipient", "sender", "smtp_host", "smtp_user", "smtp_password_env"):
                    data[key] = flask.request.form.get(f"email.{key}", "")
                try:
                    data["smtp_port"] = int(flask.request.form.get("email.smtp_port", "587"))
                except ValueError:
                    errors.append("SMTP port must be a number")
            elif section_key == "models":
                for model_type in ("strong", "weak"):
                    data.setdefault(model_type, {})
                    data[model_type]["id"] = flask.request.form.get(f"models.{model_type}.id", "")
                    try:
                        data[model_type]["temperature"] = float(
                            flask.request.form.get(f"models.{model_type}.temperature", "0.7")
                        )
                    except ValueError:
                        errors.append(f"{model_type} temperature must be a number")
            elif section_key == "pipeline":
                int_fields = [
                    "max_retries", "max_refinement_rounds", "retry_backoff_seconds",
                    "article_fetch_timeout_seconds", "llm_request_timeout_seconds", "max_themes",
                ]
                str_fields = ["schedule", "timezone"]
                for key in str_fields:
                    data[key] = flask.request.form.get(f"pipeline.{key}", "")
                for key in int_fields:
                    try:
                        data[key] = int(flask.request.form.get(f"pipeline.{key}", "0"))
                    except ValueError:
                        errors.append(f"pipeline.{key} must be an integer")
            elif section_key == "database":
                data["path"] = flask.request.form.get("database.path", "")

            # Write back
            try:
                with open(file_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            except OSError as exc:
                errors.append(f"Failed to write {yaml_file}: {exc}")

        if errors:
            for err in errors:
                flask.flash(err, "error")
            return flask.redirect("/global-config")

        flask.flash("Global configuration saved. Changes take effect on next pipeline run.", "success")
        return flask.redirect("/global-config")

    def _handle_interest_save(interest_id: Optional[int]) -> Any:
        """Handle form submission to create or update an interest."""
        name = flask.request.form.get("name", "").strip()
        start_time = flask.request.form.get("start_time", "04:00").strip()
        try:
            interval_hours = int(flask.request.form.get("interval_hours", "24"))
        except ValueError:
            interval_hours = 24

        data_length_mode = flask.request.form.get("input_data_length_mode", "full_article")
        try:
            input_word_count = int(flask.request.form.get("input_word_count", "256"))
        except ValueError:
            input_word_count = 256

        try:
            target_summary = int(flask.request.form.get("target_summary_words", "750"))
            target_script_en = int(flask.request.form.get("target_script_en_words", "1250"))
            target_script_de = int(flask.request.form.get("target_script_de_words", "1250"))
            target_brief = int(flask.request.form.get("target_brief_words", "700"))
        except ValueError:
            flask.flash("Word count targets must be integers", "error")
            if interest_id is None:
                return flask.render_template(
                    "interest_editor.html", interest=None, feeds=[],
                    length_modes=_LENGTH_MODES,
                )
            return flask.redirect(f"/interest/{interest_id}/edit")

        enable_summary = flask.request.form.get("enable_summary") == "on"
        enable_script_en = flask.request.form.get("enable_script_en") == "on"
        enable_script_de = flask.request.form.get("enable_script_de") == "on"
        enable_brief = flask.request.form.get("enable_brief") == "on"

        # Validation
        if not name:
            flask.flash("Interest name is required", "error")
            if interest_id is None:
                return flask.render_template(
                    "interest_editor.html", interest=None, feeds=[],
                    length_modes=_LENGTH_MODES,
                )
            return flask.redirect(f"/interest/{interest_id}/edit")

        if interval_hours < 1 or interval_hours > 168:
            flask.flash("Run interval must be between 1 and 168 hours", "error")
            if interest_id is None:
                return flask.render_template(
                    "interest_editor.html", interest=None, feeds=[],
                    length_modes=_LENGTH_MODES,
                )
            return flask.redirect(f"/interest/{interest_id}/edit")

        kwargs = {
            "name": name,
            "start_time": start_time,
            "interval_hours": interval_hours,
            "input_data_length_mode": data_length_mode,
            "input_word_count": input_word_count,
            "target_summary_words": target_summary,
            "target_script_en_words": target_script_en,
            "target_script_de_words": target_script_de,
            "target_brief_words": target_brief,
            "enable_summary": enable_summary,
            "enable_script_en": enable_script_en,
            "enable_script_de": enable_script_de,
            "enable_brief": enable_brief,
        }

        if interest_id is None:
            # Create new
            try:
                interest_id = _db().create_interest(**kwargs)
                logger.info("Interest '%s' created (id=%d)", name, interest_id)
            except Exception as exc:
                flask.flash(f"Failed to create interest: {exc}", "error")
                return flask.render_template(
                    "interest_editor.html", interest=None, feeds=[],
                    length_modes=_LENGTH_MODES,
                )
            flask.flash(f"Interest '{name}' created successfully", "success")
        else:
            # Update existing
            try:
                _db().update_interest(interest_id, **kwargs)
                logger.info("Interest '%s' updated (id=%d)", name, interest_id)
            except Exception as exc:
                flask.flash(f"Failed to update interest: {exc}", "error")
                return flask.redirect(f"/interest/{interest_id}/edit")
            flask.flash(f"Interest '{name}' updated successfully", "success")

        # Reschedule interests if scheduler is available
        scheduler = app.config.get("SCHEDULER")
        if scheduler:
            scheduler.reschedule_all()

        return flask.redirect("/")


# ---------------------------------------------------------------------------
# Next run computation
# ---------------------------------------------------------------------------


def _compute_next_run(interest: InterestConfig) -> Optional[str]:
    """Compute the next scheduled run time for an interest.

    Returns a display string like ``"2026-05-14 04:00"``, or ``"Paused"``.
    """
    if interest.is_paused:
        return "Paused"

    from datetime import datetime

    start_h, start_m = _parse_hhmm(interest.start_time)
    interval = max(1, min(interest.interval_hours, 168))
    now = datetime.now()

    # Find the next k*interval slot that hasn't passed yet
    for k in range(0, 24 * 7, interval):  # Check up to a week ahead
        hour = (start_h + k) % 24
        day_offset = (start_h + k) // 24
        run_time = now.replace(
            hour=hour, minute=start_m, second=0, microsecond=0
        ) + __import__("datetime").timedelta(days=day_offset)

        if run_time > now:
            return run_time.strftime("%Y-%m-%d %H:%M")

    return None


def _parse_hhmm(time_str: str) -> tuple[int, int]:
    """Parse HH:MM string into (hour, minute) integers."""
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 4, 0


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------


def run_server(
    config_dir: str,
    db: Database,
    scheduler: Any = None,
    port: Optional[int] = None,
    cert_dir: Optional[str] = None,
    admin_password: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Start the Flask web server with HTTPS.

    Args:
        config_dir: Path to the YAML config directory.
        db: Open :class:`Database` instance.
        scheduler: Optional :class:`PipelineScheduler` for run-now triggers.
        port: Override HTTPS port (from CLI or server.yaml).
        cert_dir: Override certificate directory.
        admin_password: Override admin password.
        debug: Enable Flask debug mode (not for production).
    """
    # Ensure TLS certificates exist
    cert_path, key_path = ensure_certificates(cert_dir or DEFAULT_CERT_DIR)

    app = create_app(config_dir, db)
    app.config["SCHEDULER"] = scheduler

    # Update admin password if provided
    if admin_password:
        app.config["ADMIN_PASSWORD"] = admin_password
        server = _load_server_config(config_dir)
        server.admin_password = admin_password
        _save_server_config(config_dir, server)

    used_port = port or _load_server_config(config_dir).port or DEFAULT_PORT

    logger.info("Starting web server on https://0.0.0.0:%d", used_port)
    logger.info("Certificate: %s", cert_path)

    # Use Flask+Werkzeug with SSL context
    app.run(
        host="0.0.0.0",
        port=used_port,
        ssl_context=(cert_path, key_path),
        debug=debug,
        use_reloader=False,
    )
