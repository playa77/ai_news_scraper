"""Configuration loading and validation for AI News Pipeline."""

import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .models import Config


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


def _load_dotenv_file(config_dir: str) -> None:
    """Load .env file from the config directory or parent directories."""
    config_path = Path(config_dir)
    # Try the parent directory (where .env typically lives) first
    env_file = config_path.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return
    # Try the config directory itself
    env_file = config_path / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return
    # Try current working directory
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return


def _validate_url(url: str, feed_type: str, feed_name: str) -> None:
    """Validate that a feed URL is a valid HTTP/HTTPS URL."""
    if not url.startswith(("http://", "https://")):
        raise ConfigError(
            f"Invalid URL for {feed_type} feed '{feed_name}': {url!r} "
            f"(must start with http:// or https://)"
        )


def _validate_env_vars(config_dict: dict) -> None:
    """Validate that referenced environment variables are set."""
    # Check OpenRouter API key
    openrouter_config = config_dict.get("openrouter", {})
    api_key_env = openrouter_config.get("api_key_env", "")
    if api_key_env and api_key_env not in os.environ:
        raise ConfigError(
            f"Environment variable '{api_key_env}' is not set. "
            f"This is required by openrouter.api_key_env in the config."
        )

    # Check email SMTP password
    email_config = config_dict.get("email", {})
    smtp_password_env = email_config.get("smtp_password_env", "")
    if smtp_password_env and smtp_password_env not in os.environ:
        raise ConfigError(
            f"Environment variable '{smtp_password_env}' is not set. "
            f"This is required by email.smtp_password_env in the config."
        )


def from_yaml(path: str) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A validated Config object.

    Raises:
        ConfigError: If the file is missing, YAML is invalid, validation fails,
                     or required environment variables are not set.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    # Load .env file before validating env vars
    _load_dotenv_file(str(config_path))

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config file '{path}': {e}") from e

    if raw is None:
        raise ConfigError(f"Config file '{path}' is empty or contains no YAML content.")

    # Validate required top-level sections
    required_sections = ["feeds", "models", "pipeline", "email", "database", "openrouter"]
    for section in required_sections:
        if section not in raw:
            raise ConfigError(
                f"Missing required section '{section}' in config file '{path}'."
            )

    # Validate feed URLs
    feeds = raw.get("feeds", {})
    for feed in feeds.get("news", []):
        _validate_url(feed.get("url", ""), "news", feed.get("name", "unnamed"))
    for feed in feeds.get("commentators", []):
        _validate_url(feed.get("url", ""), "commentator", feed.get("name", "unnamed"))

    # Validate environment variables are set
    _validate_env_vars(raw)

    # Parse and validate with Pydantic
    try:
        config = Config.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e

    return config
