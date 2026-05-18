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
    """Load .env file from the config directory or its parent."""
    config_path = Path(config_dir)
    # Try the parent of config_dir (where .env typically lives) first
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


REQUIRED_SECTIONS = ["models", "pipeline", "email", "database", "openrouter"]


def from_yaml(config_dir: str) -> Config:
    """Load and validate configuration from a directory of domain-specific YAML files.

    Each ``.yaml`` file in the directory becomes a top-level config section.
    The filename stem (without ``.yaml``) is used as the section key.
    For example, ``feeds.yaml`` maps to the ``feeds`` section.

    Args:
        config_dir: Path to the config directory containing ``*.yaml`` files.

    Returns:
        A validated Config object.

    Raises:
        ConfigError: If the directory is missing, no YAML files are found,
                     required sections are missing, or validation fails.
    """
    config_path = Path(config_dir)

    if not config_path.is_dir():
        raise ConfigError(f"Config directory not found: {config_dir}")

    # Load .env file before validating env vars
    _load_dotenv_file(str(config_path))

    # Load all .yaml files from the directory
    raw: dict = {}
    yaml_files = sorted(config_path.glob("*.yaml"))
    if not yaml_files:
        raise ConfigError(
            f"No .yaml config files found in '{config_dir}'. "
            f"Expected files: {', '.join(f + '.yaml' for f in REQUIRED_SECTIONS)}"
        )

    for yaml_file in yaml_files:
        section_name = yaml_file.stem
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML in '{yaml_file}': {e}") from e

        if data is None:
            raise ConfigError(f"Config file '{yaml_file}' is empty or contains no YAML content.")

        raw[section_name] = data

    # Validate required top-level sections
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ConfigError(
                f"Missing required config file '{section}.yaml' in '{config_dir}'."
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
