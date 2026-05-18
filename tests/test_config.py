"""Tests for the configuration loading and validation layer."""

import os
from pathlib import Path

import pytest
import yaml

from src.config import from_yaml, ConfigError, REQUIRED_SECTIONS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write_config_dir(config_dict, tmp_path):
    """Write each top-level key to a separate YAML file in a temp dir."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for key in config_dict:
        with open(cfg_dir / f"{key}.yaml", "w") as f:
            yaml.dump(config_dict[key], f)
    return str(cfg_dir)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config_dict():
    """Return a dictionary representing a fully valid configuration."""
    return {
        "feeds": {
            "news": [{"name": "Test News", "url": "https://example.com/rss"}],
            "commentators": [{"name": "Test Commentator", "url": "https://example.com/atom"}],
        },
        "models": {
            "strong": {"id": "deepseek/deepseek-v4-pro", "temperature": 0.7},
            "weak": {"id": "deepseek/deepseek-v4-flash", "temperature": 0.7},
        },
        "pipeline": {
            "schedule": "04:00",
            "timezone": "Europe/Berlin",
            "max_retries": 2,
            "max_refinement_rounds": 3,
            "retry_backoff_seconds": 30,
            "article_fetch_timeout_seconds": 15,
            "llm_request_timeout_seconds": 120,
        },
        "email": {
            "recipient": "test@example.com",
            "sender": "sender@example.com",
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "sender@gmail.com",
            "smtp_password_env": "GMAIL_APP_PASSWORD",
        },
        "database": {"path": "test.db"},
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
        },
    }


@pytest.fixture
def set_envs(monkeypatch):
    """Set required environment variables for tests that need them."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
    yield


@pytest.fixture
def config_file(valid_config_dict, tmp_path, set_envs):
    """Write valid config YAML files to a temporary directory and return its path."""
    return write_config_dir(valid_config_dict, tmp_path)


# ---------------------------------------------------------------------------
# 1. Successful config loading
# ---------------------------------------------------------------------------

class TestSuccessfulLoading:
    """Verify that a well-formed config file loads correctly."""

    def test_all_fields_loaded(self, config_file):
        """All fields from the YAML files should be reflected in the Config object."""
        cfg = from_yaml(config_file)

        # -- feeds --
        assert len(cfg.feeds.news) == 1
        assert cfg.feeds.news[0].name == "Test News"
        assert cfg.feeds.news[0].url == "https://example.com/rss"
        assert len(cfg.feeds.commentators) == 1
        assert cfg.feeds.commentators[0].name == "Test Commentator"
        assert cfg.feeds.commentators[0].url == "https://example.com/atom"

        # -- models --
        assert cfg.models.strong.id == "deepseek/deepseek-v4-pro"
        assert cfg.models.strong.temperature == 0.7
        assert cfg.models.weak.id == "deepseek/deepseek-v4-flash"
        assert cfg.models.weak.temperature == 0.7

        # -- pipeline --
        assert cfg.pipeline.schedule == "04:00"
        assert cfg.pipeline.timezone == "Europe/Berlin"
        assert cfg.pipeline.max_retries == 2
        assert cfg.pipeline.max_refinement_rounds == 3
        assert cfg.pipeline.retry_backoff_seconds == 30
        assert cfg.pipeline.article_fetch_timeout_seconds == 15
        assert cfg.pipeline.llm_request_timeout_seconds == 120

        # -- email --
        assert cfg.email.recipient == "test@example.com"
        assert cfg.email.sender == "sender@example.com"
        assert cfg.email.smtp_host == "smtp.gmail.com"
        assert cfg.email.smtp_port == 587
        assert cfg.email.smtp_user == "sender@gmail.com"
        assert cfg.email.smtp_password_env == "GMAIL_APP_PASSWORD"

        # -- database --
        assert cfg.database.path == "test.db"

        # -- openrouter --
        assert cfg.openrouter.api_key_env == "OPENROUTER_API_KEY"
        assert cfg.openrouter.base_url == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# 2. Missing config directory
# ---------------------------------------------------------------------------

class TestMissingFile:
    """ConfigError should be raised when the directory does not exist."""

    def test_missing_file_raises_error(self, set_envs):
        with pytest.raises(ConfigError, match="Config directory not found"):
            from_yaml("/nonexistent/path/")


# ---------------------------------------------------------------------------
# 3. Invalid YAML
# ---------------------------------------------------------------------------

class TestInvalidYaml:
    """ConfigError should be raised for malformed YAML."""

    def test_invalid_yaml_raises_error(self, valid_config_dict, tmp_path, set_envs):
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        # Overwrite one config file with bad YAML
        with open(os.path.join(cfg_dir, "feeds.yaml"), "w") as f:
            f.write(": : invalid yaml !!!\n")
        with pytest.raises(ConfigError, match="Failed to parse YAML in"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 4. Empty / null YAML
# ---------------------------------------------------------------------------

class TestEmptyYaml:
    """ConfigError should be raised when the YAML file is empty or null."""

    def test_empty_file_raises_error(self, valid_config_dict, tmp_path, set_envs):
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        # Overwrite one config file with empty content
        (Path(cfg_dir) / "feeds.yaml").write_text("")
        with pytest.raises(ConfigError, match="empty or contains no YAML content"):
            from_yaml(cfg_dir)

    def test_null_yaml_raises_error(self, valid_config_dict, tmp_path, set_envs):
        """YAML with only a comment or whitespace parses to None."""
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        # Overwrite one config file with null YAML
        (Path(cfg_dir) / "feeds.yaml").write_text("---\n")
        with pytest.raises(ConfigError, match="empty or contains no YAML content"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 5. Missing required top-level sections
# ---------------------------------------------------------------------------

class TestMissingSections:
    """Each required section should raise ConfigError when absent."""

    @pytest.fixture
    def base_config(self, valid_config_dict):
        """Return a writable copy of the valid config dict."""
        import copy
        return copy.deepcopy(valid_config_dict)

    @pytest.mark.parametrize("section", ["models", "pipeline", "email", "database", "openrouter"])
    def test_missing_section_raises_error(self, section, base_config, tmp_path, set_envs):
        del base_config[section]
        cfg_dir = write_config_dir(base_config, tmp_path)
        with pytest.raises(ConfigError, match=f"Missing required config file '{section}.yaml'"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 6. Feed validation – empty lists (feeds are now optional / DB-backed)
# ---------------------------------------------------------------------------

class TestFeedValidation:
    """Empty feed lists are accepted (feeds are loaded from the database)."""

    def test_empty_news_is_accepted(self, valid_config_dict, tmp_path, set_envs):
        """An empty news feed list should not raise an error."""
        valid_config_dict["feeds"]["news"] = []
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        cfg = from_yaml(cfg_dir)
        assert cfg.feeds is not None
        assert cfg.feeds.news == []

    def test_empty_commentators_is_accepted(self, valid_config_dict, tmp_path, set_envs):
        """An empty commentators list should not raise an error."""
        valid_config_dict["feeds"]["commentators"] = []
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        cfg = from_yaml(cfg_dir)
        assert cfg.feeds is not None
        assert cfg.feeds.commentators == []


# ---------------------------------------------------------------------------
# 7. URL validation
# ---------------------------------------------------------------------------

class TestUrlValidation:
    """Feed URLs must start with http:// or https://."""

    @pytest.mark.parametrize("feed_key,feed_name", [
        ("news", "Test News"),
        ("commentators", "Test Commentator"),
    ])
    def test_invalid_url_raises_error(self, feed_key, feed_name, valid_config_dict, tmp_path, set_envs):
        valid_config_dict["feeds"][feed_key][0]["url"] = "ftp://bad.example.com"
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        with pytest.raises(ConfigError, match="Invalid URL"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 8. Environment variable validation
# ---------------------------------------------------------------------------

class TestEnvVarValidation:
    """ConfigError when referenced env vars are not set."""

    def test_missing_openrouter_key(self, valid_config_dict, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # GMAIL_APP_PASSWORD still set so only OPENROUTER_API_KEY is missing
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            from_yaml(cfg_dir)

    def test_missing_gmail_password(self, valid_config_dict, tmp_path, monkeypatch):
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        with pytest.raises(ConfigError, match="GMAIL_APP_PASSWORD"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 9. Pipeline config defaults
# ---------------------------------------------------------------------------

class TestPipelineDefaults:
    """Optional pipeline fields should fall back to documented defaults."""

    def test_defaults_applied(self, valid_config_dict, tmp_path, set_envs):
        # Remove all optional pipeline fields
        valid_config_dict["pipeline"] = {
            "schedule": "04:00",
            "timezone": "Europe/Berlin",
        }
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)

        cfg = from_yaml(cfg_dir)

        assert cfg.pipeline.max_retries == 2
        assert cfg.pipeline.max_refinement_rounds == 3
        assert cfg.pipeline.retry_backoff_seconds == 30
        assert cfg.pipeline.article_fetch_timeout_seconds == 15
        assert cfg.pipeline.llm_request_timeout_seconds == 120


# ---------------------------------------------------------------------------
# 10. Model temperature range
# ---------------------------------------------------------------------------

class TestModelTemperature:
    """Temperature must be in [0.0, 2.0]."""

    @pytest.mark.parametrize("model_key, temp", [
        ("strong", -0.1),
        ("strong", 2.1),
        ("weak", -0.1),
        ("weak", 2.1),
    ])
    def test_out_of_range_temperature_raises_error(
        self, model_key, temp, valid_config_dict, tmp_path, set_envs
    ):
        valid_config_dict["models"][model_key]["temperature"] = temp
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        with pytest.raises(ConfigError, match="temperature"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 11. SMTP port range
# ---------------------------------------------------------------------------

class TestSmtpPort:
    """SMTP port must be between 1 and 65535 inclusive."""

    @pytest.mark.parametrize("port", [0, 65536])
    def test_invalid_port_raises_error(self, port, valid_config_dict, tmp_path, set_envs):
        valid_config_dict["email"]["smtp_port"] = port
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        with pytest.raises(ConfigError, match="smtp_port"):
            from_yaml(cfg_dir)


# ---------------------------------------------------------------------------
# 12. dotenv loading
# ---------------------------------------------------------------------------

class TestDotenvLoading:
    """When a .env file exists, env vars should be loaded from it."""

    def test_dotenv_loaded_from_config_dir(self, valid_config_dict, tmp_path, monkeypatch):
        """.env in the parent directory of the config dir should be loaded."""
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)

        env_path = Path(cfg_dir).parent / ".env"
        env_path.write_text(
            "OPENROUTER_API_KEY=dotenv-key\n"
            "GMAIL_APP_PASSWORD=dotenv-pass\n"
        )

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)

        cfg = from_yaml(cfg_dir)
        assert cfg.openrouter.api_key_env == "OPENROUTER_API_KEY"
        assert cfg.email.smtp_password_env == "GMAIL_APP_PASSWORD"

    def test_dotenv_loaded_from_cwd(self, valid_config_dict, tmp_path, monkeypatch):
        """.env in current working directory should also work."""
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)

        env_path = Path.cwd() / ".env"
        # Use a sentinel value and restore original afterward
        original_exists = env_path.exists()
        original_content = env_path.read_text() if original_exists else ""
        try:
            env_path.write_text(
                "OPENROUTER_API_KEY=dotenv-cwd-key\n"
                "GMAIL_APP_PASSWORD=dotenv-cwd-pass\n"
            )

            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
            monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)

            cfg = from_yaml(cfg_dir)
            assert cfg.openrouter.api_key_env == "OPENROUTER_API_KEY"
            assert cfg.email.smtp_password_env == "GMAIL_APP_PASSWORD"
        finally:
            if original_exists:
                env_path.write_text(original_content)
            else:
                env_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 13. Pydantic type coercion
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    """Pydantic should coerce string values to their declared types where safe."""

    def test_string_max_retries_coerced_to_int(self, valid_config_dict, tmp_path, set_envs):
        """A string "2" for max_retries should be coerced to int 2."""
        valid_config_dict["pipeline"]["max_retries"] = "2"
        cfg_dir = write_config_dir(valid_config_dict, tmp_path)
        cfg = from_yaml(cfg_dir)
        assert cfg.pipeline.max_retries == 2
        assert isinstance(cfg.pipeline.max_retries, int)
