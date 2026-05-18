"""Pydantic models for AI News Pipeline configuration."""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal


class FeedDef(BaseModel):
    """A single RSS/Atom feed definition."""
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


class FeedsConfig(BaseModel):
    """Feed list configuration (used for initial migration from YAML)."""
    news: List[FeedDef] = []
    commentators: List[FeedDef] = []


class ModelDef(BaseModel):
    """LLM model definition."""
    id: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=0.0, le=2.0)


class ModelsConfig(BaseModel):
    """Model assignments configuration."""
    strong: ModelDef
    weak: ModelDef


class PipelineConfig(BaseModel):
    """Pipeline execution configuration (global, shared across interests)."""
    schedule: str = "04:00"
    timezone: str = "Europe/Berlin"
    max_retries: int = Field(default=2, ge=0)
    max_refinement_rounds: int = Field(default=3, ge=1)
    retry_backoff_seconds: int = Field(default=30, ge=0)
    article_fetch_timeout_seconds: int = Field(default=15, ge=1)
    llm_request_timeout_seconds: int = Field(default=120, ge=1)
    max_themes: int = Field(default=10, ge=1, le=20)


class EmailConfig(BaseModel):
    """Email delivery configuration."""
    recipient: str = Field(..., min_length=1)
    sender: str = Field(..., min_length=1)
    smtp_host: str = Field(..., min_length=1)
    smtp_port: int = Field(..., ge=1, le=65535)
    smtp_user: str = Field(..., min_length=1)
    smtp_password_env: str = Field(..., min_length=1)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str = Field(..., min_length=1)


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration."""
    api_key_env: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)


class ServerConfig(BaseModel):
    """Web server configuration."""
    port: int = Field(default=8443, ge=1, le=65535)
    admin_password: str = Field(default="", min_length=0)
    cert_dir: str = Field(default="/opt/ai-news-pipeline")


InputDataLengthMode = Literal["headers_only", "word_count", "full_article"]


class InterestConfig(BaseModel):
    """Per-interest configuration stored in the database."""
    id: Optional[int] = None
    name: str = Field(..., min_length=1)
    start_time: str = Field(default="04:00", pattern=r"^\d{2}:\d{2}$")
    interval_hours: int = Field(default=24, ge=1, le=168)
    input_data_length_mode: InputDataLengthMode = "full_article"
    input_word_count: Optional[int] = Field(default=256, ge=1)
    target_summary_words: int = Field(default=750, ge=50)
    target_script_en_words: int = Field(default=1250, ge=50)
    target_script_de_words: int = Field(default=1250, ge=50)
    target_brief_words: int = Field(default=700, ge=50)
    enable_summary: bool = True
    enable_script_en: bool = True
    enable_script_de: bool = True
    enable_brief: bool = True

    @property
    def any_deliverable_enabled(self) -> bool:
        """True if at least one deliverable toggle is enabled."""
        return self.enable_summary or self.enable_script_en or self.enable_script_de or self.enable_brief

    @property
    def is_paused(self) -> bool:
        """True if all deliverable toggles are disabled."""
        return not self.any_deliverable_enabled


class Config(BaseModel):
    """Top-level configuration container."""
    feeds: Optional[FeedsConfig] = None
    models: ModelsConfig
    pipeline: PipelineConfig
    email: EmailConfig
    database: DatabaseConfig
    openrouter: OpenRouterConfig
    server: Optional[ServerConfig] = None
