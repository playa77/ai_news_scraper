"""Pydantic models for AI News Pipeline configuration."""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class FeedDef(BaseModel):
    """A single RSS/Atom feed definition."""
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


class FeedsConfig(BaseModel):
    """Feed list configuration."""
    news: List[FeedDef]
    commentators: List[FeedDef]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "FeedsConfig":
        if len(self.news) < 1:
            raise ValueError("feeds.news must contain at least 1 feed")
        if len(self.commentators) < 1:
            raise ValueError("feeds.commentators must contain at least 1 feed")
        return self


class ModelDef(BaseModel):
    """LLM model definition."""
    id: str = Field(..., min_length=1)
    temperature: float = Field(..., ge=0.0, le=2.0)


class ModelsConfig(BaseModel):
    """Model assignments configuration."""
    strong: ModelDef
    weak: ModelDef


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""
    schedule: str = "04:00"
    timezone: str = "Europe/Berlin"
    max_retries: int = Field(default=2, ge=0)
    max_refinement_rounds: int = Field(default=3, ge=1)
    retry_backoff_seconds: int = Field(default=30, ge=0)
    article_fetch_timeout_seconds: int = Field(default=15, ge=1)
    llm_request_timeout_seconds: int = Field(default=120, ge=1)


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


class Config(BaseModel):
    """Top-level configuration container."""
    feeds: FeedsConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    email: EmailConfig
    database: DatabaseConfig
    openrouter: OpenRouterConfig
