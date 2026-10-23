"""App configuration: models.json + environment-based API keys (12-factor)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """One provider slot: identity, endpoint, key reference, params."""

    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class SegmentConfig(BaseModel):
    """One pipeline segment (analysis or generation) with its failover chain."""

    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    fallbacks: list[ProviderConfig] = []

    def chain(self) -> list[ProviderConfig]:
        head = ProviderConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            params=self.params,
        )
        return [head, *self.fallbacks]


class AppConfig(BaseModel):
    analysis: SegmentConfig
    generation: SegmentConfig

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        file = path or Path(__file__).resolve().parent.parent / "config" / "models.json"
        data = json.loads(file.read_text(encoding="utf-8"))
        return cls(**data)


class MissingAPIKeyError(Exception):
    """The env var referenced by api_key_env is not set."""


def resolve_api_key(cfg: ProviderConfig) -> str:
    """Return the API key for a provider config; raise if unset.

    Mock providers have no api_key_env and return "".
    """
    if not cfg.api_key_env:
        return ""
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"环境变量 {cfg.api_key_env} 未设置（provider={cfg.provider}），跳过该 provider"
        )
    return key
