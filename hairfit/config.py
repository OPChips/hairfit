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


def load_config(path: Path | None = None) -> AppConfig:
    """Alias used by CLI/server entry points."""
    return AppConfig.load(path)


#: official endpoints per provider protocol; base_url may be omitted in config
DEFAULT_BASE_URLS: dict[str, str] = {
    # LLM (multimodal analysis)
    "openai": "https://api.openai.com/v1",
    "openai_compat": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "kimi": "https://api.moonshot.cn/v1",
    "anthropic": "https://api.anthropic.com",
    # image generation
    "gpt_image": "https://api.openai.com/v1",
    "nano_banana": "https://generativelanguage.googleapis.com/v1beta",
    "seedream": "https://ark.cn-beijing.volces.com/api/v3",
    "grok": "https://api.x.ai/v1",
    "mock": "",
}


def resolve_base_url(cfg: ProviderConfig) -> str:
    """Configured base_url wins (proxy/中转); otherwise the official endpoint."""
    if cfg.base_url:
        return cfg.base_url.rstrip("/")
    return DEFAULT_BASE_URLS.get(cfg.provider, "")


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
