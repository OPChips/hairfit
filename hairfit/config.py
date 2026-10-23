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
        app = cls(**data)
        _fill_key_env(app.analysis)
        for fb in app.analysis.fallbacks:
            _fill_key_env(fb)
        _fill_key_env(app.generation)
        for fb in app.generation.fallbacks:
            _fill_key_env(fb)
        _env_override("analysis", app.analysis)
        _env_override("generation", app.generation)
        return app


def _fill_key_env(cfg: SegmentConfig | ProviderConfig) -> None:
    """api_key_env omitted in models.json -> derive from provider name."""
    if not cfg.api_key_env:
        cfg.api_key_env = key_env_for(cfg.provider)


def _env_override(segment: str, seg: SegmentConfig) -> None:
    """HAIRFIT_{SEGMENT}_PROVIDER switches the whole segment from env alone.

    model/base_url derive from provider defaults unless explicitly given;
    the chain becomes just that provider (fallbacks cleared).
    """
    provider = os.environ.get(f"HAIRFIT_{segment.upper()}_PROVIDER", "")
    if not provider:
        return
    prefix = f"HAIRFIT_{segment.upper()}"
    defaults = DEFAULT_MODELS if segment == "analysis" else DEFAULT_IMAGE_MODELS
    seg.provider = provider
    seg.model = os.environ.get(f"{prefix}_MODEL", "") or defaults.get(provider, "mock")
    seg.base_url = os.environ.get(f"{prefix}_BASE_URL", "")
    seg.api_key_env = os.environ.get(f"{prefix}_API_KEY_ENV", "") or key_env_for(provider)
    seg.fallbacks = []


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
    # relay/aggregator — one key serves analysis + generation
    "apimart": "https://api.apimart.ai/v1",
    # image generation
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "seedream": "https://ark.cn-beijing.volces.com/api/v3",
    "grok": "https://api.x.ai/v1",
    "mock": "",
}

#: default model per provider — env-only config (HAIRFIT_*_PROVIDER) needs no model name.
#: All analysis defaults are vision-capable (as of 2026-10).
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-6.1-sol",
    "anthropic": "claude-sonnet-4-6",  # all Claude 3+ are vision-native
    "deepseek": "deepseek-flash",  # V4.1 Flash, natively multimodal
    "qwen": "qwen3.8-flash",  # latest multimodal flash on Bailian
    "zhipu": "glm-5.3flash",  # natively multimodal (image/video), 1M ctx
    "doubao": "doubao-seed-2-0-lite-260428",  # Seed 2.0 全模态, Pro id 见方舟模型列表
    "kimi": "kimi-k3",  # natively multimodal; moonshot-v1-*-vision retired 2026-08
    "apimart": "gpt-6.1-sol",  # 中转上的 OpenAI 旗舰（实测 /v1/models 确认，无裸 gpt-6.1）
    "mock": "mock",
}

#: generation segment defaults — names overlap with analysis (openai), hence split
DEFAULT_IMAGE_MODELS: dict[str, str] = {
    "openai": "gpt-image-2.5",  # GPT Images 2.5, API since 2026-09
    "gemini": "gemini-3-pro-image-preview",  # Nano Banana Pro official id
    "seedream": "doubao-seedream-5-0-pro-260628",  # Seedream 5.0 Pro，方舟当前旗舰生图
    "grok": "grok-imagine-image-2.0",  # Grok Imagine Image 2.0，docs.x.ai REST 示例 id
    "apimart": "gpt-image-2",  # apimart 图像 API 统一为异步任务（ext 别名: gpt-image-2-ext）
    "mock": "mock",
}


#: vendor-level key: providers on the same platform share one key env
_KEY_ALIASES: dict[str, str] = {
    # seedream 与 doubao 同为火山方舟（Ark），一把 key 通用
    "seedream": "HAIRFIT_DOUBAO_API_KEY",
}


def key_env_for(provider: str) -> str:
    if provider == "mock":
        return ""
    if provider in _KEY_ALIASES:
        return _KEY_ALIASES[provider]
    return f"HAIRFIT_{provider.upper()}_API_KEY"


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
