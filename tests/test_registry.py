"""Tests for provider registry failover chains."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from hairfit.config import AppConfig, MissingAPIKeyError, ProviderConfig, SegmentConfig
from hairfit.providers.base import ProviderUnavailable
from hairfit.providers.registry import build_analysis_chain, build_generation_chain


class Out(BaseModel):
    answer: str = "x"


def _seg(provider: str, fallbacks: list[ProviderConfig] | None = None) -> SegmentConfig:
    return SegmentConfig(provider=provider, model="m", api_key_env="", fallbacks=fallbacks or [])


def _cfg(provider: str, key_env: str = "") -> ProviderConfig:
    return ProviderConfig(provider=provider, model="m", api_key_env=key_env)


def _app(analysis: SegmentConfig, generation: SegmentConfig) -> AppConfig:
    return AppConfig(analysis=analysis, generation=generation)


class TestAnalysisChain:
    def test_missing_key_skipped_mock_serves(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("HAIRFIT_NOPE", raising=False)
        app = _app(_seg("openai", [_cfg("mock")]), _seg("mock"))
        app.analysis.api_key_env = "HAIRFIT_NOPE"
        chain = build_analysis_chain(app)
        assert [p.name for p in chain.providers] == ["mock"]

    async def test_failover_to_next(self) -> None:
        app = _app(_seg("mock", [_cfg("mock")]), _seg("mock"))
        chain = build_analysis_chain(app)
        first = chain.providers[0]
        assert hasattr(first, "fail_times")
        first.fail_times = 1  # type: ignore[attr-defined]
        out = await chain.analyze("p", [], Out)
        assert isinstance(out, Out)

    async def test_all_fail_raises(self) -> None:
        app = _app(_seg("mock"), _seg("mock"))
        chain = build_analysis_chain(app)
        for p in chain.providers:
            p.fail_times = 99  # type: ignore[attr-defined]
        with pytest.raises(ProviderUnavailable, match="all analysis providers"):
            await chain.analyze("p", [], Out)

    def test_no_provider_available(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("HAIRFIT_NOPE", raising=False)
        app = _app(_seg("openai"), _seg("mock"))
        app.analysis.fallbacks = []
        app.analysis.provider = "openai"
        app.analysis.api_key_env = "HAIRFIT_NOPE"
        with pytest.raises(ProviderUnavailable):
            build_analysis_chain(app)

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ProviderUnavailable, match="unknown LLM provider"):
            from hairfit.providers.registry import _build_llm

            _build_llm(_cfg("madeup"))


class TestGenerationChain:
    def test_unknown_head_skipped_mock_serves(self, monkeypatch: Any) -> None:
        # head provider name unknown -> factory raises inside build; skipped,
        # mock fallback serves
        from hairfit.config import ProviderConfig as PC

        app = _app(_seg("mock"), _seg("madeup", [PC(provider="mock", model="m")]))
        chain = build_generation_chain(app)
        assert [p.name for p in chain.providers] == ["mock"]

    def test_unknown_image_provider(self) -> None:
        from hairfit.providers.registry import _build_image

        with pytest.raises(ProviderUnavailable, match="unknown image provider"):
            _build_image(_cfg("madeup"))

    def test_missing_key_error_raised_when_skipped(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("HAIRFIT_NOPE", raising=False)
        with pytest.raises(MissingAPIKeyError):
            from hairfit.config import resolve_api_key

            resolve_api_key(ProviderConfig(provider="x", model="m", api_key_env="HAIRFIT_NOPE"))


class TestBaseUrlDefaults:
    def test_configured_wins(self) -> None:
        from hairfit.config import resolve_base_url

        cfg = ProviderConfig(provider="openai", model="m", base_url="https://proxy.example.com/v1/")
        assert resolve_base_url(cfg) == "https://proxy.example.com/v1"

    def test_official_defaults(self) -> None:
        from hairfit.config import resolve_base_url

        assert (
            resolve_base_url(ProviderConfig(provider="openai", model="m"))
            == "https://api.openai.com/v1"
        )
        assert (
            resolve_base_url(ProviderConfig(provider="anthropic", model="m"))
            == "https://api.anthropic.com"
        )
        assert (
            resolve_base_url(ProviderConfig(provider="nano_banana", model="m"))
            == "https://generativelanguage.googleapis.com/v1beta"
        )
        assert resolve_base_url(ProviderConfig(provider="mock", model="m")) == ""

    def test_adapters_use_default_when_omitted(self) -> None:
        from hairfit.providers.image.gpt_image import GPTImageDriver
        from hairfit.providers.llm.openai_compat import OpenAICompatLLM

        assert OpenAICompatLLM(_cfg("openai")).base_url == "https://api.openai.com/v1"
        assert GPTImageDriver(_cfg("gpt_image")).base_url == "https://api.openai.com/v1"


class TestEnvOverride:
    def test_provider_switch_with_defaults(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("HAIRFIT_ANALYSIS_PROVIDER", "zhipu")
        monkeypatch.setenv("HAIRFIT_GENERATION_PROVIDER", "seedream")
        app = AppConfig.load()
        assert app.analysis.provider == "zhipu"
        assert app.analysis.model == "glm-5.3flash"
        assert app.analysis.base_url == ""
        assert app.analysis.api_key_env == "HAIRFIT_ZHIPU_API_KEY"
        assert [p.provider for p in app.analysis.fallbacks] == ["mock"]
        assert app.generation.provider == "seedream"
        assert app.generation.model == "doubao-seedream-4.0"
        assert app.generation.api_key_env == "HAIRFIT_SEEDREAM_API_KEY"

    def test_explicit_model_and_base_url_win(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("HAIRFIT_ANALYSIS_PROVIDER", "openai")
        monkeypatch.setenv("HAIRFIT_ANALYSIS_MODEL", "my-model")
        monkeypatch.setenv("HAIRFIT_ANALYSIS_BASE_URL", "https://proxy.example.com/v1")
        app = AppConfig.load()
        assert app.analysis.model == "my-model"
        assert app.analysis.base_url == "https://proxy.example.com/v1"

    def test_no_env_uses_config_file(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("HAIRFIT_ANALYSIS_PROVIDER", raising=False)
        monkeypatch.delenv("HAIRFIT_GENERATION_PROVIDER", raising=False)
        app = AppConfig.load()
        assert app.analysis.provider == "openai"  # from models.json

    def test_key_env_derived_in_config_file(self, monkeypatch: Any) -> None:
        """api_key_env omitted in models.json -> derived from provider."""
        monkeypatch.delenv("HAIRFIT_ANALYSIS_PROVIDER", raising=False)
        import json as _json
        from pathlib import Path as P

        p = P("/tmp/hairfit_nokeyenv.json")
        p.write_text(
            _json.dumps(
                {
                    "analysis": {"provider": "kimi", "model": "m", "params": {}},
                    "generation": {"provider": "nano_banana", "model": "m", "params": {}},
                }
            )
        )
        app = AppConfig.load(p)
        assert app.analysis.api_key_env == "HAIRFIT_KIMI_API_KEY"
        assert app.generation.api_key_env == "HAIRFIT_GEMINI_API_KEY"  # override map
