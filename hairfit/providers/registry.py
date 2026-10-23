"""Provider factory + failover chains.

Builds ordered provider chains from config (missing API key -> skip with a
warning); wraps them so the first healthy provider serves and failures
cascade to the next.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from hairfit.config import AppConfig, MissingAPIKeyError, ProviderConfig
from hairfit.log import get_logger
from hairfit.providers.base import (
    AnalysisProvider,
    GenerationProvider,
    ProviderUnavailable,
)
from hairfit.providers.image.gpt_image import build_gpt_image
from hairfit.providers.image.grok import build_grok
from hairfit.providers.image.mock import build_mock_image
from hairfit.providers.image.nano_banana import build_nano_banana
from hairfit.providers.image.seedream import build_seedream
from hairfit.providers.llm.anthropic import build_anthropic
from hairfit.providers.llm.mock import build_mock_llm
from hairfit.providers.llm.openai_compat import build_openai_compat

log = get_logger(__name__)

_LLM_FACTORIES: dict[str, Callable[[ProviderConfig], AnalysisProvider]] = {
    "openai": build_openai_compat,
    "openai_compat": build_openai_compat,
    "deepseek": build_openai_compat,
    "qwen": build_openai_compat,
    "zhipu": build_openai_compat,
    "doubao": build_openai_compat,
    "kimi": build_openai_compat,
    "anthropic": build_anthropic,
    "mock": build_mock_llm,
}

_IMAGE_FACTORIES: dict[str, Callable[[ProviderConfig], GenerationProvider]] = {
    "openai": build_gpt_image,
    "gemini": build_nano_banana,
    "seedream": build_seedream,
    "grok": build_grok,
    "mock": build_mock_image,
}


def _build_llm(cfg: ProviderConfig) -> AnalysisProvider:
    factory = _LLM_FACTORIES.get(cfg.provider)
    if factory is None:
        raise ProviderUnavailable(f"unknown LLM provider: {cfg.provider}")
    return factory(cfg)


def _build_image(cfg: ProviderConfig) -> GenerationProvider:
    factory = _IMAGE_FACTORIES.get(cfg.provider)
    if factory is None:
        raise ProviderUnavailable(f"unknown image provider: {cfg.provider}")
    return factory(cfg)


class FailoverAnalysis:
    """Tries each configured provider in order; surfaces the last error."""

    name = "analysis_chain"

    def __init__(self, providers: list[AnalysisProvider]) -> None:
        if not providers:
            raise ProviderUnavailable("no analysis provider available (check API keys)")
        self.providers = providers

    async def analyze(
        self, prompt: str, images: list[bytes], json_schema: type[BaseModel]
    ) -> BaseModel:
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                return await p.analyze(prompt, images, json_schema)
            except ProviderUnavailable as exc:
                log.warning("analysis_provider_failed", provider=p.name, error=str(exc))
                last_exc = exc
        raise ProviderUnavailable(f"all analysis providers failed; last error: {last_exc}")

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()  # type: ignore[attr-defined]


class FailoverGeneration:
    name = "generation_chain"

    def __init__(self, providers: list[GenerationProvider]) -> None:
        if not providers:
            raise ProviderUnavailable("no generation provider available (check API keys)")
        self.providers = providers

    async def edit(self, prompt: str, images: list[bytes], size: str = "1024x1536") -> bytes:
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                return await p.edit(prompt, images, size)
            except ProviderUnavailable as exc:
                log.warning("generation_provider_failed", provider=p.name, error=str(exc))
                last_exc = exc
        raise ProviderUnavailable(f"all generation providers failed; last error: {last_exc}")

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()  # type: ignore[attr-defined]


def build_analysis_chain(app: AppConfig) -> FailoverAnalysis:
    providers: list[AnalysisProvider] = []
    for cfg in app.analysis.chain():
        try:
            providers.append(_build_llm(cfg))
        except (MissingAPIKeyError, ProviderUnavailable) as exc:
            log.warning("provider_skipped", provider=cfg.provider, detail=str(exc))
    return FailoverAnalysis(providers)


def build_generation_chain(app: AppConfig) -> FailoverGeneration:
    providers: list[GenerationProvider] = []
    for cfg in app.generation.chain():
        try:
            providers.append(_build_image(cfg))
        except (MissingAPIKeyError, ProviderUnavailable) as exc:
            log.warning("provider_skipped", provider=cfg.provider, detail=str(exc))
    return FailoverGeneration(providers)
