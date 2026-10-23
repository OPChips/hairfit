"""Provider ports (Protocols) — the only interfaces core/ depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ProviderUnavailable(Exception):
    """A provider cannot serve (missing key, network down, all retries failed)."""


class SchemaValidationError(ProviderUnavailable):
    """The provider served but its output violated the expected schema.

    Retryable with a corrective instruction — callers may re-ask the model.
    """


@runtime_checkable
class AnalysisProvider(Protocol):
    """Multimodal/text LLM: prompt (+optional images) -> validated pydantic model."""

    name: str

    async def analyze(
        self, prompt: str, images: list[bytes], json_schema: type[BaseModel]
    ) -> BaseModel: ...


@runtime_checkable
class GenerationProvider(Protocol):
    """Image editing: instruction + reference images -> edited image bytes.

    Async-task providers (e.g. seedream) poll internally until completion;
    callers always await a single result.
    """

    name: str

    async def edit(self, prompt: str, images: list[bytes], size: str = "1024x1536") -> bytes: ...
