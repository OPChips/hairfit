"""Shared plumbing for LLM HTTP adapters: client, retry, logging."""

from __future__ import annotations

import time
from typing import Any, cast

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from hairfit.config import ProviderConfig, resolve_api_key, resolve_base_url
from hairfit.log import get_logger
from hairfit.providers.base import ProviderUnavailable, SchemaValidationError

log = get_logger(__name__)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)  # timeouts, connection errors


class LLMDriver:
    """Base class for LLM HTTP adapters; subclasses implement `_request`."""

    name: str

    def __init__(self, cfg: ProviderConfig, timeout: float = 120.0) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.params: dict[str, Any] = dict(cfg.params)
        self.api_key = resolve_api_key(cfg)
        self.base_url = resolve_base_url(cfg)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _send(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        start = time.monotonic()
        resp = await self._client.post(self._endpoint(), json=payload, headers=headers)
        elapsed = time.monotonic() - start
        resp.raise_for_status()
        log.info(
            "llm_call",
            provider=self.name,
            model=self.model,
            elapsed_s=round(elapsed, 2),
            prompt_chars=len(str(payload)),
        )
        return cast(dict[str, Any], resp.json())

    def _endpoint(self) -> str:
        raise NotImplementedError

    async def analyze(
        self, prompt: str, images: list[bytes], json_schema: type[BaseModel]
    ) -> BaseModel:
        """Public API: build payload, send, parse into schema; unified failure type."""
        try:
            payload, headers = self._build_request(prompt, images)
            data = await self._send(payload, headers)
            text = self._extract_text(data)
        except ProviderUnavailable:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailable(
                f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"{self.name} transport error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — normalize to ProviderUnavailable
            raise ProviderUnavailable(f"{self.name} unexpected error: {exc}") from exc
        return self._parse(text, json_schema)

    def _build_request(
        self, prompt: str, images: list[bytes]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        raise NotImplementedError

    def _extract_text(self, data: dict[str, Any]) -> str:
        raise NotImplementedError

    def _parse(self, text: str, json_schema: type[BaseModel]) -> BaseModel:
        text = strip_json_fences(text)
        try:
            return json_schema(**json_loads_strict(text))
        except (ValidationError, ValueError) as exc:
            raise SchemaValidationError(
                f"{self.name} returned invalid JSON for {json_schema.__name__}: {exc}"
            ) from exc


def strip_json_fences(text: str) -> str:
    """Extract JSON object from a possibly markdown-fenced response."""
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        t = t[first_newline + 1 :] if first_newline != -1 else t
        end = t.rfind("```")
        if end != -1:
            t = t[:end]
    return t.strip()


def json_loads_strict(text: str) -> dict[str, Any]:
    import json

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return obj
