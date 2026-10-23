"""Shared plumbing for image-generation HTTP adapters."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from hairfit.config import ProviderConfig, resolve_api_key
from hairfit.log import get_logger
from hairfit.providers.base import ProviderUnavailable

log = get_logger(__name__)

T = TypeVar("T")

#: providers whose API is submit-task -> poll-status
POLL_INTERVAL_S = 2.0
POLL_DEADLINE_S = 300.0


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    if isinstance(exc, httpx.TransportError):
        return True
    return False


async def poll_until(
    fn: Callable[[], Awaitable[T | None]],
    interval: float = POLL_INTERVAL_S,
    deadline_s: float = POLL_DEADLINE_S,
) -> T:
    """Poll fn() until it returns non-None or the deadline passes."""
    loop = asyncio.get_event_loop()
    start = time.monotonic()
    while True:
        result = await fn()
        if result is not None:
            return result
        if time.monotonic() - start > deadline_s:
            raise ProviderUnavailable(f"polling timed out after {deadline_s}s")
        await loop.sleep(interval)


class ImageDriver:
    """Base class for image-generation adapters; subclasses implement `_request`."""

    name: str

    def __init__(self, cfg: ProviderConfig, timeout: float = 120.0) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.params: dict[str, Any] = dict(cfg.params)
        self.api_key = resolve_api_key(cfg)
        self.base_url = cfg.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def edit(self, prompt: str, images: list[bytes], size: str = "1024x1536") -> bytes:
        start = time.monotonic()
        try:
            data = await self._request(prompt, images, size)
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailable(
                f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"{self.name} transport error: {exc}") from exc
        except ProviderUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize to ProviderUnavailable
            raise ProviderUnavailable(f"{self.name} unexpected error: {exc}") from exc
        log.info(
            "image_call",
            provider=self.name,
            model=self.model,
            elapsed_s=round(time.monotonic() - start, 2),
            ref_images=len(images),
        )
        return data

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        raise NotImplementedError
