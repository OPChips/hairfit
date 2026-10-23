"""xAI images adapter (grok image family)."""

from __future__ import annotations

import base64
from typing import Any

from hairfit.providers.image.base import ImageDriver


class GrokDriver(ImageDriver):
    name = "grok"

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        resp = await self._client.post(
            f"{self.base_url}/images/generations",
            json={"model": self.model, "prompt": prompt, "size": size},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        item: dict[str, Any] = resp.json()["data"][0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            img = await self._client.get(item["url"])
            img.raise_for_status()
            return img.content
        raise ValueError("grok response has neither b64_json nor url")


def build_grok(cfg: Any) -> GrokDriver:
    return GrokDriver(cfg)
