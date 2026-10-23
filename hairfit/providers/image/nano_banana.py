"""Google Gemini image adapter (nano-banana family, generateContent)."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from hairfit.providers.image.base import ImageDriver


class NanoBananaDriver(ImageDriver):
    name = "nano_banana"

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        parts: list[dict[str, Any]] = []
        for img in images:
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img).decode()}})
        parts.append({"text": prompt})
        resp = await self._client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            json={"contents": [{"parts": parts}]},
            headers={"x-goog-api-key": self.api_key},
        )
        resp.raise_for_status()
        for candidate in resp.json().get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise ValueError("no image in Gemini response")


def build_nano_banana(cfg: Any) -> NanoBananaDriver:
    return NanoBananaDriver(cfg)
