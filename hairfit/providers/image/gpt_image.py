"""OpenAI images/edits adapter (gpt-image family, sync b64 response)."""

from __future__ import annotations

from typing import Any

from hairfit.providers.image.base import ImageDriver


class GPTImageDriver(ImageDriver):
    name = "gpt_image"

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for i, img in enumerate(images):
            files.append(("image[]", (f"image{i}.png", img, "image/png")))
        data: dict[str, str] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
        }
        data.update({k: str(v) for k, v in self.params.items() if k != "size"})
        resp = await self._client.post(f"{self.base_url}/images/edits", data=data, files=files)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        import base64

        return base64.b64decode(payload["data"][0]["b64_json"])


def build_gpt_image(cfg: Any) -> GPTImageDriver:
    return GPTImageDriver(cfg)
