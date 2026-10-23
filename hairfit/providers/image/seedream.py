"""Volcengine Ark adapter (seedream family, async task: submit -> poll)."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from hairfit.providers.image.base import ImageDriver, poll_until


class SeedreamDriver(ImageDriver):
    name = "seedream"

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        content: list[dict[str, Any]] = []
        for img in images:
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(img)}}
            )
        content.append({"type": "text", "text": prompt})
        resp = await self._client.post(
            f"{self.base_url}/contents/generations/tasks",
            json={"model": self.model, "content": content},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        task_id: str = resp.json()["id"]
        return await poll_until(lambda: self._poll_task(task_id))

    async def _poll_task(self, task_id: str) -> bytes | None:
        resp = await self._client.get(
            f"{self.base_url}/contents/generations/tasks/{task_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        task: dict[str, Any] = resp.json()
        status = task.get("status", {}).get("status") or task.get("status")
        if status == "succeeded":
            url = task["content"]["video_url"] if "video_url" in task.get("content", {}) else task["content"]["image_urls"][0]
            img = await self._client.get(url)
            img.raise_for_status()
            return img.content
        if status == "failed":
            raise ValueError(f"seedream task failed: {task}")
        return None


def _data_url(img: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(img).decode()


def build_seedream(cfg: Any) -> SeedreamDriver:
    return SeedreamDriver(cfg)
