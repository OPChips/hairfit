"""APIMart relay adapter (OpenAI-compatible relay; image API is task-based).

Contract (docs.apimart.ai, 2026-10):
- POST /v1/images/generations {model, prompt, n, size, image_urls[]} -> envelope
  {code, data: [{status: "submitted", task_id}]}
- GET /v1/tasks/{task_id} -> {code, data: {status, progress, result?}}; statuses
  pending/processing/completed/failed/cancelled; result.images[0].url[0] is a
  temporary download link (24-72h) -> must be fetched to bytes here.
- image_urls accept public URLs or base64 data URIs (up to 15, <=20MB each).
- Chat segment is plain OpenAI Chat Completions -> reuse build_openai_compat.
"""

from __future__ import annotations

import base64
from typing import Any

from hairfit.providers.base import ProviderUnavailable
from hairfit.providers.image.base import ImageDriver, poll_until

_ENVELOPE_OK = (None, 200, "200")
_TASK_DONE = ("completed",)
_TASK_FAILED = ("failed", "cancelled")

#: some models (e.g. gpt-image-2.5-*) reject pixel sizes; they take ratios only
_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9")


def _pixel_to_ratio(size: str) -> str | None:
    """Map a WxH pixel size to the closest supported aspect-ratio string."""
    try:
        w_str, h_str = size.lower().split("x")
        w, h = int(w_str), int(h_str)
    except ValueError:
        return None
    target = w / h

    def _ratio_value(r: str) -> float:
        rw, rh = r.split(":")
        return int(rw) / int(rh)

    return min(_RATIOS, key=lambda r: abs(_ratio_value(r) - target))


class ApimartDriver(ImageDriver):
    name = "apimart"

    async def _request(self, prompt: str, images: list[bytes], size: str) -> bytes:
        body: dict[str, Any] = {"model": self.model, "prompt": prompt, "n": 1, "size": size}
        if images:
            body["image_urls"] = [_data_url(img) for img in images]
        resp = await self._client.post(
            f"{self.base_url}/images/generations",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code == 400 and "size must be" in resp.text:
            # model wants aspect ratios — retry once with the closest ratio
            ratio = _pixel_to_ratio(size)
            if ratio:
                body["size"] = ratio
                resp = await self._client.post(
                    f"{self.base_url}/images/generations",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        if payload.get("code") not in _ENVELOPE_OK:
            raise ProviderUnavailable(f"apimart submit failed: {str(payload)[:300]}")
        data = payload.get("data") or []
        item = data[0] if data else {}
        task_id = item.get("task_id")
        if task_id:
            return await poll_until(lambda: self._poll_task(task_id))
        # some models return synchronously with b64
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        raise ProviderUnavailable(f"apimart unexpected response: {str(payload)[:300]}")

    async def _poll_task(self, task_id: str) -> bytes | None:
        resp = await self._client.get(
            f"{self.base_url}/tasks/{task_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        task: dict[str, Any] = resp.json().get("data") or {}
        status = task.get("status")
        if status in _TASK_DONE:
            images = (task.get("result") or {}).get("images") or []
            urls = images[0].get("url") if images else None
            if not urls:
                raise ProviderUnavailable(f"apimart task {task_id} completed without image")
            img = await self._client.get(urls[0])
            img.raise_for_status()
            return img.content
        if status in _TASK_FAILED:
            raise ProviderUnavailable(f"apimart task {task_id} {status}: {task.get('error')}")
        return None  # submitted / pending / processing


def _data_url(img: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(img).decode()


def build_apimart(cfg: Any) -> ApimartDriver:
    return ApimartDriver(cfg)
