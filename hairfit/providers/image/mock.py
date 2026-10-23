"""Deterministic mock image generator: draws a labeled gradient placeholder."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw

from hairfit.config import ProviderConfig


class MockImageDriver:
    name = "mock"

    def __init__(self, cfg: ProviderConfig, **_: Any) -> None:
        self.model = cfg.model
        self.calls = 0
        self.fail_times = 0

    async def aclose(self) -> None: ...

    async def edit(self, prompt: str, images: list[bytes], size: str = "1024x1536") -> bytes:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            from hairfit.providers.base import ProviderUnavailable

            raise ProviderUnavailable("mock image: simulated failure")
        w, h = (int(x) for x in size.lower().split("x"))
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
        top = (seed % 200, (seed >> 3) % 200, (seed >> 6) % 200)
        bottom = tuple(255 - c for c in top)
        img = Image.new("RGB", (w, h))
        px = img.load()
        assert px is not None
        for y in range(h):
            row_color = tuple(
                int(top[i] + (bottom[i] - top[i]) * y / h) for i in range(3)
            )
            for x in range(w):
                px[x, y] = row_color  # type: ignore[index]
        draw = ImageDraw.Draw(img)
        draw.rectangle((w // 4, h // 3, w * 3 // 4, h * 2 // 3), outline="white", width=4)
        draw.text((w // 2 - 60, h // 2 - 10), "MOCK GENERATED", fill="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def build_mock_image(cfg: Any) -> MockImageDriver:
    return MockImageDriver(cfg)
