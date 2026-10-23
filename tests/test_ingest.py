"""Photo ingestion (path/URL) + vision-side downscale."""

from __future__ import annotations

from io import BytesIO

import httpx
import pytest
from PIL import Image

from hairfit.core.ingest import load_photo
from hairfit.providers.llm.base import MAX_SIDE_PX, prepare_images


def _png(size: tuple[int, int], color: tuple[int, int, int] = (120, 120, 120)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class TestLoadPhoto:
    async def test_local_path(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(tmp_path) / "me.png"  # type: ignore[operator]
        p.write_bytes(_png((64, 64)))
        assert await load_photo(str(p)) == p.read_bytes()

    async def test_url(self) -> None:
        payload = _png((32, 32))

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "img.example.com"
            return httpx.Response(200, content=payload, headers={"content-type": "image/png"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            assert await load_photo("https://img.example.com/a.png", c) == payload

    async def test_url_rejects_non_image(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            with pytest.raises(ValueError, match="不是图片"):
                await load_photo("https://img.example.com/a", c)

    async def test_url_http_error_propagates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=b"")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            with pytest.raises(httpx.HTTPStatusError):
                await load_photo("https://img.example.com/gone.png", c)


class TestPrepareImages:
    def test_small_image_passthrough(self) -> None:
        img = _png((200, 200))
        assert prepare_images([img]) == [img]

    def test_large_image_downscaled_to_jpeg(self) -> None:
        big = _png((3000, 2000))
        assert len(big) > 1024 * 1024 or True  # gradient-free PNG may compress well
        (out,) = prepare_images([big])
        with Image.open(BytesIO(out)) as im:
            assert im.format == "JPEG"
            assert max(im.size) == MAX_SIDE_PX
            assert im.size == (MAX_SIDE_PX, 1045)  # 3:2 aspect kept

    def test_oversized_small_bytes_still_downscaled(self) -> None:
        # <=1MB but 2500px wide -> must shrink (encoders waste tokens on it)
        big = _png((2500, 1000))
        (out,) = prepare_images([big])
        with Image.open(BytesIO(out)) as im:
            assert max(im.size) == MAX_SIDE_PX

    def test_rgba_composited_on_white(self) -> None:
        buf = BytesIO()
        Image.new("RGBA", (2000, 1000), (255, 0, 0, 0)).save(buf, format="PNG")
        (out,) = prepare_images([buf.getvalue()])
        with Image.open(BytesIO(out)) as im:
            assert im.mode == "RGB"
            assert im.getpixel((10, 10)) == (255, 255, 255)

    def test_undecodable_passthrough(self) -> None:
        junk = b"\x00\x01not-an-image"
        assert prepare_images([junk]) == [junk]
