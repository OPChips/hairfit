"""Photo ingestion: local path or http(s) URL -> image bytes."""

from __future__ import annotations

from pathlib import Path

import httpx

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


async def load_photo(source: str, client: httpx.AsyncClient | None = None) -> bytes:
    """Load a user photo from a local path or an http(s) URL.

    Raises ValueError for non-image / oversized downloads, FileNotFoundError
    for missing paths, httpx.HTTPError for network failures.
    """
    if not source.startswith(("http://", "https://")):
        return Path(source).read_bytes()
    own = client is None
    c = client or httpx.AsyncClient(follow_redirects=True, timeout=30.0)
    try:
        resp = await c.get(source)
        resp.raise_for_status()
    finally:
        if own:
            await c.aclose()
    ctype = resp.headers.get("content-type", "")
    if ctype and not ctype.startswith("image/"):
        raise ValueError(f"URL 内容不是图片: {ctype}")
    if len(resp.content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"图片超过 {MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB 上限")
    return resp.content
