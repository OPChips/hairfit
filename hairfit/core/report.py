"""Report card composition (Pillow) — design.md 5.1.

Compositional, never one-shot: images come from the per-style generation
pipeline; all text is rendered by code (no typos, i18n-safe).
Not-recommended slots use the library reference image (model face) with ❌
and a reason — no generation spent on them.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hairfit.schemas import FaceAnalysis, StyledResult

THUMB = 360
PAD = 16
HEADER_H = 140
CARD_TEXT_H = 88
BG = (248, 246, 243)
FG = (40, 38, 36)
ACCENT = (34, 139, 87)
REJECT = (200, 60, 50)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load(data: bytes | str) -> Image.Image:
    if isinstance(data, bytes):
        img = Image.open(BytesIO(data))
    else:
        img = Image.open(data)
    return img.convert("RGB")


def _thumb(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((THUMB, THUMB))
    return img


def build_report(
    analysis: FaceAnalysis,
    results: list[StyledResult],
    image_loader: dict[str, bytes],
    rejected: list[tuple[str, str]] | None = None,
    title: str = "HairFit 发型报告",
) -> bytes:
    """Assemble the report card.

    image_loader: style_id -> image bytes (generated try-on or ref image)
    rejected: list of (style_id, reason) for not-recommended slots (ref images)
    """
    from hairfit.kb import load_hairstyles

    styles = load_hairstyles()
    rejected = rejected or []

    cols = max(len(results), len(rejected), 1)
    width = PAD + cols * (THUMB + PAD)
    rows_has_results = bool(results)
    height = HEADER_H + PAD + (THUMB + CARD_TEXT_H + PAD if rows_has_results else 0)
    height += THUMB + CARD_TEXT_H + PAD if rejected else 0
    card = Image.new("RGB", (width, max(height, 400)), BG)
    draw = ImageDraw.Draw(card)

    f_title = _font(40)
    f_body = _font(22)
    f_small = _font(18)

    draw.text((PAD, 28), title, fill=FG, font=f_title)
    summary = (
        f"脸型 {analysis.face_shape} · 比例 {analysis.face_ratio} · 发量 "
        f"{analysis.hair_volume} · 发质 {analysis.hair_texture}"
    )
    draw.text((PAD, 90), summary, fill=(90, 88, 85), font=f_body)

    y = HEADER_H
    if results:
        x = PAD
        for r in results:
            img = _thumb(_load(image_loader[r.style_id]))
            card.paste(img, (x + (THUMB - img.width) // 2, y))
            style = styles.get(r.style_id)
            mark = "✅" if not r.qc_skipped else "✅(未质检)"
            draw.text((x, y + THUMB + 6), f"{mark} {style.name}", fill=ACCENT, font=f_body)
            x += THUMB + PAD
        y += THUMB + CARD_TEXT_H
    for sid, reason in rejected:
        x = PAD
        style = styles.get(sid)
        img = _thumb(_load(image_loader[sid]))
        card.paste(img, (x, y))
        draw.text((x, y + THUMB + 6), f"❌ {style.name}：{reason[:20]}", fill=REJECT, font=f_small)
        y += THUMB + CARD_TEXT_H

    buf = BytesIO()
    card.save(buf, format="PNG")
    return buf.getvalue()


def image_loader_from_files(paths: dict[str, str]) -> dict[str, bytes]:
    """Helper: style_id -> file bytes, from StyledResult.image_paths."""
    return {sid: Path(p).read_bytes() for sid, p in paths.items()}


def image_loader_from_b64(payload: dict[str, str]) -> dict[str, bytes]:
    return {sid: base64.b64decode(b64) for sid, b64 in payload.items()}
