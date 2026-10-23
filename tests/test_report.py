"""Tests for report card composition."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from hairfit.core.report import build_report
from hairfit.kb import load_hairstyles
from hairfit.schemas import FaceAnalysis, StyledResult

STYLES = load_hairstyles()


def _png(color: tuple[int, int, int] = (200, 100, 50), size: tuple[int, int] = (400, 400)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _analysis() -> FaceAnalysis:
    return FaceAnalysis(
        face_shape="round",
        face_ratio="约 1.2:1",
        hair_volume="medium",
        hair_texture="1b",
        hairline="normal",
        notes="",
    )


class TestBuildReport:
    def test_with_results(self) -> None:
        results = [
            StyledResult(style_id="textured_crop", image_paths=["x.png"]),
            StyledResult(style_id="crew_cut", image_paths=["y.png"]),
        ]
        loader = {"textured_crop": _png(), "crew_cut": _png((50, 50, 200))}
        out = build_report(_analysis(), results, loader)
        img = Image.open(BytesIO(out))
        assert img.format == "PNG"
        assert img.width >= 2 * 360  # two columns

    def test_with_rejected_only(self) -> None:
        loader = {"buzz_cut": _png((120, 120, 120))}
        out = build_report(_analysis(), [], loader, rejected=[("buzz_cut", "不匹配发量")])
        img = Image.open(BytesIO(out))
        assert img.height > 400

    def test_empty_renders_header(self) -> None:
        out = build_report(_analysis(), [], {})
        img = Image.open(BytesIO(out))
        assert img.width > 0 and img.height > 0
