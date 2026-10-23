"""Tests for pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hairfit.schemas import (
    FaceAnalysis,
    Hairstyle,
    HairstylesKB,
    Job,
    JobStatus,
    QCScore,
    Recommendation,
    RecommendationResult,
)


def _face_analysis(**overrides: object) -> FaceAnalysis:
    data: dict = {
        "face_shape": "round",
        "face_ratio": "约 1.3:1",
        "hair_volume": "medium",
        "hair_texture": "1b",
        "hairline": "normal",
        "notes": "",
    }
    data.update(overrides)
    return FaceAnalysis(**data)


class TestFaceAnalysis:
    def test_valid(self) -> None:
        fa = _face_analysis()
        assert fa.face_shape == "round"
        assert fa.hair_texture == "1b"

    def test_invalid_texture_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _face_analysis(hair_texture="straight")

    def test_invalid_texture_number_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _face_analysis(hair_texture="5a")

    def test_invalid_face_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _face_analysis(face_shape="hexagon")


class TestRecommendationResult:
    def _rec(self, i: int) -> Recommendation:
        return Recommendation(id=f"style_{i}", reason="r")

    def test_valid(self) -> None:
        res = RecommendationResult(
            face_analysis=_face_analysis(), recommended=[self._rec(i) for i in range(3)]
        )
        assert len(res.recommended) == 3

    def test_too_few_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RecommendationResult(face_analysis=_face_analysis(), recommended=[self._rec(1)])

    def test_too_many_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RecommendationResult(
                face_analysis=_face_analysis(), recommended=[self._rec(i) for i in range(6)]
            )


class TestQCScore:
    def test_pass(self) -> None:
        qc = QCScore(face_consistency=8, style_match=7, verdict="pass")
        assert qc.passed

    def test_low_consistency_fails_even_if_verdict_pass(self) -> None:
        qc = QCScore(face_consistency=4, style_match=9, verdict="pass")
        assert not qc.passed

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QCScore(face_consistency=11, style_match=5, verdict="pass")


class TestJob:
    def test_defaults(self) -> None:
        job = Job(id="j1", style_ids=["a", "b"])
        assert job.status is JobStatus.QUEUED
        assert job.results == []


class TestHairstyle:
    def _style(self, **overrides: object) -> Hairstyle:
        data: dict = {
            "id": "textured_crop",
            "name": "纹理烫短碎盖",
            "gender": "male",
            "length": "short",
            "description": "顶部带纹理感的短碎发",
            "volume_demand": "medium",
            "hair_texture_fit": ["1a", "1b", "2a"],
            "suit_face": ["round", "square"],
            "avoid_face": ["oblong"],
            "scene_tags": ["daily"],
            "ref_image": "assets/styles/male/textured_crop.jpg",
        }
        data.update(overrides)
        return Hairstyle(**data)

    def test_valid(self) -> None:
        assert self._style().id == "textured_crop"

    def test_texture_fit_validated(self) -> None:
        with pytest.raises(ValidationError):
            self._style(hair_texture_fit=["straight"])

    def test_kb_get(self) -> None:
        kb = HairstylesKB(schema_version="1.0", items=[self._style()])
        assert kb.get("textured_crop").name == "纹理烫短碎盖"

    def test_kb_get_unknown_raises(self) -> None:
        kb = HairstylesKB(schema_version="1.0", items=[self._style()])
        with pytest.raises(KeyError):
            kb.get("nope")
