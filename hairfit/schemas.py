"""Pydantic models for LLM outputs, QC scores, jobs and knowledge bases."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

FaceShape = Literal[
    "oval", "round", "square", "heart", "diamond", "oblong", "triangle"
]
HairVolume = Literal["thin", "medium", "thick"]
Hairline = Literal["normal", "high", "receding"]

#: Andre Walker hair typing: type 1-4, subtype a/b/c (see design.md 3.4)
HairTexture = Annotated[str, Field(pattern=r"^[1-4][abc]$")]

_ID_RE = re.compile(r"^[a-z0-9_]+$")


class FaceAnalysis(BaseModel):
    face_shape: FaceShape
    face_ratio: str = Field(description="e.g. '约 1.3:1'")
    hair_volume: HairVolume
    hair_texture: HairTexture
    hairline: Hairline
    notes: str = ""


class Recommendation(BaseModel):
    id: str
    reason: str

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"invalid style id: {v!r}")
        return v


class RecommendationResult(BaseModel):
    face_analysis: FaceAnalysis
    recommended: list[Recommendation] = Field(min_length=3, max_length=5)


class QCScore(BaseModel):
    face_consistency: int = Field(ge=0, le=10)
    style_match: int = Field(ge=0, le=10)
    verdict: Literal["pass", "retry"]
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "pass" and self.face_consistency >= 6


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


class StyledResult(BaseModel):
    style_id: str
    image_paths: list[str] = []
    qc_face_consistency: int | None = None
    qc_style_match: int | None = None
    qc_skipped: bool = False  # degradation: QC unavailable, delivered unverified
    retry_used: bool = False
    error: str | None = None


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    style_ids: list[str] = []
    results: list[StyledResult] = []
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Knowledge bases
# ---------------------------------------------------------------------------


class FaceShapeRule(BaseModel):
    geometry: str
    strategy: str
    suit: list[str]
    avoid: list[str]


class FaceShapesKB(BaseModel):
    schema_version: str
    meta_rule: str
    shapes: dict[FaceShape, FaceShapeRule]
    references: list[str] = []


class Hairstyle(BaseModel):
    id: str
    name: str
    gender: Literal["male", "female"]
    length: Literal["short", "medium", "long"]
    description: str
    volume_demand: Literal["low", "low_to_medium", "medium", "medium_to_high", "high"]
    hair_texture_fit: list[HairTexture] = Field(min_length=1)
    suit_face: list[FaceShape] = []
    avoid_face: list[FaceShape] = []
    scene_tags: list[str] = []
    ref_image: str

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"invalid style id: {v!r}")
        return v


class HairstylesKB(BaseModel):
    schema_version: str
    items: list[Hairstyle]
    references: list[str] = []

    def get(self, style_id: str) -> Hairstyle:
        for item in self.items:
            if item.id == style_id:
                return item
        raise KeyError(f"unknown style id: {style_id}")
