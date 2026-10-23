"""Tests for knowledge base loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hairfit.kb import KBError, load_face_shapes, load_hairstyles, validate_knowledge_bases
from hairfit.schemas import HairstylesKB

PROJECT_KB = Path(__file__).resolve().parent.parent / "kb"


class TestRealKBs:
    """The shipped KB files must always validate."""

    def test_face_shapes_loads(self) -> None:
        kb = load_face_shapes()
        assert len(kb.shapes) == 7
        assert "平衡" in kb.meta_rule or "Balance" in kb.meta_rule

    def test_hairstyles_loads(self) -> None:
        kb = load_hairstyles()
        assert len(kb.items) == 20

    def test_shipped_kbs_have_no_problems(self) -> None:
        assert validate_knowledge_bases() == []

    def test_shipped_kbs_have_references(self) -> None:
        assert load_face_shapes().references
        assert load_hairstyles().references


class TestValidation:
    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        style = {
            "id": "dup",
            "name": "a",
            "gender": "male",
            "length": "short",
            "description": "d",
            "volume_demand": "low",
            "hair_texture_fit": ["1a"],
            "suit_face": [],
            "avoid_face": [],
            "scene_tags": [],
            "ref_image": "x.jpg",
        }
        (tmp_path / "hairstyles.json").write_text(
            json.dumps({"schema_version": "1.0", "items": [style, style]}), encoding="utf-8"
        )
        with pytest.raises((KBError, ValueError)):
            load_hairstyles(tmp_path / "hairstyles.json")

    def test_unknown_face_shape_reported(self, tmp_path: Path) -> None:
        # reuse real face_shapes.json but break hairstyle reference
        import shutil

        shutil.copy(PROJECT_KB / "face_shapes.json", tmp_path / "face_shapes.json")
        real = json.loads((PROJECT_KB / "hairstyles.json").read_text(encoding="utf-8"))
        real["items"] = real["items"][:4]
        real["items"][0]["suit_face"] = ["hexagon"]
        (tmp_path / "hairstyles.json").write_text(json.dumps(real, ensure_ascii=False), "utf-8")
        problems = validate_knowledge_bases(tmp_path)
        assert any("hexagon" in p for p in problems)


class TestGet:
    def test_get_by_id(self) -> None:
        kb: HairstylesKB = load_hairstyles()
        assert kb.get("textured_crop").gender == "male"

    def test_get_unknown(self) -> None:
        kb = load_hairstyles()
        with pytest.raises(KeyError):
            kb.get("no_such_style")
