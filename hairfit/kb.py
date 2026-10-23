"""Knowledge base loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from hairfit.log import get_logger
from hairfit.schemas import FaceShapesKB, HairstylesKB

log = get_logger(__name__)

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


class KBError(Exception):
    """Knowledge base failed validation."""


def load_face_shapes(path: Path | None = None) -> FaceShapesKB:
    file = path or KB_DIR / "face_shapes.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    kb = FaceShapesKB(**data)
    log.debug("face_shapes_kb_loaded", path=str(file), shapes=list(kb.shapes))
    return kb


def load_hairstyles(path: Path | None = None) -> HairstylesKB:
    file = path or KB_DIR / "hairstyles.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    kb = HairstylesKB(**data)
    ids = [item.id for item in kb.items]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise KBError(f"duplicate style ids: {dupes}")
    log.debug("hairstyles_kb_loaded", path=str(file), count=len(kb.items))
    return kb


def validate_knowledge_bases(kb_dir: Path | None = None) -> list[str]:
    """Validate both KBs; returns a list of problems (empty = valid)."""
    problems: list[str] = []
    base = kb_dir or KB_DIR
    try:
        load_face_shapes(base / "face_shapes.json")
    except Exception as exc:  # noqa: BLE001 — report all validation problems
        problems.append(f"face_shapes.json: {exc}")
    try:
        styles = load_hairstyles(base / "hairstyles.json")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"hairstyles.json: {exc}")
        return problems

    # coverage matrix (design.md 3.3): per gender, each length >= 2
    for gender in ("male", "female"):
        for length in ("short", "medium", "long"):
            n = sum(
                1
                for s in styles.items
                if s.gender == gender and s.length == length  # type: ignore[union-attr]
            )
            if n < 2:
                problems.append(f"coverage: {gender}/{length} has {n} styles (need >=2)")

    # referenced face shapes must exist in face_shapes KB
    try:
        faces = load_face_shapes(base / "face_shapes.json")
    except Exception:  # noqa: BLE001 — already reported above
        return problems
    for s in styles.items:
        for f in (*s.suit_face, *s.avoid_face):
            if f not in faces.shapes:
                problems.append(f"style {s.id}: unknown face shape {f!r}")
    return problems
