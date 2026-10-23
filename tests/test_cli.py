"""CLI smoke tests via the mock chain."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from hairfit.cli import main


def _photo_png(path: Path) -> Path:
    img = Image.new("RGB", (200, 200), (170, 140, 120))
    buf = BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return path


def _mock_config(tmp_path: Path) -> Path:
    cfg = {
        "analysis": {"provider": "mock", "model": "mock-vision", "params": {}},
        "generation": {"provider": "mock", "model": "mock-image", "params": {}},
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class TestCLI:
    def test_validate_kb(self) -> None:
        assert main(["validate-kb"]) == 0

    def test_recommend_mock(self, tmp_path: Path, capsys: object) -> None:
        photo = _photo_png(tmp_path / "me.png")
        code = main(["recommend", str(photo), "--config", str(_mock_config(tmp_path))])
        assert code == 0
        out = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
        assert len(out["recommended"]) == 3

    def test_generate_mock(self, tmp_path: Path) -> None:
        photo = _photo_png(tmp_path / "me.png")
        out_dir = tmp_path / "out"
        code = main(
            [
                "generate",
                str(photo),
                "--styles",
                "textured_crop,crew_cut",
                "--out",
                str(out_dir),
                "--db",
                str(tmp_path / "t.db"),
                "--config",
                str(_mock_config(tmp_path)),
            ]
        )
        assert code == 0
        assert (out_dir / "textured_crop" / "0.png").exists()
        assert (out_dir / "crew_cut" / "0.png").exists()

    def test_version(self, capsys: object) -> None:
        import pytest

        with pytest.raises(SystemExit) as e:
            main(["--version"])
        assert e.value.code == 0
