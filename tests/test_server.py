"""E2E tests for the FastAPI server on the full mock chain."""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

from hairfit.config import AppConfig, SegmentConfig
from hairfit.server import create_app


def _photo_png() -> bytes:
    img = Image.new("RGB", (256, 256), (180, 150, 130))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_app(tmp_path: Path) -> FastAPI:
    mock_seg = SegmentConfig(provider="mock", model="mock-vision")
    mock_img = SegmentConfig(provider="mock", model="mock-image")
    return create_app(
        AppConfig(analysis=mock_seg, generation=mock_img),
        db_path=tmp_path / "test.db",
    )


class TestServerE2E:
    def test_full_flow(self, tmp_path: Path) -> None:
        app = _mock_app(tmp_path)
        with TestClient(app) as client:
            # health
            assert client.get("/healthz").json()["status"] == "ok"
            prov = client.get("/healthz/providers").json()
            assert prov["generation"][0]["status"] == "no-auth"

            photo = {"photo": ("me.png", _photo_png(), "image/png")}

            # analyze
            rec = client.post("/v1/analyze", files=photo)
            assert rec.status_code == 200, rec.text
            analysis = rec.json()["face_analysis"]
            assert len(rec.json()["recommended"]) == 3

            # generate
            gen = client.post(
                "/v1/generate",
                data={"style_ids": "textured_crop,crew_cut"},
                files=photo,
            )
            assert gen.status_code == 200, gen.text
            job_id = gen.json()["job_id"]

            # poll to done
            import json as _json

            for _ in range(100):
                job = client.get(f"/v1/jobs/{job_id}").json()
                if job["status"] in ("done", "failed"):
                    break
                asyncio.run(asyncio.sleep(0.01))
            assert job["status"] == "done", job
            assert len(job["results"]) == 2

            # report
            rep = client.post(
                "/v1/report",
                data={
                    "job_id": job_id,
                    "rejected": "buzz_cut:发量不足|slick_back:拉长脸型",
                    "analysis_json": _json.dumps(analysis),
                },
            )
            assert rep.status_code == 200, rep.text
            assert rep.headers["content-type"] == "image/png"
            img = Image.open(BytesIO(rep.content))
            assert img.width > 500

    def test_unknown_style_rejected(self, tmp_path: Path) -> None:
        app = _mock_app(tmp_path)
        with TestClient(app) as client:
            r = client.post(
                "/v1/generate",
                data={"style_ids": "nope"},
                files={"photo": ("m.png", _photo_png(), "image/png")},
            )
            assert r.status_code == 422

    def test_job_404(self, tmp_path: Path) -> None:
        app = _mock_app(tmp_path)
        with TestClient(app) as client:
            assert client.get("/v1/jobs/missing").status_code == 404

    def test_report_before_done_409(self, tmp_path: Path) -> None:
        app = _mock_app(tmp_path)
        with TestClient(app) as client:
            r = client.post("/v1/report", data={"job_id": "ghost"})
            assert r.status_code == 404
