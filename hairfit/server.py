"""FastAPI server: analyze / generate / jobs / report + health."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from hairfit.config import AppConfig, MissingAPIKeyError, resolve_api_key
from hairfit.core.generate import GenerateService
from hairfit.core.jobs import JobStore
from hairfit.core.recommend import RecommendError, RecommendService
from hairfit.core.report import build_report
from hairfit.kb import load_face_shapes, load_hairstyles
from hairfit.log import configure_logging, get_logger
from hairfit.providers.base import ProviderUnavailable
from hairfit.providers.registry import build_analysis_chain, build_generation_chain
from hairfit.schemas import RecommendationResult

log = get_logger(__name__)

DB_PATH = Path("hairfit.db")


def create_app(cfg: AppConfig | None = None, db_path: Path | None = None) -> FastAPI:
    app_cfg = cfg or AppConfig.load()
    db_file = db_path or DB_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.store = await JobStore.open(db_file)
        await app.state.store.init_db()
        app.state.analysis = build_analysis_chain(app_cfg)
        app.state.generation = build_generation_chain(app_cfg)
        app.state.faces = load_face_shapes()
        app.state.styles = load_hairstyles()
        log.info("server_started", db=str(db_file))
        yield
        await app.state.store.close()
        await app.state.analysis.aclose()
        await app.state.generation.aclose()

    app = FastAPI(title="HairFit", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RecommendError)
    async def _rec_error(_: object, exc: RecommendError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ProviderUnavailable)
    async def _prov_error(_: object, exc: ProviderUnavailable) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz/providers")
    async def providers_health() -> dict[str, list[dict[str, str]]]:
        def probe(segment: object) -> list[dict[str, str]]:
            out = []
            for pcfg in segment.chain():  # type: ignore[attr-defined]
                try:
                    resolve_api_key(pcfg)
                    status = "ok" if pcfg.api_key_env else "no-auth"
                except MissingAPIKeyError:
                    status = "missing-key"
                out.append({"provider": pcfg.provider, "model": pcfg.model, "status": status})
            return out

        return {"analysis": probe(app_cfg.analysis), "generation": probe(app_cfg.generation)}

    @app.post("/v1/analyze")
    async def analyze(photo: UploadFile = File(...)) -> RecommendationResult:
        data = await photo.read()
        svc = RecommendService(app.state.analysis)
        result = await svc.analyze(data, app.state.faces, app.state.styles)
        return result

    @app.post("/v1/generate")
    async def generate(
        photo: UploadFile = File(...), style_ids: str = Form(...)
    ) -> dict[str, str]:
        ids = [s.strip() for s in style_ids.split(",") if s.strip()]
        if not ids:
            raise HTTPException(status_code=422, detail="style_ids 为空")
        for sid in ids:
            try:
                app.state.styles.get(sid)
            except KeyError:
                raise HTTPException(  # noqa: B904
                    status_code=422, detail=f"未知款式 id: {sid}"
                )
        data = await photo.read()
        job = await app.state.store.create(ids)
        svc = GenerateService(
            generation=app.state.generation,
            analysis=app.state.analysis,
            store=app.state.store,
            styles=app.state.styles,
            output_dir=Path("output") / job.id,
        )
        asyncio.create_task(svc.run(job, data))
        return {"job_id": job.id}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> object:
        job = await app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job 不存在")
        return job

    @app.post("/v1/report")
    async def report(
        job_id: str = Form(...),
        rejected: str = Form(""),
        analysis_json: str = Form(""),
    ) -> Response:
        job = await app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job 不存在")
        if job.status.value != "done":
            raise HTTPException(status_code=409, detail=f"job 未完成: {job.status.value}")
        loader: dict[str, bytes] = {}
        results = []
        for r in job.results:
            if r.error or not r.image_paths:
                continue
            loader[r.style_id] = Path(r.image_paths[0]).read_bytes()
            results.append(r)
        rejected_list = [
            (sid, reason) for sid, _, reason in (p.partition(":") for p in rejected.split("|") if p)
        ]
        for sid, _ in rejected_list:
            ref = Path(app.state.styles.get(sid).ref_image)
            if not ref.exists():
                ref = Path(__file__).resolve().parent.parent / app.state.styles.get(sid).ref_image
            loader[sid] = ref.read_bytes()
        # stateless: the caller passes back the analysis from /v1/analyze
        from hairfit.schemas import FaceAnalysis

        if analysis_json:
            analysis = FaceAnalysis(**json.loads(analysis_json))
        else:
            analysis = FaceAnalysis(
                face_shape="oval", face_ratio="-", hair_volume="medium",
                hair_texture="1b", hairline="normal", notes="",
            )
        png = build_report(analysis, results, loader, rejected=rejected_list)
        return Response(content=png, media_type="image/png")

    return app
