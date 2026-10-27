"""FastAPI server: analyze / generate / jobs / report + health."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from hairfit.config import AppConfig, MissingAPIKeyError, resolve_api_key
from hairfit.core.generate import GenerateService
from hairfit.core.ingest import load_photo
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"


async def _photo_bytes(photo: UploadFile | None, photo_url: str) -> bytes:
    """Accept either an uploaded file or an http(s) URL (mutually exclusive)."""
    if photo_url:
        if photo is not None and photo.filename:
            raise HTTPException(status_code=422, detail="photo 与 photo_url 只能二选一")
        try:
            return await load_photo(photo_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (httpx.HTTPError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"拉取图片失败: {exc}") from exc
    if photo is None or not photo.filename:
        raise HTTPException(status_code=422, detail="需要 photo 文件或 photo_url")
    return await photo.read()


def create_app(cfg: AppConfig | None = None, db_path: Path | None = None) -> FastAPI:
    app_cfg = cfg or AppConfig.load()
    db_file = db_path or DB_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.store = await JobStore.open(db_file)
        await app.state.store.init_db()
        # empty chain (no keys configured) must not prevent startup; endpoints 503
        try:
            app.state.analysis = build_analysis_chain(app_cfg)
        except ProviderUnavailable:
            app.state.analysis = None
            log.warning("analysis_chain_empty")
        try:
            app.state.generation = build_generation_chain(app_cfg)
        except ProviderUnavailable:
            app.state.generation = None
            log.warning("generation_chain_empty")
        app.state.faces = load_face_shapes()
        app.state.styles = load_hairstyles()
        log.info("server_started", db=str(db_file))
        yield
        await app.state.store.close()
        if app.state.analysis is not None:
            await app.state.analysis.aclose()
        if app.state.generation is not None:
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
    async def analyze(
        photo: UploadFile | None = File(None), photo_url: str = Form("")
    ) -> RecommendationResult:
        if app.state.analysis is None:
            raise HTTPException(status_code=503, detail="无可用分析 provider（检查 API key 配置）")
        data = await _photo_bytes(photo, photo_url)
        svc = RecommendService(app.state.analysis)
        result = await svc.analyze(data, app.state.faces, app.state.styles)
        return result

    @app.post("/v1/generate")
    async def generate(
        photo: UploadFile | None = File(None),
        photo_url: str = Form(""),
        style_ids: str = Form(...),
    ) -> dict[str, str]:
        if app.state.generation is None or app.state.analysis is None:
            raise HTTPException(status_code=503, detail="无可用 provider（检查 API key 配置）")
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
        data = await _photo_bytes(photo, photo_url)
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
        # (accepts either the full RecommendationResult or a bare FaceAnalysis)
        from pydantic import ValidationError

        from hairfit.schemas import FaceAnalysis

        if analysis_json:
            try:
                payload = json.loads(analysis_json)
                if isinstance(payload, dict) and "face_analysis" in payload:
                    payload = payload["face_analysis"]
                analysis = FaceAnalysis(**payload)
            except (ValueError, ValidationError, TypeError) as exc:
                raise HTTPException(
                    status_code=422, detail=f"analysis_json 解析失败: {exc}"
                ) from exc
        else:
            analysis = FaceAnalysis(
                face_shape="oval",
                face_ratio="-",
                hair_volume="medium",
                hair_texture="1b",
                hairline="normal",
                notes="",
            )
        png = build_report(analysis, results, loader, rejected=rejected_list)
        return Response(content=png, media_type="image/png")

    @app.get("/v1/styles")
    async def list_styles() -> list[dict[str, str]]:
        """Style metadata for clients (name + reference image URL)."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "gender": s.gender,
                "description": s.description,
                "ref_image": "/" + s.ref_image,
            }
            for s in app.state.styles.items
        ]

    @app.get("/v1/jobs/{job_id}/results/{style_id}")
    async def result_image(job_id: str, style_id: str, idx: int = 0) -> Response:
        """Serve a generated try-on image (web client has no filesystem access)."""
        job = await app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job 不存在")
        for r in job.results:
            if r.style_id != style_id:
                continue
            if r.error or not r.image_paths:
                raise HTTPException(status_code=404, detail="该款式无生成结果")
            if not 0 <= idx < len(r.image_paths):
                raise HTTPException(status_code=404, detail="图片序号越界")
            path = Path(r.image_paths[idx])
            if not path.exists():
                path = PROJECT_ROOT / path
            return FileResponse(path)
        raise HTTPException(status_code=404, detail=f"job 中无款式 {style_id}")

    # static mounts last: API routes above take precedence
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
