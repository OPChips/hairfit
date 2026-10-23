"""Generation orchestration: per-style generate -> QC -> retry; job tracking.

Design rules implemented here (design.md section 5):
- batch stage: 1 image per style; refine stage adds the 2nd image on demand
- QC failure -> one retry with corrective instruction, then deliver degraded
- single-style failure never blocks other styles
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from hairfit.core.jobs import JobStore
from hairfit.core.qc import QCService
from hairfit.log import get_logger
from hairfit.prompts import build_generation_prompt, build_retry_suffix
from hairfit.providers.base import AnalysisProvider, GenerationProvider, ProviderUnavailable
from hairfit.schemas import (
    HairstylesKB,
    Job,
    JobStatus,
    QCScore,
    StyledResult,
)

log = get_logger(__name__)

ProgressFn = Callable[[Job], Awaitable[None]]


class GenerateService:
    def __init__(
        self,
        generation: GenerationProvider,
        analysis: AnalysisProvider,
        store: JobStore,
        styles: HairstylesKB,
        output_dir: Path,
    ) -> None:
        self.generation = generation
        self.qc = QCService(analysis)
        self.store = store
        self.styles = styles
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def create_and_run(
        self,
        photo: bytes,
        style_ids: list[str],
        on_progress: ProgressFn | None = None,
    ) -> Job:
        for sid in style_ids:
            self.styles.get(sid)  # raises KeyError on unknown id
        job = await self.store.create(style_ids)
        return await self.run(job, photo, on_progress)

    async def run(self, job: Job, photo: bytes, on_progress: ProgressFn | None = None) -> Job:
        await self.store.update(job, status=JobStatus.GENERATING)
        if on_progress:
            await on_progress(job)
        for sid in job.style_ids:
            result = await self._generate_one(photo, sid)
            await self.store.update(job, add_result=result)
            if on_progress:
                await on_progress(job)
        status = JobStatus.DONE if all(r.error is None for r in job.results) else JobStatus.FAILED
        await self.store.update(job, status=status)
        if on_progress:
            await on_progress(job)
        return job

    async def _generate_one(self, photo: bytes, style_id: str) -> StyledResult:
        style = self.styles.get(style_id)
        prompt = build_generation_prompt(style)
        try:
            ref = _ref_bytes(style.ref_image)
        except (FileNotFoundError, OSError) as exc:
            log.error("ref_image_missing", style=style_id, error=str(exc))
            return StyledResult(style_id=style_id, error=str(exc))

        # round 1
        try:
            image = await self.generation.edit(prompt, [photo, ref])
        except ProviderUnavailable as exc:
            log.error("generation_failed", style=style_id, error=str(exc))
            return StyledResult(style_id=style_id, error=str(exc))

        qc = await self.qc.score(image, photo, style)
        if qc.passed:
            return await self._save(style_id, [image], qc, retry_used=False)

        # round 2: retry with corrective instruction (design.md 5)
        weak = "face" if qc.face_consistency < 6 else "style"
        retry_prompt = prompt + build_retry_suffix(qc.reason, weak)
        try:
            image2 = await self.generation.edit(retry_prompt, [photo, ref])
        except ProviderUnavailable as exc:
            log.warning("retry_failed_deliver_degraded", style=style_id, error=str(exc))
            return await self._save(style_id, [image], qc, retry_used=True, degraded=True)

        qc2 = await self.qc.score(image2, photo, style)
        chosen, final_qc = (image2, qc2) if qc2.passed else (image, qc)
        return await self._save(style_id, [chosen], final_qc, retry_used=True)

    async def _save(
        self,
        style_id: str,
        images: list[bytes],
        qc: QCScore | None = None,
        retry_used: bool = False,
        degraded: bool = False,
    ) -> StyledResult:
        paths: list[str] = []
        style_dir = self.output_dir / style_id
        style_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            # extension follows the actual bytes (e.g. apimart returns JPEG)
            with Image.open(BytesIO(img)) as im:
                fmt = (im.format or "PNG").lower()
            p = style_dir / f"{i}.{'jpg' if fmt == 'jpeg' else fmt}"
            p.write_bytes(img)
            paths.append(str(p))
        return StyledResult(
            style_id=style_id,
            image_paths=paths,
            qc_face_consistency=qc.face_consistency if qc else None,
            qc_style_match=qc.style_match if qc else None,
            qc_skipped=degraded,
            retry_used=retry_used,
        )


def _ref_bytes(ref_path: str) -> bytes:
    p = Path(ref_path)
    if not p.exists():
        # project-root-relative fallback (assets/styles/...)
        p = Path(__file__).resolve().parent.parent.parent / ref_path
    if not p.exists():
        raise FileNotFoundError(f"参考图缺失: {ref_path}（请先运行 scripts/bootstrap_refs.py）")
    return p.read_bytes()
