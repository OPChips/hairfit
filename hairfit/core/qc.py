"""QC: score a generated image against the original photo (VLM)."""

from __future__ import annotations

from typing import cast

from hairfit.log import get_logger
from hairfit.prompts import build_qc_prompt
from hairfit.providers.base import AnalysisProvider, ProviderUnavailable
from hairfit.schemas import Hairstyle, QCScore

log = get_logger(__name__)


class QCService:
    def __init__(self, analysis: AnalysisProvider) -> None:
        self.analysis = analysis

    async def score(self, generated: bytes, user_photo: bytes, style: Hairstyle) -> QCScore:
        """Two images: original first, generated second (prompt order matters)."""
        try:
            score = cast(
                QCScore,
                await self.analysis.analyze(
                    build_qc_prompt(style), [user_photo, generated], QCScore
                ),
            )
        except ProviderUnavailable as exc:
            # QC is enhancement, not on the critical path — degrade gracefully
            log.warning("qc_failed_degraded", error=str(exc), style=style.id)
            return QCScore(
                face_consistency=0,
                style_match=0,
                verdict="pass",
                reason=f"质检不可用，已降级交付: {exc}",
            )
        return score
