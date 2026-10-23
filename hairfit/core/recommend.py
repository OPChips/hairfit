"""Recommendation service: assemble prompt -> LLM -> validate -> retry."""

from __future__ import annotations

from hairfit.prompts import build_recommendation_prompt
from hairfit.providers.base import AnalysisProvider, ProviderUnavailable
from hairfit.schemas import FaceShapesKB, HairstylesKB, RecommendationResult


class RecommendError(Exception):
    """Recommendation failed after validation retries."""


class RecommendService:
    def __init__(self, analysis: AnalysisProvider, max_retries: int = 1) -> None:
        self.analysis = analysis
        self.max_retries = max_retries

    async def analyze(
        self,
        photo: bytes,
        face_kb: FaceShapesKB,
        styles_kb: HairstylesKB,
    ) -> RecommendationResult:
        prompt = build_recommendation_prompt(face_kb, styles_kb)
        valid_ids = {s.id for s in styles_kb.items}
        last_error = ""

        for attempt in range(1 + self.max_retries):
            attempt_prompt = prompt
            if last_error:
                attempt_prompt += (
                    f"\n\n注意：上一次输出存在问题：{last_error}"
                    "。请严格修正后重新输出 JSON。"
                )
            try:
                result = await self.analysis.analyze(
                    attempt_prompt, [photo], RecommendationResult
                )
            except ProviderUnavailable as exc:
                raise RecommendError(f"推荐调用失败: {exc}") from exc
            problem = self._check(result, valid_ids)
            if problem is None:
                return result
            last_error = problem

        raise RecommendError(
            f"推荐结果校验失败（重试 {self.max_retries} 次后仍无效）: {last_error}"
        )

    @staticmethod
    def _check(result: RecommendationResult, valid_ids: set[str]) -> str | None:
        bad = [r.id for r in result.recommended if r.id not in valid_ids]
        if bad:
            return f"推荐了发型库中不存在的 id: {bad}"
        return None
