"""Deterministic mock LLM for tests and key-less demos.

Behavior keyed off prompt content:
- contains "发型顾问" (recommendation prompt) -> RecommendationResult-shaped JSON
- contains "质检" (QC prompt) -> QCScore-shaped JSON
- otherwise -> generic JSON echo
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from hairfit.config import ProviderConfig
from hairfit.providers.llm.base import strip_json_fences


class MockLLM:
    name = "mock"

    def __init__(self, cfg: ProviderConfig, **_: Any) -> None:
        self.model = cfg.model
        #: tests can flip these to simulate failures
        self.fail_times = 0
        self.qc_face_consistency = 9
        self.invalid_json_times = 0
        self.calls = 0

    async def aclose(self) -> None: ...

    async def analyze(
        self, prompt: str, images: list[bytes], json_schema: type[BaseModel]
    ) -> BaseModel:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            from hairfit.providers.base import ProviderUnavailable

            raise ProviderUnavailable("mock: simulated failure")
        text = self._respond(prompt)
        if self.invalid_json_times > 0:
            self.invalid_json_times -= 1
            text = "这不是 JSON"
        try:
            return json_schema(**json.loads(strip_json_fences(text)))
        except (ValidationError, ValueError) as exc:
            from hairfit.providers.base import SchemaValidationError

            raise SchemaValidationError(f"{self.name} returned invalid JSON: {exc}") from exc

    def _respond(self, prompt: str) -> str:
        if "发型顾问" in prompt:
            return json.dumps(
                {
                    "face_analysis": {
                        "face_shape": "oval",
                        "face_ratio": "约 1.5:1",
                        "hair_volume": "medium",
                        "hair_texture": "1b",
                        "hairline": "normal",
                        "notes": "mock 分析",
                    },
                    "recommended": [
                        {"id": "textured_crop", "reason": "mock 推荐一"},
                        {"id": "french_bob_wavy", "reason": "mock 推荐二"},
                        {"id": "long_layers", "reason": "mock 推荐三"},
                    ],
                },
                ensure_ascii=False,
            )
        if "质检" in prompt:
            return json.dumps(
                {
                    "face_consistency": self.qc_face_consistency,
                    "style_match": 8,
                    "verdict": "pass" if self.qc_face_consistency >= 6 else "retry",
                    "reason": "mock 质检",
                },
                ensure_ascii=False,
            )
        return json.dumps({"echo": prompt[:50]})


def build_mock_llm(cfg: ProviderConfig) -> MockLLM:
    return MockLLM(cfg)
