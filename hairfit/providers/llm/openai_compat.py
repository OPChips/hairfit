"""OpenAI-compatible chat completions adapter.

Covers OpenAI and the OpenAI-compatible endpoints of DeepSeek, Qwen (DashScope),
Zhipu GLM, Doubao (Ark), Kimi (Moonshot), and Gemini's compatibility layer.
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel

from hairfit.config import ProviderConfig
from hairfit.providers.llm.base import LLMDriver


class OpenAICompatLLM(LLMDriver):
    name = "openai_compat"

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _build_request(
        self, prompt: str, image: bytes | None
    ) -> tuple[dict[str, Any], dict[str, str]]:
        content: list[dict[str, Any]] | str
        if image is None:
            content = prompt
        else:
            b64 = base64.b64encode(image).decode()
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        payload.update(self.params)  # temperature, max_tokens, ...
        return payload, self._headers()

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data["choices"]
        return str(choices[0]["message"]["content"])


def build_openai_compat(cfg: ProviderConfig) -> OpenAICompatLLM:
    return OpenAICompatLLM(cfg)
