"""Anthropic Messages API adapter (Claude)."""

from __future__ import annotations

from typing import Any

from hairfit.config import ProviderConfig
from hairfit.providers.llm.base import LLMDriver


class AnthropicLLM(LLMDriver):
    name = "anthropic"

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _build_request(
        self, prompt: str, images: list[bytes]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _b64(img),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        params = dict(self.params)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.pop("max_output_tokens", 2048),
            "messages": [{"role": "user", "content": content}],
        }
        payload.update(params)  # temperature, ...
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        return payload, headers

    def _extract_text(self, data: dict[str, Any]) -> str:
        blocks = data["content"]
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "\n".join(texts)


def _b64(image: bytes) -> str:
    import base64

    return base64.b64encode(image).decode()


def build_anthropic(cfg: ProviderConfig) -> AnthropicLLM:
    return AnthropicLLM(cfg)
