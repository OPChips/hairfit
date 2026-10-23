"""Contract tests for LLM adapters: request shape, response parsing, retry."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from hairfit.config import ProviderConfig
from hairfit.providers.base import ProviderUnavailable
from hairfit.providers.llm.anthropic import AnthropicLLM
from hairfit.providers.llm.base import strip_json_fences
from hairfit.providers.llm.mock import MockLLM
from hairfit.providers.llm.openai_compat import OpenAICompatLLM

PROMPT = "你是专业发型顾问……"


class Out(BaseModel):
    answer: str


def _cfg(provider: str, **kw: Any) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model="test-model",
        base_url="https://api.test/v1",
        api_key_env="",
        **kw,
    )


OPENAI_RESP = {"choices": [{"message": {"content": '```json\n{"answer": "hi"}\n```'}}]}
ANTHROPIC_RESP = {"content": [{"type": "text", "text": '{"answer": "hi"}'}]}


class TestOpenAICompat:
    async def test_request_shape_and_parse(self, respx_mock: Any, monkeypatch: Any) -> None:
        route = respx_mock.post("https://api.test/v1/chat/completions").respond(json=OPENAI_RESP)
        monkeypatch.setenv("TEST_KEY", "sk-test")
        llm = OpenAICompatLLM(
            ProviderConfig(
                provider="openai",
                model="test-model",
                base_url="https://api.test/v1",
                api_key_env="TEST_KEY",
                params={"temperature": 0.3},
            )
        )
        out = await llm.analyze(PROMPT, [], Out)
        assert out.answer == "hi"
        req = json.loads(route.calls[0].request.content)
        assert req["model"] == "test-model"
        assert req["temperature"] == 0.3
        assert req["messages"][0]["content"] == PROMPT
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"

    async def test_image_as_data_url(self, respx_mock: Any) -> None:
        route = respx_mock.post("https://api.test/v1/chat/completions").respond(json=OPENAI_RESP)
        llm = OpenAICompatLLM(_cfg("openai"))
        img = b"\xff\xd8fakejpeg"
        await llm.analyze(PROMPT, [img], Out)
        req = json.loads(route.calls[0].request.content)
        content = req["messages"][0]["content"]
        assert content[0]["type"] == "text"
        expected = "data:image/jpeg;base64," + base64.b64encode(img).decode()
        assert content[1]["image_url"]["url"] == expected

    async def test_429_then_success(self, respx_mock: Any) -> None:
        route = respx_mock.post("https://api.test/v1/chat/completions")
        route.side_effect = [
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=OPENAI_RESP),
        ]
        llm = OpenAICompatLLM(_cfg("openai"))
        out = await llm.analyze(PROMPT, [], Out)
        assert out.answer == "hi"
        assert route.call_count == 2

    async def test_400_no_retry(self, respx_mock: Any) -> None:
        route = respx_mock.post("https://api.test/v1/chat/completions").respond(
            status_code=400, json={"error": "bad request"}
        )
        llm = OpenAICompatLLM(_cfg("openai"))
        with pytest.raises(ProviderUnavailable, match="400"):
            await llm.analyze(PROMPT, [], Out)
        assert route.call_count == 1

    async def test_invalid_json_raises_unavailable(self, respx_mock: Any) -> None:
        respx_mock.post("https://api.test/v1/chat/completions").respond(
            json={"choices": [{"message": {"content": "not json"}}]}
        )
        llm = OpenAICompatLLM(_cfg("openai"))
        with pytest.raises(ProviderUnavailable, match="invalid JSON"):
            await llm.analyze(PROMPT, [], Out)


class TestAnthropic:
    async def test_request_shape_and_parse(self, respx_mock: Any) -> None:
        route = respx_mock.post("https://api.test/v1/messages").respond(json=ANTHROPIC_RESP)
        llm = AnthropicLLM(
            ProviderConfig(
                provider="anthropic",
                model="claude-test",
                base_url="https://api.test",
                api_key_env="",
                params={"max_output_tokens": 1000},
            )
        )
        out = await llm.analyze(PROMPT, [b"img"], Out)
        assert out.answer == "hi"
        req = json.loads(route.calls[0].request.content)
        assert req["max_tokens"] == 1000
        assert req["messages"][0]["content"][0]["type"] == "image"
        assert req["messages"][0]["content"][1]["text"] == PROMPT
        assert route.calls[0].request.headers["x-api-key"] == ""


class TestMock:
    async def test_deterministic_recommendation(self) -> None:
        from hairfit.schemas import RecommendationResult

        llm = MockLLM(_cfg("mock"))
        out = await llm.analyze("你是专业发型顾问……", [], RecommendationResult)
        assert len(out.recommended) == 3

    async def test_fail_times(self) -> None:
        llm = MockLLM(_cfg("mock"))
        llm.fail_times = 1
        with pytest.raises(ProviderUnavailable):
            await llm.analyze("x", [], Out)


def test_strip_fences() -> None:
    assert strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_json_fences('{"a":1}') == '{"a":1}'
    assert strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'
