"""APIMart relay: key unification + task-based image driver (mocked transport)."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from hairfit.config import (
    AppConfig,
    ProviderConfig,
    SegmentConfig,
    key_env_for,
    resolve_base_url,
)
from hairfit.providers.base import ProviderUnavailable
from hairfit.providers.image.apimart import ApimartDriver, build_apimart
from hairfit.providers.registry import build_analysis_chain, build_generation_chain

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def _seg(provider: str) -> SegmentConfig:
    return SegmentConfig(provider=provider, model="m", api_key_env="")


def _driver(handler: Any, monkeypatch: pytest.MonkeyPatch) -> ApimartDriver:
    monkeypatch.setenv("HAIRFIT_TEST_APIMART", "sk-test")
    cfg = ProviderConfig(
        provider="apimart", model="gpt-image-2", api_key_env="HAIRFIT_TEST_APIMART"
    )
    driver = build_apimart(cfg)
    driver._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    return driver


class TestKeyUnification:
    def test_apimart_one_key_both_segments(self, monkeypatch: Any) -> None:
        """One apimart key serves analysis + generation; provider = key unit."""
        monkeypatch.setenv("HAIRFIT_ANALYSIS_PROVIDER", "apimart")
        monkeypatch.setenv("HAIRFIT_GENERATION_PROVIDER", "apimart")
        app = AppConfig.load()
        assert app.analysis.model == "gpt-6.1-sol"
        assert app.generation.model == "gpt-image-2"
        assert app.analysis.api_key_env == "HAIRFIT_APIMART_API_KEY"
        assert app.generation.api_key_env == "HAIRFIT_APIMART_API_KEY"
        assert resolve_base_url(app.generation) == "https://api.apimart.ai/v1"
        assert key_env_for("apimart") == "HAIRFIT_APIMART_API_KEY"

    def test_seedream_shares_doubao_key(self) -> None:
        """seedream and doubao are both Volcano Ark -> one key per vendor."""
        assert key_env_for("seedream") == "HAIRFIT_DOUBAO_API_KEY"
        assert key_env_for("doubao") == "HAIRFIT_DOUBAO_API_KEY"

    def test_apimart_registered_in_both_factories(self) -> None:
        from hairfit.providers.registry import _IMAGE_FACTORIES, _LLM_FACTORIES

        assert "apimart" in _LLM_FACTORIES
        assert "apimart" in _IMAGE_FACTORIES
        app = AppConfig(analysis=_seg("apimart"), generation=_seg("apimart"))
        a = build_analysis_chain(app)
        g = build_generation_chain(app)
        # analysis reuses the shared OpenAI-compat adapter (provider id kept in cfg)
        assert [p.name for p in a.providers] == ["openai_compat"]
        assert [p.name for p in g.providers] == ["apimart"]


class TestApimartDriver:
    async def test_submit_poll_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        polls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "upload.apimart.ai" not in url:  # result download carries no auth
                assert request.headers["Authorization"] == "Bearer sk-test"
            if request.method == "POST" and url.endswith("/images/generations"):
                body = json.loads(request.content)
                assert body["model"] == "gpt-image-2"
                assert body["n"] == 1
                assert body["image_urls"][0].startswith("data:image/png;base64,")
                assert base64.b64decode(body["image_urls"][0].split(",", 1)[1]) == PNG
                return httpx.Response(
                    200,
                    json={"code": 200, "data": [{"status": "submitted", "task_id": "task_1"}]},
                )
            if url.endswith("/tasks/task_1"):
                polls["n"] += 1
                if polls["n"] == 1:
                    return httpx.Response(
                        200, json={"code": 200, "data": {"status": "processing", "progress": 40}}
                    )
                return httpx.Response(
                    200,
                    json={
                        "code": 200,
                        "data": {
                            "status": "completed",
                            "result": {"images": [{"url": ["https://upload.apimart.ai/i.png"]}]},
                        },
                    },
                )
            if url == "https://upload.apimart.ai/i.png":
                return httpx.Response(200, content=PNG)
            return httpx.Response(404, json={})

        driver = _driver(handler, monkeypatch)
        try:
            assert await driver.edit("crew cut", [PNG]) == PNG
            assert polls["n"] == 2
        finally:
            await driver.aclose()

    async def test_failed_task_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, json={"code": 200, "data": [{"status": "submitted", "task_id": "t"}]}
                )
            return httpx.Response(
                200,
                json={"code": 200, "data": {"status": "failed", "error": "quota exceeded"}},
            )

        driver = _driver(handler, monkeypatch)
        try:
            with pytest.raises(ProviderUnavailable, match="failed"):
                await driver.edit("p", [PNG])
        finally:
            await driver.aclose()

    async def test_error_envelope_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 1001, "message": "insufficient balance"})

        driver = _driver(handler, monkeypatch)
        try:
            with pytest.raises(ProviderUnavailable, match="submit failed"):
                await driver.edit("p", [])
        finally:
            await driver.aclose()

    async def test_sync_b64_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-task synchronous b64 response is also accepted."""

        def handler(request: httpx.Request) -> httpx.Response:
            b64 = base64.b64encode(PNG).decode()
            return httpx.Response(200, json={"code": 200, "data": [{"b64_json": b64}]})

        driver = _driver(handler, monkeypatch)
        try:
            assert await driver.edit("p", []) == PNG
        finally:
            await driver.aclose()
