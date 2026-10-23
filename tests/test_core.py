"""Unit tests for recommend / qc / generate orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hairfit.config import ProviderConfig
from hairfit.core.generate import GenerateService
from hairfit.core.jobs import JobStore
from hairfit.core.qc import QCService
from hairfit.core.recommend import RecommendError, RecommendService
from hairfit.kb import load_face_shapes, load_hairstyles
from hairfit.providers.base import AnalysisProvider, ProviderUnavailable
from hairfit.providers.image.mock import MockImageDriver
from hairfit.providers.llm.mock import MockLLM
from hairfit.schemas import FaceShapesKB, HairstylesKB, QCScore

PHOTO = b"\xff\xd8fakephoto"
FACES: FaceShapesKB = load_face_shapes()
STYLES: HairstylesKB = load_hairstyles()


def _mock_llm() -> MockLLM:
    return MockLLM(ProviderConfig(provider="mock", model="m"))


class TestRecommend:
    async def test_happy_path(self) -> None:
        svc = RecommendService(_mock_llm())
        res = await svc.analyze(PHOTO, FACES, STYLES)
        assert len(res.recommended) == 3

    async def test_invalid_id_retried_once(self) -> None:
        llm = _mock_llm()
        original = llm._respond

        def broken(prompt: str) -> str:
            text = original(prompt)
            return text.replace("textured_crop", "no_such_id", 1)

        llm._respond = broken  # type: ignore[method-assign]
        llm.invalid_json_times = 0
        # first call returns bad id; make second call valid
        state = {"n": 0}

        def flaky(prompt: str) -> str:
            state["n"] += 1
            return broken(prompt) if state["n"] == 1 else original(prompt)

        llm._respond = flaky  # type: ignore[method-assign]
        svc = RecommendService(llm)
        res = await svc.analyze(PHOTO, FACES, STYLES)
        assert "textured_crop" in [r.id for r in res.recommended]

    async def test_always_invalid_raises(self) -> None:
        llm = _mock_llm()
        original = llm._respond
        llm._respond = lambda prompt: original(prompt).replace(  # type: ignore[method-assign]
            "textured_crop", "no_such_id"
        )
        svc = RecommendService(llm, max_retries=1)
        with pytest.raises(RecommendError, match="校验失败"):
            await svc.analyze(PHOTO, FACES, STYLES)

    async def test_provider_down_raises(self) -> None:
        llm = _mock_llm()
        llm.fail_times = 99
        svc = RecommendService(llm)
        with pytest.raises(RecommendError, match="推荐调用失败"):
            await svc.analyze(PHOTO, FACES, STYLES)

    async def test_schema_violation_retried_with_errors(self) -> None:
        """null 字段等 schema 违规 -> 带校验错误重试（gpt-6.1-sol 实测案例）。"""
        llm = _mock_llm()
        original = llm._respond
        state = {"n": 0}

        def null_volume_then_valid(prompt: str) -> str:
            state["n"] += 1
            text = original(prompt)
            if state["n"] == 1:
                d = json.loads(text)
                d["face_analysis"]["hair_volume"] = None
                return json.dumps(d, ensure_ascii=False)
            return text

        llm._respond = null_volume_then_valid  # type: ignore[method-assign]
        svc = RecommendService(llm)
        res = await svc.analyze(PHOTO, FACES, STYLES)
        assert res.face_analysis.hair_volume in {"thin", "medium", "thick"}
        assert state["n"] == 2  # first attempt failed schema, second served

    async def test_schema_violation_exhausted_raises(self) -> None:
        llm = _mock_llm()
        original = llm._respond
        llm._respond = lambda prompt: (  # type: ignore[method-assign]
            lambda d: json.dumps(d | {"face_analysis": d["face_analysis"] | {"hairline": None}},
                                 ensure_ascii=False)
        )(json.loads(original(prompt)))
        svc = RecommendService(llm, max_retries=1)
        with pytest.raises(RecommendError, match="校验失败"):
            await svc.analyze(PHOTO, FACES, STYLES)


class TestQC:
    async def test_pass(self) -> None:
        qc = QCService(_mock_llm())
        score = await qc.score(b"gen", PHOTO, STYLES.get("textured_crop"))
        assert score.passed

    async def test_degrades_when_provider_down(self) -> None:
        llm = _mock_llm()
        llm.fail_times = 99
        qc = QCService(llm)
        score = await qc.score(b"gen", PHOTO, STYLES.get("textured_crop"))
        assert score.verdict == "pass"  # degraded: deliver unverified
        assert "降级" in score.reason


class _FakeAnalysis(AnalysisProvider):
    name = "fake"

    def __init__(self, score: QCScore | None = None, fail: bool = False) -> None:
        self.score = score
        self.fail = fail

    async def analyze(self, prompt: str, images: list[bytes], json_schema: type) -> object:
        if self.fail:
            raise ProviderUnavailable("qc down")
        assert json_schema is QCScore
        return self.score or QCScore(face_consistency=9, style_match=9, verdict="pass")


class TestGenerate:
    async def _service(self, tmp_path: Path, llm: MockLLM | None = None) -> GenerateService:
        store = await JobStore.open(":memory:")
        return GenerateService(
            generation=MockImageDriver(ProviderConfig(provider="mock", model="m")),
            analysis=llm or _mock_llm(),
            store=store,
            styles=STYLES,
            output_dir=tmp_path / "out",
        )

    async def test_happy_path(self, tmp_path: Path) -> None:
        svc = await self._service(tmp_path)
        job = await svc.create_and_run(PHOTO, ["textured_crop", "crew_cut"])
        assert job.status.value == "done"
        assert len(job.results) == 2
        for r in job.results:
            assert r.error is None
            assert Path(r.image_paths[0]).exists()
            assert r.qc_face_consistency is not None

    async def test_qc_retry_then_pass(self, tmp_path: Path) -> None:
        llm = _mock_llm()
        svc = await self._service(tmp_path, llm)
        # first QC low (retry), second QC high (pass)
        scores = iter([4, 9])
        original = llm._respond

        def low_then_high(prompt: str) -> str:
            if "质检" in prompt:
                v = next(scores)
                return json.dumps(
                    {
                        "face_consistency": v,
                        "style_match": 8,
                        "verdict": "pass" if v >= 6 else "retry",
                        "reason": "t",
                    }
                )
            return original(prompt)

        llm._respond = low_then_high  # type: ignore[method-assign]
        job = await svc.create_and_run(PHOTO, ["crew_cut"])
        r = job.results[0]
        assert r.retry_used is True
        assert r.qc_skipped is False

    async def test_retry_still_fails_delivers_degraded(self, tmp_path: Path) -> None:
        llm = _mock_llm()
        llm.qc_face_consistency = 2  # always fails
        svc = await self._service(tmp_path, llm)
        job = await svc.create_and_run(PHOTO, ["crew_cut"])
        r = job.results[0]
        assert r.error is None  # delivered anyway
        assert r.retry_used is True
        assert r.qc_skipped is False  # qc2 failed too -> kept round-1 image

    async def test_missing_ref_isolated_error(self, tmp_path: Path) -> None:
        import copy

        broken = HairstylesKB(schema_version="1.0", items=[copy.deepcopy(STYLES.get("crew_cut"))])
        broken.items[0].ref_image = "no/such/file.jpg"
        store = await JobStore.open(":memory:")
        svc = GenerateService(
            generation=MockImageDriver(ProviderConfig(provider="mock", model="m")),
            analysis=_mock_llm(),
            store=store,
            styles=broken,
            output_dir=tmp_path / "out",
        )
        job = await svc.create_and_run(PHOTO, ["crew_cut"])
        assert job.results[0].error is not None
        assert job.status.value == "failed"

    async def test_unknown_style_rejected(self, tmp_path: Path) -> None:
        svc = await self._service(tmp_path)
        with pytest.raises(KeyError):
            await svc.create_and_run(PHOTO, ["nope"])

    async def test_size_follows_input_aspect(self, tmp_path: Path) -> None:
        """输出尺寸跟随输入宽比（1:1 方图 -> 1024x1024，而非默认竖版）。"""
        from io import BytesIO

        from PIL import Image as PILImage

        class SizeSpy(MockImageDriver):
            sizes: list[str] = []

            async def edit(
                self, prompt: str, images: list[bytes], size: str = "1024x1536"
            ) -> bytes:
                self.sizes.append(size)
                return await super().edit(prompt, images, size)

        spy = SizeSpy(ProviderConfig(provider="mock", model="m"))
        store = await JobStore.open(":memory:")
        svc = GenerateService(
            generation=spy,
            analysis=_mock_llm(),
            store=store,
            styles=STYLES,
            output_dir=tmp_path / "out",
        )
        buf = BytesIO()
        PILImage.new("RGB", (2000, 2000), (120, 120, 120)).save(buf, format="PNG")
        await svc.create_and_run(buf.getvalue(), ["crew_cut"])
        assert spy.sizes == ["1024x1024"]
