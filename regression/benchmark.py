#!/usr/bin/env python
"""Generation-model benchmark (design.md 8.4): each provider x test faces.

Runs the generate pipeline per configured provider against every face in
regression/faces/, scores via QC (face consistency x style match), and emits
a markdown scoreboard. Requires real API keys for real providers; use
--dry-run to self-check against mock providers.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from hairfit.config import AppConfig, ProviderConfig, resolve_api_key  # noqa: E402
from hairfit.core.generate import GenerateService  # noqa: E402
from hairfit.core.jobs import JobStore  # noqa: E402
from hairfit.kb import load_hairstyles  # noqa: E402
from hairfit.log import configure_logging  # noqa: E402
from hairfit.providers.image.gpt_image import build_gpt_image  # noqa: E402
from hairfit.providers.image.grok import build_grok  # noqa: E402
from hairfit.providers.image.mock import MockImageDriver  # noqa: E402
from hairfit.providers.image.nano_banana import build_nano_banana  # noqa: E402
from hairfit.providers.image.seedream import build_seedream  # noqa: E402
from hairfit.providers.llm.mock import MockLLM  # noqa: E402
from hairfit.providers.registry import build_analysis_chain  # noqa: E402

_FACTORIES = {
    "gpt_image": build_gpt_image,
    "nano_banana": build_nano_banana,
    "seedream": build_seedream,
    "grok": build_grok,
}


def collect_generation_providers(app: AppConfig, dry_run: bool) -> list[tuple[str, object]]:
    """[(name, driver)] from the configured chain (plus fallbacks)."""
    if dry_run:
        return [("mock", MockImageDriver(ProviderConfig(provider="mock", model="mock")))]
    out = []
    for cfg in app.generation.chain():
        factory = _FACTORIES.get(cfg.provider)
        if factory is None:
            continue
        try:
            resolve_api_key(cfg)
        except Exception:  # noqa: BLE001
            print(f"  跳过 {cfg.provider}（无 key）")
            continue
        out.append((cfg.provider, factory(cfg)))
    return out


async def run_benchmark(faces_dir: Path, style_ids: list[str], dry_run: bool) -> str:
    app = AppConfig.load()
    styles_kb = load_hairstyles()
    analysis = (
        build_analysis_chain(app)
        if not dry_run
        else MockLLM(ProviderConfig(provider="mock", model="mock"))
    )
    faces = sorted(p for p in faces_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not faces:
        raise SystemExit(f"测试脸目录为空: {faces_dir}（请放入 10-20 张测试脸）")

    lines = [
        "# 生成模型基准测试",
        "",
        f"- 测试脸: {len(faces)} 张 / 款式: {', '.join(style_ids)}",
        "",
        "| provider | 面部一致性均值 | 发型匹配均值 | 成功率 |",
        "|---|---|---|---|",
    ]
    for name, driver in collect_generation_providers(app, dry_run):
        store = await JobStore.open(":memory:")
        svc = GenerateService(
            generation=driver,  # type: ignore[arg-type]
            analysis=analysis,  # type: ignore[arg-type]
            store=store,
            styles=styles_kb,
            output_dir=PROJECT / "output" / "benchmark" / name,
        )
        scores: list[tuple[int, int]] = []
        ok = 0
        total = 0
        for face in faces:
            photo = face.read_bytes()
            job = await svc.create_and_run(photo, style_ids)
            for r in job.results:
                total += 1
                if r.error is None:
                    ok += 1
                if r.qc_face_consistency is not None:
                    scores.append((r.qc_face_consistency, r.qc_style_match or 0))
        if scores:
            face_avg = sum(s[0] for s in scores) / len(scores)
            style_avg = sum(s[1] for s in scores) / len(scores)
            rate = f"{ok}/{total}"
            lines.append(f"| {name} | {face_avg:.1f} | {style_avg:.1f} | {rate} |")
        else:
            lines.append(f"| {name} | - | - | 0/{total} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模型基准测试")
    parser.add_argument("--faces", default="regression/faces", help="测试脸目录")
    parser.add_argument(
        "--styles",
        default="textured_crop,french_bob_wavy",
        help="逗号分隔款式 id",
    )
    parser.add_argument("--dry-run", action="store_true", help="mock 链路自检")
    parser.add_argument("--out", default=None, help="结果 markdown 输出路径")
    args = parser.parse_args()

    configure_logging()
    result = asyncio.run(run_benchmark(Path(args.faces), args.styles.split(","), args.dry_run))
    print(result)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
