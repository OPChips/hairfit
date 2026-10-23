#!/usr/bin/env python
"""Bootstrap hairstyle reference images (design.md 3.3 自举).

For each style whose ref_image is missing (or --force), asks the generation
provider to render a standardized studio reference from the style metadata —
uniform angle, background, lighting, natural dark hair. The generated library
matches the online generation style and is license-clean.

Requires a real generation provider key; add --dry-run for mock self-check.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from PIL import Image  # noqa: E402

from hairfit.config import AppConfig  # noqa: E402
from hairfit.kb import load_hairstyles  # noqa: E402
from hairfit.log import configure_logging  # noqa: E402
from hairfit.providers.registry import build_generation_chain  # noqa: E402

REF_PROMPT = """Studio portrait photograph of {gender_desc} demonstrating the hairstyle
"{name}": {description}. Front-facing, standard 3/4-free neutral angle, plain
light-gray studio background, even soft studio lighting, natural dark black or
dark-brown hair color (no dye), shoulders-up framing, hair fully visible with
no occlusion from hands, hats, or accessories. Photorealistic, high detail."""


def _prompt(style: object) -> str:
    g = "a man" if style.gender == "male" else "a woman"  # type: ignore[attr-defined]
    return REF_PROMPT.format(gender_desc=g, name=style.name, description=style.description)  # type: ignore[attr-defined]


async def bootstrap(force: bool, dry_run: bool) -> int:
    configure_logging()
    kb = load_hairstyles()
    chain = build_generation_chain(AppConfig.load()) if not dry_run else None
    if chain is None:
        from hairfit.config import ProviderConfig as PC
        from hairfit.providers.image.mock import MockImageDriver

        chain = MockImageDriver(PC(provider="mock", model="mock"))  # type: ignore[assignment]

    root = PROJECT
    done = 0
    for style in kb.items:
        out = root / style.ref_image
        if out.exists() and not force:
            continue
        img_bytes = await chain.edit(_prompt(style), [], size="768x1024")  # type: ignore[union-attr]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img_bytes)
        # verify it decodes
        Image.open(out).verify()
        done += 1
        print(f"✓ {style.id} -> {out}")
    print(f"完成：生成 {done} 张参考图")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="自举生成发型参考图")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的参考图")
    parser.add_argument("--dry-run", action="store_true", help="mock 链路自检")
    args = parser.parse_args()
    return asyncio.run(bootstrap(args.force, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
