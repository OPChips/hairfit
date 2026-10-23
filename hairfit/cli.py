"""HairFit command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hairfit import __version__
from hairfit.kb import validate_knowledge_bases
from hairfit.log import configure_logging


def cmd_validate_kb(_args: argparse.Namespace) -> int:
    problems = validate_knowledge_bases()
    if problems:
        print("知识库校验失败：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("知识库校验通过 (kb/face_shapes.json, kb/hairstyles.json)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    configure_logging(verbose=args.verbose)
    from hairfit.server import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    import asyncio

    from hairfit.config import load_config
    from hairfit.core.recommend import RecommendService
    from hairfit.providers.registry import build_analysis_chain
    from hairfit.schemas import RecommendationResult

    configure_logging(verbose=args.verbose)
    photo = Path(args.photo).read_bytes()

    async def run() -> RecommendationResult:
        cfg = load_config(Path(args.config))
        chain = build_analysis_chain(cfg)
        svc = RecommendService(chain)
        return await svc.analyze(photo)

    result = asyncio.run(run())
    print(result.model_dump_json(indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    import asyncio

    from hairfit.config import load_config
    from hairfit.core.generate import GenerateService
    from hairfit.core.jobs import JobStore
    from hairfit.kb import load_hairstyles
    from hairfit.providers.registry import build_analysis_chain, build_generation_chain

    configure_logging(verbose=args.verbose)
    photo = Path(args.photo).read_bytes()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    async def run() -> int:
        cfg = load_config(Path(args.config))
        store = await JobStore.open(Path(args.db))
        await store.init_db()
        svc = GenerateService(
            generation=build_generation_chain(cfg),
            analysis=build_analysis_chain(cfg),
            store=store,
            styles=load_hairstyles(),
            output_dir=out_dir,
        )
        job = await svc.create_and_run(photo=photo, style_ids=args.styles.split(","))
        if job.status.value == "failed":
            print(f"生成失败: {job.error}", file=sys.stderr)
            return 1
        for r in job.results:
            flag = "qc_ok" if not r.qc_skipped else "qc_skipped"
            print(f"{r.style_id}: {', '.join(r.image_paths) or r.error} [{flag}]")
        return 0

    return asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hairfit", description="AI 发型推荐与换发")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug 日志")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-kb", help="校验知识库")
    p.set_defaults(func=cmd_validate_kb)

    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8321)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("recommend", help="分析照片并推荐发型")
    p.add_argument("photo", help="用户照片路径")
    p.add_argument("--config", default=None, help="models.json 路径")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("generate", help="按款式生成换发图")
    p.add_argument("photo", help="用户照片路径")
    p.add_argument("--styles", required=True, help="逗号分隔的款式 id")
    p.add_argument("--out", default="output", help="输出目录")
    p.add_argument("--db", default="hairfit.db", help="job 数据库路径")
    p.add_argument("--config", default=None, help="models.json 路径")
    p.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
