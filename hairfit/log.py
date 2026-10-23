"""Structured logging configuration (structlog, JSON)."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO, cast

import structlog


class _LazyStderr:
    """Resolve sys.stderr at write time (tests replace it per-test)."""

    def write(self, msg: str) -> None:
        sys.stderr.write(msg)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str) -> Any:  # satisfy the TextIO shape
        return getattr(sys.stderr, name)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # stderr: stdout belongs to command output (e.g. recommend prints JSON)
    logging.basicConfig(format="%(message)s", level=level, stream=sys.stderr)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if verbose else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(cast(TextIO, _LazyStderr())),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
