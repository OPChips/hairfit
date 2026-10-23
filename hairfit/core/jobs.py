"""Job persistence: SQLite (WAL) via aiosqlite."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite

from hairfit.log import get_logger
from hairfit.schemas import Job, JobStatus, StyledResult

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class JobStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    @classmethod
    async def open(cls, path: Path | str = ":memory:") -> JobStore:
        conn = await aiosqlite.connect(str(path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_SCHEMA)
        await conn.commit()
        return cls(conn)

    async def init_db(self) -> None:
        await self.db.execute(_SCHEMA)
        await self.db.commit()

    async def close(self) -> None:
        await self.db.close()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    async def create(self, style_ids: list[str], job_id: str | None = None) -> Job:
        job = Job(id=job_id or JobStore.new_id(), style_ids=style_ids)
        await self._write(job)
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self.db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        return Job(**json.loads(row["payload"])) if row else None

    async def update(
        self,
        job: Job,
        *,
        status: JobStatus | None = None,
        add_result: StyledResult | None = None,
        error: str | None = None,
    ) -> Job:
        if status is not None:
            job.status = status
        if add_result is not None:
            job.results = [*job.results, add_result]
        if error is not None:
            job.error = error
        job.updated_at = datetime.now()
        await self._write(job)
        return job

    async def _write(self, job: Job) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO jobs (id, status, payload, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                job.id,
                job.status.value,
                job.model_dump_json(),
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
            ),
        )
        await self.db.commit()
