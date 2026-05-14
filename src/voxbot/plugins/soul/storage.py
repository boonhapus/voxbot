"""
Crash-safe JSON store via write-ahead log.

Design:
  * Base snapshot in the main JSON file.
  * WAL at ``{file.name}-wal`` — append-only JSONL of individual operations.
  * In-memory ``_cache`` stays in sync with the WAL so reads are O(1).
  * On init: replay WAL over base snapshot → crash recovery.
  * Checkpoint: atomically flush cache to main file, then truncate WAL.
"""
from typing import Any
import enum
import asyncio
import contextlib
import datetime as dt
import json
import os
import pathlib
import tempfile

import pydantic

type Cache = dict[str, list[Record]]


class NoEntryFound(Exception):
    """Raised when no matching entry is found in the FileStore."""


class Op(enum.StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class Record(pydantic.BaseModel):
    """Represents a row managed in the FileStore."""
    partition_key: str
    unique_key: str
    created_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.timezone.utc).timestamp()))
    updated_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.timezone.utc).timestamp()))
    data: dict[str, Any]

    def compare(self, other: Record) -> bool:
        """Override to customize comparison."""
        in_same_bucket = self.partition_key == other.partition_key
        is_same_entry  = self.data.get(self.unique_key) == other.data.get(self.unique_key)
        return in_same_bucket and is_same_entry

    def to_json(self) -> dict[str, Any]:
        data = self.data.copy()
        data["created_at"] = self.created_at
        data["updated_at"] = self.updated_at
        return data
    

class WalRecord(pydantic.BaseModel):
    """Represents a row managed in the FileStore."""
    record: Record
    operation: Op


# ── helpers ──────────────────────────────────────────────────────────────────


class FileStorage:
    """
    Crash-safe JSON store via write-ahead log.

    All file I/O is offloaded to a thread so the event loop stays unblocked.
    """

    def __init__(self, path: pathlib.Path, *, record_cls: type[Record] = Record) -> None:
        self.path = path
        self.wal_path = self.path.with_name(f"{path.name}-wal")
        self.record_cls = record_cls
        self._lock = asyncio.Lock()
        self._cache: Cache | None = None

    # ── private API ───────────────────────────────────────────────────────────────────

    async def _ensure_recovered(self) -> None:
        """Replay the WAL to the store on first load."""
        if self._cache is not None:
            return

        await asyncio.to_thread(self._replay_wal)

    @staticmethod
    def _read_json(path: pathlib.Path) -> Cache:
        """Load and parse the JSON file, returning an empty dict if missing or blank."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        return json.loads(text) if text.strip() else {}

    @staticmethod
    def _atomic_write(path: pathlib.Path, data: bytes) -> None:
        """Write *data* to *path* atomically (POSIX)."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(tmp, path)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def _append_wal(self, record: WalRecord) -> None:
        """Write an entry to the WAL."""
        with self.wal_path.open(mode="a", encoding="utf-8", newline="") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
    
    def _replay_wal(self) -> None:
        """Replay the write-ahead log into the in-memory cache, then checkpoint."""
        self._cache = self._read_json(self.path)

        try:
            text = self.wal_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        
        if not text:
            return

        for line in text.splitlines():
            w = WalRecord.model_validate(json.loads(line.strip()))
            bucket = self._cache.setdefault(w.record.partition_key, [])

            if w.operation == Op.DELETE:
                bucket[:] = [r for r in bucket if not r.compare(w.record)]

            if w.operation == Op.UPSERT:
                for idx, r in enumerate(bucket):
                    if r.compare(w.record):
                        bucket[idx] = w.record
                        break
                else:
                    bucket.append(w.record)
        
        self._checkpoint()

    def _checkpoint(self) -> None:
        """Flush cache to the base file and clear the WAL."""
        assert self._cache is not None, "FileStore is in an invalid state."
        data = {k: [r.model_dump() for r in v] for k, v in self._cache.items()}

        self._atomic_write(self.path, data=json.dumps(data, indent=2, sort_keys=True).encode())

        # Safe to truncate: base file is durable thanks to fsync + rename above.
        self.wal_path.write_text("", encoding="utf-8")

    # ── public API ────────────────────────────────────────────────────────────────────

    async def read(self) -> Cache:
        """Return a snapshot of all stored data, replaying the WAL first if needed."""
        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."
            return dict(self._cache)

    async def store(self, data: Cache) -> None:
        """Flush data to the file store."""
        async with self._lock:
            self._cache = dict(data)
            await asyncio.to_thread(self._checkpoint)

    async def upsert(self, record: Record) -> dict[str, Any]:
        """Insert or update a record within its partition, persisting via the WAL."""
        now = int(dt.datetime.now(tz=dt.timezone.utc).timestamp())

        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."

            bucket = self._cache.setdefault(record.partition_key, [])

            for entry in bucket:
                if entry.compare(record):
                    entry.data = record.data
                    entry.updated_at = now
                    break

            else:
                bucket.append(record)

            await asyncio.to_thread(self._append_wal, WalRecord(record=record, operation=Op.UPSERT))

        return entry.data

    async def delete(self, record: Record) -> dict[str, Any]:
        """Remove a matching record from its partition, persisting the deletion via the WAL."""
        now = int(dt.datetime.now(tz=dt.timezone.utc).timestamp())

        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."

            bucket = self._cache.get(record.partition_key, [])

            for idx, entry in enumerate(bucket):
                if entry.compare(record):
                    removed = bucket.pop(idx)
                    removed.updated_at = now

                    await asyncio.to_thread(self._append_wal, WalRecord(record=removed, operation=Op.DELETE))
                    return removed.data

            raise NoEntryFound()
