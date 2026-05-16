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
import asyncio
import contextlib
import datetime as dt
import enum
import json
import os
import pathlib
import tempfile

from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.exceptions import MemoryNotFoundError
from agent_memory_client.filters import Namespace, UserId
from agent_memory_client.models import ClientMemoryRecord, MemoryRecord, MemoryTypeEnum
import pydantic

from .settings import soul_settings

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
    created_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.UTC).timestamp()))
    updated_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.UTC).timestamp()))
    data: dict[str, Any]

    def compare(self, other: Record) -> bool:
        """Override to customize comparison."""
        in_same_bucket = self.partition_key == other.partition_key
        is_same_entry = self.data.get(self.unique_key) == other.data.get(self.unique_key)
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
        now = int(dt.datetime.now(tz=dt.UTC).timestamp())

        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."

            bucket = self._cache.setdefault(record.partition_key, [])

            to_persist = record
            to_persist.updated_at = now

            # UPDATE candidates
            for entry in bucket:
                if entry.compare(record):
                    entry.data = record.data
                    entry.updated_at = record.updated_at
                    to_persist = entry
                    break

            # or INSERT
            else:
                bucket.append(record)

            await asyncio.to_thread(self._append_wal, WalRecord(record=to_persist, operation=Op.UPSERT))

        return to_persist.data

    async def delete(self, record: Record) -> dict[str, Any]:
        """Remove a matching record from its partition, persisting the deletion via the WAL."""
        now = int(dt.datetime.now(tz=dt.UTC).timestamp())

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


class RedisAgentMemoryServer:
    """Redis Agent Memory Server storage backend.

    Talks to ``redislabs/agent-memory-server`` via the ``agent-memory-client`` SDK.

    Because ``ClientMemoryRecord`` has no free-form metadata field, the
    non-``fact`` keys of ``Record.data`` are serialised as JSON and appended
    to the AMS ``text`` field after a tab separator.

    Maps:
      * ``partition_key`` → ``user_id``
      * ``unique_key``   → never sent; always ``"fact"``
      * ``data["fact"]`` → ``text`` (first segment before tab)
      * ``data`` (rest)  → JSON suffix on ``text`` after tab
      * ``data["category"]`` → duplicated as ``topics[0]`` for discoverability
    """

    _SEP = "\t"

    def __init__(
        self,
        url: str | None = None,
        namespace: str | None = None,
        *,
        record_cls: type[Record] = Record,
    ) -> None:
        self.url = url or soul_settings.memory_server_url
        self.namespace = namespace or soul_settings.memory_namespace
        self.record_cls = record_cls
        self._client_instance: MemoryAPIClient | None = None

    # ── private API ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_fact(text: str) -> str:
        return text.split(RedisAgentMemoryServer._SEP, 1)[0] if RedisAgentMemoryServer._SEP in text else text

    @staticmethod
    def _extract_metadata(text: str) -> dict[str, Any]:
        if RedisAgentMemoryServer._SEP not in text:
            return {}
        return json.loads(text.split(RedisAgentMemoryServer._SEP, 1)[1])

    def _record_from_ams(self, memory: MemoryRecord) -> Record:
        text = memory.text
        fact = self._extract_fact(text)
        data = self._extract_metadata(text)
        data["fact"] = fact
        if memory.topics:
            data.setdefault("category", memory.topics[0])

        return self.record_cls(
            partition_key=memory.user_id or "",
            unique_key="fact",
            created_at=int(memory.created_at.timestamp()) if memory.created_at else 0,
            updated_at=int(memory.updated_at.timestamp()) if memory.updated_at else 0,
            data=data,
        )

    def _record_to_ams(self, record: Record) -> ClientMemoryRecord:
        data = record.data.copy()
        fact = data.pop("fact", "")
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return ClientMemoryRecord(
            text=f"{fact}{self._SEP}{payload}",
            memory_type=MemoryTypeEnum.SEMANTIC,
            user_id=record.partition_key,
            topics=[data.get("category")] if data.get("category") else None,
            namespace=self.namespace,
        )

    async def _client(self) -> MemoryAPIClient:
        if self._client_instance is None:
            config = MemoryClientConfig(base_url=self.url, default_namespace=self.namespace)
            self._client_instance = MemoryAPIClient(config)
        return self._client_instance

    # ── public API ────────────────────────────────────────────────────────────────────

    async def read(self) -> Cache:
        """Return a snapshot of all stored data, grouped by partition key."""
        client = await self._client()
        result: Cache = {}
        offset = 0
        limit = 100

        while True:
            results = await client.search_long_term_memory(
                text="",
                limit=limit,
                offset=offset,
                namespace=Namespace(eq=self.namespace),
            )
            for mem in results.memories:
                uid = mem.user_id or ""
                result.setdefault(uid, []).append(self._record_from_ams(mem))

            if len(results.memories) < limit:
                break
            offset += limit

        return result

    async def store(self, data: Cache) -> None:
        """Replace all data in the store with *data*."""
        client = await self._client()

        offset = 0
        limit = 100
        while True:
            results = await client.search_long_term_memory(
                text="",
                limit=limit,
                offset=offset,
                namespace=Namespace(eq=self.namespace),
            )
            if not results.memories:
                break
            await client.delete_long_term_memories([m.id for m in results.memories])
            if len(results.memories) < limit:
                break
            offset += limit

        batch = [self._record_to_ams(record) for partition_key, records in data.items() for record in records]
        if batch:
            await client.create_long_term_memory(batch)

    async def upsert(self, record: Record) -> dict[str, Any]:
        """Insert or update a record within its partition."""
        client = await self._client()
        fact = record.data.get("fact", "")
        ams_rec = self._record_to_ams(record)

        results = await client.search_long_term_memory(
            text=fact,
            limit=50,
            user_id=UserId(eq=record.partition_key),
            namespace=Namespace(eq=self.namespace),
        )

        for memory in results.memories:
            if self._extract_fact(memory.text) == fact:
                try:
                    await client.edit_long_term_memory(
                        memory_id=memory.id,
                        updates={"text": ams_rec.text, "topics": ams_rec.topics},
                    )
                except MemoryNotFoundError:
                    await client.create_long_term_memory([ams_rec])
                return record.data

        await client.create_long_term_memory([ams_rec])
        return record.data

    async def delete(self, record: Record) -> dict[str, Any]:
        """Remove a matching record from its partition."""
        client = await self._client()
        fact = record.data.get("fact", "")

        results = await client.search_long_term_memory(
            text=fact,
            limit=50,
            user_id=UserId(eq=record.partition_key),
            namespace=Namespace(eq=self.namespace),
        )

        for memory in results.memories:
            if self._extract_fact(memory.text) == fact:
                await client.delete_long_term_memories([memory.id])
                return record.data

        raise NoEntryFound()
