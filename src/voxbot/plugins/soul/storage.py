from typing import Any, ClassVar, Protocol
import asyncio
import contextlib
import datetime as dt
import enum
import json
import math
import os
import pathlib
import re
import tempfile

from agent_memory_client.filters import Namespace, UserId
import agent_memory_client as redis_ams
import agent_memory_client.exceptions as redis_errors
import pydantic

from .embedding import GoogleEmbeddingProvider, Vector
from .settings import soul_settings

type Cache = dict[str, list["Record"]]


class NoEntryFound(Exception):
    """Raised when no matching entry is found in the store."""


class Op(enum.StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class Record(pydantic.BaseModel):
    """Represents a row managed in the store.

    ``unique_key`` and ``semantic_key`` name the fields inside ``data`` that
    carry the record's identity and semantic payload respectively.
    """

    partition_key: str
    unique_key: str
    semantic_key: str
    created_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.UTC).timestamp()))
    updated_at: int = pydantic.Field(default_factory=lambda: int(dt.datetime.now(tz=dt.UTC).timestamp()))
    data: dict[str, Any]
    embedding: Vector = pydantic.Field(default_factory=list)

    @property
    def partition(self) -> str:
        """Fetch the partition value."""
        return str(self.data[self.partition_key])

    @property
    def pk(self) -> str:
        """Fetch the unique key value."""
        return str(self.data[self.unique_key])

    @property
    def semantic(self) -> str:
        """Fetch the semantic key value."""
        return str(self.data[self.semantic_key])

    def compare(self, other: Record) -> bool:
        """Return ``True`` when records belong to the same partition AND their
        unique or semantic keys carry equal values."""
        if self.partition != other.partition:
            return False

        same_semantic = self.semantic == other.semantic  # fmt: skip
        same_unique   = self.pk == other.pk  # fmt: skip

        return same_semantic or same_unique

    def to_json(self) -> dict[str, Any]:
        data = self.data.copy()
        data["created_at"] = self.created_at
        data["updated_at"] = self.updated_at
        return data


class WalRecord(pydantic.BaseModel):
    """Represents a row managed in the FileStore."""

    record: Record
    operation: Op


class StorageT(Protocol):
    async def read(self) -> Cache: ...

    async def upsert(self, record: Record) -> dict[str, Any]: ...

    async def delete(self, record: Record) -> dict[str, Any]: ...

    async def semantic_search(
        self,
        partition: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Record]: ...


class FileStorage:
    """
    Crash-safe JSON store via write-ahead log.

    All file I/O is offloaded to a thread so the event loop stays unblocked.
    """

    _SEMANTIC_RELEVANCE_PROFILES: ClassVar[dict[str, float]] = {
        "loose": 0.20,
        "balanced": 0.30,
        "strict": 0.40,
    }

    def __init__(
        self,
        path: pathlib.Path,
        *,
        record_cls: type[Record] = Record,
    ) -> None:
        self.path = path
        self.wal_path = self.path.with_name(f"{path.name}-wal")
        self.record_cls = record_cls
        self.embedding_provider = GoogleEmbeddingProvider()

        self._min_score = self._SEMANTIC_RELEVANCE_PROFILES.get(soul_settings.memory_semantic_relevance, 0.30)
        self._lock = asyncio.Lock()
        self._cache: Cache | None = None

    # ── private API ───────────────────────────────────────────────────────────────────

    async def _ensure_recovered(self) -> None:
        """Replay the WAL to the store on first load."""
        if self._cache is not None:
            return

        await asyncio.to_thread(self._replay_wal)

    @staticmethod
    def _read_json(path: pathlib.Path) -> dict[str, list[Any]]:
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
        raw = self._read_json(self.path)

        self._cache = {
            partition: [self.record_cls.model_validate(record) for record in records]
            for partition, records in raw.items()
        }

        try:
            text = self.wal_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return

        if not text:
            return

        for line in text.splitlines():
            w = WalRecord.model_validate(json.loads(line.strip()))
            bucket = self._cache.setdefault(w.record.partition, [])

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

        self.wal_path.write_text("", encoding="utf-8")

    @staticmethod
    def _cosine_similarity(lhs: Vector, rhs: Vector) -> float:
        if not lhs or not rhs or len(lhs) != len(rhs):
            return 0.0

        dot = sum(left * right for left, right in zip(lhs, rhs, strict=False))
        lhs_norm = math.sqrt(sum(v * v for v in lhs))
        rhs_norm = math.sqrt(sum(v * v for v in rhs))
        if lhs_norm == 0 or rhs_norm == 0:
            return 0.0

        return dot / (lhs_norm * rhs_norm)

    # ── public API ────────────────────────────────────────────────────────────────────

    async def read(self) -> Cache:
        """Return a snapshot of all stored data, replaying the WAL first if needed."""
        await self._ensure_recovered()
        assert self._cache is not None, "FileStore is in an invalid state."
        return dict(self._cache)

    async def semantic_search(
        self,
        partition: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Record]:
        """Return semantic matches for a partition, ranked best-first."""
        if not (cleaned_query := query.strip()) or limit <= 0:
            return []

        query_embedding = await self.embedding_provider.embed_query(cleaned_query)

        await self._ensure_recovered()
        assert self._cache is not None, "FileStore is in an invalid state."
        records = list(self._cache.get(partition, []))

        scored: list[tuple[float, Record]] = [
            (score, record)
            for record in records
            if (score := self._cosine_similarity(query_embedding, record.embedding)) >= self._min_score
        ]

        return [r for _, r in sorted(scored, key=lambda s: s[0], reverse=True)[:limit]]

    async def store(self, data: Cache) -> None:
        """Flush data to the file store."""
        async with self._lock:
            self._cache = dict(data)
            await asyncio.to_thread(self._checkpoint)

    async def upsert(self, record: Record) -> dict[str, Any]:
        """Insert or update a record within its partition, persisting via the WAL."""
        now = int(dt.datetime.now(tz=dt.UTC).timestamp())

        # Generate the embedding before UPSERTing.
        embedding = await self.embedding_provider.embed_document(record.semantic)

        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."

            bucket = self._cache.setdefault(record.partition, [])

            to_persist = record
            to_persist.updated_at = now
            to_persist.embedding = embedding

            for entry in bucket:
                if entry.compare(record):
                    record.data[entry.unique_key] = entry.pk

                    entry.data = record.data
                    entry.updated_at = now
                    entry.embedding = embedding
                    to_persist = entry
                    break

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

            bucket = self._cache.get(record.partition, [])

            for idx, entry in enumerate(bucket):
                if entry.compare(record):
                    removed = bucket.pop(idx)
                    removed.updated_at = now

                    await asyncio.to_thread(self._append_wal, WalRecord(record=removed, operation=Op.DELETE))
                    return removed.data

            raise NoEntryFound()


class RedisAgentMemoryServer:
    """
    Redis Agent Memory Server storage backend.

    Talks to ``redislabs/agent-memory-server`` via the ``agent-memory-client`` SDK.

    ``ClientMemoryRecord`` has no free-form metadata field, so data fields are
    stored in the ``entities`` list using a ``vx_`` prefix convention.  Only the
    semantic payload (the value at ``data[semantic_key]``) is placed in ``text``,
    keeping the vectorised content clean.

    The ``unique_key`` and ``semantic_key`` names are stored as synthetic entities
    (``vx__unique_key``, ``vx__semantic_key``) so records can be reconstructed
    on read-back.
    """

    _ENT_PREFIX = "vx_"
    _SEMANTIC_RELEVANCE_PROFILES: ClassVar[dict[str, float | None]] = {
        "loose": 0.90,
        "balanced": 0.75,
        "strict": 0.60,
    }

    def __init__(
        self,
        url: str | None = None,
        namespace: str | None = None,
        *,
        record_cls: type[Record] = Record,
    ) -> None:
        self.namespace = namespace or soul_settings.memory_namespace
        self.record_cls = record_cls
        self.client = redis_ams.MemoryAPIClient(
            config=redis_ams.MemoryClientConfig(
                base_url=url or soul_settings.memory_server_url,
                default_namespace=self.namespace,
            )
        )

        self._distance_threshold = self._SEMANTIC_RELEVANCE_PROFILES.get(soul_settings.memory_semantic_relevance)
        self._lock = asyncio.Lock()

    # ── private API ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _entity_encode(v: str) -> str:
        return v.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=")

    @staticmethod
    def _entity_decode(v: str) -> str:
        return re.sub(r"\\([\\,=])", r"\1", v)

    @staticmethod
    def _metadata_from_entities(entities: list[str] | None) -> dict[str, Any]:
        p = RedisAgentMemoryServer._ENT_PREFIX
        return {
            k: RedisAgentMemoryServer._entity_decode(v)
            for e in (entities or [])
            if e.startswith(p)
            for k, _, v in [e.removeprefix(p).partition("=")]
        }

    @staticmethod
    def _entities_from_data(data: dict[str, Any]) -> list[str]:
        p = RedisAgentMemoryServer._ENT_PREFIX
        return [f"{p}{k}={RedisAgentMemoryServer._entity_encode(str(v))}" for k, v in data.items()]

    def _record_from_ams(self, memory: redis_ams.models.MemoryRecord) -> Record:
        data = self._metadata_from_entities(memory.entities)

        return self.record_cls(
            partition_key=data.pop("__partition_key"),
            unique_key=data.pop("__unique_key"),
            semantic_key=data.pop("__semantic_key"),
            created_at=int(memory.created_at.timestamp()),
            updated_at=int(memory.updated_at.timestamp()),
            data=data,
        )

    def _record_to_ams(self, record: Record) -> redis_ams.models.ClientMemoryRecord:
        data = record.data.copy()
        data["__partition_key"] = record.partition_key
        data["__unique_key"] = record.unique_key
        data["__semantic_key"] = record.semantic_key

        return redis_ams.models.ClientMemoryRecord(
            text=record.semantic,
            memory_type=redis_ams.models.MemoryTypeEnum.SEMANTIC,
            user_id=record.partition,
            entities=self._entities_from_data(data),
            namespace=self.namespace,
        )

    # ── public API ────────────────────────────────────────────────────────────────────

    async def read(self) -> Cache:
        """Return a snapshot of all stored data, grouped by partition key."""
        LIMIT = 100
        OFFSET = 0
        cache: Cache = {}

        while True:
            r = await self.client.search_long_term_memory(
                text="",
                limit=LIMIT,
                offset=OFFSET,
                namespace=Namespace(eq=self.namespace),
            )

            for memory in r.memories:
                record = self._record_from_ams(memory)
                cache.setdefault(record.partition, []).append(record)

            if len(r.memories) < LIMIT:
                break

            OFFSET += LIMIT

        return cache

    async def semantic_search(
        self,
        partition: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Record]:
        """Return semantic matches for a partition, ranked best-first."""
        if not (cleaned_query := query.strip()):
            return []

        r = await self.client.search_long_term_memory(
            text=cleaned_query,
            limit=limit,
            user_id=UserId(eq=partition),
            namespace=Namespace(eq=self.namespace),
            distance_threshold=self._distance_threshold,
        )

        return [self._record_from_ams(memory) for memory in r.memories]

    async def store(self, data: Cache) -> None:
        """Replace all data in the store with *data*."""
        LIMIT = 100

        async with self._lock:
            # ── TRUNCATE ─────────────────────────────────────────────────────
            OFFSET = 0

            while True:
                r = await self.client.search_long_term_memory(
                    text="",
                    limit=LIMIT,
                    offset=OFFSET,
                    namespace=Namespace(eq=self.namespace),
                )

                if not r.memories:
                    break

                await self.client.delete_long_term_memories([m.id for m in r.memories])

                if len(r.memories) < LIMIT:
                    break

                OFFSET += LIMIT

            # ── INSERT ───────────────────────────────────────────────────────

            if batch := [self._record_to_ams(record) for records in data.values() for record in records]:
                await self.client.create_long_term_memory(batch)

    async def upsert(self, record: Record) -> dict[str, Any]:
        """Insert or update a record within its partition.

        Notes on divergence from ``FileStorage.upsert``:
          * AMS auto-generates embeddings from ``text`` on the server side,
            so no client-side embedding call is needed.
          * AMS sets ``updated_at`` server-side on PATCH (edit), and the
            ``ClientMemoryRecord`` model provides a client-side default on
            create.  No local timestamp management is required.
          * No embedding vector is returned by AMS, so ``Record.embedding``
            is left empty for this backend.
        """
        now = int(dt.datetime.now(tz=dt.UTC).timestamp())

        async with self._lock:
            r = await self.client.search_long_term_memory(
                text="",
                limit=100,
                user_id=UserId(eq=record.partition),
                namespace=Namespace(eq=self.namespace),
            )

            to_persist = record
            to_persist.updated_at = now

            ams_rec = self._record_to_ams(record)

            for memory in r.memories:
                candidate = self._record_from_ams(memory)
                candidate.updated_at = now

                if candidate.compare(record):
                    record.data[candidate.unique_key] = candidate.pk

                    candidate.data = record.data
                    candidate.updated_at = now
                    to_persist = candidate

                    ams_rec = self._record_to_ams(candidate)

                    try:
                        await self.client.edit_long_term_memory(
                            memory_id=memory.id,
                            updates={"text": ams_rec.text, "topics": ams_rec.topics, "entities": ams_rec.entities},
                        )
                    except redis_errors.MemoryNotFoundError:
                        await self.client.create_long_term_memory([ams_rec])

                    break
            else:
                await self.client.create_long_term_memory([ams_rec])

            return to_persist.data

    async def delete(self, record: Record) -> dict[str, Any]:
        """Remove a matching record from its partition."""
        async with self._lock:
            r = await self.client.search_long_term_memory(
                text="",
                limit=100,
                user_id=UserId(eq=record.partition),
                namespace=Namespace(eq=self.namespace),
            )

            for memory in r.memories:
                candidate = self._record_from_ams(memory)

                if candidate.compare(record):
                    await self.client.delete_long_term_memories([memory.id])
                    return candidate.data

        raise NoEntryFound()
