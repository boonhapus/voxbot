"""
Crash-safe JSON store via write-ahead log.

Design:
  * Base snapshot in the main JSON file.
  * WAL at ``{file.name}-wal`` — append-only JSONL of individual operations.
  * In-memory ``_cache`` stays in sync with the WAL so reads are O(1).
  * On init: replay WAL over base snapshot → crash recovery.
  * Checkpoint: atomically flush cache to main file, then truncate WAL.
"""

from typing import Any, Protocol, TypedDict
import asyncio
import contextlib
import datetime as dt
import enum
import functools
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile

from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.exceptions import MemoryNotFoundError
from agent_memory_client.filters import Namespace, Topics, UserId
from agent_memory_client.models import ClientMemoryRecord, MemoryRecord, MemoryTypeEnum
import pydantic

from .settings import soul_settings

type Cache = dict[str, list[Record]]
type Vector = list[float]


class SemanticRelevanceProfile(TypedDict):
    json_min_score: float
    redis_distance_threshold: float | None


_HASH_EMBED_DIM = 256
_SEMANTIC_RELEVANCE: dict[str, SemanticRelevanceProfile] = {
    "loose": {"json_min_score": 0.20, "redis_distance_threshold": 0.90},
    "balanced": {"json_min_score": 0.30, "redis_distance_threshold": 0.75},
    "strict": {"json_min_score": 0.40, "redis_distance_threshold": 0.60},
}


def _semantic_relevance_profile() -> SemanticRelevanceProfile:
    key = soul_settings.memory_semantic_relevance.strip().casefold()
    return _SEMANTIC_RELEVANCE.get(key, _SEMANTIC_RELEVANCE["balanced"])


def _json_min_score_threshold() -> float:
    return _semantic_relevance_profile()["json_min_score"]


def _redis_distance_threshold() -> float | None:
    return _semantic_relevance_profile()["redis_distance_threshold"]


def _hashed_embedding(text: str) -> Vector:
    """Deterministic local embedding fallback when remote embeddings are unavailable."""
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    vector = [0.0] * _HASH_EMBED_DIM
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % _HASH_EMBED_DIM
        sign = 1.0 if (digest[2] & 1) else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingProvider(Protocol):
    async def embed_document(self, text: str) -> Vector: ...

    async def embed_query(self, text: str) -> Vector: ...


class GoogleEmbeddingProvider:
    """Gemini embedding wrapper with graceful fallback when unavailable."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or soul_settings.memory_embedding_model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Any | None = None

    def _get_client(self) -> Any | None:
        if not self.api_key:
            return None

        if self._client is not None:
            return self._client

        try:
            from google import genai
        except Exception:
            return None

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _build_config(task_type: str) -> Any | None:
        try:
            from google.genai import types
        except Exception:
            return None
        return types.EmbedContentConfig(task_type=task_type)

    def _embed(self, text: str, *, task_type: str) -> Vector:
        payload = text.strip()
        if not payload:
            return _hashed_embedding(text)

        client = self._get_client()
        if client is None:
            return _hashed_embedding(payload)

        try:
            kwargs: dict[str, Any] = {}
            if config := self._build_config(task_type):
                kwargs["config"] = config

            response = client.models.embed_content(
                model=self.model,
                contents=[payload],
                **kwargs,
            )

            embeddings = getattr(response, "embeddings", None) or []
            if embeddings:
                values = getattr(embeddings[0], "values", None)
                if values:
                    return [float(v) for v in values]
        except Exception:
            pass

        return _hashed_embedding(payload)

    async def embed_document(self, text: str) -> Vector:
        fn = functools.partial(self._embed, text=text, task_type="RETRIEVAL_DOCUMENT")
        return await asyncio.to_thread(fn)

    async def embed_query(self, text: str) -> Vector:
        fn = functools.partial(self._embed, text=text, task_type="RETRIEVAL_QUERY")
        return await asyncio.to_thread(fn)


def _cosine_similarity(lhs: Vector, rhs: Vector) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0

    dot = sum(left * right for left, right in zip(lhs, rhs, strict=False))
    lhs_norm = math.sqrt(sum(v * v for v in lhs))
    rhs_norm = math.sqrt(sum(v * v for v in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0

    return dot / (lhs_norm * rhs_norm)


def _keyword_similarity(query: str, candidate: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9_]+", query.casefold()))
    candidate_tokens = set(re.findall(r"[a-z0-9_]+", candidate.casefold()))

    if not query_tokens or not candidate_tokens:
        return 0.0

    return len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)


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
    embedding: Vector = pydantic.Field(default_factory=list)

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

    def __init__(
        self,
        path: pathlib.Path,
        *,
        record_cls: type[Record] = Record,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.path = path
        self.wal_path = self.path.with_name(f"{path.name}-wal")
        self.record_cls = record_cls
        self.embedding_provider = embedding_provider or GoogleEmbeddingProvider()
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
        fact = str(record.data.get("fact", "")).strip()
        if not fact:
            msg = "record.data['fact'] cannot be empty for semantic memory storage"
            raise ValueError(msg)
        record.embedding = await self.embedding_provider.embed_document(fact)

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
                    entry.embedding = record.embedding
                    to_persist = entry
                    break

            # or INSERT
            else:
                bucket.append(record)

            await asyncio.to_thread(self._append_wal, WalRecord(record=to_persist, operation=Op.UPSERT))

        return to_persist.data

    async def semantic_search(
        self,
        partition_key: str,
        query: str,
        *,
        category: str | None = None,
        limit: int = 20,
    ) -> list[Record]:
        """Return semantic matches for a partition, ranked best-first."""
        cleaned_query = query.strip()
        if not cleaned_query or limit <= 0:
            return []

        query_embedding = await self.embedding_provider.embed_query(cleaned_query)

        async with self._lock:
            await self._ensure_recovered()
            assert self._cache is not None, "FileStore is in an invalid state."
            records = list(self._cache.get(partition_key, []))

        if category is not None:
            records = [r for r in records if r.data.get("category") == category]

        threshold = _json_min_score_threshold()
        scored: list[tuple[float, Record]] = []

        for record in records:
            fact = str(record.data.get("fact", "")).strip()
            if not fact:
                continue

            doc_embedding = record.embedding
            if not doc_embedding:
                doc_embedding = await self.embedding_provider.embed_document(fact)
                record.embedding = doc_embedding

            score = _cosine_similarity(query_embedding, doc_embedding)

            # Fall back to lexical overlap when embeddings are unavailable.
            if score == 0.0:
                score = _keyword_similarity(cleaned_query, fact)

            if score >= threshold:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

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


_ENT_PREFIX = "vx_"
"""Prefix for synthetic entities so they don't collide with real entities."""


class RedisAgentMemoryServer:
    """Redis Agent Memory Server storage backend.

    Talks to ``redislabs/agent-memory-server`` via the ``agent-memory-client`` SDK.

    ``ClientMemoryRecord`` has no free-form metadata field, so the non-``fact``
    keys of ``Record.data`` are stored in the ``entities`` list using a
    ``key=value`` convention (prefixed with ``vx_``).  Only the fact itself
    is placed in ``text``, keeping the vectorised content clean.

    Maps:
      * ``partition_key``  → ``user_id``
      * ``unique_key``    → never sent; always ``"fact"``
      * ``data["fact"]``  → ``text``
      * ``data["category"]`` → ``topics[0]``
      * ``data`` (rest)   → ``entities`` as ``vx_key=value`` strings
    """

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
    def _metadata_from_entities(entities: list[str] | None) -> dict[str, Any]:
        if not entities:
            return {}
        result: dict[str, Any] = {}
        for e in entities:
            if e.startswith(_ENT_PREFIX):
                k, _, v = e[len(_ENT_PREFIX) :].partition("=")
                result[k] = v
        return result

    @staticmethod
    def _entities_from_metadata(data: dict[str, Any]) -> list[str]:
        return [f"{_ENT_PREFIX}{k}={v}" for k, v in data.items()]

    def _record_from_ams(self, memory: MemoryRecord) -> Record:
        data = self._metadata_from_entities(memory.entities)
        data["fact"] = memory.text
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
        data.pop("category", None)
        categories = [record.data["category"]] if record.data.get("category") else None
        return ClientMemoryRecord(
            text=fact,
            memory_type=MemoryTypeEnum.SEMANTIC,
            user_id=record.partition_key,
            topics=categories,
            entities=self._entities_from_metadata(data) or None,
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
            if memory.text == fact:
                try:
                    await client.edit_long_term_memory(
                        memory_id=memory.id,
                        updates={"text": ams_rec.text, "topics": ams_rec.topics, "entities": ams_rec.entities},
                    )
                except MemoryNotFoundError:
                    await client.create_long_term_memory([ams_rec])
                return record.data

        await client.create_long_term_memory([ams_rec])
        return record.data

    async def semantic_search(
        self,
        partition_key: str,
        query: str,
        *,
        category: str | None = None,
        limit: int = 20,
    ) -> list[Record]:
        """Return semantic matches for a partition, ranked best-first."""
        cleaned_query = query.strip()
        if not cleaned_query or limit <= 0:
            return []

        client = await self._client()
        topics_filter = Topics(any=[category]) if category else None

        results = await client.search_long_term_memory(
            text=cleaned_query,
            limit=limit,
            user_id=UserId(eq=partition_key),
            namespace=Namespace(eq=self.namespace),
            topics=topics_filter,
            distance_threshold=_redis_distance_threshold(),
        )
        return [self._record_from_ams(memory) for memory in results.memories]

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
            if memory.text == fact:
                await client.delete_long_term_memories([memory.id])
                return record.data

        raise NoEntryFound()
