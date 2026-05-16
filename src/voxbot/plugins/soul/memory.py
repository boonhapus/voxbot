from typing import Any, cast
import secrets

import discord

from voxbot.store import runtime

from . import storage
from .errors import NoMemoryFound
from .settings import soul_settings


class MemoryService:
    """
    High-level CRUD facade for per-user memories.

    Selects the storage backend (``FileStorage`` or ``RedisAgentMemoryServer``)
    based on ``soul_settings.memory_backend``.  Each memory is a structured
    fact (confidence, source) keyed by Discord user ID and
    optionally searchable via semantic (embedding) similarity.
    """

    def __init__(self) -> None:
        if soul_settings.memory_backend == "redis":
            self.storage: storage.StorageT = storage.RedisAgentMemoryServer()
        else:
            self.storage: storage.StorageT = storage.FileStorage(path=runtime.extend("soul/people.json"))

    @staticmethod
    def _resolve_member(message: discord.Message, *, person_ident: str | int | None = None) -> discord.Member:
        if person_ident is None:
            return cast(discord.Member, message.author)

        person_ident = str(person_ident).casefold()

        for candidate in (message.author, *message.mentions):
            names = (candidate.name, candidate.global_name, candidate.display_name, candidate.id)

            if person_ident in map(str.casefold, map(str, names)):
                return cast(discord.Member, candidate)

        return cast(discord.Member, message.author)

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────────────

    async def summary(self, message: discord.Message | None) -> str:
        """Return a bulleted summary of all memories about the person."""
        if message is None:
            return "- No current author; this is a background identity check."

        query = message.content.strip()

        memories = await self.recall(
            message,
            person=message.author.id,
            query=query or None,
            limit=20,
        )

        if not memories:
            memories = await self.recall(message, person=message.author.id)

        if not memories:
            return "- No memories for this user."

        return "\n".join(f"- [{m.get('memory_id', '???????')}] {m['fact']}" for m in memories[-20:])

    async def recall(
        self,
        message: discord.Message,
        *,
        person: str | int | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return stored facts about the person, optionally ranked by semantic query."""
        member = self._resolve_member(message, person_ident=person)

        if query is not None and query.strip():
            records = await self.storage.semantic_search(
                partition=str(member.id),
                query=query,
                limit=limit,
            )
            return [r.data for r in records]

        buffer = await self.storage.read()
        records = buffer.get(str(member.id), [])

        return [r.data for r in records]

    async def remember(
        self,
        message: discord.Message,
        fact: str,
        person: str | int | None = None,
    ) -> dict[str, Any]:
        """Insert or update a fact about the person, returning the persisted data."""
        member = self._resolve_member(message, person_ident=person)

        record = storage.Record.model_validate(
            {
                "unique_key": "memory_id",
                "partition_key": "person_id",
                "semantic_key": "fact",
                "data": {
                    "memory_id": secrets.token_hex(4)[:7],
                    "person_id": str(member.id),
                    "fact": fact,
                    "person": member.name,
                    "confidence": "explicit",
                    "source": "discord",
                    "message_id": message.id if message else None,
                    "channel_id": message.channel.id if message else None,
                },
            }
        )

        return await self.storage.upsert(record)

    async def forget(
        self,
        message: discord.Message,
        fact: str = "",
        person: str | int | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete a memory for the person, raising NoMemoryFound if absent.

        If *memory_id* is provided the record is removed by its unique identifier.
        Otherwise, the *fact* string is used as a semantic query and the top
        matching memory is deleted.
        """
        member = self._resolve_member(message, person_ident=person)
        partition = str(member.id)
        memory_id = memory_id.strip() or None if memory_id is not None else None

        if memory_id is None and fact.strip():
            # Find the best semantic match first, then delete by concrete ID so the
            # final removal remains deterministic.
            candidates = await self.storage.semantic_search(partition=partition, query=fact, limit=1)

            if not candidates:
                raise NoMemoryFound()

            memory_id = str(candidates[0].data.get("memory_id") or "").strip() or None

        record_data: dict[str, Any] = {
            "unique_key": "memory_id",
            "partition_key": "person_id",
            "semantic_key": "fact",
            "data": {
                "memory_id": memory_id,
                "person_id": partition,
                "fact": "" if memory_id is not None else fact,
            },
        }

        record = storage.Record.model_validate(record_data)

        try:
            return await self.storage.delete(record)
        except storage.NoEntryFound as exc:
            raise NoMemoryFound() from exc


Memories = MemoryService()
