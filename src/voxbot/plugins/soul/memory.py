from typing import Any, Literal, cast

import discord

from voxbot.store import runtime

from . import storage
from .errors import NoMemoryFound

MemoryCategoryT = Literal[
    "birthday",
    "job",
    "holiday",
    "preference",
    "relationship",
    "life_event",
    "identity",
    "other",
]


class MemoryService:
    """

    Storage types:
      HOT  - :memory: , Redis
      COLD - file.json , sqlite.db
    """

    def __init__(self) -> None:
        self.storage = storage.FileStorage(path=runtime.extend("soul/people.json"))

    @staticmethod
    def _partition_key(person: discord.Member | discord.User) -> str:
        # Prefer stable IDs so memories remain attached across username changes.
        return str(person.id)

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

        memories = await self.recall(message, person=message.author.id)

        if not memories:
            return "- No memories for this user."

        return "\n".join(f"- {m['category']}: {m['fact']}" for m in memories[-20:])

    async def recall(
        self,
        message: discord.Message,
        *,
        category: MemoryCategoryT | None = None,
        person: str | int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all stored facts about the person, optionally filtered by category."""
        member = self._resolve_member(message, person_ident=person)
        buffer = await self.storage.read()
        records = buffer.get(self._partition_key(member), [])

        if category is not None:
            records = [r for r in records if r.data.get("category") == category]

        return [r.data for r in records]

    async def remember(
        self,
        message: discord.Message,
        fact: str,
        category: MemoryCategoryT = "other",
        person: str | int | None = None,
    ) -> dict[str, Any]:
        """Insert or update a fact about the person, returning the persisted data."""
        member = self._resolve_member(message, person_ident=person)

        record = storage.Record.model_validate(
            {
                "partition_key": self._partition_key(member),
                "unique_key": "fact",
                "data": {
                    "category": category,
                    "person": member.name,
                    "person_id": member.id,
                    "fact": fact,
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
        fact: str,
        category: MemoryCategoryT | None = None,
        person: str | int | None = None,
    ) -> dict[str, Any]:
        """Delete a matching fact for the person, raising NoMemoryFound if absent."""
        member = self._resolve_member(message, person_ident=person)

        record_data = {
            "partition_key": self._partition_key(member),
            "unique_key": "fact",
            "data": {
                "category": category,
                "fact": fact,
            },
        }

        record = storage.Record.model_validate(record_data)

        try:
            return await self.storage.delete(record)
        except storage.NoEntryFound as exc:
            raise NoMemoryFound() from exc


Memories = MemoryService()
