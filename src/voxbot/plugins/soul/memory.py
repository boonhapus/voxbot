from typing import Any, Literal, cast

import discord

from voxbot.store import runtime

from . import storage

MemoryCategory = Literal[
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
    def _resolve_member(message: discord.Message, *, person_ident: str | int | None = None) -> discord.Member:
        if person_ident is None:
            return cast(discord.Member, message.author)

        person_ident = str(person_ident).casefold()

        for candidate in (message.author, *message.mentions):
            names = (candidate.name, candidate.global_name, candidate.display_name, candidate.id)

            if person_ident in map(str.casefold, map(str, names)):
                return cast(discord.Member, message.author)

        return cast(discord.Member, message.author)

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────────────

    async def summary(self, message: discord.Message | None) -> str:
        if message is None:
            return "- No current author; this is a background identity check."

        member = self._resolve_member(message, person_ident=message.author.id)
        buffer = await self.storage.read()

        if not buffer or not buffer.get(member.name):
            return "- No memories for this user."

        lines: list[str] = []

        for record in buffer[member.name][-20:]:
            lines.append(f"- {record.data['category']}: {record.data['fact']}")

        return "\n".join(lines)

    async def remember(
        self,
        message: discord.Message,
        fact: str,
        category: MemoryCategory = "other",
        person: str | int | None = None,
    ) -> dict[str, Any]:
        member = self._resolve_member(message, person_ident=person)

        record = storage.Record.model_validate({
            "partition_key": member.name,
            "unique_key": "fact",
            "data": {
                "category": category,
                "fact": fact,
                "confidence": "explicit",
                "source": "discord",
                "message_id": message.id if message else None,
                "channel_id": message.channel.id if message else None,
            },
        })

        return await self.storage.upsert(record)

    async def forget(
        self,
        message: discord.Message,
        fact: str,
        category: MemoryCategory | None = None,
        person: str | int | None = None,
    ) -> dict[str, Any]:
        member = self._resolve_member(message, person_ident=person)

        record = storage.Record.model_validate({
            "partition_key": member.name,
            "unique_key": "fact",
            "data": {
                "category": category,
                "fact": fact,
            },
        })

        return await self.storage.delete(record)


Memories = MemoryService()
