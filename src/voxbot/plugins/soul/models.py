from typing import Annotated, Literal
import dataclasses

import discord
import pydantic

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


@dataclasses.dataclass
class DiscordDeps:
    """External data pushed into every agent call."""

    bot: discord.Client
    message: discord.Message | None = None


class SilentAction(pydantic.BaseModel):
    kind: Literal["silent"] = "silent"


class TextAction(pydantic.BaseModel):
    kind: Literal["text"] = "text"
    delivery: Literal["channel", "reply"] = "channel"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()][:3]


class ReactAction(pydantic.BaseModel):
    kind: Literal["react"] = "react"
    emoji: str

    @pydantic.field_validator("emoji")
    @classmethod
    def _strip_emoji(cls, emoji: str) -> str:
        emoji = emoji.strip()
        if not emoji:
            msg = "emoji cannot be empty"
            raise ValueError(msg)

        return emoji


class ThreadAction(pydantic.BaseModel):
    kind: Literal["thread"] = "thread"
    title: str = "Side thread"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("title")
    @classmethod
    def _strip_title(cls, title: str) -> str:
        title = " ".join(title.strip().split())
        return title[:100] or "Side thread"

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()][:3]


DiscordAction = Annotated[SilentAction | TextAction | ReactAction | ThreadAction, pydantic.Field(discriminator="kind")]


class DiscordResponse(pydantic.BaseModel):
    actions: list[DiscordAction] = pydantic.Field(default_factory=list)
