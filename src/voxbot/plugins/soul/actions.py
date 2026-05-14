from typing import Annotated, Literal

import discord
import pydantic
import structlog

_LOGGER = structlog.get_logger(__name__)


class BotAIAction(pydantic.BaseModel):
    """The base class for any automated bot action."""

    async def do(self, message: discord.Message) -> None:
        """Override to allow the bot to take an action."""
        pass


class SilentAction(BotAIAction):
    kind: Literal["silent"] = "silent"

    async def do(self, message: discord.Message) -> None:
        """Nothing."""
        pass


class TextAction(BotAIAction):
    kind: Literal["text"] = "text"
    delivery: Literal["channel", "reply"] = "channel"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()][:3]

    async def do(self, message: discord.Message) -> None:
        """Send a message back."""
        for idx, content in enumerate(self.content):
            if idx == 0 and self.delivery == "reply":
                await message.reply(content, mention_author=False)
            else:
                await message.channel.send(content)


class ReactAction(BotAIAction):
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

    async def do(self, message: discord.Message) -> None:
        await message.add_reaction(self.emoji)


class ThreadAction(BotAIAction):
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

    async def do(self, message: discord.Message) -> None:
        if not self.content:
            return
        
        if message.channel.guild is None:
            await TextAction(content=self.content).do(message)
            return

        if isinstance(message.channel, discord.Thread):
            thread = message.channel
        else:
            try:
                thread = await message.create_thread(name=self.title)
            except discord.HTTPException as exc:
                _LOGGER.warning("thread_create_failed", error=str(exc), title=self.title, message_id=message.id)
                await TextAction(content=self.content).do(message)
                return

        for content in self.content:
            await thread.send(content)

type _BotAIAction = SilentAction | TextAction | ReactAction | ThreadAction
type BotAIActionT = Annotated[_BotAIAction, pydantic.Field(discriminator="kind")]
