from typing import Annotated, Literal
import abc
import asyncio

import discord
import pydantic
import structlog

from voxbot.bot import VoxBot
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)


# ── PROTOCOL ──────────────────────────────────────────────────────────────────────────


class BotAIAction(pydantic.BaseModel, abc.ABC):
    """
    The base class for any automated bot action.

    The docstring is sent along with the prompt to the LLM. Strong docstrings
    describe not only the functionality, but when to use the action.

    pydantic.Field() members should also have their `description` parameter
    sent as well.
    """

    @abc.abstractmethod
    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        """Override to allow the bot to take an action."""
        ...

    async def run(self, *, bot: VoxBot, message: discord.Message) -> None:
        """Runs the action."""
        try:
            await self.do(bot=bot, message=message)
        except discord.HTTPException as exc:
            assert hasattr(self, "kind"), "Protocol member `kind` not defined."
            _LOGGER.warning("bot_ui_action_failed", error=str(exc), message=message.id, action=self.kind)


# ── BOT ACTIONS ───────────────────────────────────────────────────────────────────────


class Silent(BotAIAction):
    """Choose not to respond to the message."""

    kind: Literal["silent"] = "silent"

    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        """Do nothing."""
        pass


class Respond(BotAIAction):
    """Respond with one or more messages."""

    kind: Literal["respond"] = "respond"
    delivery: Literal["channel", "reply"] = "channel"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()]

    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        """Send a message back."""
        TYPING_SPEED_WPM = 90

        async with message.channel.typing():
            for idx, content in enumerate(self.content):
                n_words = len(content.split(" "))

                # SIMULATE TYPING
                await asyncio.sleep(n_words * TYPING_SPEED_WPM * 60)

                if idx == 0 and self.delivery == "reply":
                    await message.reply(content, mention_author=False)
                else:
                    await message.channel.send(content)


class React(BotAIAction):
    """Add or remove an emoji reaction to the message."""

    kind: Literal["react"] = "react"
    emoji: str
    mode: Literal["add", "remove"] = "add"

    @pydantic.field_validator("emoji")
    @classmethod
    def _strip_emoji(cls, emoji: str) -> str:
        if not (emoji := emoji.strip()):
            msg = "emoji cannot be empty"
            raise ValueError(msg)

        return emoji

    async def do(self, bot: VoxBot, message: discord.Message) -> None:
        try:
            if self.mode == "add":
                await message.add_reaction(self.emoji)
            else:
                await message.remove_reaction(self.emoji, member=bot.me)
        except discord.NotFound:
            pass


class RenameSelf(BotAIAction):
    """Change your display name."""

    kind: Literal["rename_self"] = "rename_self"
    name: str = pydantic.Field(max_length=32)

    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        guild = bot.get_guild(settings.debug_guild) if message.guild is None else message.guild
        assert guild is not None, "Guild cannot be None."
        await guild.me.edit(nick=self.name, reason=f"VoxBot renamed themself as a result of {message.id}")


type _BotAIAction = Silent | Respond | React | RenameSelf
type BotAIActionT = Annotated[_BotAIAction, pydantic.Field(discriminator="kind")]
