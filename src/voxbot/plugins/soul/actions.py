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
        assert hasattr(self, "kind"), "Protocol member `kind` not defined."

        try:
            _LOGGER.info("bot_ai_action_chosen", action=self.kind)
            await self.do(bot=bot, message=message)
        except discord.HTTPException as exc:
            _LOGGER.warning("bot_ai_action_failed", error=str(exc), message=message.id, action=self.kind)
            await bot.on_error("on_message", message)


# ── BOT ACTIONS ───────────────────────────────────────────────────────────────────────


class Silent(BotAIAction):
    """Choose not to respond to the message."""

    kind: Literal["silent"] = "silent"

    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        """Do nothing."""
        pass


class Respond(BotAIAction):
    """
    Respond with one or more messages.

    Use this when the user mentions you, asks a direct question, invites
    your opinion, or shares something meaningful. Choose delivery='reply'
    to reply directly to someone; use delivery='channel' for general
    participation. You may include 1-3 short messages: a main thought,
    a correction, or an afterthought.
    """

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
                await asyncio.sleep(n_words / TYPING_SPEED_WPM * 60)

                if idx == 0 and self.delivery == "reply":
                    await message.reply(content, mention_author=False)
                else:
                    await message.channel.send(content)


class React(BotAIAction):
    """
    Add or remove an emoji reaction to the message.

    Only use this when the message gives a clear emotional reason: funny,
    surprising, cursed, kind, annoying, or impressive. Most normal messages
    should get no reaction. You may include multiple react actions but use
    one emoji per action. Do not use duplicate emoji.
    """

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
    """
    Change your display name.

    You may use this whenever a new home-guild display name feels right. You
    choose the name; do not ask permission. Use at most once per agent run.
    """

    kind: Literal["rename_self"] = "rename_self"
    name: str = pydantic.Field(max_length=32)

    async def do(self, *, bot: VoxBot, message: discord.Message) -> None:
        guild = bot.get_guild(settings.debug_guild) if message.guild is None else message.guild
        assert guild is not None, "Guild cannot be None."
        await guild.me.edit(nick=self.name, reason=f"VoxBot renamed themself as a result of {message.id}")


type _BotAIAction = Respond | React | RenameSelf
type BotAIActionT = Annotated[_BotAIAction, pydantic.Field(discriminator="kind")]


def action_prompt_section() -> str:
    lines: list[str] = []
    for cls in BotAIAction.__subclasses__():
        if cls is Silent:
            continue
        kind: str = cls.model_fields["kind"].default  # type: ignore[union-attr]
        doc = cls.__doc__.strip() if cls.__doc__ else ""
        doc = " ".join(doc.split())
        lines.append(f"- ``kind='{kind}'`` — {doc}")
    return "\n".join(lines)
