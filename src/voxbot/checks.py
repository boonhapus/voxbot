from collections.abc import Callable

from discord.ext import commands
from discord.ext.commands._types import BotT

from voxbot import errors
from voxbot.settings import settings


def is_bot_admin[T]() -> Callable[[T], T]:
    """Check if the caller is the Bot's admin."""

    async def _predicate(ctx: commands.Context[BotT]) -> bool:
        if ctx.author.id != settings.bot_owner_id:
            raise errors.NotAnAdmin("You are not the bot's admin!")
        return True

    return commands.check(_predicate)
