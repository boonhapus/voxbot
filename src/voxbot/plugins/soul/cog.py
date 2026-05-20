import asyncio

from discord.ext import commands
from pydantic_ai.messages import ModelMessage
import discord
import structlog

from voxbot.bot import VoxBot
from voxbot.settings import settings

from . import ai, memory, utils
from .settings import soul_settings

_LOGGER = structlog.get_logger(__name__)


class SoulCog(commands.GroupCog, name="soul"):
    """It's a chatbot!"""

    def __init__(self, bot: VoxBot) -> None:
        self.bot = bot
        self.conversations: dict[int, list[ModelMessage]] = {}

    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Called when cog is loaded."""
        _LOGGER.info("soul_cog_loaded")

    # ── EVENT LISTENERS ───────────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Notify owner on first ready event after startup."""
        sha = settings.voxbot_release_sha
        url = f"https://github.com/boonhapus/voxbot/commit/{sha}" if sha else "https://github.com/boonhapus/voxbot"

        message = await self.bot.dad.send(
            f"👋 Hey! I just came back online.\n\nRelease: [`{sha[:7] if sha else 'unknown'}`]({url})",
            suppress_embeds=True,
        )

        await memory.Memories.remember(
            message=message,
            fact="Voxbot just came back online and notified me.",
            person=self.bot.dad.name,
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Route whitelisted messages through the AI soul agent and dispatch resulting actions."""
        if message.author.bot:
            return

        is_channel_whitelisted = message.channel.id in soul_settings.channel_ids
        is_private_whitelisted = message.guild is None and message.author.id == settings.bot_owner_id

        if not (is_channel_whitelisted or is_private_whitelisted):
            return

        try:
            r = await ai.soul_agent.run(
                message.content,
                deps=ai.DiscordDeps(bot=self.bot, message=message),
                output_type=ai.DiscordResponse,
                message_history=self.conversations.get(message.channel.id, []),
            )

            self.conversations[message.channel.id] = utils.trim_conversation(r.all_messages())

            # DISPATCH ACTIONS
            _ = await asyncio.gather(*[a.run(bot=self.bot, message=message) for a in r.output.actions])

        except Exception as exc:
            exc_type, tb = type(exc), exc.__traceback__

            assert tb is not None, "Not handling an active Exception."

            _LOGGER.error(
                "chat_error", exc=exc_type, error=str(exc), user=message.author.id, channel=message.channel.id
            )

            await message.reply("My circuits are a bit fried right now. Try again?", mention_author=False)
            await self.bot.on_error("on_message", message)
