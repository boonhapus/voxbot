
from pydantic_ai.messages import ModelMessage
from discord.ext import commands
import discord
import structlog

from voxbot.settings import settings
from voxbot import error_reports

from .settings import soul_settings
from . import ai, memory, utils

_LOGGER = structlog.get_logger(__name__)


class SoulCog(commands.GroupCog, name="soul"):
    """It's a chatbot!"""

    def __init__(self, bot: commands.Bot) -> None:
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
        if not (owner := self.bot.get_user(settings.bot_owner_id)):
            return

        sha = settings.voxbot_release_sha
        short = sha[:7] if sha else "unknown"
        link = f"https://github.com/boonhapus/voxbot/commit/{sha}" if sha else "https://github.com/boonhapus/voxbot"

        message = await owner.send(
            f"Hey! I just came back online.\n"
            f"Release: [`{short}`]({link})"
        )

        try:
            await memory.Memories.remember(
                message=message,
                fact="Voxbot just came back online and notified me.",
                category="life_event",
                person=owner.name,
            )
        except Exception:
            _LOGGER.warning("startup_memory_store_failed")

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
            async with message.channel.typing():
                r = await ai.soul_agent.run(
                    message.content,
                    deps=ai.DiscordDeps(bot=self.bot, message=message),
                    output_type=ai.DiscordResponse,
                    message_history=self.conversations.get(message.channel.id, []),
                )

                self.conversations[message.channel.id] = utils.trim_conversation(r.all_messages())

                # DISPATCH ACTIONS
                for action in r.output.actions:
                    try:
                        await action.do(message=message)
                    except discord.HTTPException as exc:
                        _LOGGER.warning(
                            "soul_action_failed",
                            error=str(exc), message=message.id, action=action.kind,
                        )

        except Exception as exc:
            _LOGGER.error("chat_error", exc=type(exc), error=str(exc), user=message.author.id, channel=message.channel.id)
            await message.reply("My circuits are a bit fried right now. Try again?", mention_author=False)

            await error_reports.dm_owner_error_report(
                self.bot,
                subject="Soul chat error",
                title="Soul chat error",
                details={
                    "User": f"{message.author} ({message.author.id})",
                    "Channel": getattr(message.channel, "mention", str(message.channel.id)),
                    "Message": repr(message.content),
                    "Details": f"{type(exc).__name__}: {exc}",
                },
                filename="soul_error_log.txt",
                error=exc,
            )
