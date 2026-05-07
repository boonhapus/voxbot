from pydantic_ai.messages import ModelMessage
from discord.ext import commands
import discord
import structlog

from . import ai

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
    async def on_message(self, message: discord.Message) -> None:
        """Process an incoming message."""
        # 1. Filter: No bots, and must be a DM
        if message.author.bot or not isinstance(message.channel, discord.DMChannel):
            return
        
        if (user_id := message.author.id) not in self.conversations:
            self.conversations[user_id] = []

        async with message.channel.typing():
            try:
                r = await ai.soul_agent.run(
                    message.content,
                    deps=ai.DiscordDeps(message=message),
                    output_type=ai.DiscordResponse,
                    message_history=self.conversations[user_id]
                )

                self.conversations[user_id] = r.all_messages()

                if r.output and r.output.content:
                    if r.output.delivery == "reply":
                        await message.reply(r.output.content, mention_author=False)
                    else:
                        await message.channel.send(r.output.content)

            except Exception as e:
                _LOGGER.error("gemini_error", error=str(e), user_id=user_id)
                await message.reply("My circuits are a bit fried right now. Try again?")
