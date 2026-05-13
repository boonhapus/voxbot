from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from discord.ext import commands
import discord
import structlog

from voxbot.settings import settings

from . import ai

_LOGGER = structlog.get_logger(__name__)
_MAX_CONVERSATION_TURNS = 20


def _primary_owner_id() -> int | None:
    raw = settings.bot_owner_id
    if not raw:
        return None
    for token in raw.replace(",", " ").split():
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _trim_conversation(messages: list[ModelMessage]) -> list[ModelMessage]:
    user_request_indexes = [
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]

    if len(user_request_indexes) <= _MAX_CONVERSATION_TURNS:
        return messages

    return messages[user_request_indexes[-_MAX_CONVERSATION_TURNS] :]


class SoulCog(commands.GroupCog, name="soul"):
    """It's a chatbot!"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.conversations: dict[int, list[ModelMessage]] = {}
        self._startup_notified = False

    @staticmethod
    def _conversation_key(message: discord.Message) -> int:
        return message.channel.id

    @staticmethod
    def _is_scoped_channel(message: discord.Message) -> bool:
        if not settings.soul_channel_ids:
            return False
        return str(message.channel.id) in settings.soul_channel_ids

    async def _send_text_action(self, message: discord.Message, action: ai.TextAction) -> None:
        for idx, content in enumerate(action.content):
            if idx == 0 and action.delivery == "reply":
                await message.reply(content, mention_author=False)
            else:
                await message.channel.send(content)

    async def _send_react_action(self, message: discord.Message, action: ai.ReactAction) -> None:
        try:
            await message.add_reaction(action.emoji)
        except discord.HTTPException as e:
            _LOGGER.warning("reaction_failed", error=str(e), emoji=action.emoji, message_id=message.id)

    async def _send_thread_action(self, message: discord.Message, action: ai.ThreadAction) -> None:
        if not action.content:
            return

        if isinstance(message.channel, discord.Thread):
            thread = message.channel
        elif isinstance(message.channel, discord.DMChannel):
            await self._send_text_action(message, ai.TextAction(content=action.content))
            return
        else:
            try:
                thread = await message.create_thread(name=action.title)
            except discord.HTTPException as e:
                _LOGGER.warning("thread_create_failed", error=str(e), title=action.title, message_id=message.id)
                await self._send_text_action(message, ai.TextAction(content=action.content))
                return

        for content in action.content:
            await thread.send(content)

    async def _send_actions(self, message: discord.Message, response: ai.DiscordResponse) -> None:
        for action in response.actions:
            if isinstance(action, ai.SilentAction):
                continue
            if isinstance(action, ai.TextAction):
                await self._send_text_action(message, action)
            elif isinstance(action, ai.ReactAction):
                await self._send_react_action(message, action)
            elif isinstance(action, ai.ThreadAction):
                await self._send_thread_action(message, action)

    # ── EVENT LISTENERS ───────────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Notify owner on first ready event after startup."""
        if self._startup_notified:
            return
        self._startup_notified = True

        owner_id = _primary_owner_id()
        if owner_id is None:
            _LOGGER.warning("startup_notify_skipped", reason="no_owner_configured")
            return

        try:
            owner = self.bot.get_user(owner_id) or await self.bot.fetch_user(owner_id)
        except discord.HTTPException:
            _LOGGER.warning("startup_notify_fetch_failed", owner_id=owner_id)
            return

        try:
            await owner.send("Hey! I just came back online.")
        except discord.HTTPException:
            _LOGGER.warning("startup_notify_dm_failed", owner_id=owner_id)

        try:
            await ai.memory_service.remember(
                message=None,
                fact="Voxbot just came back online and notified me.",
                category="life_event",
                person_id=str(owner_id),
                person_name=owner.display_name,
            )
        except Exception:
            _LOGGER.warning("startup_memory_store_failed")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Process an incoming message."""
        if message.author.bot or not self._is_scoped_channel(message):
            return

        conversation_id = self._conversation_key(message)
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        try:
            async with message.channel.typing():
                r = await ai.soul_agent.run(
                    message.content,
                    deps=ai.DiscordDeps(bot=self.bot, message=message),
                    output_type=ai.DiscordResponse,
                    message_history=self.conversations[conversation_id],
                )

            self.conversations[conversation_id] = _trim_conversation(r.all_messages())
            if r.output:
                await self._send_actions(message, r.output)

        except Exception as e:
            _LOGGER.error("chat_error", error=str(e), user_id=message.author.id, conversation_id=conversation_id)
            try:
                await message.reply("My circuits are a bit fried right now. Try again?", mention_author=False)
            except discord.HTTPException:
                await message.channel.send("My circuits are a bit fried right now. Try again?")
