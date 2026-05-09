from pydantic_ai import Agent, ModelSettings, RunContext
import discord

from voxbot.settings import settings

from .identity import IdentityService
from .memory import MEMORY_DIR, PEOPLE_FILE, MemoryService
from .models import (
    DiscordAction,
    DiscordDeps,
    DiscordResponse,
    MemoryCategory,
    ReactAction,
    SilentAction,
    TextAction,
    ThreadAction,
)
from .prompts import build_persona_prompt

__all__ = [
    "DiscordAction",
    "DiscordDeps",
    "DiscordResponse",
    "MEMORY_DIR",
    "PEOPLE_FILE",
    "MemoryCategory",
    "ReactAction",
    "SilentAction",
    "TextAction",
    "ThreadAction",
    "soul_agent",
]


memory_service = MemoryService()
identity_service = IdentityService(settings.soul_home_guild_id)

soul_agent = Agent(
    settings.txt_model,
    deps_type=DiscordDeps,
    end_strategy="graceful",
    model_settings=ModelSettings(temperature=0.4),
)


def _current_context(message: discord.Message | None) -> str:
    if message is None:
        return """
- Mode: background identity check
- No Discord message is being handled.
""".strip()

    return f"""
- Author: {message.author.display_name} ({message.author.name}, id: {message.author.id})
- Channel type: {message.channel.type}
- Message id: {message.id}
""".strip()


@soul_agent.system_prompt
def _persona(ctx: RunContext[DiscordDeps]) -> str:
    message = ctx.deps.message
    return build_persona_prompt(
        current_context=_current_context(message),
        identity_summary=identity_service.summary(ctx.deps.bot),
        memory_summary=memory_service.summary(message),
    )


@soul_agent.tool
async def react_to_message(ctx: RunContext[DiscordDeps], emoji: str) -> str:
    """
    React to the user's message with one emoji.

    Prefer returning react actions in your final response. Use this tool only when
    you need to react before returning your final actions.

    Use this rarely. Do not call this for normal conversation.
    Only react when Voxbot has a clear emotional response.
    Pick an emoji that matches Voxbot's personality and mood.
    """
    if ctx.deps.message is None:
        return "No reaction added because there is no current Discord message."

    try:
        await ctx.deps.message.add_reaction(emoji)
        return f"Added {emoji} reaction."
    except discord.HTTPException:
        return "Failed to add reaction (invalid emoji or permissions)."


@soul_agent.tool
async def change_own_display_name(ctx: RunContext[DiscordDeps], display_name: str, reason: str | None = None) -> str:
    """
    Change Voxbot's display name in the configured home guild.

    This changes the bot's server nickname, not the global account username.
    Choose any name that fits Voxbot's mood, as long as it is Discord-valid.
    Names must be non-empty, 32 characters or fewer, and contain no control
    characters or newlines.
    """
    return await identity_service.change_display_name(ctx.deps.bot, display_name, reason)


@soul_agent.tool
async def remember_person_fact(
    ctx: RunContext[DiscordDeps],
    fact: str,
    category: MemoryCategory = "other",
    person_id: str | None = None,
    person_name: str | None = None,
) -> str:
    """
    Remember an explicitly stated, stable fact about a person.

    Use this for facts that will still matter later, such as birthdays, jobs,
    durable preferences, major life events, and holiday participation.
    Do not store guesses, jokes, secrets, temporary moods, or sensitive facts
    unless the user clearly asks Voxbot to remember them.

    If person_id is omitted, the fact is stored for the current message author.
    """
    return await memory_service.remember(ctx.deps.message, fact, category, person_id, person_name)


@soul_agent.tool
async def forget_person_fact(
    ctx: RunContext[DiscordDeps],
    fact_fragment: str = "",
    category: MemoryCategory | None = None,
    person_id: str | None = None,
    person_name: str | None = None,
) -> str:
    """
    Forget saved facts when a user asks Voxbot to forget something or corrects stale information.

    Provide category when the user wants a class of facts forgotten, such as birthday or holiday.
    Provide fact_fragment when removing one specific memory. If person_id is omitted, the current
    message author is used.
    """
    return await memory_service.forget(ctx.deps.message, fact_fragment, category, person_id, person_name)
