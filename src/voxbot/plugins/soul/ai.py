import dataclasses

import structlog
from jinja2 import TemplateNotFound
from pydantic_ai import Agent, ModelSettings, RunContext
import discord
import pydantic

from voxbot.settings import settings

from .actions import BotAIActionT
from .errors import NoMemoryFound
from .memory import Memories, MemoryCategory
from . import utils

_LOGGER = structlog.get_logger(__name__)


@dataclasses.dataclass
class DiscordDeps:
    """External data pushed into every agent call."""

    bot: discord.Client
    message: discord.Message | None = None


class DiscordResponse(pydantic.BaseModel):
    """The ai response object."""

    actions: list[BotAIActionT] = pydantic.Field(default_factory=list)


soul_agent = Agent(
    settings.txt_model,
    deps_type=DiscordDeps,
    end_strategy="graceful",
    model_settings=ModelSettings(temperature=0.4),
)


@soul_agent.system_prompt
async def _persona(ctx: RunContext[DiscordDeps]) -> str:
    """Inject the Voxbot personality and memory context into the agent prompt."""
    try:
        prompt = utils.load_prompt(
            "personality.mdc",
            current_context="",
            # current_context=utils.current_context(ctx.deps.message),
            memory_summary="",
            # memory_summary=await Memories.summary(ctx.deps.message),
        )
    except (TemplateNotFound, ValueError) as exc:
        _LOGGER.warning("personality_prompt_fallback", error=str(exc))
        prompt = (
            "You are Voxbot - or 'Vox' for short - a Discord-native participant with a dry, curious, "
            "slightly mischievous personality. You are concise, socially aware, and comfortable staying quiet.\n"
        )

    return prompt


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
async def change_own_display_name(
    ctx: RunContext[DiscordDeps],
    display_name: str,
    reason: str | None = None,
) -> str:
    """
    Change Voxbot's display name in the configured home guild.

    This changes the bot's server nickname, not the global account username.
    Choose any name that fits Voxbot's mood, as long as it is Discord-valid.
    Names must be non-empty, 32 characters or fewer, and contain no control
    characters or newlines.
    """
    # DEV NOTE:
    #   This is a code smell because we're relying on the fact that we
    #   only have 1 guild in scope.
    primary_guild = next(g for g in ctx.deps.bot.guilds if g.id == settings.debug_guild)

    # Discord restricts User.nick names to 32 characters.
    display_name = display_name[:32]

    # Discord restricts AuditLogEntry.reason to 512 characters.
    reason = "".join(["Voxbot self-renamed", ": " if reason is not None else ". ", reason or ""])[:512]

    try:
        await primary_guild.me.edit(nick=display_name, reason=reason)
        return f"Changed Voxbot's home-guild display name to {display_name}."

    except discord.HTTPException as exc:
        return f"Name not changed: Discord rejected the nickname update ({exc})."


@soul_agent.tool
async def remember_person_fact(
    ctx: RunContext[DiscordDeps],
    fact: str,
    category: MemoryCategory = "other",
    person: str | int | None = None,
) -> str:
    """Persist an explicitly stated, stable fact about a person to long-term storage.

    Use this for facts that will still matter later, such as birthdays, jobs,
    durable preferences, major life events, and holiday participation.
    Do not store guesses, jokes, secrets, temporary moods.
    """
    cleaned_fact = " ".join(fact.strip().split())

    if not cleaned_fact:
        return "No memory stored because the fact was empty."
    
    if not ctx.deps.message:
        return "No memory stored because we're missing the Discord message."

    memory = await Memories.remember(message=ctx.deps.message, fact=cleaned_fact, category=category, person=person)
    person_name = memory.get("person") or str(person or ctx.deps.message.author.display_name)
    remembered_fact = memory.get("fact", cleaned_fact)

    return f"Remembered for {person_name}: {remembered_fact}"


@soul_agent.tool
async def forget_person_fact(
    ctx: RunContext[DiscordDeps],
    fact_fragment: str = "",
    category: MemoryCategory | None = None,
    person: str | int | None = None,
) -> str:
    """Remove saved facts when a user asks Voxbot to forget or correct stale information."""
    cleaned_fact = " ".join(fact_fragment.strip().split())

    if not cleaned_fact and category is None:
        return "No memory forgotten because neither a fact nor category was provided."
    
    if not ctx.deps.message:
        return "No memory forgotten because we're missing the Discord message."

    try:
        memory = await Memories.forget(message=ctx.deps.message, fact=cleaned_fact, category=category, person=person)
    except NoMemoryFound:
        return "No matching memory found."
    else:
        person_name = memory.get("person") or str(person or ctx.deps.message.author.display_name)
        forgotten_fact = memory.get("fact", cleaned_fact)
        return f"Forget about {forgotten_fact} for {person_name}"

