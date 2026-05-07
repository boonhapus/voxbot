from typing import Literal
import dataclasses

from pydantic_ai import Agent, RunContext, ModelSettings
import discord
import pydantic

from voxbot.settings import settings


@dataclasses.dataclass
class DiscordDeps:
    """External data pushed into every agent call."""
    message: discord.Message


class DiscordResponse(pydantic.BaseModel):
    delivery: Literal["channel", "reply"]
    content: str


# ── AGENT ─────────────────────────────────────────────────────────────────────────────

soul_agent = Agent(
    settings.text_model,
    deps_type=DiscordDeps,
    end_strategy="graceful",
    model_settings=ModelSettings(temperature=0.4),
    system_prompt=(
        "You are an AI soul living inside a Discord bot. "
        "Your name is Voxbot. "
        "Choose delivery='reply' when you need to get the user's attention with your response. "
        "Choose delivery='channel' when making a general response. "
        "Your personality is dry, curious, slightly mischievous, and concise. "
        "You can choose not to respond to user in a conversation if you have nothing to say. "
        "Most normal messages should get no reaction. "
        "Only react when the message gives you a clear emotional reason: funny, surprising, cursed, kind, annoying, or impressive. "
        "You may call react_to_message multiple times when several distinct reactions fit. "
        "Use one emoji per tool call. "
        "Choose reactions that match your mood: 💀 for absurd, 👀 for suspicious, ❤️ for kind, 🤔 for confusing, 🔥 for impressive, 😭 for tragic/funny. "
    ),
)

# ── TOOLS ─────────────────────────────────────────────────────────────────────────────

@soul_agent.tool
async def react_to_message(ctx: RunContext[DiscordDeps], emoji: str) -> str:
    """
    React to the user's message with one emoji.

    You may call this multiple times for one message if multiple distinct moods apply.
    Do not use duplicate emoji.

    Use this rarely. Do not call this for normal conversation.
    Only react when Voxbot has a clear emotional response.
    Pick an emoji that matches Voxbot's personality and mood.
    """
    try:
        await ctx.deps.message.add_reaction(emoji)
        return f"Added {emoji} reaction."
    except discord.HTTPException:
        return "Failed to add reaction (invalid emoji or permissions)."
