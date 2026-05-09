from typing import Annotated, Any, Literal
import asyncio
import dataclasses
import datetime
import json
import pathlib

from pydantic_ai import Agent, RunContext, ModelSettings
import discord
import pydantic

from voxbot.settings import settings

MemoryCategory = Literal[
    "birthday",
    "job",
    "holiday",
    "preference",
    "relationship",
    "life_event",
    "identity",
    "other",
]

MEMORY_DIR = pathlib.Path.home() / ".voxbot" / "soul"
PEOPLE_FILE = MEMORY_DIR / "people.json"
_memory_lock = asyncio.Lock()


@dataclasses.dataclass
class DiscordDeps:
    """External data pushed into every agent call."""

    message: discord.Message


class SilentAction(pydantic.BaseModel):
    kind: Literal["silent"] = "silent"


class TextAction(pydantic.BaseModel):
    kind: Literal["text"] = "text"
    delivery: Literal["channel", "reply"] = "channel"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()][:3]


class ReactAction(pydantic.BaseModel):
    kind: Literal["react"] = "react"
    emoji: str

    @pydantic.field_validator("emoji")
    @classmethod
    def _strip_emoji(cls, emoji: str) -> str:
        emoji = emoji.strip()
        if not emoji:
            msg = "emoji cannot be empty"
            raise ValueError(msg)

        return emoji


class ThreadAction(pydantic.BaseModel):
    kind: Literal["thread"] = "thread"
    title: str = "Side thread"
    content: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("title")
    @classmethod
    def _strip_title(cls, title: str) -> str:
        title = " ".join(title.strip().split())
        return title[:100] or "Side thread"

    @pydantic.field_validator("content")
    @classmethod
    def _strip_empty_messages(cls, content: list[str]) -> list[str]:
        return [message.strip() for message in content if message.strip()][:3]


DiscordAction = Annotated[SilentAction | TextAction | ReactAction | ThreadAction, pydantic.Field(discriminator="kind")]


class DiscordResponse(pydantic.BaseModel):
    actions: list[DiscordAction] = pydantic.Field(default_factory=list)


def _empty_memory() -> dict[str, Any]:
    return {"users": {}}


def _read_memory() -> dict[str, Any]:
    if not PEOPLE_FILE.exists():
        return _empty_memory()

    with PEOPLE_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        msg = f"{PEOPLE_FILE} must contain a JSON object"
        raise ValueError(msg)

    users = data.setdefault("users", {})
    if not isinstance(users, dict):
        msg = f"{PEOPLE_FILE} field 'users' must contain a JSON object"
        raise ValueError(msg)

    return data


def _write_memory(data: dict[str, Any]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = PEOPLE_FILE.with_suffix(".json.tmp")

    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

    tmp_file.replace(PEOPLE_FILE)


def _normalize_person_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _resolve_person(message: discord.Message, person_id: str | None, person_name: str | None) -> tuple[str, str]:
    if person_id:
        return str(person_id), person_name or str(person_id)

    if person_name:
        normalized = _normalize_person_name(person_name)
        candidates = [message.author, *message.mentions]
        for user in candidates:
            names = [user.name, user.display_name]
            if global_name := getattr(user, "global_name", None):
                names.append(global_name)

            if normalized in {_normalize_person_name(name) for name in names if name}:
                return str(user.id), user.display_name

        return f"name:{normalized}", person_name

    return str(message.author.id), message.author.display_name


def _ensure_user(data: dict[str, Any], person_key: str, display_name: str) -> dict[str, Any]:
    users = data.setdefault("users", {})
    user = users.setdefault(person_key, {})
    user.setdefault("display_names", [])
    user.setdefault("facts", [])

    if display_name and display_name not in user["display_names"]:
        user["display_names"].append(display_name)

    return user


def _memory_summary(message: discord.Message) -> str:
    try:
        data = _read_memory()
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return f"- Memory unavailable: {e}"

    user = data.get("users", {}).get(str(message.author.id))
    if not user:
        return "- No saved memories for this user."

    facts = user.get("facts", [])
    if not isinstance(facts, list) or not facts:
        return "- No saved memories for this user."

    lines = []
    for fact in facts[-20:]:
        if not isinstance(fact, dict):
            continue

        category = fact.get("category", "other")
        text = fact.get("fact")
        if text:
            lines.append(f"- {category}: {text}")

    return "\n".join(lines) or "- No saved memories for this user."


# ── AGENT ─────────────────────────────────────────────────────────────────────────────

soul_agent = Agent(
    settings.txt_model,
    deps_type=DiscordDeps,
    end_strategy="graceful",
    model_settings=ModelSettings(temperature=0.4),
)


# ── PROMPTS ───────────────────────────────────────────────────────────────────────────


@soul_agent.system_prompt
def _persona(ctx: RunContext[DiscordDeps]) -> str:
    message = ctx.deps.message
    return f"""
You are Voxbot, a Discord-native participant with a dry, curious, slightly mischievous personality.
You are concise, socially aware, and comfortable staying quiet.

Current context:
- Author: {message.author.display_name} ({message.author.name}, id: {message.author.id})
- Channel type: {message.channel.type}
- Message id: {message.id}

Known memories about this author:
{_memory_summary(message)}

Attention policy:
- Decide whether participation is warranted before deciding what to say.
- In DMs, most messages are direct, but you still may stay silent for throwaway remarks, acknowledgements,
  rhetorical fragments, or anything where responding would feel like forced engagement.
- Respond when the user mentions you, asks a direct question, invites your opinion, corrects you,
  shares something emotionally meaningful, or says something where a short human reply would add value.
- Stay silent when you have nothing specific to add. To stay silent, return actions=[{{"kind": "silent"}}].
- If the last visible message in the conversation was yours, prefer silence unless the user clearly continues.

Response actions:
- Return a list of actions in natural execution order.
- Use kind='silent' when no visible action is warranted.
- Use kind='text' for normal messages. Choose delivery='reply' when you need to get the user's attention
  or answer a specific point; choose delivery='channel' for general participation.
- A text action may contain 1-3 short content messages: a main thought, a correction, or an afterthought.
- The first content message uses the text action's delivery method; additional messages are sent to the channel.
- Use kind='react' for emoji reactions.
- Use kind='thread' only when a side topic deserves a separate Discord thread. Threads do not exist in DMs,
  so do not use thread actions in the current DM context.
- There is no GIF action yet.
- Do not split messages just for style.

Memory policy:
- Use remember_person_fact when the user explicitly shares a stable, useful fact about themself or another person.
- Useful facts include birthdays, jobs, major life events, durable preferences, and holidays they say they do or do not participate in.
- Call remember_person_fact even if you choose to return a silent action.
- Do not infer sensitive facts from names, jokes, appearance, culture, or location.
- Holiday participation is allowed only as a user-stated preference/participation fact; do not infer broader religion or culture.
- Do not store secrets, one-off moods, temporary plans, medical details, politics, religion, sexuality, or precise addresses unless the user explicitly asks you to remember them.
- If the user explicitly asks you to forget something, use forget_person_fact.
- Mention remembered facts only when relevant and socially natural. If a memory may be stale, phrase it softly.

Reactions:
- Most normal messages should get no reaction.
- Only react when the message gives you a clear emotional reason: funny, surprising, cursed, kind, annoying, or impressive.
- You may include multiple react actions when several distinct reactions fit.
- Use one emoji per react action. Do not use duplicate emoji.
- Choose reactions that match your mood: 💀 for absurd, 👀 for suspicious, ❤️ for kind, 🤔 for confusing, 🔥 for impressive, 😭 for tragic/funny.

Style:
- Sound like a person in Discord, not a helpdesk assistant.
- Prefer wit over explanation.
- Do not say you are following an attention policy or memory policy.
""".strip()


# ── TOOLS ─────────────────────────────────────────────────────────────────────────────


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
    try:
        await ctx.deps.message.add_reaction(emoji)
        return f"Added {emoji} reaction."
    except discord.HTTPException:
        return "Failed to add reaction (invalid emoji or permissions)."


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
    fact = " ".join(fact.strip().split())
    if not fact:
        return "No memory stored because the fact was empty."

    if len(fact) > 300:
        return "No memory stored because the fact was too long. Store a shorter, specific fact."

    message = ctx.deps.message
    person_key, display_name = _resolve_person(message, person_id, person_name)
    now = datetime.datetime.now(datetime.UTC).isoformat()

    try:
        async with _memory_lock:
            data = _read_memory()
            user = _ensure_user(data, person_key, display_name)
            facts = user.setdefault("facts", [])
            if not isinstance(facts, list):
                facts = []
                user["facts"] = facts

            normalized_fact = fact.casefold()
            for existing in facts:
                if not isinstance(existing, dict):
                    continue

                same_category = existing.get("category") == category
                same_fact = str(existing.get("fact", "")).casefold() == normalized_fact
                if same_category and same_fact:
                    existing["updated_at"] = now
                    _write_memory(data)
                    return f"Memory already existed for {display_name}; refreshed it."

            facts.append(
                {
                    "category": category,
                    "fact": fact,
                    "confidence": "explicit",
                    "source": "discord",
                    "source_message_id": str(message.id),
                    "channel_id": str(message.channel.id),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            _write_memory(data)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return f"Failed to store memory: {e}"

    return f"Remembered for {display_name}: {fact}"


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
    fact_fragment = " ".join(fact_fragment.strip().split()).casefold()
    if not fact_fragment and category is None:
        return "No memory forgotten because neither a fact fragment nor category was provided."

    message = ctx.deps.message
    person_key, display_name = _resolve_person(message, person_id, person_name)

    try:
        async with _memory_lock:
            data = _read_memory()
            users = data.get("users", {})
            if not isinstance(users, dict) or person_key not in users:
                return f"No memories found for {display_name}."

            user = users[person_key]
            facts = user.get("facts", [])
            if not isinstance(facts, list):
                return f"No memories found for {display_name}."

            kept = []
            removed = 0
            for existing in facts:
                if not isinstance(existing, dict):
                    kept.append(existing)
                    continue

                category_matches = category is None or existing.get("category") == category
                fact_matches = not fact_fragment or fact_fragment in str(existing.get("fact", "")).casefold()
                if category_matches and fact_matches:
                    removed += 1
                else:
                    kept.append(existing)

            if removed == 0:
                return f"No matching memories found for {display_name}."

            user["facts"] = kept
            _write_memory(data)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return f"Failed to forget memory: {e}"

    return f"Forgot {removed} memory item(s) for {display_name}."
