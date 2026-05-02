"""LLM helpers — generates short in-character TTS lines from user prompts."""

import pydantic_ai
import structlog

from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """
You write short voice lines for a text-to-speech bot.

Rules:
- Output ONLY the spoken line itself. No quotes, no narration, no stage directions, no labels.
- Keep it concise: a single sentence, ideally under 20 words. Two sentences only if needed.
- If a Dota 2 hero is named, write the line in that hero's established voice and personality from Dota 2.
- Otherwise produce a neutral, natural spoken line that fits the user's prompt.
- Never refuse, never explain, never apologize. Just write the line.
"""

_agent: pydantic_ai.Agent | None = None


def _get_agent() -> pydantic_ai.Agent:
    global _agent

    if _agent is None:
        _agent = pydantic_ai.Agent(f"google-gla:{settings.gemini_model}", system_prompt=_SYSTEM_PROMPT)

    return _agent


async def generate_line(prompt: str, dota_hero: str | None = None) -> str:
    """Generate a short in-character TTS line from a user prompt."""
    if not settings.google_api_key:
        raise RuntimeError("google_api_key is not configured")

    parts: list[str] = []

    if dota_hero:
        parts.append(f"Dota 2 hero: {dota_hero}")

    parts.append(f"Prompt: {prompt}")
    user_message = "\n".join(parts)

    result = await _get_agent().run(user_message)
    text = "".join(part.content for part in result.response.parts if isinstance(part, pydantic_ai.TextPart))
    line = text.strip().strip('"').strip("'").strip()
    _LOGGER.info("ai_line_generated", hero=dota_hero, prompt_len=len(prompt), output_len=len(line))
    return line
