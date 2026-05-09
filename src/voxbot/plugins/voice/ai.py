"""AI helpers for generated voice lines."""

from pydantic_ai import Agent
import structlog

from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)

LINE_SYSTEM_PROMPT = """\
You write short voice lines for a text-to-speech bot.

Rules:
- Output ONLY the spoken line itself. No quotes, no narration, no stage directions, no labels.
- Keep it concise: a single sentence, ideally under 20 words. Two sentences only if needed.
- If a Dota 2 hero is named, write the line in that hero's established voice and personality from Dota 2.
- Otherwise produce a neutral, natural spoken line that fits the user's prompt.
- Never refuse, never explain, never apologize. Just write the line.
"""

voice_line_agent = Agent(settings.txt_model, system_prompt=LINE_SYSTEM_PROMPT)


async def generate_line(prompt: str, dota_hero: str | None = None) -> str:
    """Generate a short in-character TTS line from a user prompt."""
    parts: list[str] = []

    if dota_hero:
        parts.append(f"Dota 2 hero: {dota_hero}")

    parts.append(f"Prompt: {prompt}")

    result = await voice_line_agent.run("\n".join(parts))
    line = result.output.strip().strip('"').strip("'").strip()
    _LOGGER.info("ai_line_generated", hero=dota_hero, prompt_len=len(prompt), output_len=len(line))
    return line
