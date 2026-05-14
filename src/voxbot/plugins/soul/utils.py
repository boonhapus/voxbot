from typing import Any
import functools as ft
import pathlib

import jinja2
import structlog
import yaml

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
import discord

_LOGGER = structlog.get_logger(__name__)
_PROMPT_DIR = pathlib.Path(__file__).parent.joinpath("prompts")


@ft.cache
def _env_for(loader_root: pathlib.Path) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(loader_root),
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_prompt(template_path: str, **variables: Any) -> str:
    """
    Load a prompt from the prompts directory.

    If the path is in a subdirectory, that directory becomes the Jinja
    loader root so includes resolve relative to it.
    """
    path = pathlib.Path(template_path)
    env = _env_for(_PROMPT_DIR)

    source, _, _ = env.loader.get_source(env, path.name)

    if source.count("---\n") < 2:
        raise ValueError(f"'{template_path}' is missing frontmatter delimiters")

    _, _, rest = source.partition("---\n")
    m, _, body = rest.partition("---\n")

    metadata = yaml.safe_load(m) or {}

    _LOGGER.debug("Constructing prompt", prompt_path=template_path, prompt_version=metadata.get("version"))

    return env.from_string(body.strip()).render(**variables)


def current_context(message: discord.Message | None) -> str:
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


def trim_conversation(messages: list[ModelMessage], *, max_turns: int = 20) -> list[ModelMessage]:
    user_request_indexes = [
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]

    if len(user_request_indexes) <= max_turns:
        return messages

    return messages[user_request_indexes[-max_turns] :]
