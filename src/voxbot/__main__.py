import importlib.metadata
import logging
from typing import Annotated

import cyclopts
import disnake
from disnake.ext import commands
import structlog

from voxbot import model, settings

_LOGGER = structlog.get_logger(__name__)


async def _on_ready(bot: disnake.Bot) -> None:
    _LOGGER.info("bot_online")


def setup_logging() -> None:
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


cli = cyclopts.App(help="Voxtral TTS Runner")


@cli.default
def main() -> int:
    _LOGGER.info("initializing")

    # Load settings from .env
    app_settings = settings.Settings()

    # Create model with Mistral config
    voice_model = model.VoxModel(
        mistral_api_key=app_settings.mistral_api_key,
        mistral_model=app_settings.mistral_model,
    )

    # Set up bot with intents
    intents = disnake.Intents.GUILD_VOICE_STATES | disnake.Intents.GUILD_MESSAGES
    bot = commands.Bot(
        command_prefix="!",
        intents=intents,
        test_guilds=[int(app_settings.debug_guild)] if app_settings.debug_guild else None,
    )
    bot.vox_model = voice_model

    # Register ready event
    @bot.event
    async def on_ready():
        _LOGGER.info("bot_online", version=importlib.metadata.version("voxbot"))

    # Load cogs
    bot.load_extensions("voxbot.plugins")

    _LOGGER.info("starting")

    try:
        bot.run(app_settings.discord_token)
    except KeyboardInterrupt:
        _LOGGER.info("shutdown")
        return 0
    except Exception:
        _LOGGER.exception("error")
        return 1
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(cli())
