import importlib.metadata
import logging
from typing import Annotated

import crescent
import cyclopts
import dotenv
import hikari
import hikariwave
import structlog
from hikari import Snowflake

from voxbot import model

_LOGGER = structlog.get_logger(__name__)


async def _on_starting(event: hikari.StartingEvent) -> None:
    _LOGGER.info("bot_starting", version=importlib.metadata.version("voxbot"))


async def _on_started(event: hikari.StartedEvent) -> None:
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
def main(
    token: Annotated[str, cyclopts.Parameter(env_var="DISCORD_TOKEN")],
    mistral_api_key: Annotated[str, cyclopts.Parameter(env_var="MISTRAL_API_KEY")],
    debug_guild: Annotated[str | None, cyclopts.Parameter(env_var="DEBUG_GUILD")] = None,
) -> int:
    _LOGGER.info("initializing")

    config = model.Config(token=token, mistral_api_key=mistral_api_key)
    voice_model = model.VoxModel(config=config)

    intents = hikari.Intents.GUILD_VOICE_STATES | hikari.Intents.GUILD_MESSAGES
    bot = hikari.GatewayBot(token=token, intents=intents)
    voice_model.voice_client = hikariwave.VoiceClient(bot)

    bot.event_manager.subscribe(hikari.StartingEvent, _on_starting)
    bot.event_manager.subscribe(hikari.StartedEvent, _on_started)

    guild = Snowflake(debug_guild) if debug_guild else None
    client = crescent.Client(bot, voice_model, default_guild=guild)
    client.plugins.load("voxbot.plugins.voice")

    _LOGGER.info("starting")

    try:
        bot.run()
    except KeyboardInterrupt:
        _LOGGER.info("shutdown")
        return 0
    except Exception:
        _LOGGER.exception("error")
        return 1
    return 0


if __name__ == "__main__":
    dotenv.load_dotenv()
    setup_logging()
    raise SystemExit(cli())
