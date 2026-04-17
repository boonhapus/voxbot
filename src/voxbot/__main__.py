import logging
from pathlib import Path
from typing import Annotated

import cyclopts
import dotenv
import hikari
import structlog

import crescent
import hikariwave

from voxbot.model import Config
from voxbot.model import VoxModel
from voxbot.model import LOGGER

dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_model: VoxModel | None = None


async def on_starting(event: hikari.StartingEvent) -> None:
    global _model
    if _model:
        _model._log.info("bot_starting", version="2026.4.17")


async def on_started(event: hikari.StartedEvent) -> None:
    global _model
    if _model:
        _model._log.info(
            "bot_online",
            user=str(event.app.get_me()),
            guilds=len(event.app.guilds),
        )


cli = cyclopts.App(help="Voxtral TTS Runner")


@cli.default
def main(
    token: Annotated[str, cyclopts.Parameter(env_var="DISCORD_TOKEN")],
    mistral_api_key: Annotated[str, cyclopts.Parameter(env_var="MISTRAL_API_KEY")],
) -> int:
    global _model

    config = Config(token=token, mistral_api_key=mistral_api_key)
    model = VoxModel(config)
    _model = model

    intents = hikari.Intents.GUILD_VOICE_STATES | hikari.Intents.GUILD_MESSAGES
    bot = hikari.GatewayBot(token=token, intents=intents)
    model.voice_client = hikariwave.VoiceClient(bot)

    client = crescent.Client(bot, model)
    client.plugins.load_folder("voxbot.plugins")

    bot.event_manager.subscribe(hikari.StartingEvent, on_starting)
    bot.event_manager.subscribe(hikari.StartedEvent, on_started)

    setup_logging()
    model._log.info("starting")

    try:
        bot.run()
        return 0
    except KeyboardInterrupt:
        model._log.info("shutdown")
        return 0
    except Exception:
        model._log.exception("error")
        return 1


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


if __name__ == "__main__":
    raise SystemExit(cli())
