from typing import Annotated
import logging

import crescent
import cyclopts
import dotenv
import hikari
import hikariwave
import structlog

from voxbot import model

_LOGGER = structlog.get_logger(__name__)


class Client(crescent.Client[hikari.GatewayBot, model.VoxModel]):
    """ """

    async def start(self, intents: hikari.Intents) -> None:
        self.bot.event_manager.subscribe(hikari.StartingEvent, self._on_starting)
        self.bot.event_manager.subscribe(hikari.StartedEvent, self._on_started)
        await super().start(intents)

    async def _on_starting(self, event: hikari.StartingEvent) -> None:
        _LOGGER.info("bot_starting", version="2026.4.17")

    async def _on_started(self, event: hikari.StartedEvent) -> None:
        _LOGGER.info("bot_online", user=str(event.app.get_me()), guilds=len(event.app.guilds))


cli = cyclopts.App(help="Voxtral TTS Runner")


@cli.default
def main(
    token: Annotated[str, cyclopts.Parameter(env_var="DISCORD_TOKEN")],
    mistral_api_key: Annotated[str, cyclopts.Parameter(env_var="MISTRAL_API_KEY")],
) -> int:
    _LOGGER.info("initializing")

    config = model.Config(token=token, mistral_api_key=mistral_api_key)
    model = model.VoxModel(config)

    intents = hikari.Intents.GUILD_VOICE_STATES | hikari.Intents.GUILD_MESSAGES
    bot = hikari.GatewayBot(token=token, intents=intents)
    model.voice_client = hikariwave.VoiceClient(bot)

    client = Client(bot, model)
    client.plugins.load_folder("voxbot.plugins")

    _LOGGER.info("starting")

    try:
        client.run()
    except KeyboardInterrupt:
        _LOGGER.info("shutdown")
        return 0
    except Exception:
        _LOGGER.exception("error")
        return 1
    return 0


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
    dotenv.load_dotenv()
    setup_logging()

    raise SystemExit(cli())
