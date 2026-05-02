import logging

import cyclopts
import structlog

_LOGGER = structlog.get_logger(__name__)


def setup_logging() -> None:
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
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
    setup_logging()
    _LOGGER.info("starting")

    from voxbot.bot import VoxBot
    from voxbot.settings import settings

    bot = VoxBot()

    try:
        bot.run(settings.discord_token)
    except KeyboardInterrupt:
        _LOGGER.info("shutdown")
        return 0
    except Exception:
        _LOGGER.exception("error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
