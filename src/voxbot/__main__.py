import asyncio
import logging

import cyclopts
import structlog

from voxbot import __project__

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


cli = cyclopts.App(
    name=__project__.__name__.title(),
    help="A memory-capable TTS Discord bot built with discord.py, Songbird (Rust), and powered by Mistral and Gemini.",
    version=__project__.__version__,
)


@cli.command
async def bot() -> int:
    """Run the Discord bot."""
    from voxbot.settings import settings
    from voxbot.store import runtime
    from voxbot.bot import VoxBot

    setup_logging()

    _LOGGER.info("starting")

    bot = VoxBot()

    try:
        await bot.start(settings.discord_token)

    except KeyboardInterrupt:
        _LOGGER.info("shutdown")
        return 0

    except Exception:
        _LOGGER.exception("error")
        return 1

    finally:
        if not bot.is_closed():
            await bot.close()

    return bot.exit_code


@cli.command
async def worker() -> int:
    """Run the external docket process."""
    from voxbot.runtime.worker import run_worker

    setup_logging()

    _LOGGER.info("starting_worker")

    exit_code = await run_worker()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(cli())
