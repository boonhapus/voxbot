import asyncio
import logging
import signal

import cyclopts
import structlog

_LOGGER = structlog.get_logger(__name__)


def setup_ssl() -> None:
    import os
    import certifi
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)


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
    setup_ssl()
    setup_logging()
    _LOGGER.info("starting")
    return asyncio.run(run_discord_bot())


async def run_discord_bot() -> int:
    from voxbot.bot import VoxBot
    from voxbot.settings import settings

    bot = VoxBot()
    _install_bot_signal_handlers(bot)

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
def worker() -> int:
    setup_ssl()
    setup_logging()
    _LOGGER.info("starting_worker")

    from voxbot.runtime.worker import run_worker

    return asyncio.run(run_worker())


def _install_bot_signal_handlers(bot) -> None:
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                lambda sig=sig: asyncio.create_task(
                    bot.request_shutdown(reason=f"{sig.name} received", exit_code=0)
                ),
            )
        except (NotImplementedError, RuntimeError):
            continue


if __name__ == "__main__":
    raise SystemExit(cli())
