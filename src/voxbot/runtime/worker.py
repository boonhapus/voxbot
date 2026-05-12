from __future__ import annotations

import asyncio
from contextlib import suppress
import signal

from docket import Docket, Worker
import structlog

from voxbot.runtime.docket import register_pure_background_tasks
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings

_LOGGER = structlog.get_logger(__name__)


async def run_worker() -> int:
    """Run the external Docket worker process."""
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    health = RedisHealthRuntime(settings, key_prefix="voxbot:worker:health")
    await health.start()

    try:
        if not settings.docket_enabled:
            _LOGGER.info("docket_worker_disabled")
            await health.mark_ready(True)
            await stop_event.wait()
            return 0

        docket_url = settings.docket_url or settings.redis_url
        async with Docket(name=settings.docket_name, url=docket_url) as docket:
            task_count = register_pure_background_tasks(docket)
            _LOGGER.info("docket_worker_registered_tasks", task_count=task_count)

            async with Worker(docket) as worker:
                await health.mark_ready(True)
                worker_task = asyncio.create_task(worker.run_forever(), name="voxbot-docket-worker")
                stop_task = asyncio.create_task(stop_event.wait(), name="voxbot-worker-stop")

                done, pending = await asyncio.wait(
                    {worker_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task

                if worker_task in done:
                    await worker_task
                else:
                    worker_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await worker_task

        return 0
    except KeyboardInterrupt:
        _LOGGER.info("docket_worker_interrupted")
        return 0
    except Exception as err:
        await health.record_error(str(err))
        _LOGGER.exception("docket_worker_failed")
        return 1
    finally:
        await health.stop()


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue
