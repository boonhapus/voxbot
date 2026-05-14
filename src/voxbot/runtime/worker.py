import asyncio
import signal

import docket
import structlog

from voxbot.runtime.docket import DurableTasks
from voxbot.runtime.health import RedisHealthRuntime
from voxbot.settings import settings
from voxbot import __project__

_LOGGER = structlog.get_logger(__name__)


async def run_worker() -> int:
    """Run the external Docket worker process."""
    stop_event = asyncio.Event()


    # ── LIFECYCLE METHODS ─────────────────────────────────────────────────────────────

    loop = asyncio.get_running_loop()

    for sig_num in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig_num, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue


    # ── LOOP FOREVER ──────────────────────────────────────────────────────────────────

    health = RedisHealthRuntime(key_prefix="voxbot:worker:health")

    await health.start()

    try:
        async with docket.Docket(name=__project__.__name__, url=settings.redis_url) as d:
            _LOGGER.info("docket_worker_registered_tasks", task_count=len(DurableTasks.get_tasks()))

            async with docket.Worker(d) as worker:
                await health.mark_ready(True)

                try:
                    async with asyncio.TaskGroup() as g:
                        worker_task = g.create_task(worker.run_forever(), name="voxbot-docket-worker")

                        # Wait for a SIGINT / SIGTERM.
                        await stop_event.wait()

                        # This will raise asyncio.CancelledError, breaking us from the loop.
                        worker_task.cancel()

                except asyncio.CancelledError:
                    _LOGGER.info("docket_worker_shutting_down")

        return 0

    except KeyboardInterrupt:
        _LOGGER.info("docket_worker_interrupted")
        return 0

    except Exception as exc:
        await health.record_error(exc=exc)
        _LOGGER.exception("docket_worker_failed")
        return 1

    finally:
        await health.stop()
