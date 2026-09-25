import asyncio
import os
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

"""
Load shedding for the inference endpoints.

Every model call is CPU-bound and runs in a thread, so without a limit a
burst of requests piles up in the threadpool: each in-flight call spawns
its own team of OpenMP threads, the cores are oversubscribed, and every
later request, including the cheap ones, waits behind an ever-growing
first-come first-served line. The gate below admits a bounded number of
inference calls per worker process and keeps a bounded queue in front of
them; a request that finds the queue full is refused at once with 503 and
``Retry-After``, which gives the caller a back-pressure signal instead of
an unbounded wait. Endpoints that run no model bypass the gate entirely.

The gate lives in the event loop, so the check costs nothing and the
refused request never touches a thread. Handlers therefore are ``async``
and hand their service call to :meth:`InferenceGate.run`.
"""

CONCURRENCY_ENV = "ML_GATEWAY_INFERENCE_CONCURRENCY"
QUEUE_ENV = "ML_GATEWAY_INFERENCE_QUEUE"
# One inference at a time per worker: with ``OMP_NUM_THREADS`` intra-op
# threads per call and one worker per ``OMP_NUM_THREADS`` cores, that is
# exactly what keeps every core busy without oversubscribing them.
_DEFAULT_CONCURRENCY = 1
# At a few hundred milliseconds per call, 32 queued requests bound the wait
# to a handful of seconds.
_DEFAULT_QUEUE = 32
RETRY_AFTER_SECONDS = 1

P = ParamSpec("P")
T = TypeVar("T")


class InferenceGate:
    """Per-process admission control for CPU-bound inference calls.

    Attributes:
        concurrency: Calls allowed to run at the same time.
        queue: Calls allowed to wait for a running slot; one more is
            refused with 503.
        running: Calls currently executing.
        waiting: Calls currently waiting for a slot.
    """

    def __init__(self, concurrency: int, queue: int) -> None:
        self.configure(concurrency, queue)

    @classmethod
    def from_env(cls) -> "InferenceGate":
        return cls(
            int(os.environ.get(CONCURRENCY_ENV, _DEFAULT_CONCURRENCY)),
            int(os.environ.get(QUEUE_ENV, _DEFAULT_QUEUE)),
        )

    def configure(self, concurrency: int, queue: int) -> None:
        """Set the limits; only meaningful while no call is in flight."""
        if concurrency < 1 or queue < 0:
            raise ValueError("concurrency must be at least 1 and queue at least 0")
        self.concurrency = concurrency
        self.queue = queue
        self.running = 0
        self.waiting = 0
        # Created on first use, inside the worker's event loop: the module
        # is imported in the gunicorn master before the fork.
        self._semaphore: asyncio.Semaphore | None = None

    def _slots(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    async def run(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run ``fn`` in the threadpool once a slot is free, or refuse with 503."""
        slots = self._slots()
        if slots.locked() and self.waiting >= self.queue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Overloaded: {self.running} inference calls running and {self.waiting} queued "
                    f"in this worker. Retry after {RETRY_AFTER_SECONDS} second."
                ),
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        self.waiting += 1
        try:
            # A client that hangs up while queued cancels this wait; the
            # ``finally`` keeps the count honest either way.
            await slots.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        try:
            return await run_in_threadpool(fn, *args, **kwargs)
        finally:
            self.running -= 1
            slots.release()


INFERENCE_GATE = InferenceGate.from_env()

# OpenAPI documentation shared by every gated endpoint.
OVERLOADED_RESPONSE: dict[int | str, dict[str, Any]] = {
    503: {
        "description": (
            f"Overloaded: more than {QUEUE_ENV} inference calls are already queued in the "
            "worker. Retry after the number of seconds in the Retry-After header."
        ),
    },
}
