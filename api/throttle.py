import asyncio
import math
import os
import time
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
MAX_WAIT_ENV = "ML_GATEWAY_INFERENCE_MAX_WAIT"
# One inference at a time per worker: with ``OMP_NUM_THREADS`` intra-op
# threads per call and one worker per ``OMP_NUM_THREADS`` cores, that is
# exactly what keeps every core busy without oversubscribing them.
_DEFAULT_CONCURRENCY = 1
# The real limit is the wait budget below; the queue length is a hard cap
# for the moments before the worker knows how long a call takes.
_DEFAULT_QUEUE = 256
# Refuse a call once the work already queued would take longer than this
# to drain, judged from the worker's own recent inference times. Set it
# under the caller's timeout (Vulnerability-Lookup's proxy waits 10 s), so
# a call that is admitted is a call that gets answered.
_DEFAULT_MAX_WAIT_SECONDS = 8.0
# Weight of the latest call in the running average of the service time.
_SERVICE_TIME_ALPHA = 0.2

P = ParamSpec("P")
T = TypeVar("T")


class InferenceGate:
    """Per-process admission control for CPU-bound inference calls.

    Attributes:
        concurrency: Calls allowed to run at the same time.
        queue: Hard cap on the calls allowed to wait for a running slot.
        max_wait: Seconds of queued work beyond which a call is refused.
        service_time: Running average of one call's duration in seconds,
            ``None`` until the first call completes.
        running: Calls currently executing.
        waiting: Calls currently waiting for a slot.
        served: Calls admitted and completed since the process started.
        refused: Calls refused with 503 since the process started.
    """

    def __init__(self, concurrency: int, queue: int, max_wait: float = _DEFAULT_MAX_WAIT_SECONDS) -> None:
        self.configure(concurrency, queue, max_wait)

    @classmethod
    def from_env(cls) -> "InferenceGate":
        return cls(
            int(os.environ.get(CONCURRENCY_ENV, _DEFAULT_CONCURRENCY)),
            int(os.environ.get(QUEUE_ENV, _DEFAULT_QUEUE)),
            float(os.environ.get(MAX_WAIT_ENV, _DEFAULT_MAX_WAIT_SECONDS)),
        )

    def configure(self, concurrency: int, queue: int, max_wait: float = _DEFAULT_MAX_WAIT_SECONDS) -> None:
        """Set the limits; only meaningful while no call is in flight."""
        if concurrency < 1 or queue < 0 or max_wait <= 0:
            raise ValueError("concurrency must be at least 1, queue at least 0 and max_wait positive")
        self.concurrency = concurrency
        self.queue = queue
        self.max_wait = max_wait
        self.service_time: float | None = None
        self.running = 0
        self.waiting = 0
        self.served = 0
        self.refused = 0
        # Created on first use, inside the worker's event loop: the module
        # is imported in the gunicorn master before the fork.
        self._semaphore: asyncio.Semaphore | None = None

    def _slots(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    def expected_wait(self) -> float:
        """Seconds a new call would wait for a slot, from the queued work."""
        if self.service_time is None:
            return 0.0
        return self.waiting * self.service_time / self.concurrency

    def retry_after(self) -> int:
        """Whole seconds until a slot is likely to free up (at least 1)."""
        return max(1, math.ceil(self.service_time or 0.0))

    async def run(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run ``fn`` in the threadpool once a slot is free, or refuse with 503."""
        slots = self._slots()
        if slots.locked() and (self.waiting >= self.queue or self.expected_wait() >= self.max_wait):
            self.refused += 1
            retry_after = self.retry_after()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Overloaded: {self.running} inference calls running and {self.waiting} queued in this "
                    f"worker, about {self.expected_wait():.1f} s of work ahead (limit {self.max_wait:g} s). "
                    f"Retry after {retry_after} second{'s' if retry_after > 1 else ''}."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        self.waiting += 1
        try:
            # A client that hangs up while queued cancels this wait; the
            # ``finally`` keeps the count honest either way.
            await slots.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(fn, *args, **kwargs)
        finally:
            self.running -= 1
            slots.release()
        self.observe(time.perf_counter() - started)
        self.served += 1
        return result

    def observe(self, seconds: float) -> None:
        """Fold one completed call's duration into the service-time average."""
        if self.service_time is None:
            self.service_time = seconds
        else:
            self.service_time += _SERVICE_TIME_ALPHA * (seconds - self.service_time)


INFERENCE_GATE = InferenceGate.from_env()

# OpenAPI documentation shared by every gated endpoint.
OVERLOADED_RESPONSE: dict[int | str, dict[str, Any]] = {
    503: {
        "description": (
            f"Overloaded: the work already queued in the worker would take longer than {MAX_WAIT_ENV} "
            f"seconds to drain (or {QUEUE_ENV} calls are waiting). Retry after the number of seconds in "
            "the Retry-After header."
        ),
    },
}
