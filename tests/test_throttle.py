import asyncio
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api import throttle
from api.routers.classification_router import router
from api.services import classification_service
from api.throttle import INFERENCE_GATE, InferenceGate

"""
Tests for the inference gate: bounded concurrency and queue per worker,
503 with Retry-After beyond that, and the wiring of the routers.
"""


@pytest.fixture(autouse=True)
def restore_gate() -> Iterator[None]:
    limits = (INFERENCE_GATE.concurrency, INFERENCE_GATE.queue, INFERENCE_GATE.max_wait)
    yield
    INFERENCE_GATE.configure(*limits)


class Blocker:
    """A stand-in for inference that blocks until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0

    def __call__(self, value: str) -> str:
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        return value


def test_gate_refuses_when_queue_is_full() -> None:
    gate = InferenceGate(concurrency=1, queue=1)
    blocker = Blocker()

    async def scenario() -> tuple[str, str, HTTPException]:
        first = asyncio.create_task(gate.run(blocker, "first"))
        await asyncio.to_thread(blocker.started.wait, 5)
        second = asyncio.create_task(gate.run(blocker, "second"))
        await asyncio.sleep(0.05)
        assert (gate.running, gate.waiting) == (1, 1)
        with pytest.raises(HTTPException) as refused:
            await gate.run(blocker, "third")
        blocker.release.set()
        return await first, await second, refused.value

    first, second, refused = asyncio.run(scenario())
    assert (first, second) == ("first", "second")
    assert refused.status_code == 503
    assert refused.headers == {"Retry-After": "1"}
    assert "1 inference calls running and 1 queued" in refused.detail
    assert blocker.calls == 2
    assert (gate.running, gate.waiting) == (0, 0)


def test_gate_forgets_a_caller_that_gives_up_while_queued() -> None:
    gate = InferenceGate(concurrency=1, queue=1)
    blocker = Blocker()

    async def scenario() -> str:
        first = asyncio.create_task(gate.run(blocker, "first"))
        await asyncio.to_thread(blocker.started.wait, 5)
        second = asyncio.create_task(gate.run(blocker, "second"))
        await asyncio.sleep(0.05)
        assert gate.waiting == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert gate.waiting == 0
        blocker.release.set()
        return await first

    assert asyncio.run(scenario()) == "first"
    assert blocker.calls == 1


def test_gate_limits_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(throttle.CONCURRENCY_ENV, "3")
    monkeypatch.setenv(throttle.QUEUE_ENV, "7")
    monkeypatch.setenv(throttle.MAX_WAIT_ENV, "2.5")
    gate = InferenceGate.from_env()
    assert (gate.concurrency, gate.queue, gate.max_wait) == (3, 7, 2.5)
    with pytest.raises(ValueError):
        InferenceGate(concurrency=0, queue=1)
    with pytest.raises(ValueError):
        InferenceGate(concurrency=1, queue=1, max_wait=0)


def test_gate_refuses_on_the_wait_budget_not_the_count() -> None:
    gate = InferenceGate(concurrency=1, queue=1000, max_wait=1.0)
    blocker = Blocker()

    async def scenario() -> tuple[str, HTTPException]:
        first = asyncio.create_task(gate.run(blocker, "first"))
        await asyncio.to_thread(blocker.started.wait, 5)
        # Pretend three earlier calls are queued and each call takes 0.4 s:
        # 1.2 s of work ahead exceeds the 1 s budget.
        gate.service_time = 0.4
        gate.waiting = 3
        with pytest.raises(HTTPException) as refused:
            await gate.run(blocker, "late")
        gate.waiting = 0
        # Two queued calls are 0.8 s of work: admitted.
        gate.waiting = 2
        assert gate.expected_wait() == pytest.approx(0.8)
        gate.waiting = 0
        blocker.release.set()
        return await first, refused.value

    first, refused = asyncio.run(scenario())
    assert first == "first"
    assert "about 1.2 s of work ahead (limit 1 s)" in refused.detail
    assert refused.headers == {"Retry-After": "1"}
    assert gate.service_time == pytest.approx(0.4, abs=0.2)  # the first call's duration was folded in
    assert gate.refused == 1


def test_service_time_is_a_running_average() -> None:
    gate = InferenceGate(concurrency=2, queue=10, max_wait=5.0)
    assert gate.expected_wait() == 0.0
    gate.observe(1.0)
    assert gate.service_time == 1.0
    gate.observe(0.0)
    assert gate.service_time == pytest.approx(0.8)
    gate.waiting = 4
    assert gate.expected_wait() == pytest.approx(1.6)  # 4 calls, 0.8 s each, 2 at a time
    gate.service_time = 2.4
    assert gate.retry_after() == 3


class StubSeverityClassifier:
    model_name = "stub/severity-model"
    revision = "0123456789abcdef0123456789abcdef01234567"
    quantized = False

    def __init__(self) -> None:
        self.blocker = Blocker()

    def predict(self, description: str) -> dict[str, Any]:
        self.blocker(description)
        return {"severity": "High", "confidence": 0.5}


@pytest.fixture()
def classifier(monkeypatch: pytest.MonkeyPatch) -> StubSeverityClassifier:
    stub = StubSeverityClassifier()
    monkeypatch.setattr(classification_service, "get_model_instance", lambda model_name: stub)
    classification_service._predict_cache.clear()
    return stub


def test_endpoint_sheds_load_and_recovers(classifier: StubSeverityClassifier) -> None:
    INFERENCE_GATE.configure(concurrency=1, queue=0)
    app = FastAPI()
    app.include_router(router)
    outcome: list[int] = []

    with TestClient(app) as client:

        def occupy_the_slot() -> None:
            outcome.append(client.post("/classify/severity", json={"description": "first"}).status_code)

        occupant = threading.Thread(target=occupy_the_slot)
        occupant.start()
        assert classifier.blocker.started.wait(5)

        refused = client.post("/classify/severity", json={"description": "second"})
        assert refused.status_code == 503
        assert refused.headers["Retry-After"] == "1"
        assert refused.json()["detail"].startswith("Overloaded: 1 inference calls running and 0 queued")
        # The root never touches the gate.
        assert client.get("/").status_code == 200

        classifier.blocker.release.set()
        occupant.join(5)
        assert outcome == [200]
        assert client.post("/classify/severity", json={"description": "third"}).json()["severity"] == "High"
    assert classifier.blocker.calls == 2
