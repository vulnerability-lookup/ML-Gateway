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
    limits = (INFERENCE_GATE.concurrency, INFERENCE_GATE.queue)
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
    assert refused.headers == {"Retry-After": str(throttle.RETRY_AFTER_SECONDS)}
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
    gate = InferenceGate.from_env()
    assert (gate.concurrency, gate.queue) == (3, 7)
    with pytest.raises(ValueError):
        InferenceGate(concurrency=0, queue=1)


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
        assert refused.headers["Retry-After"] == str(throttle.RETRY_AFTER_SECONDS)
        assert refused.json()["detail"].startswith("Overloaded: 1 inference calls running and 0 queued")
        # The root never touches the gate.
        assert client.get("/").status_code == 200

        classifier.blocker.release.set()
        occupant.join(5)
        assert outcome == [200]
        assert client.post("/classify/severity", json={"description": "third"}).json()["severity"] == "High"
    assert classifier.blocker.calls == 2
