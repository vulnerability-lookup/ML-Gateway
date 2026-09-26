import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.classification_router import router as classification_router
from api.routers.stats_router import router as stats_router
from api.services import classification_service, retrieval_service
from api.throttle import INFERENCE_GATE

"""
Tests for ``GET /stats``: cache hit and miss counters and the inference
gate's counters, per worker.
"""


class StubSeverityClassifier:
    model_name = "stub/severity-model"
    revision = "0123456789abcdef0123456789abcdef01234567"
    quantized = False

    def predict(self, description: str) -> dict[str, Any]:
        return {"severity": "Medium", "confidence": 0.7}


@pytest.fixture(autouse=True)
def fresh_counters(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(classification_service, "get_model_instance", lambda model_name: StubSeverityClassifier())
    for cached in (
        classification_service._cached_predict,
        classification_service._cached_predict_attack,
        retrieval_service._cached_embed,
    ):
        cached.cache_clear()
    limits = (INFERENCE_GATE.concurrency, INFERENCE_GATE.queue, INFERENCE_GATE.max_wait)
    INFERENCE_GATE.configure(*limits)
    yield
    INFERENCE_GATE.configure(*limits)


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(classification_router)
    app.include_router(stats_router)
    return TestClient(app)


def test_stats_count_cache_hits_and_served_calls(client: TestClient) -> None:
    before = client.get("/stats").json()
    assert before["caches"]["severity"] == {"hits": 0, "misses": 0, "size": 0, "maxsize": 10_000}
    assert before["inference"]["served"] == 0

    for _ in range(3):
        assert client.post("/classify/severity", json={"description": "same text"}).status_code == 200
    client.post("/classify/severity", json={"description": "other text"})

    after = client.get("/stats").json()
    assert after["pid"] == before["pid"]
    assert after["caches"]["severity"] == {"hits": 2, "misses": 2, "size": 2, "maxsize": 10_000}
    assert set(after["caches"]) == {"severity", "attack_techniques", "embeddings"}
    inference = after["inference"]
    assert inference.pop("service_time_ms") >= 0
    assert inference == {
        "concurrency": INFERENCE_GATE.concurrency,
        "queue": INFERENCE_GATE.queue,
        "max_wait_seconds": INFERENCE_GATE.max_wait,
        "expected_wait_seconds": 0.0,
        "running": 0,
        "waiting": 0,
        "served": 4,
        "refused": 0,
    }
    assert before["inference"]["service_time_ms"] is None


def test_stats_count_refusals_and_never_queue(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(INFERENCE_GATE, "_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(INFERENCE_GATE, "waiting", INFERENCE_GATE.queue)
    assert client.post("/classify/severity", json={"description": "x"}).status_code == 503
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["inference"]["refused"] == 1
