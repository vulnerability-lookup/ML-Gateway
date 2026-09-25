from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.availability import DISABLED_ENDPOINTS_ENV, disabled_endpoints
from api.routers.classification_router import router
from api.services import classification_service

"""
Tests for the operator switch that takes endpoints out of service.
"""


class StubSeverityClassifier:
    model_name = "stub/severity-model"
    revision = "0123456789abcdef0123456789abcdef01234567"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, description: str) -> dict[str, Any]:
        self.calls += 1
        return {"severity": "Low", "confidence": 0.9}


@pytest.fixture()
def classifier(monkeypatch: pytest.MonkeyPatch) -> StubSeverityClassifier:
    stub = StubSeverityClassifier()
    monkeypatch.setattr(classification_service, "get_model_instance", lambda model_name: stub)
    monkeypatch.setattr(classification_service, "get_attack_model_instance", lambda model_name: stub)
    classification_service._predict_cache.clear()
    return stub


@pytest.fixture()
def client(classifier: StubSeverityClassifier) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_listed_endpoint_is_refused_without_touching_the_model(
    client: TestClient, classifier: StubSeverityClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DISABLED_ENDPOINTS_ENV, " /classify/attack-techniques , ")
    refused = client.post("/classify/attack-techniques", json={"description": "x"})
    assert refused.status_code == 503
    assert "Retry-After" not in refused.headers
    assert refused.json()["detail"] == (
        f"Endpoint /classify/attack-techniques is disabled on this gateway ({DISABLED_ENDPOINTS_ENV})."
    )
    assert classifier.calls == 0
    # Unlisted endpoints keep working.
    assert client.post("/classify/severity", json={"description": "x"}).json()["severity"] == "Low"
    assert client.get("/").status_code == 200


def test_nothing_is_disabled_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DISABLED_ENDPOINTS_ENV, raising=False)
    assert disabled_endpoints() == set()
    assert client.post("/classify/severity", json={"description": "x"}).status_code == 200


def test_placeholders_and_trailing_slashes_do_not_matter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        DISABLED_ENDPOINTS_ENV,
        "/retrieve/attack-biencoder/technique/{id},/retrieve/attack-biencoder/related/",
    )
    assert disabled_endpoints() == {
        "/retrieve/attack-biencoder/technique/{}",
        "/retrieve/attack-biencoder/related",
    }
