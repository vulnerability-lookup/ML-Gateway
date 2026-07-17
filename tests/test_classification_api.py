from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.classification_router import router
from api.schemas import AttackTechniquesRequest
from api.services import classification_service

"""
End-to-end tests for the classification endpoints.

The model layer is stubbed out: ``get_model_instance`` /
``get_attack_model_instance`` are monkeypatched in the service module, so
the tests exercise the router -> service path (validation, caching,
response shaping) without downloading or running any Hugging Face model.
``api.main`` is deliberately not imported — it preloads the real models at
import time.
"""

SEVERITY_MODEL = "stub/severity-model"
ATTACK_MODEL = "stub/attack-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"

RANKING: list[dict[str, Any]] = [
    {
        "technique": "T1190",
        "name": "Exploit Public-Facing Application",
        "score": 0.7147,
        "predicted": True,
    },
    {
        "technique": "T1059",
        "name": "Command and Scripting Interpreter",
        "score": 0.7063,
        "predicted": True,
    },
    {"technique": "T1505", "name": "Server Software Component", "score": 0.4834, "predicted": False},
]


class StubSeverityClassifier:
    model_name = SEVERITY_MODEL
    revision = REVISION

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, description: str) -> dict[str, Any]:
        self.calls += 1
        return {"severity": "Critical", "confidence": 0.9876}


class StubAttackClassifier:
    model_name = ATTACK_MODEL
    revision = REVISION

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, description: str) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(entry) for entry in RANKING]


@pytest.fixture(autouse=True)
def clear_prediction_caches() -> None:
    # The TTL caches are module-level; entries produced with one test's stub
    # must not leak into the next test.
    classification_service._predict_cache.clear()
    classification_service._attack_predict_cache.clear()


@pytest.fixture()
def severity_stub(monkeypatch: pytest.MonkeyPatch) -> StubSeverityClassifier:
    stub = StubSeverityClassifier()

    def get_instance(model_name: str) -> StubSeverityClassifier:
        if model_name != SEVERITY_MODEL:
            raise ValueError(f"Unknown model: {model_name}")
        return stub

    monkeypatch.setattr(classification_service, "get_model_instance", get_instance)
    return stub


@pytest.fixture()
def attack_stub(monkeypatch: pytest.MonkeyPatch) -> StubAttackClassifier:
    stub = StubAttackClassifier()

    def get_instance(model_name: str) -> StubAttackClassifier:
        if model_name != ATTACK_MODEL:
            raise ValueError(f"Unknown model: {model_name}")
        return stub

    monkeypatch.setattr(
        classification_service, "get_attack_model_instance", get_instance
    )
    return stub


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_severity_classification(
    client: TestClient, severity_stub: StubSeverityClassifier
) -> None:
    response = client.post(
        "/classify/severity",
        json={"description": "Remote code execution.", "model": SEVERITY_MODEL},
    )
    assert response.status_code == 200
    assert response.json() == {
        "severity": "Critical",
        "confidence": 0.9876,
        "model": SEVERITY_MODEL,
        "model_revision": REVISION,
        "error": None,
    }


def test_severity_unknown_model(
    client: TestClient, severity_stub: StubSeverityClassifier
) -> None:
    response = client.post(
        "/classify/severity",
        json={"description": "Remote code execution.", "model": "nope/unknown"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["severity"] is None
    assert body["error"] == "Unknown model: nope/unknown"
    assert severity_stub.calls == 0


def test_attack_techniques_ranking(
    client: TestClient, attack_stub: StubAttackClassifier
) -> None:
    response = client.post(
        "/classify/attack-techniques",
        json={
            "description": "Remote code execution.",
            "model": ATTACK_MODEL,
            "top_k": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["techniques"] == RANKING[:2]
    assert body["model"] == ATTACK_MODEL
    assert body["model_revision"] == REVISION
    assert body["error"] is None


def test_attack_techniques_unknown_model(
    client: TestClient, attack_stub: StubAttackClassifier
) -> None:
    response = client.post(
        "/classify/attack-techniques",
        json={"description": "Remote code execution.", "model": "nope/unknown"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["techniques"] == []
    assert body["error"] == "Unknown model: nope/unknown"
    assert attack_stub.calls == 0


def test_attack_techniques_top_k_rejects_zero(
    client: TestClient, attack_stub: StubAttackClassifier
) -> None:
    response = client.post(
        "/classify/attack-techniques",
        json={
            "description": "Remote code execution.",
            "model": ATTACK_MODEL,
            "top_k": 0,
        },
    )
    assert response.status_code == 422


def test_varying_top_k_does_not_rerun_inference(
    client: TestClient, attack_stub: StubAttackClassifier
) -> None:
    for top_k in (1, 3, 2):
        response = client.post(
            "/classify/attack-techniques",
            json={
                "description": "Remote code execution.",
                "model": ATTACK_MODEL,
                "top_k": top_k,
            },
        )
        assert len(response.json()["techniques"]) == top_k
    assert attack_stub.calls == 1


def test_returned_ranking_is_copied_from_cache(
    attack_stub: StubAttackClassifier,
) -> None:
    request = AttackTechniquesRequest(
        description="Remote code execution.", model=ATTACK_MODEL, top_k=1
    )
    first = classification_service.classify_attack_techniques(request)
    first["techniques"][0]["score"] = 999.0

    second = classification_service.classify_attack_techniques(request)
    assert second["techniques"][0]["score"] == RANKING[0]["score"]
    assert attack_stub.calls == 1


def test_technique_names_table() -> None:
    from api.models.attack_model import technique_names

    names = technique_names()
    assert names["T1190"] == "Exploit Public-Facing Application"
    assert all(
        key.startswith("T") and key.removeprefix("T").replace(".", "").isdigit()
        for key in names
    )
