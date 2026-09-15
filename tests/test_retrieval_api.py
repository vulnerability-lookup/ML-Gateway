from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from api.routers.retrieval_router import router
from api.services import retrieval_service

"""
End-to-end tests for the bi-encoder retrieval endpoints.

The encoder is stubbed with a tiny deterministic "embedding" (each text
maps to a fixed unit vector), while the real on-disk vector store runs
against a temporary directory — so the tests cover the router -> service
-> store path without downloading or running the model.
"""

MODEL = "stub/biencoder"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DIM = 4

X, Y, Z, W = np.eye(DIM, dtype=np.float32)
TEXT_VECTORS: dict[str, NDArray[np.float32]] = {
    "sql injection": X,
    "blind sql injection": (X + 0.2 * Y) / np.linalg.norm(X + 0.2 * Y),
    "buffer overflow": Y,
    "phishing": Z,
}
TECHNIQUE_VECTORS: dict[str, tuple[NDArray[np.float32], bool]] = {
    "T1190": (X, True),  # trained
    "T1566": (Z, False),  # scored from the STIX text, out-of-vocabulary
}


class StubBiEncoder:
    model_name = MODEL
    revision = REVISION
    dimension = DIM
    logit_scale = 10.0
    logit_bias = -5.0

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_vulnerabilities(self, descriptions: list[str]) -> NDArray[np.float32]:
        self.embed_calls += 1
        return np.stack([TEXT_VECTORS[text] for text in descriptions])

    def technique_vector(self, technique_id: str) -> tuple[NDArray[np.float32], bool] | None:
        return TECHNIQUE_VECTORS.get(technique_id)

    def probability(self, cosine: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.logit_scale * cosine + self.logit_bias))))


@pytest.fixture()
def encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> StubBiEncoder:
    stub = StubBiEncoder()

    def get_instance(model_name: str) -> StubBiEncoder:
        if model_name != MODEL:
            raise ValueError(f"Unknown model: {model_name}")
        return stub

    monkeypatch.setattr(retrieval_service, "get_biencoder_instance", get_instance)
    monkeypatch.setenv(retrieval_service.INDEX_DIR_ENV, str(tmp_path / "index"))
    retrieval_service._stores.clear()
    retrieval_service._embed_cache.clear()
    return stub


@pytest.fixture()
def client(encoder: StubBiEncoder) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def index(client: TestClient, *items: tuple[str, str], model: str = MODEL) -> dict[str, Any]:
    response = client.post(
        "/index/attack-biencoder",
        json={"items": [{"id": id_, "text": text} for id_, text in items], "model": model},
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_index_reports_count_and_provenance(client: TestClient) -> None:
    body = index(client, ("CVE-1", "sql injection"), ("CVE-2", "buffer overflow"))
    assert body == {
        "indexed": 2,
        "count": 2,
        "model": MODEL,
        "model_revision": REVISION,
        "error": None,
    }
    body = index(client, ("CVE-1", "blind sql injection"))
    assert body["indexed"] == 1
    assert body["count"] == 2


def test_index_unknown_model(client: TestClient, encoder: StubBiEncoder) -> None:
    body = index(client, ("CVE-1", "sql injection"), model="nope/unknown")
    assert body["indexed"] == 0
    assert body["error"] == "Unknown model: nope/unknown"
    assert encoder.embed_calls == 0


def test_index_rejects_ids_with_whitespace(client: TestClient) -> None:
    response = client.post(
        "/index/attack-biencoder",
        json={"items": [{"id": "CVE 1", "text": "sql injection"}]},
    )
    assert response.status_code == 422


def test_index_is_persisted_in_the_configured_directory(
    client: TestClient, tmp_path: Path
) -> None:
    index(client, ("CVE-1", "sql injection"))
    store_dir = tmp_path / "index" / "stub__biencoder"
    assert (store_dir / "ids.txt").read_text() == "CVE-1\n"
    assert (store_dir / "vectors.f16").stat().st_size == DIM * 2
    assert '"model_revision": "%s"' % REVISION in (store_dir / "meta.json").read_text()


def test_technique_retrieval_scores_with_affine_sigmoid(client: TestClient) -> None:
    index(
        client,
        ("CVE-1", "sql injection"),
        ("CVE-2", "blind sql injection"),
        ("CVE-3", "buffer overflow"),
    )
    response = client.get("/retrieve/attack-biencoder/technique/t1190", params={"top_k": 2, "model": MODEL})
    assert response.status_code == 200
    body = response.json()
    assert body["technique"] == "T1190"
    assert body["name"] == "Exploit Public-Facing Application"
    assert body["in_vocabulary"] is True
    assert [hit["id"] for hit in body["results"]] == ["CVE-1", "CVE-2"]
    # cosine 1.0 -> sigmoid(10 * 1 - 5) = 0.9933
    assert body["results"][0]["score"] == pytest.approx(0.9933, abs=1e-3)
    assert body["model"] == MODEL
    assert body["model_revision"] == REVISION
    assert body["error"] is None


def test_technique_retrieval_flags_out_of_vocabulary(client: TestClient) -> None:
    index(client, ("CVE-1", "phishing"), ("CVE-2", "sql injection"))
    body = client.get("/retrieve/attack-biencoder/technique/T1566", params={"model": MODEL}).json()
    assert body["in_vocabulary"] is False
    assert body["name"] == "Phishing"
    assert body["results"][0]["id"] == "CVE-1"
    assert body["error"] is None


def test_technique_retrieval_unknown_technique(client: TestClient) -> None:
    body = client.get("/retrieve/attack-biencoder/technique/T9999", params={"model": MODEL}).json()
    assert body["results"] == []
    assert body["in_vocabulary"] is False
    assert body["error"] == "Unknown technique: T9999"
    assert body["model_revision"] == REVISION


def test_technique_retrieval_rejects_bad_top_k(client: TestClient) -> None:
    for top_k in (0, 1001):
        response = client.get(
            "/retrieve/attack-biencoder/technique/T1190", params={"top_k": top_k, "model": MODEL}
        )
        assert response.status_code == 422


def test_related_by_id_excludes_itself(client: TestClient) -> None:
    index(
        client,
        ("CVE-1", "sql injection"),
        ("CVE-2", "blind sql injection"),
        ("CVE-3", "buffer overflow"),
    )
    response = client.post("/retrieve/attack-biencoder/related", json={"id": "CVE-1", "model": MODEL})
    assert response.status_code == 200
    body = response.json()
    assert [hit["id"] for hit in body["results"]] == ["CVE-2", "CVE-3"]
    assert body["results"][0]["score"] == pytest.approx(0.9806, abs=1e-3)
    assert body["results"][1]["score"] == pytest.approx(0.0, abs=1e-3)
    assert body["error"] is None


def test_related_by_text_embeds_and_caches(
    client: TestClient, encoder: StubBiEncoder
) -> None:
    index(client, ("CVE-1", "sql injection"), ("CVE-2", "buffer overflow"))
    calls_after_index = encoder.embed_calls
    for _ in range(2):
        body = client.post(
            "/retrieve/attack-biencoder/related",
            json={"text": "blind sql injection", "top_k": 1, "model": MODEL},
        ).json()
        assert [hit["id"] for hit in body["results"]] == ["CVE-1"]
    assert encoder.embed_calls == calls_after_index + 1


def test_related_unknown_id(client: TestClient) -> None:
    body = client.post("/retrieve/attack-biencoder/related", json={"id": "CVE-404", "model": MODEL}).json()
    assert body["results"] == []
    assert body["error"] == "Unknown id: CVE-404"


def test_related_requires_exactly_one_query(client: TestClient) -> None:
    for payload in ({}, {"id": "CVE-1", "text": "sql injection"}, {"text": "  "}):
        response = client.post(
            "/retrieve/attack-biencoder/related", json={**payload, "model": MODEL}
        )
        assert response.status_code == 422


def test_revision_mismatch_is_reported(
    client: TestClient, encoder: StubBiEncoder, tmp_path: Path
) -> None:
    index(client, ("CVE-1", "sql injection"))
    # Simulate a newer model being served over an index built earlier.
    encoder.revision = "f" * 40
    retrieval_service._stores.clear()
    body = client.get("/retrieve/attack-biencoder/technique/T1190", params={"model": MODEL}).json()
    assert body["results"] == []
    assert "rebuild the index" in body["error"]
    assert body["model_revision"] == "f" * 40
    body = index(client, ("CVE-2", "buffer overflow"))
    assert body["indexed"] == 0
    assert "rebuild the index" in body["error"]


def test_technique_text_table() -> None:
    from api.models.biencoder_model import technique_texts

    texts = technique_texts()
    assert texts["T1190"].startswith("Exploit Public-Facing Application. ")
    assert "(Citation:" not in texts["T1190"]
    assert all("." in text for text in texts.values())
