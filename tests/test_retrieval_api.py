from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from api import security
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
INDEX_TOKEN = "s3cret-index-token"
AUTH = {"Authorization": f"Bearer {INDEX_TOKEN}"}

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

    # What ``technique_vector`` resolves trained techniques from.
    trained_technique_texts = {"T1190": "Adversaries may attempt to exploit a weakness in an Internet-facing host."}

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_vulnerabilities(self, descriptions: list[str]) -> NDArray[np.float32]:
        self.embed_calls += 1
        if not descriptions:
            return np.zeros((0, DIM), dtype=np.float32)
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
    monkeypatch.setenv(security.INDEX_TOKEN_ENV, INDEX_TOKEN)
    monkeypatch.delenv(retrieval_service.INDEX_MAX_ITEMS_ENV, raising=False)
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
        headers=AUTH,
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


ONE_ITEM = {"items": [{"id": "CVE-1", "text": "sql injection"}], "model": MODEL}


def test_index_without_token_is_unauthorized(client: TestClient, encoder: StubBiEncoder) -> None:
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic " + INDEX_TOKEN},
                    {"Authorization": "Bearer"}):
        response = client.post("/index/attack-biencoder", json=ONE_ITEM, headers=headers)
        assert response.status_code == 401, headers
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json()["detail"] == "Missing or invalid bearer token."
    assert encoder.embed_calls == 0


def test_index_with_token_is_accepted(client: TestClient) -> None:
    response = client.post("/index/attack-biencoder", json=ONE_ITEM, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["indexed"] == 1
    # The scheme is case-insensitive, the token is not.
    response = client.post(
        "/index/attack-biencoder", json=ONE_ITEM, headers={"Authorization": f"bearer {INDEX_TOKEN}"}
    )
    assert response.status_code == 200
    response = client.post(
        "/index/attack-biencoder", json=ONE_ITEM, headers={"Authorization": f"Bearer {INDEX_TOKEN.upper()}"}
    )
    assert response.status_code == 401


def test_index_refused_when_no_token_configured(
    client: TestClient, encoder: StubBiEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    for value in (None, ""):
        if value is None:
            monkeypatch.delenv(security.INDEX_TOKEN_ENV)
        else:
            monkeypatch.setenv(security.INDEX_TOKEN_ENV, value)
        # Even a request carrying a token is refused: there is nothing to check it against.
        response = client.post("/index/attack-biencoder", json=ONE_ITEM, headers=AUTH)
        assert response.status_code == 503
        assert "ML_GATEWAY_INDEX_TOKEN is not set" in response.json()["detail"]
    assert encoder.embed_calls == 0


def test_read_endpoints_need_no_token(client: TestClient) -> None:
    assert client.get("/retrieve/attack-biencoder/technique/T1190", params={"model": MODEL}).status_code == 200
    response = client.post(
        "/retrieve/attack-biencoder/related", json={"text": "sql injection", "model": MODEL}
    )
    assert response.status_code == 200


def test_index_ceiling(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(retrieval_service.INDEX_MAX_ITEMS_ENV, "2")
    assert index(client, ("CVE-1", "sql injection"), ("CVE-2", "buffer overflow"))["count"] == 2
    response = client.post(
        "/index/attack-biencoder",
        json={"items": [{"id": "CVE-3", "text": "phishing"}], "model": MODEL},
        headers=AUTH,
    )
    assert response.status_code == 507
    assert "ML_GATEWAY_INDEX_MAX_ITEMS=2" in response.json()["detail"]
    # Updating an already indexed ID does not grow the index and stays allowed.
    assert index(client, ("CVE-1", "blind sql injection"))["count"] == 2
    # A batch that mixes known and new IDs is refused as a whole.
    response = client.post(
        "/index/attack-biencoder",
        json={"items": [{"id": "CVE-1", "text": "sql injection"}, {"id": "CVE-3", "text": "phishing"}], "model": MODEL},
        headers=AUTH,
    )
    assert response.status_code == 507
    # Nothing was written by the refused calls.
    assert index(client)["count"] == 2
    # Raising the ceiling lets the refused ID in.
    monkeypatch.setenv(retrieval_service.INDEX_MAX_ITEMS_ENV, "3")
    assert index(client, ("CVE-3", "phishing"))["count"] == 3


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
        headers=AUTH,
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


def test_technique_list_covers_trained_and_bundled_techniques(client: TestClient) -> None:
    response = client.get("/retrieve/attack-biencoder/techniques", params={"model": MODEL})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL
    assert body["model_revision"] == REVISION
    assert body["error"] is None
    by_id = {entry["technique"]: entry for entry in body["techniques"]}
    assert by_id["T1190"] == {"technique": "T1190", "name": "Exploit Public-Facing Application", "in_vocabulary": True}
    assert by_id["T1566"] == {"technique": "T1566", "name": "Phishing", "in_vocabulary": False}
    assert [entry["technique"] for entry in body["techniques"]] == sorted(by_id)
    assert sum(entry["in_vocabulary"] for entry in body["techniques"]) == 1
    # Every listed technique is one the technique search can resolve.
    assert len(body["techniques"]) > 600


def test_technique_list_unknown_model(client: TestClient) -> None:
    response = client.get("/retrieve/attack-biencoder/techniques", params={"model": "nope"})
    assert response.status_code == 200
    body = response.json()
    assert body["techniques"] == []
    assert body["error"].startswith("Unknown model")


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
