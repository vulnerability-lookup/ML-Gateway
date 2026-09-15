import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from cachetools import TTLCache, cached
from numpy.typing import NDArray

from api.models.attack_model import technique_names
from api.models.biencoder_model import AttackBiEncoder, get_biencoder_instance
from api.schemas import IndexRequest, RelatedRequest
from api.store.vector_store import VectorStore

"""
The retrieval service owns the bi-encoder index: it embeds vulnerability
descriptions into the per-model vector store and runs the two search
directions the classification head cannot answer — which vulnerabilities
for a technique, and which vulnerabilities behave like this one.
"""

# Root directory of the on-disk indexes; one sub-directory per model. Must
# be a persistent volume shared by every worker process.
INDEX_DIR_ENV = "ML_GATEWAY_INDEX_DIR"
_DEFAULT_INDEX_DIR = "index"

_stores: dict[str, VectorStore] = {}
_stores_lock = Lock()


def index_directory() -> Path:
    return Path(os.environ.get(INDEX_DIR_ENV, _DEFAULT_INDEX_DIR))


def _store_directory(model_name: str) -> Path:
    return index_directory() / re.sub(r"[^A-Za-z0-9._-]+", "__", model_name)


def get_store(encoder: AttackBiEncoder) -> VectorStore:
    """Open (once per process) the index matching the served model revision.

    Raises :class:`api.store.vector_store.IndexRevisionMismatch` (a
    ``ValueError``) when an index built with another revision sits in the
    directory; it is not cached, so the error is reported on every call
    until the operator rebuilds the index.
    """
    with _stores_lock:
        store = _stores.get(encoder.model_name)
        if store is None:
            store = VectorStore(
                _store_directory(encoder.model_name),
                encoder.dimension,
                encoder.model_name,
                encoder.revision,
            )
            _stores[encoder.model_name] = store
        return store


def _resolve(model_name: str) -> tuple[AttackBiEncoder, VectorStore]:
    encoder = get_biencoder_instance(model_name)
    return encoder, get_store(encoder)


def _error(model_name: str, message: str, **payload: Any) -> dict[str, Any]:
    # Surface the served revision when the model itself resolved, so a
    # revision-mismatch error still tells the caller what is being served.
    revision = None
    try:
        revision = get_biencoder_instance(model_name).revision
    except ValueError:
        pass
    return {**payload, "model": model_name, "model_revision": revision, "error": message}


def index_vulnerabilities(request: IndexRequest) -> dict[str, Any]:
    """Embed each item and upsert its vector under its ID.

    Returns a dict shaped like :class:`api.schemas.IndexResponse`.
    """
    try:
        encoder, store = _resolve(request.model)
    except ValueError as e:
        return _error(request.model, str(e), indexed=0, count=0)
    vectors = encoder.embed_vulnerabilities([item.text for item in request.items])
    store.upsert([item.id for item in request.items], vectors)
    return {
        "indexed": len(request.items),
        "count": store.count,
        "model": encoder.model_name,
        "model_revision": encoder.revision,
    }


def retrieve_by_technique(technique_id: str, top_k: int, model_name: str) -> dict[str, Any]:
    """Rank indexed vulnerabilities for one technique.

    Scores are ``sigmoid(logit_scale * cosine + logit_bias)``, the
    training-time CVE -> technique probability, so a vulnerability's score
    for a technique here equals what the CVE -> technique direction would
    give it. Returns a dict shaped like
    :class:`api.schemas.TechniqueRetrievalResponse`.
    """
    technique_id = technique_id.strip().upper()
    base = {"technique": technique_id, "name": technique_names().get(technique_id)}
    try:
        encoder, store = _resolve(model_name)
    except ValueError as e:
        return _error(model_name, str(e), **base, in_vocabulary=False, results=[])
    resolved = encoder.technique_vector(technique_id)
    if resolved is None:
        return _error(
            model_name,
            f"Unknown technique: {technique_id}",
            **base,
            in_vocabulary=False,
            results=[],
        )
    vector, in_vocabulary = resolved
    hits = store.search(vector, top_k)
    return {
        **base,
        "in_vocabulary": in_vocabulary,
        "results": [
            {"id": id_, "score": round(encoder.probability(cosine), 4)} for id_, cosine in hits
        ],
        "model": encoder.model_name,
        "model_revision": encoder.revision,
    }


# Query texts repeat (a vulnerability page reloaded, retries), so cache the
# embedding — never the search result, which changes as the index grows.
_embed_cache: TTLCache = TTLCache(maxsize=10_000, ttl=3600)
_embed_cache_lock = Lock()


@cached(_embed_cache, lock=_embed_cache_lock)
def _cached_embed(model_name: str, text: str) -> NDArray[np.float32]:
    return get_biencoder_instance(model_name).embed_vulnerabilities([text])[0]


def retrieve_related(request: RelatedRequest) -> dict[str, Any]:
    """Nearest indexed vulnerabilities by plain cosine.

    Returns a dict shaped like :class:`api.schemas.RelatedResponse`.
    """
    try:
        encoder, store = _resolve(request.model)
    except ValueError as e:
        return _error(request.model, str(e), results=[])
    exclude: str | None = None
    if request.id is not None:
        stored = store.get(request.id)
        if stored is None:
            return _error(request.model, f"Unknown id: {request.id}", results=[])
        vector = stored
        exclude = request.id
    else:
        assert request.text is not None  # enforced by the schema validator
        vector = _cached_embed(request.model, request.text)
    hits = store.search(vector, request.top_k, exclude=exclude)
    return {
        "results": [{"id": id_, "score": round(cosine, 4)} for id_, cosine in hits],
        "model": encoder.model_name,
        "model_revision": encoder.revision,
    }
