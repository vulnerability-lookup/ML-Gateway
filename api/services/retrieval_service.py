import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from cachetools import TTLCache, cached
from fastapi import HTTPException, status
from numpy.typing import NDArray

from api.models.attack_model import technique_names
from api.models.biencoder_model import (
    AttackBiEncoder,
    get_biencoder_instance,
    technique_texts,
)
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
# Ceiling on the number of distinct IDs the index endpoint may grow the
# index to. Each ID costs about 1.5 KB on disk and in the page cache, so
# this bounds what an ingesting client can make the gateway store.
INDEX_MAX_ITEMS_ENV = "ML_GATEWAY_INDEX_MAX_ITEMS"
_DEFAULT_INDEX_MAX_ITEMS = 5_000_000

_stores: dict[str, VectorStore] = {}
_stores_lock = Lock()


def index_directory() -> Path:
    return Path(os.environ.get(INDEX_DIR_ENV, _DEFAULT_INDEX_DIR))


def index_max_items() -> int:
    return int(os.environ.get(INDEX_MAX_ITEMS_ENV, _DEFAULT_INDEX_MAX_ITEMS))


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
    # Only IDs not yet indexed grow the index; re-sending an indexed ID
    # replaces its vector and is always allowed, so a full index can still
    # be kept up to date.
    new_ids = {item.id for item in request.items if not store.contains(item.id)}
    ceiling = index_max_items()
    if store.count + len(new_ids) > ceiling:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=(
                f"Index full: {store.count} IDs indexed, adding {len(new_ids)} would exceed "
                f"{INDEX_MAX_ITEMS_ENV}={ceiling}."
            ),
        )
    vectors = encoder.embed_vulnerabilities([item.text for item in request.items])
    store.upsert([item.id for item in request.items], vectors)
    return {
        "indexed": len(request.items),
        "count": store.count,
        "model": encoder.model_name,
        "model_revision": encoder.revision,
    }


def list_techniques(model_name: str) -> dict[str, Any]:
    """Every technique ``retrieve_by_technique`` can rank for, sorted by ID.

    The trained vocabulary comes from the texts shipped with the weights,
    the rest from the bundled STIX-derived text table; both are exactly the
    sources ``AttackBiEncoder.technique_vector`` resolves from, so this list
    and that lookup cannot disagree. Returns a dict shaped like
    :class:`api.schemas.TechniqueListResponse`.
    """
    try:
        encoder = get_biencoder_instance(model_name)
    except ValueError as e:
        return _error(model_name, str(e), techniques=[])
    trained = set(encoder.trained_technique_texts)
    names = technique_names()
    return {
        "techniques": [
            {"technique": technique_id, "name": names.get(technique_id), "in_vocabulary": technique_id in trained}
            for technique_id in sorted(trained | set(technique_texts()))
        ],
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


@cached(_embed_cache, lock=_embed_cache_lock, info=True)
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
