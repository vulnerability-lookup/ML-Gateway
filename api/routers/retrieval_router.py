from typing import Any

from fastapi import APIRouter, Depends, Path, Query

from api.availability import require_enabled

from api.schemas import (
    DEFAULT_BIENCODER_MODEL,
    IndexRequest,
    IndexResponse,
    RelatedRequest,
    RelatedResponse,
    TechniqueListResponse,
    TechniqueRetrievalResponse,
)
from api.security import require_index_token
from api.services.retrieval_service import (
    index_vulnerabilities,
    list_techniques,
    retrieve_by_technique,
    retrieve_related,
)
from api.throttle import INFERENCE_GATE, OVERLOADED_RESPONSE

"""
Retrieval endpoints backed by the ATT&CK bi-encoder and its vector index.
Like the classification endpoints, the ones that embed or search go
through the inference gate (a thread per call, 503 once the per-worker
queue is full). The technique list runs no model and answers directly.
"""

# Every route can be taken out of service with ML_GATEWAY_DISABLED_ENDPOINTS.
router = APIRouter(dependencies=[Depends(require_enabled)])


@router.post(
    "/index/attack-biencoder",
    response_model=IndexResponse,
    dependencies=[Depends(require_index_token)],
    responses={
        401: {"description": "Missing or invalid bearer token."},
        503: {
            "description": (
                "Indexing disabled (no ML_GATEWAY_INDEX_TOKEN configured), or overloaded: "
                "the response then carries a Retry-After header."
            )
        },
        507: {"description": "Index full: ML_GATEWAY_INDEX_MAX_ITEMS would be exceeded."},
    },
)
async def index_endpoint(request: IndexRequest) -> dict[str, Any]:
    """Embed vulnerability descriptions and upsert them into the index.

    Requires ``Authorization: Bearer <ML_GATEWAY_INDEX_TOKEN>``. Request
    body: ``{"items": [{"id": "CVE-…", "text": "…"}, …], "model":
    "<optional-model-id>"}``. Call once per record at ingest and again
    whenever a description changes; re-sending an ID replaces its vector.
    Vectors are only comparable within one model revision, which the
    response reports as ``model_revision``. The call is refused with 507
    when adding the new IDs would push the index past
    ``ML_GATEWAY_INDEX_MAX_ITEMS``.
    """
    return await INFERENCE_GATE.run(index_vulnerabilities, request)


@router.get("/retrieve/attack-biencoder/techniques", response_model=TechniqueListResponse)
async def technique_list_endpoint(
    model: str = Query(default=DEFAULT_BIENCODER_MODEL),
) -> dict[str, Any]:
    """List every technique the technique-retrieval endpoint can rank for.

    The bi-encoder's trained vocabulary is reported with
    ``in_vocabulary: true``; the other enterprise techniques are scored
    from their official ATT&CK text and rank noticeably worse. Sorted by
    technique ID. Lets a client build a technique index without shipping
    its own copy of the ATT&CK tables. Runs no model, so it answers even
    while the inference queue is full.
    """
    return list_techniques(model)


@router.get(
    "/retrieve/attack-biencoder/technique/{technique_id}",
    response_model=TechniqueRetrievalResponse,
    responses=OVERLOADED_RESPONSE,
)
async def technique_retrieval_endpoint(
    technique_id: str = Path(description="MITRE ATT&CK technique ID, e.g. 'T1190'."),
    top_k: int = Query(default=10, ge=1, le=1000),
    model: str = Query(default=DEFAULT_BIENCODER_MODEL),
) -> dict[str, Any]:
    """Rank indexed vulnerabilities for one ATT&CK technique.

    Scores are the training-time probability ``sigmoid(logit_scale · cosine
    + logit_bias)``. Techniques the model was not trained on are scored
    from their official ATT&CK text and reported with
    ``in_vocabulary: false``; they rank noticeably worse. This is a
    similarity search, not a classification.
    """
    return await INFERENCE_GATE.run(retrieve_by_technique, technique_id, top_k, model)


@router.post("/retrieve/attack-biencoder/related", response_model=RelatedResponse, responses=OVERLOADED_RESPONSE)
async def related_endpoint(request: RelatedRequest) -> dict[str, Any]:
    """Find the indexed vulnerabilities nearest to one vulnerability.

    Request body: ``{"id": "CVE-…"}`` for an indexed vulnerability (itself
    excluded from the results) or ``{"text": "…"}`` for a free description,
    plus optional ``top_k`` and ``model``. Ranked by plain cosine; a search
    aid with no measured accuracy, not a classification.
    """
    return await INFERENCE_GATE.run(retrieve_related, request)
