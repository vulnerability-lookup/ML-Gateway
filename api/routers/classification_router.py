from typing import Any

from fastapi import APIRouter, Depends

from api.availability import require_enabled

from api.schemas import (
    AttackTechniquesRequest,
    AttackTechniquesResponse,
    SeverityRequest,
    SeverityResponse,
)
from api.services.classification_service import (
    classify_attack_techniques,
    classify_severity,
)
from api.throttle import INFERENCE_GATE, OVERLOADED_RESPONSE

"""
This module sets up the API route(s) using FastAPI's APIRouter.
We define the request schema with Pydantic and the endpoint function:
"""

# Every route can be taken out of service with ML_GATEWAY_DISABLED_ENDPOINTS.
router = APIRouter(dependencies=[Depends(require_enabled)])


@router.get("/")
async def root() -> str:
    return "OK"


# The classification endpoints hand their synchronous, CPU-bound torch
# inference to the inference gate, which runs it in a thread (so the event
# loop stays free) and refuses the call with 503 once the per-worker queue
# is full (see ``api.throttle``).


@router.post("/classify/severity", response_model=SeverityResponse, responses=OVERLOADED_RESPONSE)
async def severity_classification_endpoint(
    request: SeverityRequest,
) -> dict[str, Any]:
    """Classify a vulnerability description's severity.

    Request body: ``{"description": "<text>", "model": "<optional-model-id>"}``.

    The response includes the prediction (``severity``, ``confidence``)
    together with the model identifier and the Hugging Face commit SHA of
    the loaded snapshot (``model``, ``model_revision``), so callers can pin
    and audit which exact weights produced the result.
    """
    return await INFERENCE_GATE.run(classify_severity, request)


@router.post(
    "/classify/attack-techniques", response_model=AttackTechniquesResponse, responses=OVERLOADED_RESPONSE
)
async def attack_techniques_endpoint(
    request: AttackTechniquesRequest,
) -> dict[str, Any]:
    """Rank MITRE ATT&CK techniques for a vulnerability description.

    Request body: ``{"description": "<text>", "model": "<optional-model-id>",
    "top_k": <optional-int>}``.

    Multi-label classification: every technique in the model's vocabulary is
    scored independently (sigmoid) and the top-k are returned ranked by
    score, each with its ATT&CK ID, official name, score, and whether it
    clears the 0.5 prediction threshold. The response also carries the model
    identifier and Hugging Face snapshot SHA, like ``/classify/severity``.
    """
    return await INFERENCE_GATE.run(classify_attack_techniques, request)
