from typing import Any

from fastapi import APIRouter

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

"""
This module sets up the API route(s) using FastAPI's APIRouter.
We define the request schema with Pydantic and the endpoint function:
"""

router = APIRouter()


@router.get("/")
async def root() -> str:
    return "OK"


# The classification endpoints are plain ``def``, not ``async def``: FastAPI
# then runs them in its threadpool, so the synchronous CPU-bound torch
# inference cannot block the event loop for every other request.


@router.post("/classify/severity", response_model=SeverityResponse)
def severity_classification_endpoint(
    request: SeverityRequest,
) -> dict[str, Any]:
    """Classify a vulnerability description's severity.

    Request body: ``{"description": "<text>", "model": "<optional-model-id>"}``.

    The response includes the prediction (``severity``, ``confidence``)
    together with the model identifier and the Hugging Face commit SHA of
    the loaded snapshot (``model``, ``model_revision``), so callers can pin
    and audit which exact weights produced the result.
    """
    return classify_severity(request)


@router.post("/classify/attack-techniques", response_model=AttackTechniquesResponse)
def attack_techniques_endpoint(
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
    return classify_attack_techniques(request)
