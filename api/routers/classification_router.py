from typing import Any

from fastapi import APIRouter

from api.schemas import SeverityRequest, SeverityResponse
from api.services.classification_service import classify_severity

"""
This module sets up the API route(s) using FastAPI's APIRouter.
We define the request schema with Pydantic and the endpoint function:
"""

router = APIRouter()


@router.get("/")
async def root() -> str:
    return "OK"


@router.post("/classify/severity", response_model=SeverityResponse)
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
    return classify_severity(request)
