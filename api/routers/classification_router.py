from fastapi import APIRouter

from api.schemas import SeverityRequest
from api.services.classification_service import classify_severity

"""
This module sets up the API route(s) using FastAPI's APIRouter.
We define the request schema with Pydantic and the endpoint function:
"""

router = APIRouter()


@router.get("/")
async def root():
    return "OK"


@router.post("/classify/severity")
async def severity_classification_endpoint(request: SeverityRequest):
    """
    Endpoint to classify vulnerability severity.
    Expects JSON: {"description": "\<text\>"}
    Returns JSON: {"severity": "\<label\>", "confidence": \<float\>}
    """
    return classify_severity(request)
