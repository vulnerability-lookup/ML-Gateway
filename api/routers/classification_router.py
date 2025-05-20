from fastapi import APIRouter
from pydantic import BaseModel

from api.services.classification_service import classify_severity

"""
This module sets up the API route(s) using FastAPI’s APIRouter.
We define the request schema with Pydantic and the endpoint function:
"""

router = APIRouter()

class SeverityRequest(BaseModel):
    description: str

@router.post("/classify/severity")
async def severity_endpoint(request: SeverityRequest):
    """
    Endpoint to classify vulnerability severity.
    Expects JSON: {"description": "<text>"}
    Returns JSON: {"severity": "<label>", "confidence": <float>}
    """
    result = classify_severity(request.description)
    return result
