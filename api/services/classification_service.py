from functools import lru_cache

from api.models.severity_model import get_model_instance
from api.schemas import SeverityRequest

"""
The service layer contains the business logic for classification.
It receives the input text, selects the appropriate model, and formats the output.
"""

# Per-worker LRU for inference results. Vulnerability descriptions repeat often
# in practice (re-enrichment, retries, shared CVE text), so caching by
# (model, description) avoids re-running the model for identical inputs.
_PREDICT_CACHE_SIZE = 10_000


@lru_cache(maxsize=_PREDICT_CACHE_SIZE)
def _cached_predict(model_name: str, description: str) -> dict:
    return get_model_instance(model_name).predict(description)


def classify_severity(request: SeverityRequest):
    """
    Classify the severity of a vulnerability description.
    Returns a dict with 'severity' and 'confidence'.
    """
    try:
        output = _cached_predict(request.model, request.description)
    except ValueError as e:
        return {
            "severity": None,
            "confidence": 0.0,
            "error": str(e),
        }

    if output:
        # Extract label and score from the model output
        label = output.get("severity")
        score = output.get("confidence", 0.0)
    else:
        label, score = None, 0.0
    return {"severity": label, "confidence": float(score)}
