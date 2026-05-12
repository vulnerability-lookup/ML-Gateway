from threading import Lock

from cachetools import TTLCache, cached

from api.models.severity_model import get_model_instance
from api.schemas import SeverityRequest

"""
The service layer contains the business logic for classification.
It receives the input text, selects the appropriate model, and formats the output.
"""

# Per-worker TTL cache for inference results. Vulnerability descriptions repeat
# often in practice (re-enrichment, retries, shared CVE text), so caching by
# (model, description) avoids re-running the model for identical inputs. The
# TTL bounds how long stale entries can sit in memory between requests, in
# addition to the size cap.
_PREDICT_CACHE_SIZE = 10_000
_PREDICT_CACHE_TTL_SECONDS = 3600  # 1 hour

_predict_cache: TTLCache = TTLCache(
    maxsize=_PREDICT_CACHE_SIZE, ttl=_PREDICT_CACHE_TTL_SECONDS
)
# cachetools holds this lock only around cache reads and writes, not around
# the wrapped inference call, so concurrent requests still run in parallel.
_predict_cache_lock = Lock()


@cached(_predict_cache, lock=_predict_cache_lock)
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
