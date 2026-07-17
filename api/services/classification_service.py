from threading import Lock
from typing import Any

from cachetools import TTLCache, cached

from api.models.attack_model import TechniqueScore, get_attack_model_instance
from api.models.severity_model import Prediction, get_model_instance
from api.schemas import AttackTechniquesRequest, SeverityRequest

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
def _cached_predict(model_name: str, description: str) -> Prediction:
    return get_model_instance(model_name).predict(description)


def classify_severity(request: SeverityRequest) -> dict[str, Any]:
    """Classify the severity of a vulnerability description.

    Returns a dict shaped like :class:`api.schemas.SeverityResponse`. In
    addition to the prediction (``severity``, ``confidence``), the result
    carries the provenance of the model that produced it: the model
    identifier and the Hugging Face commit SHA of the loaded snapshot,
    so callers can identify exactly which weights were used.
    """
    # Resolve the classifier up front so we can surface ``model`` and
    # ``model_revision`` even on the error path, and so unknown-model errors
    # don't go through the prediction cache.
    try:
        classifier = get_model_instance(request.model)
    except ValueError as e:
        return {
            "severity": None,
            "confidence": 0.0,
            "model": request.model,
            "model_revision": None,
            "error": str(e),
        }

    output = _cached_predict(request.model, request.description)
    return {
        "severity": output.get("severity"),
        "confidence": float(output.get("confidence", 0.0)),
        "model": classifier.model_name,
        "model_revision": classifier.revision,
    }


# Separate cache for the multi-label endpoint: cachetools keys entries by
# call arguments only (not by function), so sharing ``_predict_cache``
# would let a (model, description) pair collide across endpoints.
_attack_predict_cache: TTLCache = TTLCache(
    maxsize=_PREDICT_CACHE_SIZE, ttl=_PREDICT_CACHE_TTL_SECONDS
)
_attack_predict_cache_lock = Lock()


@cached(_attack_predict_cache, lock=_attack_predict_cache_lock)
def _cached_predict_attack(
    model_name: str, description: str
) -> list[TechniqueScore]:
    return get_attack_model_instance(model_name).predict(description)


def classify_attack_techniques(request: AttackTechniquesRequest) -> dict[str, Any]:
    """Rank ATT&CK techniques for a vulnerability description.

    Returns a dict shaped like :class:`api.schemas.AttackTechniquesResponse`.
    The full vocabulary ranking is computed (and cached) once per
    (model, description); ``top_k`` only slices the cached result, so
    varying it does not re-run inference.
    """
    try:
        classifier = get_attack_model_instance(request.model)
    except ValueError as e:
        return {
            "techniques": [],
            "model": request.model,
            "model_revision": None,
            "error": str(e),
        }

    ranking = _cached_predict_attack(request.model, request.description)
    return {
        "techniques": ranking[: request.top_k],
        "model": classifier.model_name,
        "model_revision": classifier.revision,
    }
