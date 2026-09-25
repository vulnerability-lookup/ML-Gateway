import os
import re

from fastapi import HTTPException, Request, status

"""
Operator switch to take endpoints out of service without a code change.

``ML_GATEWAY_DISABLED_ENDPOINTS`` lists route paths, comma-separated, as
they appear in the API documentation (``/classify/attack-techniques``,
``/retrieve/attack-biencoder/technique/{id}``, …). A call to a listed
endpoint is refused with 503 in the event loop, before any body is
handled by a model or the inference gate is consulted, so it costs the
gateway nothing. There is no ``Retry-After`` header: the refusal lasts
until the operator restarts the gateway without the entry. The variable is
read on every request, like the index token, so there is no cached state
to get out of step with the environment.
"""

DISABLED_ENDPOINTS_ENV = "ML_GATEWAY_DISABLED_ENDPOINTS"

_PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _normalise(path: str) -> str:
    """``/a/{technique_id}`` and ``/a/{id}`` name the same route."""
    return _PLACEHOLDER.sub("{}", path.strip().rstrip("/"))


def disabled_endpoints() -> set[str]:
    raw = os.environ.get(DISABLED_ENDPOINTS_ENV, "")
    return {_normalise(item) for item in raw.split(",") if item.strip()}


def require_enabled(request: Request) -> None:
    """FastAPI dependency: refuse routes listed in ``ML_GATEWAY_DISABLED_ENDPOINTS``."""
    route = request.scope.get("route")
    template = getattr(route, "path", request.url.path)
    if _normalise(template) in disabled_endpoints():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Endpoint {template} is disabled on this gateway ({DISABLED_ENDPOINTS_ENV}).",
        )
