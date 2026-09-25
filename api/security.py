import hmac
import os

from fastapi import Header, HTTPException, status

"""
Authentication for the endpoints that change state.

The read endpoints stay open: they only serve rankings and never write. The
index endpoint appends to the on-disk store, so it requires a bearer token
shared with the ingesting Vulnerability-Lookup instance. The token comes
from the environment and is read on every request, so it can never be
"unset but accepting": with no token configured the endpoint refuses every
call with 503 until the operator sets one.
"""

INDEX_TOKEN_ENV = "ML_GATEWAY_INDEX_TOKEN"


async def require_index_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: ``Authorization: Bearer <ML_GATEWAY_INDEX_TOKEN>``.

    ``async`` so it runs in the event loop rather than costing a threadpool
    hop per request; it only compares strings.
    """
    expected = os.environ.get(INDEX_TOKEN_ENV, "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Indexing is disabled: {INDEX_TOKEN_ENV} is not set on the gateway. "
                "Set it to a shared secret and send it as a bearer token."
            ),
        )
    scheme, _, presented = (authorization or "").partition(" ")
    # Constant-time comparison so response timing reveals nothing about the
    # token; the scheme check is cheap and not secret.
    if (
        scheme.lower() != "bearer"
        or not presented.strip()
        or not hmac.compare_digest(presented.strip().encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
