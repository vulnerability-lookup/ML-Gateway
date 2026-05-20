from pydantic import BaseModel, ConfigDict, Field


class SeverityRequest(BaseModel):
    description: str
    model: str = Field(
        default="CIRCL/vulnerability-severity-classification-RoBERTa-base",
        description="Hugging Face model identifier to use for classification.",
    )


class SeverityResponse(BaseModel):
    """Response payload for ``POST /classify/severity``.

    Beyond the prediction itself, the response carries the *provenance* of
    the model that produced it: the Hugging Face repository identifier and
    the commit SHA of the snapshot that was loaded. This lets callers pin
    and audit which exact weights generated a given classification — useful
    when models are retrained or replaced on the Hub.
    """

    # Pydantic v2 reserves the ``model_`` prefix for its own internals and
    # warns about user-defined fields colliding with it. Disable that
    # protection so the API can expose a clean, descriptive
    # ``model_revision`` field.
    model_config = ConfigDict(protected_namespaces=())

    severity: str | None = Field(
        description=(
            "Predicted severity label (e.g. 'Low', 'Medium', 'High', "
            "'Critical'). ``None`` when classification could not be "
            "performed."
        ),
    )
    confidence: float = Field(
        description=(
            "Softmax probability of the predicted class, rounded to four "
            "decimals. ``0.0`` when classification fails."
        ),
    )
    model: str = Field(
        description=(
            "Hugging Face model identifier that produced this prediction."
        ),
    )
    model_revision: str | None = Field(
        description=(
            "Commit SHA of the model snapshot resolved at load time on the "
            "Hugging Face Hub. ``None`` when the snapshot does not carry "
            "revision metadata (for example, models loaded from a local "
            "path)."
        ),
    )
    error: str | None = Field(
        default=None,
        description=(
            "Human-readable error message when classification could not be "
            "performed. Absent on success."
        ),
    )
