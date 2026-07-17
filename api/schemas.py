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


class AttackTechniquesRequest(BaseModel):
    description: str
    model: str = Field(
        default="CIRCL/vulnerability-attack-technique-classification-roberta-base",
        description=(
            "Hugging Face model identifier to use for ATT&CK technique "
            "classification."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top-ranked techniques to return.",
    )


class TechniqueScore(BaseModel):
    technique: str = Field(
        description="MITRE ATT&CK technique ID (e.g. 'T1190').",
    )
    name: str | None = Field(
        description=(
            "Official ATT&CK technique name (e.g. 'Exploit Public-Facing "
            "Application'). ``None`` if the ID is not in the bundled "
            "ATT&CK name table."
        ),
    )
    score: float = Field(
        description=(
            "Sigmoid probability for this technique, rounded to four "
            "decimals. Multi-label: scores are independent and do not sum "
            "to 1."
        ),
    )
    predicted: bool = Field(
        description=(
            "True when the score is at least 0.5 — the threshold the "
            "model's training metrics use for a positive prediction."
        ),
    )


class AttackTechniquesResponse(BaseModel):
    """Response payload for ``POST /classify/attack-techniques``.

    Carries the same model provenance as :class:`SeverityResponse` (model
    identifier and Hugging Face snapshot SHA) so callers can pin and audit
    which exact weights produced the ranking.
    """

    model_config = ConfigDict(protected_namespaces=())

    techniques: list[TechniqueScore] = Field(
        description=(
            "Top-k techniques ranked by score, best first. Empty when "
            "classification could not be performed."
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
