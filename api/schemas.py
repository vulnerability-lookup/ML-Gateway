from pydantic import BaseModel, ConfigDict, Field, model_validator


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


DEFAULT_BIENCODER_MODEL = "CIRCL/vulnerability-attack-technique-biencoder"


class IndexItem(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^\S+$",
        description=(
            "Identifier the vector is stored under (e.g. 'CVE-2021-44077'). "
            "Re-sending an ID replaces its vector."
        ),
    )
    text: str = Field(
        min_length=1,
        description="Vulnerability description to embed.",
    )


class IndexRequest(BaseModel):
    """Request payload for ``POST /index/attack-biencoder``."""

    items: list[IndexItem] = Field(
        max_length=1000,
        description="Vulnerabilities to embed and upsert into the index.",
    )
    model: str = Field(
        default=DEFAULT_BIENCODER_MODEL,
        description="Hugging Face bi-encoder identifier whose index to write.",
    )


class IndexResponse(BaseModel):
    """Response payload for ``POST /index/attack-biencoder``."""

    model_config = ConfigDict(protected_namespaces=())

    indexed: int = Field(
        description="Number of items embedded and upserted by this call.",
    )
    count: int = Field(
        description="Number of distinct IDs in the index after this call.",
    )
    model: str = Field(
        description="Hugging Face model identifier whose index was written.",
    )
    model_revision: str | None = Field(
        description=(
            "Commit SHA of the model snapshot the index vectors were computed "
            "with. The index is only valid for this revision."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when nothing was indexed.",
    )


class RetrievedVulnerability(BaseModel):
    id: str = Field(description="Identifier the vulnerability was indexed under.")
    score: float = Field(
        description=(
            "Similarity score, rounded to four decimals. For technique "
            "retrieval this is the training-time probability "
            "sigmoid(logit_scale * cosine + logit_bias); for related-"
            "vulnerability retrieval it is the plain cosine."
        ),
    )


class TechniqueRetrievalResponse(BaseModel):
    """Response payload for ``GET /retrieve/attack-biencoder/technique/{id}``.

    This is a similarity search, not a classification: the paper only
    measures the CVE -> technique direction, so the ranking should be
    presented as a search aid.
    """

    model_config = ConfigDict(protected_namespaces=())

    technique: str = Field(description="MITRE ATT&CK technique ID (e.g. 'T1190').")
    name: str | None = Field(
        description=(
            "Official ATT&CK technique name. ``None`` if the ID is not in "
            "the bundled ATT&CK name table."
        ),
    )
    in_vocabulary: bool = Field(
        description=(
            "True when the technique is one the bi-encoder was trained on. "
            "Other techniques are scored from their official ATT&CK text "
            "and rank noticeably worse; interfaces should flag them."
        ),
    )
    results: list[RetrievedVulnerability] = Field(
        description="Top-k indexed vulnerabilities for the technique, best first.",
    )
    model: str = Field(description="Hugging Face model identifier used for the search.")
    model_revision: str | None = Field(
        description="Commit SHA of the model snapshot the index was built with.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when the search could not run.",
    )


class RelatedRequest(BaseModel):
    """Request payload for ``POST /retrieve/attack-biencoder/related``.

    Exactly one of ``id`` (an indexed vulnerability) or ``text`` (a free
    description, embedded on the fly) must be given.
    """

    id: str | None = Field(
        default=None,
        description="Identifier of an indexed vulnerability to search around.",
    )
    text: str | None = Field(
        default=None,
        description="Vulnerability description to search around.",
    )
    model: str = Field(
        default=DEFAULT_BIENCODER_MODEL,
        description="Hugging Face bi-encoder identifier whose index to search.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of nearest vulnerabilities to return.",
    )

    @model_validator(mode="after")
    def _exactly_one_query(self) -> "RelatedRequest":
        if (self.id is None) == (self.text is None):
            raise ValueError("Provide exactly one of 'id' or 'text'.")
        if self.id is not None and not self.id.strip():
            raise ValueError("'id' must not be empty.")
        if self.text is not None and not self.text.strip():
            raise ValueError("'text' must not be empty.")
        return self


class RelatedResponse(BaseModel):
    """Response payload for ``POST /retrieve/attack-biencoder/related``.

    Nearest neighbours by plain cosine in the bi-encoder space — a search
    aid with no measured accuracy, not a classification.
    """

    model_config = ConfigDict(protected_namespaces=())

    results: list[RetrievedVulnerability] = Field(
        description=(
            "Nearest indexed vulnerabilities, best first. When searching by "
            "``id`` the queried vulnerability itself is excluded."
        ),
    )
    model: str = Field(description="Hugging Face model identifier used for the search.")
    model_revision: str | None = Field(
        description="Commit SHA of the model snapshot the index was built with.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when the search could not run.",
    )
