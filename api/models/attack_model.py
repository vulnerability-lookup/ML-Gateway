import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
This module defines a model wrapper for ATT&CK technique classification.

Unlike severity classification, this is a *multi-label* task: the model
scores every technique in its vocabulary independently (sigmoid, not
softmax), and the caller receives the full ranking rather than a single
argmax class. The label vocabulary is read from the model's own
``id2label`` config, so no per-model label registry is needed here.
"""

# Human-readable technique names (id -> name), extracted from the MITRE
# enterprise ATT&CK STIX bundle (attack-stix-data). Regenerate by dumping
# ``external_id: name`` for every ``attack-pattern`` object in
# ``enterprise-attack.json``.
_TECHNIQUE_NAMES_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "attack_technique_names.json"
)


@lru_cache(maxsize=1)
def technique_names() -> dict[str, str]:
    with open(_TECHNIQUE_NAMES_FILE, encoding="utf-8") as f:
        names: dict[str, str] = json.load(f)
    return names


class RankedTechnique(TypedDict):
    """One ranked technique returned by :meth:`AttackTechniqueClassifier.predict`."""

    technique: str
    name: str | None
    score: float
    predicted: bool


class AttackTechniqueClassifier:
    """Wraps a Hugging Face multi-label model for ATT&CK technique prediction.

    Attributes:
        model_name: Hugging Face repository identifier (e.g.
            ``CIRCL/vulnerability-attack-technique-classification-roberta-base``).
        labels: Ordered list of technique IDs matching the model's logits,
            taken from the model config's ``id2label``.
        revision: Commit SHA of the snapshot that was loaded from the Hugging
            Face Hub, or ``None`` if the source did not carry revision
            metadata (e.g. a local path). Returned to API clients so they
            can pin and audit which exact weights produced a prediction.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Disable dropout etc.
        # transformers stamps the resolved snapshot SHA onto
        # ``model.config._commit_hash`` during ``from_pretrained``; see
        # SeverityClassifier for the caveat about it being private API.
        self.revision: str | None = getattr(self.model.config, "_commit_hash", None)
        id_to_label = self.model.config.id2label
        self.labels: list[str] = [
            id_to_label[index] for index in sorted(id_to_label)
        ]

    def predict(self, description: str) -> list[RankedTechnique]:
        """Return every vocabulary technique ranked by sigmoid probability.

        Techniques with a probability of at least 0.5 carry
        ``predicted: True`` — the same threshold the trainer uses for its
        F1 metrics; the rest of the ranking shows how the model orders the
        remaining candidates.
        """
        inputs = self.tokenizer(
            description, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits[0])

        names = technique_names()
        ranked_indices = torch.argsort(probabilities, descending=True).tolist()
        ranking: list[RankedTechnique] = []
        for index in ranked_indices:
            # Derive ``predicted`` from the rounded score that clients see,
            # so the response never shows ``score: 0.5`` with
            # ``predicted: false`` (or the reverse) at the threshold.
            score = round(probabilities[index].item(), 4)
            ranking.append(
                {
                    "technique": self.labels[index],
                    "name": names.get(self.labels[index]),
                    "score": score,
                    "predicted": score >= 0.5,
                }
            )
        return ranking


# Multi-label models this endpoint may load. Labels come from each model's
# config, so unlike ``LABELS`` in severity_model.py this is a plain
# allow-list — its purpose is to keep the public endpoint from loading
# arbitrary Hub repositories.
ATTACK_MODELS = [
    "CIRCL/vulnerability-attack-technique-classification-roberta-base",
]

_model_cache: dict[str, AttackTechniqueClassifier] = {}


def get_attack_model_instance(model_name: str) -> AttackTechniqueClassifier:
    if model_name not in _model_cache:
        if model_name not in ATTACK_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        _model_cache[model_name] = AttackTechniqueClassifier(model_name)
    return _model_cache[model_name]


def preload_models() -> None:
    """Load every model in ``ATTACK_MODELS`` into the in-memory cache.

    Called at import time so that running gunicorn with ``--preload``
    populates the cache in the master process; forked workers then share
    the weights via copy-on-write instead of each loading their own copy.
    """
    for model_name in ATTACK_MODELS:
        get_attack_model_instance(model_name)
