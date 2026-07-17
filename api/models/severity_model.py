from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
This module defines a model wrapper for the severity classification.
"""


class Prediction(TypedDict):
    """Inference result returned by :meth:`SeverityClassifier.predict`."""

    severity: str
    confidence: float


class SeverityClassifier:
    """Wraps a Hugging Face sequence-classification model for severity prediction.

    Attributes:
        model_name: Hugging Face repository identifier (e.g.
            ``CIRCL/vulnerability-severity-classification-RoBERTa-base``).
        labels: Ordered list of class labels matching the model's logits.
        revision: Commit SHA of the snapshot that was loaded from the Hugging
            Face Hub, or ``None`` if the source did not carry revision
            metadata (e.g. a local path). Returned to API clients so they
            can pin and audit which exact weights produced a prediction.
    """

    def __init__(self, model_name: str, labels: list[str]):
        self.model_name = model_name
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Disable dropout etc.
        # transformers stamps the resolved snapshot SHA onto
        # ``model.config._commit_hash`` during ``from_pretrained``, for both
        # fresh downloads and cached snapshots. The leading underscore marks
        # it as private API, but it has been the de facto way to surface the
        # loaded revision across recent major versions of transformers.
        self.revision: str | None = getattr(
            self.model.config, "_commit_hash", None
        )

    def predict(self, description: str) -> "Prediction":
        inputs = self.tokenizer(
            description, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get top class index and confidence. ``Tensor.item()`` is typed as
        # ``int | float``; argmax always yields an integral tensor.
        predicted_index = int(torch.argmax(probabilities, dim=-1).item())
        return {
            "severity": self.labels[predicted_index],
            "confidence": round(probabilities[0][predicted_index].item(), 4),
        }


# Dictionary of known models
_model_cache: dict[str, SeverityClassifier] = {}

# Label sets can vary by language or domain
LABELS = {
    "CIRCL/vulnerability-severity-classification-RoBERTa-base": [
        "Low",
        "Medium",
        "High",
        "Critical",
    ],
    "CIRCL/vulnerability-severity-classification-chinese-macbert-base": [
        "低",
        "中",
        "高",
    ],
    "CIRCL/vulnerability-severity-classification-russian-ruRoberta-large": [
        "Low",
        "Medium",
        "High",
        "Critical",
    ],
}


def get_model_instance(model_name: str) -> SeverityClassifier:
    if model_name not in _model_cache:
        labels = LABELS.get(model_name)
        if not labels:
            raise ValueError(f"Unknown model: {model_name}")
        _model_cache[model_name] = SeverityClassifier(model_name, labels)
    return _model_cache[model_name]


def preload_models() -> None:
    """Load every model in ``LABELS`` into the in-memory cache.

    Called at import time so that running gunicorn with ``--preload`` populates
    the cache in the master process; forked workers then share the weights via
    copy-on-write instead of each loading their own copy.
    """
    for model_name in LABELS:
        get_model_instance(model_name)
